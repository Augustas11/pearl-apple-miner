// F1 K3-NA prototype host (SPEC v0.2 §5.1, §8 P1, §9 F1).
//
// Subcommands:
//   run JOBDIR            run the production K3 kernel (variant 2) on JOBDIR/{job.json,A.bin,B.bin} for every case in
//                         job.json; writes JOBDIR/out_<case>.bin = ctr[2] ++ block slots ++ share slots (u32 LE, incl.
//                         guard slots). Correctness only: no GPU lock.
//   perf [MxNxK ...] [rounds=R]
//                         paired alternating rounds: int8bench (C stored, context), baseline (C store disabled),
//                         V6 fold-only, full K3. Takes /tmp/pmm-gpu-bench.lock per shape batch.
import Metal
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let SLOT_WORDS = 26
let GUARD_SLOTS = 8
let CANARY: UInt32 = 0xA5A5A5A5
let srcPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("k3.metal").path

func sh(_ cmd: String) -> String {
  let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh"); p.arguments = ["-c", cmd]
  let pipe = Pipe(); p.standardOutput = pipe; try? p.run(); p.waitUntilExit()
  return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
}

func pipeline(variant: Int) -> MTLComputePipelineState {
  let src: String
  do { src = try String(contentsOfFile: srcPath, encoding: .utf8) } catch { print("cannot read \(srcPath): \(error)"); exit(1) }
  let o = MTLCompileOptions(); o.languageVersion = .version4_0
  o.preprocessorMacros = ["K3_VARIANT": NSNumber(value: variant)]
  do {
    let lib = try dev.makeLibrary(source: src, options: o)
    return try dev.makeComputePipelineState(function: lib.makeFunction(name: "k3")!)
  } catch { print("COMPILE/PIPELINE ERROR (variant \(variant)):\n\(error)"); exit(1) }
}

struct Params {   // mirrors K3Params in k3.metal (32 u32 words)
  var w = [UInt32](repeating: 0, count: 32)
  init(M: Int, N: Int, K: Int, capBlock: Int, capShare: Int, key: [UInt32], boundBlock: [UInt32], boundShare: [UInt32]) {
    w[0] = UInt32(M); w[1] = UInt32(N); w[2] = UInt32(K); w[3] = UInt32(capBlock); w[4] = UInt32(capShare)
    for i in 0..<8 { w[8 + i] = key[i]; w[16 + i] = boundBlock[i]; w[24 + i] = boundShare[i] }
  }
}

struct Job {
  var M, N, K: Int
  var a, b: MTLBuffer
  var sink = dev.makeBuffer(length: 16)!
  var c: MTLBuffer
}

func encode(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ job: Job, _ prm: Params,
            ctr: MTLBuffer, blk: MTLBuffer, shr: MTLBuffer) {
  precondition(job.M % 128 == 0 && job.N % 64 == 0 && job.K % 128 == 0, "M%128, N%64, K%128 must be 0")
  var p = prm.w
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps)
  e.setBuffer(job.a, offset: 0, index: 0); e.setBuffer(job.b, offset: 0, index: 1)
  e.setBytes(&p, length: 128, index: 2)
  e.setBuffer(ctr, offset: 0, index: 3); e.setBuffer(blk, offset: 0, index: 4); e.setBuffer(shr, offset: 0, index: 5)
  e.setBuffer(job.sink, offset: 0, index: 6); e.setBuffer(job.c, offset: 0, index: 7)
  e.dispatchThreadgroups(MTLSize(width: job.N / 64, height: job.M / 128, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.endEncoding()
}

func runOnce(_ ps: MTLComputePipelineState, _ job: Job, _ prm: Params, ctr: MTLBuffer, blk: MTLBuffer, shr: MTLBuffer) -> Double {
  let cb = queue.makeCommandBuffer()!
  encode(cb, ps, job, prm, ctr: ctr, blk: blk, shr: shr)
  cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print("GPU ERROR: \(err)"); exit(1) }
  return cb.gpuEndTime - cb.gpuStartTime
}

// MARK: - run (correctness)
func readFile(_ p: String) -> Data {
  guard let d = FileManager.default.contents(atPath: p) else { print("cannot read \(p)"); exit(1) }
  return d
}
func u32s(_ any: Any?) -> [UInt32] { (any as! [NSNumber]).map { $0.uint32Value } }

func runJob(_ dir: String) {
  let js = try! JSONSerialization.jsonObject(with: readFile(dir + "/job.json")) as! [String: Any]
  let M = (js["m"] as! NSNumber).intValue, N = (js["n"] as! NSNumber).intValue, K = (js["k"] as! NSNumber).intValue
  let key = u32s(js["key"])
  let A = readFile(dir + "/A.bin"), B = readFile(dir + "/B.bin")
  precondition(A.count == M * K && B.count == K * N, "operand sizes do not match job.json")
  let job = A.withUnsafeBytes { pa in B.withUnsafeBytes { pb in
    Job(M: M, N: N, K: K, a: dev.makeBuffer(bytes: pa.baseAddress!, length: A.count)!,
        b: dev.makeBuffer(bytes: pb.baseAddress!, length: B.count)!, c: dev.makeBuffer(length: 16)!) } }
  let ps = pipeline(variant: 2)
  for cs in js["cases"] as! [[String: Any]] {
    let name = cs["name"] as! String
    let capB = (cs["cap_block"] as! NSNumber).intValue, capS = (cs["cap_share"] as! NSNumber).intValue
    let prm = Params(M: M, N: N, K: K, capBlock: capB, capShare: capS, key: key,
                     boundBlock: u32s(cs["bound_block"]), boundShare: u32s(cs["bound_share"]))
    let nb = (capB + GUARD_SLOTS) * SLOT_WORDS, ns = (capS + GUARD_SLOTS) * SLOT_WORDS
    let ctr = dev.makeBuffer(length: 8)!, blk = dev.makeBuffer(length: nb * 4)!, shr = dev.makeBuffer(length: ns * 4)!
    memset(ctr.contents(), 0, 8)
    for (buf, n) in [(blk, nb), (shr, ns)] { let p = buf.contents().bindMemory(to: UInt32.self, capacity: n); for i in 0..<n { p[i] = CANARY } }
    let t = runOnce(ps, job, prm, ctr: ctr, blk: blk, shr: shr)
    var out = Data(bytes: ctr.contents(), count: 8)
    out.append(Data(bytes: blk.contents(), count: nb * 4)); out.append(Data(bytes: shr.contents(), count: ns * 4))
    FileManager.default.createFile(atPath: dir + "/out_\(name).bin", contents: out)
    let c = ctr.contents().bindMemory(to: UInt32.self, capacity: 2)
    print(String(format: "  case %-28@ block_ctr=%u share_ctr=%u  gpu %.2f ms", name as NSString, c[0], c[1], t * 1e3))
  }
}

// MARK: - perf
let LOCK = "/tmp/pmm-gpu-bench.lock"
func loadavg1() -> Double {
  let s = sh("sysctl -n vm.loadavg").replacingOccurrences(of: "{", with: "").split(separator: " ")
  return Double(s.first ?? "0") ?? 0
}
func envLine(_ tag: String) -> String {
  "  [\(tag)] \(sh("date -u +%FT%TZ")) loadavg \(sh("sysctl -n vm.loadavg")) | power: \(sh("pmset -g batt | head -2 | tr '\\n' ' ' | tr '\\t' ' '"))"
}
var flags: [String] = []
func acquireLock(_ label: String) {
  var waited = 0
  while loadavg1() > 8 && waited < 1800 {
    if waited % 300 == 0 { print("  [load] loadavg1 \(loadavg1()) > 8, waiting (waited \(waited) s)") }
    sleep(30); waited += 30
  }
  if loadavg1() > 8 {
    let f = "FLAG \(label): loadavg1 \(loadavg1()) > 8 after 30 min wait; proceeding"
    print("  [load] " + f); flags.append(f)
  }
  var lw = 0
  while mkdir(LOCK, 0o755) != 0 {
    if lw % 60 == 0 { print("  [lock] \(LOCK) busy, retrying every 15 s (waited \(lw) s)") }
    sleep(15); lw += 15
  }
  print("  [lock] acquired after \(lw) s")
  print(envLine("before"))
}
func releaseLock() {
  print(envLine("after"))
  rmdir(LOCK)
  print("  [lock] released")
}

func median(_ x: [Double]) -> Double { let s = x.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count/2 - 1] + s[s.count/2]) / 2 }
// Percentile bootstrap 90% CI of the median (fixed-seed xorshift, 20000 resamples).
func bootCI(_ x: [Double]) -> (Double, Double) {
  var st: UInt64 = 0x9E3779B97F4A7C15
  func nxt() -> UInt64 { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st }
  var meds: [Double] = []
  for _ in 0..<20000 { var r: [Double] = []; for _ in 0..<x.count { r.append(x[Int(nxt() % UInt64(x.count))]) }; meds.append(median(r)) }
  meds.sort()
  return (meds[Int(0.05 * Double(meds.count))], meds[Int(0.95 * Double(meds.count)) - 1])
}
func randOperands(_ n: Int) -> MTLBuffer {
  let buf = dev.makeBuffer(length: n)!
  arc4random_buf(buf.contents(), n)
  let p = buf.contents().bindMemory(to: Int8.self, capacity: n)
  for i in 0..<n where p[i] == -128 { p[i] = 127 }   // noised range [-127, 127]
  return buf
}
// bound with all bits below 2^e set (U256 LE words)
func boundBelowPow2(_ e: Int) -> [UInt32] {
  var w = [UInt32](repeating: 0, count: 8)
  for i in 0..<8 { let lo = i * 32; if e >= lo + 32 { w[i] = 0xFFFFFFFF } else if e > lo { w[i] = (1 << UInt32(e - lo)) - 1 } }
  return w
}

func perf(_ shapes: [(Int, Int, Int)], rounds: Int) {
  print("== F1 perf: device \(dev.name), \(sh("sysctl -n hw.model")), \(sh("sysctl -n machdep.cpu.brand_string")), macOS \(sh("sw_vers -productVersion")) (\(sh("sw_vers -buildVersion"))) ==")
  print("variants: int8bench = bench/int8bench.swift kernel (C stored, context only); base = same matmul2d 128x64 single run(),")
  print("          C store disabled (cooperative-tensor destination + conditional sink) [P1 BASELINE]; fold = K loop RK=128 + V6 fold;")
  print("          k3 = fold + keyed BLAKE3 + 2x U256 compare + atomic found slots (block cap 4, share cap 64)")
  print("timing: GPU timestamps per command buffer; 1 warm-up each; \(rounds) paired rounds, order reversed on odd rounds")
  let names = ["int8bench", "base", "fold", "k3"]
  let vids = [3, 0, 1, 2]
  let pss = vids.map { pipeline(variant: $0) }   // compile before taking the lock
  for (M, N, K) in shapes {
    let tiles = M * N / 64
    let log2t = Int(log2(Double(tiles)).rounded())
    let shareE = 256 + 2 - log2t   // ~4 expected share finds per job
    let key: [UInt32] = (0..<8).map { _ in arc4random() }
    let prm = Params(M: M, N: N, K: K, capBlock: 4, capShare: 64, key: key, boundBlock: boundBelowPow2(200), boundShare: boundBelowPow2(shareE))
    let job = Job(M: M, N: N, K: K, a: randOperands(M * K), b: randOperands(K * N), c: dev.makeBuffer(length: M * N * 4)!)
    let ctr = dev.makeBuffer(length: 8)!, blk = dev.makeBuffer(length: 4 * SLOT_WORDS * 4)!, shr = dev.makeBuffer(length: 64 * SLOT_WORDS * 4)!
    let ops = 2.0 * Double(M) * Double(N) * Double(K)
    print("\nshape \(M)x\(N)x\(K)  ops=2mnk=\(String(format: "%.3e", ops))  tiles=\(tiles)  share bound=2^\(shareE)-1 (E[shares/job]~4)  block bound=2^200-1")
    acquireLock("\(M)x\(N)x\(K)")
    for ps in pss { memset(ctr.contents(), 0, 8); _ = runOnce(ps, job, prm, ctr: ctr, blk: blk, shr: shr) }
    var t = [[Double]](repeating: [], count: 4)
    var shares: [UInt32] = [], blocks: [UInt32] = []
    for r in 0..<rounds {
      let order = r % 2 == 0 ? Array(0..<4) : Array((0..<4).reversed())
      for i in order {
        memset(ctr.contents(), 0, 8)
        t[i].append(runOnce(pss[i], job, prm, ctr: ctr, blk: blk, shr: shr))
        if vids[i] == 2 { let c = ctr.contents().bindMemory(to: UInt32.self, capacity: 2); blocks.append(c[0]); shares.append(c[1]) }
      }
    }
    releaseLock()
    print("  variant      median TOPS   min TOPS   max TOPS   raw GPU ms per round")
    for i in 0..<4 {
      let x = t[i]
      print(String(format: "  %-10@ %10.2f %10.2f %10.2f   ", names[i] as NSString, ops / median(x) / 1e12, ops / x.max()! / 1e12, ops / x.min()! / 1e12)
            + x.map { String(format: "%.2f", $0 * 1e3) }.joined(separator: " "))
    }
    func ratioLine(_ label: String, _ num: Int, _ den: Int) {   // throughput ratio num/den per round = t_den / t_num
      let r = (0..<rounds).map { t[den][$0] / t[num][$0] }
      let (lo, hi) = bootCI(r)
      print(String(format: "  %-34@ median %.4f  90%% CI [%.4f, %.4f] half-width %.4f  (min %.4f max %.4f)",
                   label as NSString, median(r), lo, hi, (hi - lo) / 2, r.min()!, r.max()!))
    }
    ratioLine("P1 k3/base (throughput)", 3, 1)
    ratioLine("fold/base (throughput)", 2, 1)
    ratioLine("k3/fold (throughput)", 3, 2)
    ratioLine("base/int8bench (throughput)", 1, 0)
    let ov = (0..<rounds).map { (t[3][$0] / t[2][$0] - 1) * 100 }
    let (ol, oh) = bootCI(ov)
    print(String(format: "  hash+compare+atomics overhead (t_k3/t_fold-1): median %+.2f%%  90%% CI [%+.2f%%, %+.2f%%]", median(ov), ol, oh))
    print("  k3 find counters per run: share \(shares.map(String.init).joined(separator: ",")) | block \(blocks.map(String.init).joined(separator: ","))")
  }
  if !flags.isEmpty { print("\nFLAGS:"); flags.forEach { print("  " + $0) } }
}

// MARK: - main
let args = Array(CommandLine.arguments.dropFirst())
switch args.first ?? "" {
case "run":
  guard args.count == 2 else { print("usage: k3 run JOBDIR"); exit(2) }
  print("== K3 run \(args[1]) (device \(dev.name)) ==")
  runJob(args[1])
case "perf":
  var rounds = 31
  var shapes: [(Int, Int, Int)] = []
  for a in args.dropFirst() {
    if a.hasPrefix("rounds=") { rounds = Int(a.dropFirst(7))! } else {
      let p = a.split(separator: "x").map { Int($0)! }; shapes.append((p[0], p[1], p[2]))
    }
  }
  if shapes.isEmpty { shapes = [(4096, 4096, 4096), (8192, 8192, 4096)] }
  precondition(rounds >= 15, "F1 needs >= 15 paired rounds")
  perf(shapes, rounds: rounds)
default:
  print("usage: k3 run JOBDIR | perf [MxNxK ...] [rounds=R]")
}
