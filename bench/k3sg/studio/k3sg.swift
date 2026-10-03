// K3-SG host (SPEC §5.3 / §5.4 / §8): fp32 simdgroup_matrix Pearl v3 mining kernel for Apple7-9 (M1-M4).
// Structure follows bench/f1_k3/k3.swift (same job-dir format, slot layout, perf protocol).
//
// Subcommands (cfg = BMxBNxBKxWMxWNxPF; PF = prefetch mode, see k3sg.metal):
//   probe                       simdgroup_matrix lane-layout probe + Pearl PeriodicPattern legality; exit 1 on failure
//   run JOBDIR [cfgs=a,b]       production kernel (variant 3) on JOBDIR/{job.json,A.bin,B.bin}, every case, every cfg;
//                               writes JOBDIR/out_<case>[__<cfg>].bin; if JOBDIR/tiles.bin exists (oracle tiles) every
//                               case is checked bit-exact here; exit 1 on any mismatch. No GPU lock.
//   perf [MxNxK ...] [rounds=R] [cfg=C] [nolock] [mlx=PYTHON] [noint8bench]
//                               paired alternating rounds: int8bench (Metal 4 matmul2d, C stored), MLX fp32 matmul
//                               (subprocess, wall time), base_f32 (P3 baseline), base_i8, fold, k3
//   sweep MxNxK [rounds=R] cfgs=a,b,... [nolock]   k3 only, alternating; prints "BEST <cfg>"
//   sustain SECS MxNxK [cfg=C] [nolock]            k3 back to back; one line per 10 s window
import Metal
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
guard let dev = MTLCreateSystemDefaultDevice() else { print("no Metal device"); exit(1) }
let queue = dev.makeCommandQueue()!
let SLOT_WORDS = 26
let GUARD_SLOTS = 8
let CANARY: UInt32 = 0xA5A5A5A5
let DEFAULT_CFG = "64x64x16x2x2x1"
let ROWS_PATTERN = [0, 8, 16, 24]
let COLS_PATTERN = [0, 1, 8, 9, 16, 17, 24, 25]
let exeDir: URL = {
  let p = Bundle.main.executablePath ?? CommandLine.arguments[0]
  return URL(fileURLWithPath: p).resolvingSymlinksInPath().deletingLastPathComponent()
}()
func srcPath(_ name: String) -> String {
  if let d = ProcessInfo.processInfo.environment["K3SG_SRC_DIR"] { return d + "/" + name }
  return exeDir.appendingPathComponent(name).path
}

func sh(_ cmd: String) -> String {
  let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh"); p.arguments = ["-c", cmd]
  let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe; try? p.run(); p.waitUntilExit()
  return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
}

struct Cfg: CustomStringConvertible {
  var BM, BN, BK, WM, WN, PF: Int
  var NT: Int { 32 * WM * WN }
  var description: String { "\(BM)x\(BN)x\(BK)x\(WM)x\(WN)x\(PF)" }
  init(_ s: String) {
    let p = s.split(separator: "x").map { Int($0)! }
    precondition(p.count == 6, "cfg must be BMxBNxBKxWMxWNxPF, got \(s)")
    BM = p[0]; BN = p[1]; BK = p[2]; WM = p[3]; WN = p[4]; PF = p[5]
    precondition(BM == 32 * WM && BN == 32 * WN && 128 % BK == 0 && BK % 8 == 0 && (0...2).contains(PF), "illegal cfg \(s)")
  }
}

func readSource(_ name: String) -> String {
  do { return try String(contentsOfFile: srcPath(name), encoding: .utf8) } catch { print("cannot read \(srcPath(name)): \(error)"); exit(1) }
}
func pipeline(variant: Int, cfg: Cfg) -> MTLComputePipelineState {
  let o = MTLCompileOptions(); o.languageVersion = .version3_1
  o.preprocessorMacros = ["VARIANT": NSNumber(value: variant), "BM": NSNumber(value: cfg.BM), "BN": NSNumber(value: cfg.BN),
                          "BK": NSNumber(value: cfg.BK), "WM": NSNumber(value: cfg.WM), "WN": NSNumber(value: cfg.WN),
                          "PF": NSNumber(value: cfg.PF)]
  do {
    let lib = try dev.makeLibrary(source: readSource("k3sg.metal"), options: o)
    let ps = try dev.makeComputePipelineState(function: lib.makeFunction(name: "k3sg")!)
    precondition(ps.maxTotalThreadsPerThreadgroup >= cfg.NT, "pipeline max threads \(ps.maxTotalThreadsPerThreadgroup) < \(cfg.NT)")
    return ps
  } catch { print("COMPILE/PIPELINE ERROR (variant \(variant), cfg \(cfg)):\n\(error)"); exit(1) }
}

struct Params {   // mirrors K3Params in k3sg.metal (32 u32 words)
  var w = [UInt32](repeating: 0, count: 32)
  init(M: Int, N: Int, K: Int, capBlock: Int, capShare: Int, key: [UInt32], boundBlock: [UInt32], boundShare: [UInt32]) {
    w[0] = UInt32(M); w[1] = UInt32(N); w[2] = UInt32(K); w[3] = UInt32(capBlock); w[4] = UInt32(capShare)
    for i in 0..<8 { w[8 + i] = key[i]; w[16 + i] = boundBlock[i]; w[24 + i] = boundShare[i] }
  }
}

let sinkBuf = dev.makeBuffer(length: 16)!
func encode(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ cfg: Cfg, M: Int, N: Int, K: Int,
            a: MTLBuffer, b: MTLBuffer, _ prm: Params, ctr: MTLBuffer, blk: MTLBuffer, shr: MTLBuffer) {
  precondition(M % cfg.BM == 0 && N % cfg.BN == 0 && K % 128 == 0, "M % \(cfg.BM), N % \(cfg.BN), K % 128 must be 0")
  var p = prm.w
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps)
  e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
  e.setBytes(&p, length: 128, index: 2)
  e.setBuffer(ctr, offset: 0, index: 3); e.setBuffer(blk, offset: 0, index: 4); e.setBuffer(shr, offset: 0, index: 5)
  e.setBuffer(sinkBuf, offset: 0, index: 6)
  e.dispatchThreadgroups(MTLSize(width: N / cfg.BN, height: M / cfg.BM, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: cfg.NT, height: 1, depth: 1))
  e.endEncoding()
}
func finish(_ cb: MTLCommandBuffer) -> Double {
  cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print("GPU ERROR: \(err)"); exit(1) }
  return cb.gpuEndTime - cb.gpuStartTime
}
func runOnce(_ ps: MTLComputePipelineState, _ cfg: Cfg, M: Int, N: Int, K: Int, a: MTLBuffer, b: MTLBuffer, _ prm: Params,
             ctr: MTLBuffer, blk: MTLBuffer, shr: MTLBuffer) -> Double {
  let cb = queue.makeCommandBuffer()!
  encode(cb, ps, cfg, M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr)
  return finish(cb)
}

// MARK: - PeriodicPattern (port of zk-pow proof_utils.rs from_list / offset_is_valid, as bench/f1_k3/oracle.py)
func patternShape(_ pattern: [Int]) -> [(Int, Int)]? {
  guard let f = pattern.first, f == 0, zip(pattern, pattern.dropFirst()).allSatisfy({ $0 < $1 }) else { return nil }
  var p = pattern
  var shape: [(Int, Int)] = []
  while p.count > 1 {
    var found = false
    for per in 1..<p.count where p.count % per == 0 {
      let s = p[per]
      if (0..<(p.count - per)).allSatisfy({ p[$0] + s == p[$0 + per] }) {
        shape.append((s, p.count / per)); p = Array(p[0..<per]); found = true; break
      }
    }
    if !found { return nil }   // "Pattern is not periodic"
  }
  shape.reverse()
  let period = shape.last.map { $0.0 * $0.1 } ?? 1
  if shape.count > 3 { return nil }
  while shape.count < 3 { shape.append((period, 1)) }
  return shape
}
func offsetIsValid(_ shape: [(Int, Int)], _ off: Int) -> Bool {
  var o = off
  for (stride, len) in shape.reversed() { o %= stride * len; if o >= stride { return false } }
  return true
}

// MARK: - probe
func probe() -> Bool {
  print("== K3-SG layout probe: device \(dev.name), \(sh("sysctl -n hw.model")), macOS \(sh("sw_vers -productVersion")) (\(sh("sw_vers -buildVersion"))) ==")
  for (fam, name) in [(MTLGPUFamily.apple7, "apple7"), (.apple8, "apple8"), (.apple9, "apple9"), (.metal3, "metal3")] {
    print("  supportsFamily(\(name)) = \(dev.supportsFamily(fam))")
  }
  let o = MTLCompileOptions(); o.languageVersion = .version3_1
  let ps: MTLComputePipelineState
  do {
    let lib = try dev.makeLibrary(source: readSource("k3sg.metal"), options: o)
    ps = try dev.makeComputePipelineState(function: lib.makeFunction(name: "sg_probe")!)
  } catch { print("PROBE COMPILE ERROR:\n\(error)"); return false }
  print("  sg_probe threadExecutionWidth = \(ps.threadExecutionWidth)")
  var inp = [Float](repeating: 0, count: 128)
  var bq = [[Float]](repeating: [Float](repeating: 0, count: 8), count: 8)
  for r in 0..<8 { for c in 0..<8 { inp[r * 8 + c] = Float(r * 8 + c); bq[r][c] = Float((r * 3 + c * 5) % 7 - 3); inp[64 + r * 8 + c] = bq[r][c] } }
  var in8 = [Int8](repeating: 0, count: 256)
  for i in 0..<256 { in8[i] = Int8(truncatingIfNeeded: i - 128) }
  let bIn = dev.makeBuffer(bytes: inp, length: 512)!, bOut = dev.makeBuffer(length: 448 * 4)!, bIn8 = dev.makeBuffer(bytes: in8, length: 256)!
  let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps); e.setBuffer(bIn, offset: 0, index: 0); e.setBuffer(bOut, offset: 0, index: 1); e.setBuffer(bIn8, offset: 0, index: 2)
  e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  e.endEncoding(); _ = finish(cb)
  let out = Array(UnsafeBufferPointer(start: bOut.contents().bindMemory(to: Float.self, capacity: 448), count: 448))
  var ok = true
  func check(_ cond: Bool, _ msg: String) { print("  \(cond ? "PASS" : "FAIL") \(msg)"); if !cond { ok = false } }

  // test 1: coords per (lane, element)
  var coord = [[(Int, Int)]](repeating: [(0, 0), (0, 0)], count: 32)
  var seen = Set<Int>()
  for l in 0..<32 { for e in 0..<2 { let v = Int(out[l * 2 + e]); coord[l][e] = (v / 8, v % 8); seen.insert(v) } }
  check(seen.count == 64 && (0..<64).allSatisfy { Float(Int(out[$0])) == out[$0] }, "test1 load: 32 lanes x 2 thread_elements cover all 64 fragment elements exactly once")
  print("  measured layout (lane: (row,col) of thread_elements()[0], [1]):")
  for l0 in stride(from: 0, to: 32, by: 8) {
    print("    " + (l0..<l0 + 8).map { l in "\(l):(\(coord[l][0].0),\(coord[l][0].1))(\(coord[l][1].0),\(coord[l][1].1))" }.joined(separator: " "))
  }
  // test 2: inverse via injection
  var inv = true
  for l in 0..<32 { for e in 0..<2 { let (r, c) = coord[l][e]; if out[64 + r * 8 + c] != Float(1000 + 2 * l + e) { inv = false } } }
  check(inv, "test2 inject+store: simdgroup_store places thread_elements()[e] of lane l at the test1 coordinate")
  // test 3: MMA result layout
  var mm = true
  for l in 0..<32 { for e in 0..<2 {
    let (r, c) = coord[l][e]; var s: Float = 0; for t in 0..<8 { s += Float(r * 8 + t) * bq[t][c] }
    if out[128 + l * 2 + e] != s { mm = false } } }
  check(mm, "test3 fp32 MMA: accumulator thread_elements() follow the same layout (C = M x Bq exact)")
  // test 4: int8 -> float staging conversion
  check((0..<256).allSatisfy { out[192 + $0] == Float($0 - 128) }, "test4 char->float conversion exact for all 256 int8 values")
  // MLX get_coord formula
  var mlx = true
  for l in 0..<32 {
    let qid = l / 4, fm = (qid & 4) + ((l / 2) % 4), fn = (qid & 2) * 2 + (l % 2) * 2
    if coord[l][0] != (fm, fn) || coord[l][1] != (fm, fn + 1) { mlx = false }
  }
  check(mlx, "layout == MLX steel mma.h get_coord (1 row x 2 adjacent cols; fm=(qid&4)+((lane>>1)&3), fn=(qid&2)*2+(lane&1)*2) as hard-coded in k3sg.metal")

  // PeriodicPattern legality for the 32x32 simdgroup tile (4x4 fragments)
  var rowsPat: [Int]? = nil, colsPat: [Int]? = nil
  var origins = Set<[Int]>(), cover = [Int](repeating: 0, count: 1024)
  var product = true, same = true
  for l in 0..<32 {
    var S = Set<[Int]>()
    for i in 0..<4 { for j in 0..<4 { for e in 0..<2 { S.insert([8 * i + coord[l][e].0, 8 * j + coord[l][e].1]) } } }
    let R = Set(S.map { $0[0] }).sorted(), Cc = Set(S.map { $0[1] }).sorted()
    if S.count != R.count * Cc.count || S.count != 32 { product = false }
    for x in S { cover[x[0] * 32 + x[1]] += 1 }
    let rp = R.map { $0 - R[0] }, cp = Cc.map { $0 - Cc[0] }
    if rowsPat == nil { rowsPat = rp; colsPat = cp } else if rowsPat! != rp || colsPat! != cp { same = false }
    origins.insert([R[0], Cc[0]])
  }
  check(product, "every lane's 32-element set is a product rows x cols")
  check(same, "all 32 lanes share one normalized (rows_pattern, cols_pattern)")
  check(cover.allSatisfy { $0 == 1 }, "the 32 lane sets partition the 32x32 simdgroup tile")
  let rp = rowsPat ?? [], cp = colsPat ?? []
  print("  measured rows_pattern \(rp)  cols_pattern \(cp)")
  if let rs = patternShape(rp), let cs = patternShape(cp) {
    print("  from_list shapes: rows \(rs)  cols \(cs)")
    let h = rp.count, w = cp.count
    check(h % 2 == 0 && w % 2 == 0 && h * w >= 32 && h * w <= 256, "h=\(h), w=\(w) even and 32 <= h*w=\(h * w) <= 256")
    let rper = rs[2].0 * rs[2].1, cper = cs[2].0 * cs[2].1
    let vr = (0..<32).filter { offsetIsValid(rs, $0) }, vc = (0..<32).filter { offsetIsValid(cs, $0) }
    var expOrig = Set<[Int]>(); for r in vr { for c in vc { expOrig.insert([r, c]) } }
    check(rper == 32 && cper == 32, "pattern periods rows \(rper), cols \(cper) == simdgroup tile 32 (tiles align with 32-aligned simdgroup tiles)")
    check(origins == expOrig, "lane origins == Pearl valid offsets in [0,32)^2 (rows \(vr), cols \(vc))")
  } else {
    check(false, "measured patterns are legal PeriodicPatterns (<= 3 dims, start 0, sorted)")
  }
  check(rp == ROWS_PATTERN && cp == COLS_PATTERN, "measured pattern == committed MiningConfiguration rows \(ROWS_PATTERN) cols \(COLS_PATTERN)")
  print("PROBE: " + (ok ? "PASS" : "FAIL (refuse to mine on this device)"))
  return ok
}

// MARK: - run (correctness)
func readFile(_ p: String) -> Data {
  guard let d = FileManager.default.contents(atPath: p) else { print("cannot read \(p)"); exit(1) }
  return d
}
func u32s(_ any: Any?) -> [UInt32] { (any as! [NSNumber]).map { $0.uint32Value } }
func u256LE(_ h: ArraySlice<UInt32>, _ b: [UInt32]) -> Bool {
  let h = Array(h)
  for i in stride(from: 7, through: 0, by: -1) { if h[i] < b[i] { return true }; if h[i] > b[i] { return false } }
  return true
}

/// Swift port of bench/f1_k3/harness.py check_case against oracle tiles (t_rows, t_cols, jp[16], hash[8]) per tile.
func checkCase(_ tiles: [ArraySlice<UInt32>], bb: [UInt32], bs: [UInt32], capB: Int, capS: Int, out: [UInt32]) -> (Bool, String) {
  var byXY: [[UInt32]: ArraySlice<UInt32>] = [:]
  for t in tiles { byXY[[t[t.startIndex], t[t.startIndex + 1]]] = t }
  var errs: [String] = []
  let nb = (capB + GUARD_SLOTS) * SLOT_WORDS
  let parts: [(String, Int, Int, ArraySlice<UInt32>, [UInt32])] = [
    ("block", Int(out[0]), capB, out[2..<(2 + nb)], bb), ("share", Int(out[1]), capS, out[(2 + nb)...], bs)]
  var summary: [String] = []
  for (label, ctr, cap, arr, bound) in parts {
    let exp = Set(tiles.filter { u256LE($0[($0.startIndex + 18)...], bound) }.map { [$0[$0.startIndex], $0[$0.startIndex + 1]] })
    if ctr != exp.count { errs.append("\(label) counter \(ctr) != oracle \(exp.count)") }
    let written = min(ctr, cap)
    var seen = Set<[UInt32]>()
    for i in 0..<written {
      let s = Array(arr[(arr.startIndex + i * SLOT_WORDS)..<(arr.startIndex + (i + 1) * SLOT_WORDS)])
      let xy = [s[0], s[1]]
      guard let t = byXY[xy] else { errs.append("\(label) slot \(i): \(xy) is not a Pearl tile origin"); continue }
      if seen.contains(xy) { errs.append("\(label) slot \(i): duplicate tile \(xy)") }
      seen.insert(xy)
      if Array(t) != s { errs.append("\(label) slot \(i) tile \(xy): transcript/hash mismatch") }
      if !exp.contains(xy) { errs.append("\(label) slot \(i) tile \(xy): not a find per oracle") }
    }
    if ctr <= cap && seen != exp { errs.append("\(label) slot set != oracle find set (\(seen.count) vs \(exp.count))") }
    if !arr[(arr.startIndex + written * SLOT_WORDS)...].allSatisfy({ $0 == CANARY }) {
      errs.append("\(label): write outside the \(written) valid slots (capacity \(cap), guard \(GUARD_SLOTS))")
    }
    summary.append("\(label) ctr=\(ctr) (oracle \(exp.count), cap \(cap))")
  }
  return (errs.isEmpty, summary.joined(separator: ", ") + (errs.isEmpty ? "" : " ERRORS: " + errs.prefix(8).joined(separator: "; ")))
}

func runJob(_ dir: String, cfgs: [Cfg]) -> Int {
  let js = try! JSONSerialization.jsonObject(with: readFile(dir + "/job.json")) as! [String: Any]
  let M = (js["m"] as! NSNumber).intValue, N = (js["n"] as! NSNumber).intValue, K = (js["k"] as! NSNumber).intValue
  let key = u32s(js["key"])
  let A = readFile(dir + "/A.bin"), B = readFile(dir + "/B.bin")
  precondition(A.count == M * K && B.count == K * N, "operand sizes do not match job.json")
  let a = A.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: A.count)! }
  let b = B.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: B.count)! }
  var tiles: [ArraySlice<UInt32>]? = nil
  if FileManager.default.fileExists(atPath: dir + "/tiles.bin") {
    let tw = readFile(dir + "/tiles.bin").withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    precondition(tw.count == (M * N / 32) * SLOT_WORDS, "tiles.bin size \(tw.count) != \(M * N / 32) tiles")
    tiles = (0..<(M * N / 32)).map { tw[($0 * SLOT_WORDS)..<(($0 + 1) * SLOT_WORDS)] }
  }
  var fails = 0
  for (ci, cfg) in cfgs.enumerated() {
    if M % cfg.BM != 0 || N % cfg.BN != 0 { print("  cfg \(cfg): SKIP (job \(M)x\(N) not a multiple of the threadgroup tile)"); continue }
    let ps = pipeline(variant: 3, cfg: cfg)
    for cs in js["cases"] as! [[String: Any]] {
      let name = cs["name"] as! String
      let capB = (cs["cap_block"] as! NSNumber).intValue, capS = (cs["cap_share"] as! NSNumber).intValue
      let bb = u32s(cs["bound_block"]), bs = u32s(cs["bound_share"])
      let prm = Params(M: M, N: N, K: K, capBlock: capB, capShare: capS, key: key, boundBlock: bb, boundShare: bs)
      let nb = (capB + GUARD_SLOTS) * SLOT_WORDS, ns = (capS + GUARD_SLOTS) * SLOT_WORDS
      let ctr = dev.makeBuffer(length: 8)!, blk = dev.makeBuffer(length: nb * 4)!, shr = dev.makeBuffer(length: ns * 4)!
      memset(ctr.contents(), 0, 8)
      for (buf, n) in [(blk, nb), (shr, ns)] { let p = buf.contents().bindMemory(to: UInt32.self, capacity: n); for i in 0..<n { p[i] = CANARY } }
      let t = runOnce(ps, cfg, M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr)
      var out = Data(bytes: ctr.contents(), count: 8)
      out.append(Data(bytes: blk.contents(), count: nb * 4)); out.append(Data(bytes: shr.contents(), count: ns * 4))
      let suffix = ci == 0 ? "" : "__\(cfg)"
      FileManager.default.createFile(atPath: dir + "/out_\(name)\(suffix).bin", contents: out)
      let c = ctr.contents().bindMemory(to: UInt32.self, capacity: 2)
      var line = String(format: "  cfg %@ case %-26@ block_ctr=%u share_ctr=%u gpu %.2f ms", cfg.description as NSString, name as NSString, c[0], c[1], t * 1e3)
      if let tiles = tiles {
        let ow = out.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let (ok, summ) = checkCase(tiles, bb: bb, bs: bs, capB: capB, capS: capS, out: ow)
        line += "  \(ok ? "PASS" : "FAIL") \(summ)"
        if !ok { fails += 1 }
      }
      print(line)
    }
  }
  if tiles != nil { print("  JOB \(dir): \(fails == 0 ? "PASS" : "FAIL (\(fails) cases)")") }
  return fails
}

// MARK: - perf helpers
let LOCK = "/tmp/pmm-gpu-bench.lock"
var useLock = true
func loadavg1() -> Double {
  let s = sh("sysctl -n vm.loadavg").replacingOccurrences(of: "{", with: "").split(separator: " ")
  return Double(s.first ?? "0") ?? 0
}
func envLine(_ tag: String) -> String {
  "  [\(tag)] \(sh("date -u +%FT%TZ")) loadavg \(sh("sysctl -n vm.loadavg")) | power: \(sh("pmset -g batt 2>/dev/null | head -2 | tr '\\n' ' ' | tr '\\t' ' '")) | thermal: \(sh("pmset -g therm 2>/dev/null | grep -i -E 'level|limit' | tr '\\n' ' '"))"
}
var flags: [String] = []
func acquireLock(_ label: String) {
  if !useLock { print("  [lock] nolock (caller owns the machine window)"); print(envLine("before")); return }
  var waited = 0
  while loadavg1() > 8 && waited < 1800 {
    if waited % 300 == 0 { print("  [load] loadavg1 \(loadavg1()) > 8, waiting (waited \(waited) s)") }
    sleep(30); waited += 30
  }
  if loadavg1() > 8 { let f = "FLAG \(label): loadavg1 \(loadavg1()) > 8 after 30 min wait; proceeding"; print("  [load] " + f); flags.append(f) }
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
  if useLock { rmdir(LOCK); print("  [lock] released") }
}
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count/2 - 1] + s[s.count/2]) / 2 }
func bootCI(_ x: [Double]) -> (Double, Double) {   // percentile bootstrap 90% CI of the median, fixed seed, 20000 resamples
  var st: UInt64 = 0x9E3779B97F4A7C15
  func nxt() -> UInt64 { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st }
  var meds: [Double] = []
  for _ in 0..<20000 { var r: [Double] = []; for _ in 0..<x.count { r.append(x[Int(nxt() % UInt64(x.count))]) }; meds.append(median(r)) }
  meds.sort()
  return (meds[Int(0.05 * Double(meds.count))], meds[Int(0.95 * Double(meds.count)) - 1])
}
func randI8(_ n: Int) -> MTLBuffer {
  let buf = dev.makeBuffer(length: n)!
  arc4random_buf(buf.contents(), n)
  let p = buf.contents().bindMemory(to: Int8.self, capacity: n)
  for i in 0..<n where p[i] == -128 { p[i] = 127 }   // noised range [-127, 127]
  return buf
}
func toF32(_ b8: MTLBuffer, _ n: Int) -> MTLBuffer {
  let buf = dev.makeBuffer(length: n * 4)!
  let s = b8.contents().bindMemory(to: Int8.self, capacity: n), d = buf.contents().bindMemory(to: Float.self, capacity: n)
  for i in 0..<n { d[i] = Float(s[i]) }
  return buf
}
func boundBelowPow2(_ e: Int) -> [UInt32] {
  var w = [UInt32](repeating: 0, count: 8)
  for i in 0..<8 { let lo = i * 32; if e >= lo + 32 { w[i] = 0xFFFFFFFF } else if e > lo { w[i] = (1 << UInt32(e - lo)) - 1 } }
  return w
}
func perfParams(M: Int, N: Int, K: Int) -> (Params, Int) {
  let tiles = M * N / 32
  let shareE = 256 + 2 - Int(log2(Double(tiles)).rounded())   // ~4 expected share finds per job
  let key: [UInt32] = (0..<8).map { _ in arc4random() }
  return (Params(M: M, N: N, K: K, capBlock: 4, capShare: 64, key: key, boundBlock: boundBelowPow2(200), boundShare: boundBelowPow2(shareE)), shareE)
}
func header(_ what: String) {
  print("== K3-SG \(what): device \(dev.name), \(sh("sysctl -n hw.model")), \(sh("sysctl -n machdep.cpu.brand_string")), macOS \(sh("sw_vers -productVersion")) (\(sh("sw_vers -buildVersion"))) ==")
}

/// MLX fp32 matmul in a persistent subprocess, timed per request (wall ms).
final class MLXRef {
  let p = Process(); let inp = Pipe(); let outp = Pipe(); var buf = Data()
  init?(python: String, M: Int, N: Int, K: Int) {
    p.executableURL = URL(fileURLWithPath: python)
    p.arguments = [srcPath("mlx_ref.py"), "\(M)", "\(N)", "\(K)"]
    p.standardInput = inp; p.standardOutput = outp; p.standardError = outp
    do { try p.run() } catch { print("  MLX reference unavailable: \(error)"); return nil }
    guard let l = readLine(), l.hasPrefix("ready") else { print("  MLX reference failed to start (verbatim): \(buf.isEmpty ? "" : String(decoding: buf, as: UTF8.self))"); p.terminate(); return nil }
    print("  MLX reference: \(l)")
  }
  func readLine() -> String? {
    while true {
      if let i = buf.firstIndex(of: 10) { let l = String(decoding: buf[buf.startIndex..<i], as: UTF8.self); buf = Data(buf[(i + 1)...]); return l }
      let d = outp.fileHandleForReading.availableData
      if d.isEmpty { return nil }
      buf.append(d)
    }
  }
  func run() -> Double {
    inp.fileHandleForWriting.write("run\n".data(using: .utf8)!)
    guard let l = readLine(), let ms = Double(l) else { print("MLX reference died"); exit(1) }
    return ms / 1e3
  }
  func quit() { inp.fileHandleForWriting.write("quit\n".data(using: .utf8)!); p.waitUntilExit() }
}

func int8benchPipeline() -> MTLComputePipelineState? {
  guard #available(macOS 26.0, *) else { print("  int8bench (Metal 4 matmul2d) unavailable: macOS < 26"); return nil }
  let o = MTLCompileOptions(); o.languageVersion = .version4_0
  do {
    let lib = try dev.makeLibrary(source: readSource("int8ref.metal"), options: o)
    return try dev.makeComputePipelineState(function: lib.makeFunction(name: "int8bench")!)
  } catch { print("  int8bench (Metal 4 matmul2d) unavailable on this device, verbatim error:\n\(error)"); return nil }
}

func perf(_ shapes: [(Int, Int, Int)], rounds: Int, cfg: Cfg, python: String?, withInt8: Bool) {
  header("perf (cfg \(cfg))")
  print("variants: int8bench = Metal 4 matmul2d int8 128x64 (C stored; context); mlx = MLX mx.matmul fp32 (wall time incl. dispatch);")
  print("          base_f32 = plain fp32 simdgroup_matrix GEMM, same tiling, fp32 operands, C store disabled [P3 BASELINE];")
  print("          base_i8 = same tiling with int8 operands converted while staging, one fp32 accumulator, no fold;")
  print("          fold = production K loop (fresh fp32 acc per 128 chunk -> int32 acc -> XOR fold -> rotl13), no hash;")
  print("          k3 = fold + keyed BLAKE3 + 2x U256 compare + atomic found slots (block cap 4, share cap 64)")
  print("timing: GPU timestamps per command buffer (mlx: wall); 1 warm-up each; \(rounds) paired rounds, order reversed on odd rounds")
  let i8ps = withInt8 ? int8benchPipeline() : nil
  let sg = [1, 0, 2, 3].map { pipeline(variant: $0, cfg: cfg) }   // base_f32, base_i8, fold, k3 (compile before the lock)
  for (M, N, K) in shapes {
    let (prm, shareE) = perfParams(M: M, N: N, K: K)
    let a8 = randI8(M * K), b8 = randI8(K * N), af = toF32(a8, M * K), bf = toF32(b8, K * N)
    let ctr = dev.makeBuffer(length: 8)!, blk = dev.makeBuffer(length: 4 * SLOT_WORDS * 4)!, shr = dev.makeBuffer(length: 64 * SLOT_WORDS * 4)!
    let cbuf = i8ps != nil ? dev.makeBuffer(length: M * N * 4) : nil
    let ops = 2.0 * Double(M) * Double(N) * Double(K)
    print("\nshape \(M)x\(N)x\(K)  ops=2mnk=\(String(format: "%.3e", ops))  tiles=\(M * N / 32)  share bound=2^\(shareE)-1 (E[shares/job]~4)  block bound=2^200-1")
    let mlx = python.flatMap { MLXRef(python: $0, M: M, N: N, K: K) }
    var names: [String] = [], runs: [() -> Double] = []
    if let ps = i8ps, let cb = cbuf {
      func i8cmd() -> MTLCommandBuffer {
        var mnk: [UInt32] = [UInt32(M), UInt32(N), UInt32(K)]
        let c = queue.makeCommandBuffer()!, e = c.makeComputeCommandEncoder()!
        e.setComputePipelineState(ps); e.setBuffer(a8, offset: 0, index: 0); e.setBuffer(b8, offset: 0, index: 1)
        e.setBuffer(cb, offset: 0, index: 2); e.setBytes(&mnk, length: 12, index: 3)
        e.dispatchThreadgroups(MTLSize(width: N / 64, height: M / 128, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        e.endEncoding(); return c
      }
      let probeCb = i8cmd(); probeCb.commit(); probeCb.waitUntilCompleted()   // trial run: a GPU error drops the column
      if let err = probeCb.error { print("  int8bench trial run failed on this device, column dropped; verbatim error: \(err)") } else {
        names.append("int8bench")
        runs.append { finish(i8cmd()) }
      }
    }
    if let m = mlx { names.append("mlx"); runs.append { m.run() } }
    let sgNames = ["base_f32", "base_i8", "fold", "k3"]
    var k3Shares: [UInt32] = []
    for (vi, ps) in sg.enumerated() {
      names.append(sgNames[vi])
      let (a, b) = vi == 0 ? (af, bf) : (a8, b8)
      runs.append {
        memset(ctr.contents(), 0, 8)
        let t = runOnce(ps, cfg, M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr)
        if vi == 3 { k3Shares.append(ctr.contents().bindMemory(to: UInt32.self, capacity: 2)[1]) }
        return t
      }
    }
    let nv = names.count
    acquireLock("\(M)x\(N)x\(K)")
    for r in runs { _ = r() }
    k3Shares.removeAll()
    var t = [[Double]](repeating: [], count: nv)
    for r in 0..<rounds {
      let order = r % 2 == 0 ? Array(0..<nv) : Array((0..<nv).reversed())
      for i in order { t[i].append(runs[i]()) }
    }
    releaseLock()
    mlx?.quit()
    print("  variant      median TOPS   min TOPS   max TOPS   raw ms per round")
    for i in 0..<nv {
      let x = t[i]
      print(String(format: "  %-10@ %10.2f %10.2f %10.2f   ", names[i] as NSString, ops / median(x) / 1e12, ops / x.max()! / 1e12, ops / x.min()! / 1e12)
            + x.map { String(format: "%.2f", $0 * 1e3) }.joined(separator: " "))
    }
    func ratioLine(_ label: String, _ num: String, _ den: String) {
      guard let a = names.firstIndex(of: num), let b = names.firstIndex(of: den) else { return }
      let r = (0..<rounds).map { t[b][$0] / t[a][$0] }
      let (lo, hi) = bootCI(r)
      print(String(format: "  %-30@ median %.4f  90%% CI [%.4f, %.4f] half-width %.4f  (min %.4f max %.4f)",
                   label as NSString, median(r), lo, hi, (hi - lo) / 2, r.min()!, r.max()!))
    }
    ratioLine("P3 k3/base_f32 (throughput)", "k3", "base_f32")
    ratioLine("k3/base_i8", "k3", "base_i8")
    ratioLine("fold/base_i8", "fold", "base_i8")
    ratioLine("k3/fold", "k3", "fold")
    ratioLine("base_i8/base_f32", "base_i8", "base_f32")
    ratioLine("k3/mlx_fp32", "k3", "mlx")
    ratioLine("base_f32/mlx_fp32", "base_f32", "mlx")
    ratioLine("k3/int8bench", "k3", "int8bench")
    if let a = names.firstIndex(of: "k3"), let b = names.firstIndex(of: "base_f32") {
      let med = median((0..<rounds).map { t[b][$0] / t[a][$0] })
      print("  P3 gate (k3 >= 0.80x same-round base_f32): \(med >= 0.80 ? "PASS" : "FAIL") (median \(String(format: "%.4f", med)))")
    }
    print("  k3 share finds per run: \(k3Shares.map(String.init).joined(separator: ","))")
  }
  if !flags.isEmpty { print("\nFLAGS:"); flags.forEach { print("  " + $0) } }
}

func sweep(_ M: Int, _ N: Int, _ K: Int, rounds: Int, cfgs: [Cfg]) {
  header("cfg sweep \(M)x\(N)x\(K)")
  let pss = cfgs.map { pipeline(variant: 3, cfg: $0) }
  let (prm, _) = perfParams(M: M, N: N, K: K)
  let a = randI8(M * K), b = randI8(K * N)
  let ctr = dev.makeBuffer(length: 8)!, blk = dev.makeBuffer(length: 4 * SLOT_WORDS * 4)!, shr = dev.makeBuffer(length: 64 * SLOT_WORDS * 4)!
  let ops = 2.0 * Double(M) * Double(N) * Double(K)
  acquireLock("sweep")
  for (i, ps) in pss.enumerated() { _ = runOnce(ps, cfgs[i], M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr) }
  var t = [[Double]](repeating: [], count: cfgs.count)
  for r in 0..<rounds {
    let order = r % 2 == 0 ? Array(0..<cfgs.count) : Array((0..<cfgs.count).reversed())
    for i in order { memset(ctr.contents(), 0, 8); t[i].append(runOnce(pss[i], cfgs[i], M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr)) }
  }
  releaseLock()
  var best = 0
  for i in 0..<cfgs.count {
    print(String(format: "  k3 cfg %-16@ median %.2f TOPS  (min %.2f, max %.2f)", cfgs[i].description as NSString,
                 ops / median(t[i]) / 1e12, ops / t[i].max()! / 1e12, ops / t[i].min()! / 1e12))
    if median(t[i]) < median(t[best]) { best = i }
  }
  print("BEST \(cfgs[best])")
}

func sustain(_ secs: Int, _ M: Int, _ N: Int, _ K: Int, cfg: Cfg) {
  header("sustained k3 \(M)x\(N)x\(K) cfg \(cfg) for \(secs) s")
  let ps = pipeline(variant: 3, cfg: cfg)
  let (prm, _) = perfParams(M: M, N: N, K: K)
  let a = randI8(M * K), b = randI8(K * N)
  let ctr = dev.makeBuffer(length: 8)!, blk = dev.makeBuffer(length: 4 * SLOT_WORDS * 4)!, shr = dev.makeBuffer(length: 64 * SLOT_WORDS * 4)!
  let ops = 2.0 * Double(M) * Double(N) * Double(K)
  acquireLock("sustain")
  _ = runOnce(ps, cfg, M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr)
  let t0 = Date(); var wStart = Date(); var wJobs = 0; var wGpu = 0.0; var win = 0; var all: [Double] = []
  var shares = Set<UInt32>()
  print("  window  t_end_s  jobs  gpu_TOPS  wall_TOPS  share_finds/job")
  while Date().timeIntervalSince(t0) < Double(secs) {
    memset(ctr.contents(), 0, 8)
    wGpu += runOnce(ps, cfg, M: M, N: N, K: K, a: a, b: b, prm, ctr: ctr, blk: blk, shr: shr); wJobs += 1
    shares.insert(ctr.contents().bindMemory(to: UInt32.self, capacity: 2)[1])
    let el = Date().timeIntervalSince(wStart)
    if el >= 10 {
      win += 1
      let g = ops * Double(wJobs) / wGpu / 1e12, w = ops * Double(wJobs) / el / 1e12
      all.append(g)
      print(String(format: "  %6d %8.1f %5d %9.2f %10.2f  %@", win, Date().timeIntervalSince(t0), wJobs, g, w, shares.sorted().map(String.init).joined(separator: ",") as NSString))
      wStart = Date(); wJobs = 0; wGpu = 0; shares.removeAll()
    }
  }
  releaseLock()
  if all.count >= 2 {
    let n = max(1, all.count / 5)   // first vs last ~20% of windows
    let first = all.prefix(n).reduce(0, +) / Double(n), last = all.suffix(n).reduce(0, +) / Double(n)
    print(String(format: "  first %d windows %.2f TOPS, last %d windows %.2f TOPS, change %+.1f%%; min %.2f max %.2f", n, first, n, last, (last / first - 1) * 100, all.min()!, all.max()!))
  }
}

// MARK: - main
let args = Array(CommandLine.arguments.dropFirst())
func opt(_ k: String) -> String? { args.first { $0.hasPrefix(k + "=") }.map { String($0.dropFirst(k.count + 1)) } }
func shapeArgs() -> [(Int, Int, Int)] {
  args.dropFirst().filter { $0.first!.isNumber && $0.contains("x") && !$0.contains("=") }.map { a in
    let p = a.split(separator: "x").map { Int($0)! }; return (p[0], p[1], p[2]) }
}
useLock = !args.contains("nolock")
let defCfg = Cfg(opt("cfg") ?? DEFAULT_CFG)
switch args.first ?? "" {
case "probe":
  exit(probe() ? 0 : 1)
case "run":
  guard args.count >= 2 else { print("usage: k3sg run JOBDIR [cfgs=a,b]"); exit(2) }
  let cfgs = (opt("cfgs")?.split(separator: ",").map { Cfg(String($0)) }) ?? [defCfg]
  print("== K3-SG run \(args[1]) (device \(dev.name)) cfgs \(cfgs) ==")
  exit(runJob(args[1], cfgs: cfgs) == 0 ? 0 : 1)
case "perf":
  let rounds = Int(opt("rounds") ?? "31")!
  precondition(rounds >= 15, "needs >= 15 paired rounds")
  var shapes = shapeArgs()
  if shapes.isEmpty { shapes = [(4096, 4096, 4096), (8192, 8192, 4096)] }
  perf(shapes, rounds: rounds, cfg: defCfg, python: opt("mlx"), withInt8: !args.contains("noint8bench"))
case "sweep":
  let s = shapeArgs().first ?? (4096, 4096, 4096)
  let cfgs = (opt("cfgs") ?? DEFAULT_CFG).split(separator: ",").map { Cfg(String($0)) }
  sweep(s.0, s.1, s.2, rounds: Int(opt("rounds") ?? "5")!, cfgs: cfgs)
case "sustain":
  guard args.count >= 3, let secs = Int(args[1]) else { print("usage: k3sg sustain SECS MxNxK [cfg=C]"); exit(2) }
  let p = args[2].split(separator: "x").map { Int($0)! }
  sustain(secs, p[0], p[1], p[2], cfg: defCfg)
default:
  print("usage: k3sg probe | run JOBDIR [cfgs=..] | perf [MxNxK ..] [rounds=R] [cfg=C] [nolock] [mlx=PY] [noint8bench] | sweep MxNxK cfgs=.. | sustain SECS MxNxK [cfg=C]")
  exit(2)
}
