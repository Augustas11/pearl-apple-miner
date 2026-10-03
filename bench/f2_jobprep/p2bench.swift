// F2 end-to-end P2 proxy: K3-like GPU kernel fed by pmkcore jobs built into shared MTLBuffers.
//
// Mode "gpu":  K3-alone wall rate. The same A/B buffers, back to back, 2 command buffers in flight.
// Mode "pipe": the real job loop. 3 rotating Bᵀ slots (shared MTLBuffers). The CPU (pmkcore, 9 threads)
//              builds job i+1 — fresh Bᵀ, Merkle root, cert-v3 seeds — straight into slot.contents()
//              while the GPU runs earlier jobs. Slots are recycled from addCompletedHandler.
// P2 = (Σ2mnk / elapsed in pipe) / (Σ2mnk / elapsed in gpu). Rounds alternate gpu/pipe.
//
// GPU kernel = R6 "V6" (matmul2d int8→int32, 128×64 tile, RK=128, uint4 pointer XOR-fold + rotl13 into
// a 16-word per-lane transcript, no C store). It lacks K1/K2 noise and the final BLAKE3/compare (F1),
// so it is a proxy for K3 timing, not a miner. It reads the Bᵀ slot as its k×n B operand (same bytes,
// same cost); the transposition belongs to K2.
//
// Build: see README.md.  Run: p2bench [MxNxK ...] [--jobs J] [--rounds R]
import Metal
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let copts = MTLCompileOptions(); copts.languageVersion = .version4_0

func sh(_ cmd: String) -> String {
  let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh"); p.arguments = ["-c", cmd]
  let pipe = Pipe(); p.standardOutput = pipe; try? p.run(); p.waitUntilExit()
  return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
}
func load1() -> Double { Double(sh("sysctl -n vm.loadavg").split(separator: " ")[1]) ?? .nan }

let LOCK = "/tmp/pmm-gpu-bench.lock"
func acquireLock() {
  var waitedLoad = 0
  while load1() > 8 && waitedLoad < 1800 {
    if waitedLoad % 60 == 0 { print("  [load] 1-min loadavg \(load1()) > 8, waiting (waited \(waitedLoad) s)") }
    sleep(15); waitedLoad += 15
  }
  if load1() > 8 { print("  [load] FLAG: loadavg still \(load1()) after 30 min; proceeding") }
  var waited = 0
  while mkdir(LOCK, 0o755) != 0 {
    if waited % 60 == 0 { print("  [lock] \(LOCK) busy, retrying every 15 s (waited \(waited) s)") }
    sleep(15); waited += 15
  }
  print("  [lock] acquired after \(waited) s; loadavg before: \(sh("sysctl -n vm.loadavg"))")
}
func releaseLock() {
  print("  [lock] releasing; loadavg after: \(sh("sysctl -n vm.loadavg"))")
  rmdir(LOCK)
}

let src = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
typedef tensor<device int8_t, dextents<int32_t,2>, tensor_inline> TI8;
kernel void k3proxy(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device uint* tr [[buffer(2)]],
                    constant uint4& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 tgcount [[threadgroups_per_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  uint M = dims.x, N = dims.y, K = dims.z;
  uint lin = tgid.y * tgcount.x + tgid.x;
  uint m0 = tgid.y * 128, n0 = tgid.x * 64;
  TI8 A(a, dextents<int32_t,2>(K, M));
  TI8 B(b, dextents<int32_t,2>(N, K));
  constexpr auto d = matmul2d_descriptor(128, 64, 128, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA0 = A.slice(0, m0); auto tB0 = B.slice(n0, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), int>();
  #pragma unroll
  for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  uint jp[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  uint nfull = K / 128;
  for (uint ch = 0; ch < nfull; ++ch) {
    auto tA = A.slice(ch * 128, m0); auto tB = B.slice(n0, ch * 128);
    op.run(tA, tB, cT);
    thread const uint4* p4 = (thread const uint4*)&cT[0];
    uint4 v = p4[0];
    #pragma unroll
    for (ushort i = 1; i < 16; ++i) v ^= p4[i];
    uint xf = v.x ^ v.y ^ v.z ^ v.w;
    uint s = ch & 15u;
    #pragma unroll
    for (uint j = 0; j < 16; ++j) jp[j] = (j == s) ? (rotate(jp[j], 13u) ^ xf) : jp[j];
  }
  uint g = lin * 128 + tid;
  #pragma unroll
  for (uint j = 0; j < 16; ++j) tr[g * 16 + j] = jp[j];
}
"""
let ps: MTLComputePipelineState = {
  do {
    let lib = try dev.makeLibrary(source: src, options: copts)
    return try dev.makeComputePipelineState(function: lib.makeFunction(name: "k3proxy")!)
  } catch { print("COMPILE ERROR:\n\(error)"); exit(1) }
}()

func encode(_ cb: MTLCommandBuffer, a: MTLBuffer, b: MTLBuffer, tr: MTLBuffer, M: Int, N: Int, K: Int) {
  var dims: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), 0]
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps)
  e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(tr, offset: 0, index: 2)
  e.setBytes(&dims, length: 16, index: 3)
  e.dispatchThreadgroups(MTLSize(width: N / 64, height: M / 128, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.endEncoding()
}

func check(_ rc: Int32, _ what: String) {
  if rc != 0 { print("\(what) failed: \(rc) \(String(cString: pmkcore_strerror(rc)))"); exit(1) }
}

func header() -> [UInt8] { (0..<76).map { UInt8(truncatingIfNeeded: $0 &* 37 &+ 11) } }
func config(_ k: Int) -> [UInt8] {
  var c = [UInt8](repeating: 0, count: 52)
  withUnsafeBytes(of: UInt32(k).littleEndian) { for i in 0..<4 { c[i] = $0[i] } }
  c[4] = 128
  c.replaceSubrange(8..<14, with: [7, 1, 3, 1, 0, 0])    // rows [0,8,64,72]
  c.replaceSubrange(14..<20, with: [0, 3, 3, 3, 0, 0])   // cols [0..3,16..19,32..35,48..51]
  return c
}

final class Stats: @unchecked Sendable {
  let lock = NSLock(); var gpuBusy = 0.0; var err: Error?
  func add(_ cb: MTLCommandBuffer) { lock.lock(); gpuBusy += cb.gpuEndTime - cb.gpuStartTime; if let e = cb.error { err = e }; lock.unlock() }
}

func median(_ x: [Double]) -> Double { let s = x.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }

// MARK: - main
var shapes: [(Int, Int, Int)] = []
var jobs = 70, rounds = 3
var it = CommandLine.arguments.dropFirst().makeIterator()
while let a = it.next() {
  if a == "--jobs" { jobs = Int(it.next()!)! } else if a == "--rounds" { rounds = Int(it.next()!)! }
  else { let p = a.split(separator: "x").map { Int($0)! }; shapes.append((p[0], p[1], p[2])) }
}
if shapes.isEmpty { shapes = [(4096, 4096, 2048), (4096, 4096, 4096), (4096, 4096, 8192), (8192, 8192, 2048), (8192, 8192, 4096), (8192, 8192, 8192)] }

check(pmkcore_init(9), "pmkcore_init(9)")
print("== F2 end-to-end P2 proxy (device \(dev.name), \(sh("sysctl -n machdep.cpu.brand_string")), macOS \(sh("sw_vers -productVersion")), \(sh("pmset -g batt | head -1"))) ==")
print("date: \(sh("date")); pmkcore threads: 9 (one core reserved for the orchestrator); \(rounds) rounds x \(jobs) jobs per mode per shape; rounds alternate gpu/pipe")

var summary: [String] = []
for (M, N, K) in shapes {
  let aLen = Int(pmkcore_padded_len(UInt64(M), UInt64(K))), bLen = Int(pmkcore_padded_len(UInt64(N), UInt64(K)))
  let A = dev.makeBuffer(length: aLen, options: .storageModeShared)!
  let slots = (0..<3).map { _ in dev.makeBuffer(length: bLen, options: .storageModeShared)! }
  let tr = dev.makeBuffer(length: (M / 128) * (N / 64) * 128 * 64, options: .storageModeShared)!
  var tmpl = PmkTemplate()
  let t0 = Date()
  check(header().withUnsafeBufferPointer { h in config(K).withUnsafeBufferPointer { c in
    pmkcore_template_init(h.baseAddress, c.baseAddress, UInt32(M), UInt32(N), A.contents().assumingMemoryBound(to: UInt8.self), UInt64(aLen), 1, &tmpl)
  } }, "pmkcore_template_init")
  let tmplMs = Date().timeIntervalSince(t0) * 1e3
  var job = PmkJob()
  for s in slots {   // first-touch every slot outside the timed rounds (steady state, not page-fault cost)
    check(pmkcore_build_job(&tmpl, s.contents().assumingMemoryBound(to: UInt8.self), UInt64(bLen), &job), "pmkcore_build_job")
  }
  let ops = 2.0 * Double(M) * Double(N) * Double(K)
  print(String(format: "\nshape %dx%dx%d  ops/job %.3e  B^T slot %.1f MiB  template (A) build %.1f ms", M, N, K, ops, Double(bLen) / 1048576, tmplMs))

  func gpuOnly() -> (Double, Double) {
    let st = Stats(); let inflight = DispatchSemaphore(value: 2); let done = DispatchGroup()
    let t = Date()
    for _ in 0..<jobs {
      inflight.wait(); done.enter()
      let cb = queue.makeCommandBuffer()!
      encode(cb, a: A, b: slots[0], tr: tr, M: M, N: N, K: K)
      cb.addCompletedHandler { cb in st.add(cb); inflight.signal(); done.leave() }
      cb.commit()
    }
    done.wait()
    if let e = st.err { print("GPU ERROR: \(e)"); exit(1) }
    return (Date().timeIntervalSince(t), st.gpuBusy)
  }
  func pipe() -> (Double, Double, [Double]) {
    let st = Stats(); let free = DispatchSemaphore(value: 3); let done = DispatchGroup()
    var builds: [Double] = []
    let t = Date()
    for i in 0..<jobs {
      free.wait(); done.enter()
      let slot = slots[i % 3]
      let tb = Date()
      var j = PmkJob()
      check(pmkcore_build_job(&tmpl, slot.contents().assumingMemoryBound(to: UInt8.self), UInt64(bLen), &j), "pmkcore_build_job")
      builds.append(Date().timeIntervalSince(tb))
      let cb = queue.makeCommandBuffer()!
      encode(cb, a: A, b: slot, tr: tr, M: M, N: N, K: K)
      cb.addCompletedHandler { cb in st.add(cb); free.signal(); done.leave() }
      cb.commit()
    }
    done.wait()
    if let e = st.err { print("GPU ERROR: \(e)"); exit(1) }
    return (Date().timeIntervalSince(t), st.gpuBusy, builds)
  }

  acquireLock()
  _ = gpuOnly()   // warm-up
  var gW = 0.0, pW = 0.0, ratios: [Double] = [], allBuilds: [Double] = []
  for r in 0..<rounds {
    let order = r % 2 == 0 ? ["gpu", "pipe"] : ["pipe", "gpu"]
    var gr = 0.0, pr = 0.0
    for mode in order {
      if mode == "gpu" {
        let (w, busy) = gpuOnly(); gW += w; gr = Double(jobs) * ops / w / 1e12
        print(String(format: "  round %d gpu : wall %.3f s  %.2f TOPS  (gpu busy %.0f%%)", r, w, gr, busy / w * 100))
      } else {
        let (w, busy, b) = pipe(); pW += w; pr = Double(jobs) * ops / w / 1e12; allBuilds += b
        print(String(format: "  round %d pipe: wall %.3f s  %.2f TOPS  (gpu busy %.0f%%; build median %.2f ms, max %.2f ms)",
                     r, w, pr, busy / w * 100, median(b) * 1e3, b.max()! * 1e3))
      }
    }
    ratios.append(pr / gr)
  }
  releaseLock()
  let gRate = Double(rounds * jobs) * ops / gW / 1e12, pRate = Double(rounds * jobs) * ops / pW / 1e12
  let line = String(format: "%5d %5d %5d | K3-alone %6.2f TOPS | pipeline %6.2f TOPS | P2 %.3f (per-round %@) | build med %.2f ms vs GPU %.2f ms/job",
                    M, N, K, gRate, pRate, pRate / gRate, ratios.map { String(format: "%.3f", $0) }.joined(separator: ","),
                    median(allBuilds) * 1e3, gW / Double(rounds * jobs) * 1e3)
  print("  " + line)
  summary.append(line)
}
print("\n== summary (\(rounds * jobs) jobs per mode per shape) ==")
for s in summary { print("  " + s) }
