// R6 microbenchmark: does reading int32 matmul2d accumulators every rank=128 K-steps
// (Pearl v3 XOR-fold jackpot) slow the M5 GPU Neural Accelerator path?
//
// Subcommands:
//   probe                 dump cooperative-tensor layout + Pearl PeriodicPattern validity
//   correct               bit-exact checks vs CPU int64 oracle (accumulators at every 128-K boundary + transcripts)
//   perf [shapes...]      throughput matrix, rounds alternate variants; shapes like 4096x4096x4096
//   sustain SECS [MxNxK]  sustained V2 loop logging TOPS every 10 s
import Metal
import Foundation

// MARK: - setup
setvbuf(stdout, nil, _IOLBF, 0)
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let copts = MTLCompileOptions(); copts.languageVersion = .version4_0
let RANK = 128
let TILES: [(Int, Int)] = [(64, 32), (128, 64)]
let RKS = [16, 32, 64, 128]

func compile(_ src: String) -> MTLLibrary {
  do { return try dev.makeLibrary(source: src, options: copts) }
  catch { print("COMPILE ERROR:\n\(error)"); exit(1) }
}
func pso(_ lib: MTLLibrary, _ name: String) -> MTLComputePipelineState {
  do { return try dev.makeComputePipelineState(function: lib.makeFunction(name: name)!) }
  catch { print("PIPELINE ERROR \(name): \(error)"); exit(1) }
}
func sh(_ cmd: String) -> String {
  let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh"); p.arguments = ["-c", cmd]
  let pipe = Pipe(); p.standardOutput = pipe; try? p.run(); p.waitUntilExit()
  return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - GPU lock (another agent may benchmark concurrently)
let LOCK = "/tmp/pmm-gpu-bench.lock"
func acquireLock() {
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

// MARK: - kernel source
let HDR = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
typedef tensor<device int8_t, dextents<int32_t,2>, tensor_inline> TI8;
typedef tensor<device int, dextents<int32_t,2>, tensor_inline> TI32;
// Morton decode of linear threadgroup index into (tile_x, tile_y); lx, ly = log2 of grid dims.
inline uint2 morton_xy(uint lin, uint lx, uint ly) {
  uint x = 0, y = 0, mn = min(lx, ly);
  for (uint i = 0; i < mn; ++i) { x |= ((lin >> (2*i)) & 1u) << i; y |= ((lin >> (2*i+1)) & 1u) << i; }
  uint rest = lin >> (2*mn);
  if (lx > ly) x |= rest << mn; else y |= rest << mn;
  return uint2(x, y);
}
"""

struct Variant: Hashable {
  var v: Int        // 0 baseline, 1 K-loop no readout, 2 K-loop + in-register fold, 3 K-loop + threadgroup fold
  var bm: Int, bn: Int, rk: Int
  var morton = false
  var dump = false   // debug: write accumulators at every 128-K boundary
  var pair = false   // V2: lane fragment too small -> fold with lane^1 (simd_shuffle_xor)
  var name: String {
    var s = "V\(v)_\(bm)x\(bn)"
    if v != 0 { s += "_rk\(rk)" }
    if morton { s += "_mort" }
    if dump { s += "_dump" }
    return s
  }
  // V3 fallback pattern: 2 rows x W contiguous cols per active thread
  var v3E: Int { max(32, bm * bn / 128) }
  var v3W: Int { v3E / 2 }
  var v3Active: Int { (bm / 2) * (bn / v3W) }
}

func kernelSource(_ x: Variant) -> String {
  let fn = x.name
  let common = """
  uint M = dims.x, N = dims.y, K = dims.z;
  uint lin = tgid.y * tgcount.x + tgid.x;
  uint2 t = \(x.morton ? "morton_xy(lin, dims.w & 255u, dims.w >> 8)" : "tgid");
  uint m0 = t.y * \(x.bm), n0 = t.x * \(x.bn);
  TI8 A(a, dextents<int32_t,2>(K, M));
  TI8 B(b, dextents<int32_t,2>(N, K));
  TI32 C(c, dextents<int32_t,2>(N, M));
  (void)tr; (void)dump; (void)tid; (void)lane;
"""
  if x.v == 0 {
    return """
kernel void \(fn)(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int* c [[buffer(2)]],
                 device uint* tr [[buffer(3)]], device int* dump [[buffer(4)]], constant uint4& dims [[buffer(5)]],
                 uint2 tgid [[threadgroup_position_in_grid]], uint2 tgcount [[threadgroups_per_grid]],
                 ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
\(common)
  constexpr auto d = matmul2d_descriptor(\(x.bm), \(x.bn), static_cast<int>(dynamic_extent));
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, m0); auto tB = B.slice(n0, 0); auto tC = C.slice(n0, m0);
  op.run(tA, tB, tC);
}

"""
  }
  func regFold(_ slot: String) -> String { """
      {
      uint xf = 0;
      #pragma unroll
      for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) xf ^= as_type<uint>(cT[i]);
\(x.pair ? "      xf ^= simd_shuffle_xor(xf, (ushort)1);\n" : "")
      uint s = (\(slot)) & 15u;
      #pragma unroll
      for (uint j = 0; j < 16; ++j) jp[j] = (j == s) ? (rotate(jp[j], 13u) ^ xf) : jp[j];
      }
""" }
  var fold = ""
  if x.v == 6 {
    // V6: probe showed every element valid and storage contiguous int32 (&cT[i] == &cT[0] + i); fold via raw
    // pointer with compile-time count, no per-element is_valid_element/get_element_pointer calls.
    let cap = x.bm * x.bn / 128
    fold = """
      {
      thread const uint4* p4 = (thread const uint4*)&cT[0];
      uint4 v = p4[0];
      #pragma unroll
      for (ushort i = 1; i < \(cap / 4); ++i) v ^= p4[i];
      uint xf = v.x ^ v.y ^ v.z ^ v.w;
\(x.pair ? "      xf ^= simd_shuffle_xor(xf, (ushort)1);\n" : "")
      uint s = ch & 15u;
      #pragma unroll
      for (uint j = 0; j < 16; ++j) jp[j] = (j == s) ? (rotate(jp[j], 13u) ^ xf) : jp[j];
      }
"""
  } else if x.v == 2 {
    fold = regFold("ch")
  } else if x.v == 5 {
    fold = """
      uint xf = as_type<uint>(cT[0]);   // DIAGNOSTIC ONLY (not Pearl-valid): touch a single accumulator element
"""
  } else if x.v == 3 {
    fold = """
      cT.store(TG);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint xf = 0;
      if (tid < \(x.v3Active)) {
        uint rp = tid / \(x.bn / x.v3W), cg = tid % \(x.bn / x.v3W);
        threadgroup const int* r0p = tgm + (2*rp) * \(x.bn) + cg * \(x.v3W);
        for (ushort j = 0; j < \(x.v3W); ++j) {
          ushort jj = (j + lane) & \(x.v3W - 1);   // stagger to spread threadgroup banks; XOR order is irrelevant
          xf ^= as_type<uint>(r0p[jj]) ^ as_type<uint>(r0p[\(x.bn) + jj]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
"""
  }
  if x.v == 3 || x.v == 5 {
    fold += "\n" + """
      uint s = ch & 15u;
      #pragma unroll
      for (uint j = 0; j < 16; ++j) jp[j] = (j == s) ? (rotate(jp[j], 13u) ^ xf) : jp[j];
"""
  }
  let dumpCode = x.dump ? """
      #pragma unroll
      for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        auto ix = cT.get_multidimensional_index(i);
        dump[((ulong)ch * M + m0 + ix[1]) * N + n0 + ix[0]] = cT[i];
      }
""" : ""
  let plain = """
  for (uint ch = 0; ch < nfull; ++ch) {
    for (uint kk = 0; kk < \(RANK); kk += \(x.rk)) {
      auto tA = A.slice(ch * \(RANK) + kk, m0); auto tB = B.slice(n0, ch * \(RANK) + kk);
      op.run(tA, tB, cT);
    }
\(fold)
\(dumpCode)
  }
"""
  // V4: software-pipelined fold. Chunk ch accumulates into a fresh cooperative tensor cN while the lane
  // folds acc_{ch-1} (held in cT); then cT += cN. Exact: same int32 sums, same fold order.
  let pipelined = """
  auto cN = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), int>();
  for (uint ch = 0; ch < nfull; ++ch) {
    #pragma unroll
    for (ushort i = 0; i < cN.get_capacity(); ++i) if (cN.is_valid_element(i)) cN[i] = 0;
    for (uint kk = 0; kk < \(RANK); kk += \(x.rk)) {
      auto tA = A.slice(ch * \(RANK) + kk, m0); auto tB = B.slice(n0, ch * \(RANK) + kk);
      op.run(tA, tB, cN);
    }
    if (ch > 0) \(regFold("ch - 1"))
    #pragma unroll
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] += cN[i];
  }
  if (nfull > 0) \(regFold("nfull - 1"))
"""
  let tg = x.v == 3 ? """
  threadgroup int tgm[\(x.bm * x.bn)];
  tensor<threadgroup int, dextents<int32_t,2>, tensor_inline> TG(tgm, dextents<int32_t,2>(\(x.bn), \(x.bm)));
""" : ""
  let tail = x.v == 1 ? """
  auto tC = C.slice(n0, m0);
  cT.store(tC);
""" : """
  uint g = lin * 128 + tid;
  #pragma unroll
  for (uint j = 0; j < 16; ++j) tr[g * 16 + j] = jp[j];
"""
  return """
kernel void \(fn)(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int* c [[buffer(2)]],
                 device uint* tr [[buffer(3)]], device int* dump [[buffer(4)]], constant uint4& dims [[buffer(5)]],
                 uint2 tgid [[threadgroup_position_in_grid]], uint2 tgcount [[threadgroups_per_grid]],
                 ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
\(common)
\(tg)
  constexpr auto d = matmul2d_descriptor(\(x.bm), \(x.bn), \(x.rk), false, false, false,
                                         matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA0 = A.slice(0, m0); auto tB0 = B.slice(n0, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), int>();
  #pragma unroll
  for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  uint jp[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  (void)jp;
  uint nfull = K / \(RANK);
\(x.v == 4 ? pipelined : plain)
  for (uint kc = nfull * \(RANK); kc < K; kc += \(x.rk)) {   // trailing k % r: never enters the transcript
    auto tA = A.slice(kc, m0); auto tB = B.slice(n0, kc);
    op.run(tA, tB, cT);
  }
\(tail)
}

"""
}

// MARK: - layout probe
struct Lane { var cells: [(Int, Int)] }   // (row, col) tile-local
func probeLayout(bm: Int, bn: Int, rk: Int) -> [[(Int, Int)]] {
  let src = HDR + """
kernel void probe(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int* out [[buffer(2)]],
                  ushort tid [[thread_index_in_threadgroup]]) {
  TI8 A(a, dextents<int32_t,2>(\(rk), \(bm)));
  TI8 B(b, dextents<int32_t,2>(\(bn), \(rk)));
  constexpr auto d = matmul2d_descriptor(\(bm), \(bn), \(rk), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, 0); auto tB = B.slice(0, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int>();
  int cap = cT.get_capacity();
  out[tid * 1025] = cap;
  for (int i = 0; i < cap && i < 256; ++i) {
    auto ix = cT.get_multidimensional_index(i);
    out[tid * 1025 + 1 + i * 4 + 0] = cT.is_valid_element(i) ? 1 : 0;
    out[tid * 1025 + 1 + i * 4 + 1] = ix[0];   // dim0 of destination = column (N)
    out[tid * 1025 + 1 + i * 4 + 2] = ix[1];   // dim1 = row (M)
  }
}
"""
  let lib = compile(src), ps = pso(lib, "probe")
  let ba = dev.makeBuffer(length: bm * rk)!, bb = dev.makeBuffer(length: bn * rk)!, bo = dev.makeBuffer(length: 128 * 1025 * 4)!
  let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps); e.setBuffer(ba, offset: 0, index: 0); e.setBuffer(bb, offset: 0, index: 1); e.setBuffer(bo, offset: 0, index: 2)
  e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let p = bo.contents().bindMemory(to: Int32.self, capacity: 128 * 1025)
  var lanes: [[(Int, Int)]] = []
  for t in 0..<128 {
    let cap = Int(p[t * 1025]); var cells: [(Int, Int)] = []
    for i in 0..<min(cap, 256) where p[t * 1025 + 1 + i * 4] == 1 {
      cells.append((Int(p[t * 1025 + 3 + i * 4]), Int(p[t * 1025 + 2 + i * 4])))
    }
    lanes.append(cells)
  }
  return lanes
}

// Port of zk-pow PeriodicPattern::{from_list, to_bytes, from_bytes, offset_is_valid} (proof_utils.rs:100-245).
struct PPattern: Equatable { var shape: [(UInt32, UInt32)]
  static func == (l: PPattern, r: PPattern) -> Bool { l.shape.map { [$0.0, $0.1] } == r.shape.map { [$0.0, $0.1] } }
  var period: UInt32 { shape.last!.0 * shape.last!.1 }
  var size: UInt32 { shape.reduce(1) { $0 * $1.1 } }
  func offsetIsValid(_ o: UInt32) -> Bool {
    var off = o
    for (s, l) in shape.reversed() { off %= s * l; if off >= s { return false } }
    return true
  }
  func toList() -> [UInt32] {
    var res: [UInt32] = [0]
    for (s, l) in shape { var nr: [UInt32] = []; for i in 0..<l { for r in res { nr.append(r + i * s) } }; res = nr }
    return res
  }
  var desc: String { shape.filter { $0.1 > 1 }.map { "(\($0.0),\($0.1))" }.joined() }
  static func fromBytes(_ d: [UInt32]) -> PPattern? {
    var shape: [(UInt32, UInt32)] = []; var minStride: UInt32 = 1; var done = false
    for i in 0..<3 {
      let factor = 1 + d[2*i], length = 1 + d[2*i+1]
      if length == 1 || done { if !(factor == 1 && length == 1) { return nil }; done = true }
      else if factor <= 1 && minStride != 1 { return nil }
      if !(UInt64(minStride) <= UInt64(1 << 24) / (UInt64(factor) * UInt64(length))) { return nil }
      let stride = factor * minStride; shape.append((stride, length)); minStride = stride * length
    }
    return PPattern(shape: shape)
  }
  func toBytes() -> [UInt32]? {
    var d: [UInt32] = []; var minStride: UInt32 = 1
    for (s, l) in shape {
      if s % minStride != 0 { return nil }
      let f = s / minStride
      if f < 1 || f > 256 || l < 1 || l > 256 { return nil }
      d += [f - 1, l - 1]; minStride = s * l
    }
    return d
  }
  static func fromList(_ pat: [UInt32]) -> (PPattern?, String) {
    if pat.isEmpty { return (nil, "empty") }
    for i in 1..<max(pat.count, 1) where pat[i-1] >= pat[i] { return (nil, "not sorted/unique") }
    if pat[0] != 0 { return (nil, "does not start at 0") }
    var p = pat; var sv: [(UInt32, UInt32)] = []
    while p.count > 1 {
      var found = false
      for period in 1..<p.count where p.count % period == 0 {
        let s = p[period]
        if (0..<(p.count - period)).allSatisfy({ p[$0] + s == p[$0 + period] }) {
          sv.append((s, UInt32(p.count / period))); p = Array(p[0..<period]); found = true; break
        }
      }
      if !found { return (nil, "not periodic") }
    }
    sv.reverse()
    if sv.count > 3 { return (nil, "needs \(sv.count) > 3 dims") }
    let per = sv.last.map { $0.0 * $0.1 } ?? 1
    while sv.count < 3 { sv.append((per, 1)) }
    let r = PPattern(shape: sv)
    guard let b = r.toBytes(), let back = fromBytes(b), back == r else { return (nil, "fails serialization roundtrip") }
    return (r, "ok")
  }
}

struct LayoutVerdict { var valid: Bool; var group: Int; var h: Int; var w: Int; var rows: PPattern?; var cols: PPattern?; var note: String }

// Check that lane groups (tid ^ mask for mask < group) form a valid Pearl hash-tile partition of the BMxBN tile.
func checkGroups(_ lanes: [[(Int, Int)]], bm: Int, bn: Int, group: Int) -> LayoutVerdict {
  var rp: PPattern? = nil, cp: PPattern? = nil, h = 0, w = 0
  var origins = Set<[Int]>(), covered = 0
  for g in stride(from: 0, to: 128, by: group) {
    var cells: [(Int, Int)] = []; for t in g..<(g + group) { cells += lanes[t] }
    let rows = Array(Set(cells.map { $0.0 })).sorted(), cols = Array(Set(cells.map { $0.1 })).sorted()
    if Set(cells.map { [$0.0, $0.1] }).count != cells.count { return LayoutVerdict(valid: false, group: group, h: 0, w: 0, rows: nil, cols: nil, note: "duplicate cells") }
    if rows.count * cols.count != cells.count { return LayoutVerdict(valid: false, group: group, h: rows.count, w: cols.count, rows: nil, cols: nil, note: "lane set is not a rows x cols product") }
    let (r, rn) = PPattern.fromList(rows.map { UInt32($0 - rows[0]) }), (c, cn) = PPattern.fromList(cols.map { UInt32($0 - cols[0]) })
    guard let r = r, let c = c else { return LayoutVerdict(valid: false, group: group, h: rows.count, w: cols.count, rows: nil, cols: nil, note: "rows: \(rn); cols: \(cn)") }
    if rp == nil { rp = r; cp = c; h = rows.count; w = cols.count }
    if r != rp! || c != cp! { return LayoutVerdict(valid: false, group: group, h: h, w: w, rows: rp, cols: cp, note: "lanes use different patterns") }
    origins.insert([rows[0], cols[0]]); covered += cells.count
  }
  let rP = rp!, cP = cp!
  var note = "ok"
  if h % 2 != 0 || w % 2 != 0 { note = "h or w odd" }
  else if h * w < 32 || h * w > 256 { note = "h*w=\(h*w) outside [32,256]" }
  else if bm % Int(rP.period) != 0 || bn % Int(cP.period) != 0 { note = "tile not a multiple of pattern period" }
  else if covered != bm * bn { note = "lanes do not cover tile" }
  else {
    let vr = (0..<bm).filter { rP.offsetIsValid(UInt32($0)) }, vc = (0..<bn).filter { cP.offsetIsValid(UInt32($0)) }
    var expect = Set<[Int]>(); for r in vr { for c in vc { expect.insert([r, c]) } }
    if expect != origins { note = "lane origins != Pearl valid offsets" }
  }
  return LayoutVerdict(valid: note == "ok", group: group, h: h, w: w, rows: rP, cols: cP, note: note)
}

func analyze(bm: Int, bn: Int, rk: Int, verbose: Bool) -> (LayoutVerdict, [[(Int, Int)]]) {
  let lanes = probeLayout(bm: bm, bn: bn, rk: rk)
  if verbose {
    print("tile \(bm)x\(bn) rk=\(rk): capacity(valid) per lane = \(Set(lanes.map { $0.count }))")
    for t in [0, 1, 2, 31, 32, 64, 96, 127] {
      let rows = Array(Set(lanes[t].map { $0.0 })).sorted(), cols = Array(Set(lanes[t].map { $0.1 })).sorted()
      print("  lane \(t): rows \(rows) x cols \(cols)")
    }
  }
  var verdict = checkGroups(lanes, bm: bm, bn: bn, group: 1)
  if verbose { print("  single lane: \(verdict.valid ? "VALID" : "INVALID") h=\(verdict.h) w=\(verdict.w) rows=\(verdict.rows?.desc ?? "-") cols=\(verdict.cols?.desc ?? "-") [\(verdict.note)]") }
  if !verdict.valid {
    for gsz in [2, 4] {
      let v = checkGroups(lanes, bm: bm, bn: bn, group: gsz)
      if verbose { print("  lane group of \(gsz) (simd_shuffle_xor): \(v.valid ? "VALID" : "INVALID") h=\(v.h) w=\(v.w) rows=\(v.rows?.desc ?? "-") cols=\(v.cols?.desc ?? "-") [\(v.note)]") }
      if v.valid { verdict = v; break }
    }
  }
  return (verdict, lanes)
}

// MARK: - variant registry + execution
var pairForTile: [String: Bool] = [:]
var lanesFor: [String: [[(Int, Int)]]] = [:]   // key "bm x bn x rk"
func prepareLayouts(verbose: Bool) {
  for (bm, bn) in TILES {
    for rk in RKS {
      let (v, lanes) = analyze(bm: bm, bn: bn, rk: rk, verbose: verbose && rk == 128)
      lanesFor["\(bm)x\(bn)x\(rk)"] = lanes
      if rk == 128 { pairForTile["\(bm)x\(bn)"] = v.valid && v.group == 2 }
      if !v.valid { print("  WARN: \(bm)x\(bn) rk=\(rk) layout not a valid pattern even with lane groups") }
    }
  }
  // layouts independent of rk? (checked against rk=128)
  for (bm, bn) in TILES {
    let ref = lanesFor["\(bm)x\(bn)x128"]!.map { $0.map { [$0.0, $0.1] } }
    let same = RKS.allSatisfy { lanesFor["\(bm)x\(bn)x\($0)"]!.map { $0.map { [$0.0, $0.1] } } == ref }
    if verbose { print("tile \(bm)x\(bn): destination layout identical for rk in \(RKS): \(same)") }
  }
}

func buildLibrary(_ vs: [Variant]) -> [Variant: MTLComputePipelineState] {
  let t0 = Date()
  let lib = compile(HDR + vs.map(kernelSource).joined())
  var r: [Variant: MTLComputePipelineState] = [:]
  for v in vs { r[v] = pso(lib, v.name) }
  print(String(format: "  compiled %d kernels in %.1f s", vs.count, Date().timeIntervalSince(t0)))
  return r
}

func log2i(_ x: Int) -> Int { var l = 0; while (1 << l) < x { l += 1 }; precondition(1 << l == x, "morton needs power-of-2 grid"); return l }

struct Bufs { var a, b, c, tr, dump: MTLBuffer }
func encode(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ v: Variant, _ bufs: Bufs, M: Int, N: Int, K: Int) {
  precondition(M % v.bm == 0 && N % v.bn == 0 && K % 16 == 0 && (v.v == 0 || K % v.rk == 0))
  let gx = N / v.bn, gy = M / v.bm
  var dims: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), v.morton ? UInt32(log2i(gx) | (log2i(gy) << 8)) : 0]
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps)
  for (i, b) in [bufs.a, bufs.b, bufs.c, bufs.tr, bufs.dump].enumerated() { e.setBuffer(b, offset: 0, index: i) }
  e.setBytes(&dims, length: 16, index: 5)
  e.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.endEncoding()
}
func runOnce(_ ps: MTLComputePipelineState, _ v: Variant, _ bufs: Bufs, M: Int, N: Int, K: Int) -> Double {
  let cb = queue.makeCommandBuffer()!
  encode(cb, ps, v, bufs, M: M, N: N, K: K)
  cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print("GPU ERROR \(v.name): \(err)"); exit(1) }
  return cb.gpuEndTime - cb.gpuStartTime
}
func tileXY(_ lin: Int, gx: Int, gy: Int, morton: Bool) -> (Int, Int) {
  if !morton { return (lin % gx, lin / gx) }
  let lx = log2i(gx), ly = log2i(gy), mn = min(lx, ly); var x = 0, y = 0
  for i in 0..<mn { x |= ((lin >> (2*i)) & 1) << i; y |= ((lin >> (2*i+1)) & 1) << i }
  let rest = lin >> (2*mn); if lx > ly { x |= rest << mn } else { y |= rest << mn }
  return (x, y)
}

func randInt8(_ n: Int) -> [Int8] {
  var a = [Int8](repeating: 0, count: n)
  a.withUnsafeMutableBytes { arc4random_buf($0.baseAddress!, n) }
  for i in 0..<n where a[i] == -128 { a[i] = 127 }   // full Pearl range [-127, 127]
  return a
}

// MARK: - CPU oracle (int64)
// Returns cumulative accumulators at every 128-K boundary: acc[ch][i*N+j] (checked to fit int32), plus final C (full K).
func oracle(A: [Int8], B: [Int8], M: Int, N: Int, K: Int) -> (bounds: [[Int32]], full: [Int64]) {
  var Bt = [Int8](repeating: 0, count: N * K)
  for l in 0..<K { for j in 0..<N { Bt[j * K + l] = B[l * N + j] } }
  let nch = K / RANK
  var acc = [Int64](repeating: 0, count: M * N)
  var bounds = [[Int32]](repeating: [], count: nch)
  var chunk = [Int64](repeating: 0, count: M * N)
  let rangeEnds = Array(stride(from: RANK, through: K, by: RANK)) + (K % RANK != 0 ? [K] : [])
  var start = 0
  for (ci, end) in rangeEnds.enumerated() {
    A.withUnsafeBufferPointer { pa in Bt.withUnsafeBufferPointer { pb in chunk.withUnsafeMutableBufferPointer { pc in
      let pcb = pc.baseAddress!
      DispatchQueue.concurrentPerform(iterations: M) { i in
        for j in 0..<N {
          var s: Int64 = 0
          for l in start..<end { s += Int64(pa[i * K + l]) * Int64(pb[j * K + l]) }
          pcb[i * N + j] = s
        }
      }
    }}}
    for x in 0..<(M * N) { acc[x] += chunk[x] }
    if ci < nch {
      bounds[ci] = acc.map { v in precondition(v >= Int64(Int32.min) && v <= Int64(Int32.max), "int32 overflow"); return Int32(v) }
    }
    start = end
  }
  return (bounds, acc)
}
func rotl13(_ x: UInt32) -> UInt32 { (x << 13) | (x >> 19) }
// Pearl jackpot transcript (mine.rs:87-107) for an element set.
func transcript(_ cells: [(Int, Int)], _ bounds: [[Int32]], N: Int) -> [UInt32] {
  var jp = [UInt32](repeating: 0, count: 16)
  for (ch, acc) in bounds.enumerated() {
    var x: UInt32 = 0
    for (r, c) in cells { x ^= UInt32(bitPattern: acc[r * N + c]) }
    jp[ch % 16] = rotl13(jp[ch % 16]) ^ x
  }
  return jp
}

// MARK: - correctness
func correctness() {
  print("== correctness (device \(dev.name)) ==")
  prepareLayouts(verbose: false)
  var vs: [Variant] = []
  for (bm, bn) in TILES {
    let pair = pairForTile["\(bm)x\(bn)"]!
    vs.append(Variant(v: 0, bm: bm, bn: bn, rk: 0))
    for rk in RKS {
      vs.append(Variant(v: 1, bm: bm, bn: bn, rk: rk))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, morton: true, pair: pair))
      vs.append(Variant(v: 3, bm: bm, bn: bn, rk: rk))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, dump: true, pair: pair))   // accumulators at every boundary
      vs.append(Variant(v: 4, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 6, bm: bm, bn: bn, rk: rk, pair: pair))
    }
  }
  let ps = buildLibrary(vs)
  var allOK = true
  for (M, N, K) in [(256, 256, 8192), (128, 128, 65536), (256, 128, 2048 + 64)] {
    let A = randInt8(M * K), B = randInt8(K * N)
    let t0 = Date(); let (bounds, full) = oracle(A: A, B: B, M: M, N: N, K: K)
    print(String(format: "shape m=%d n=%d k=%d  (oracle %.1f s, %d boundaries, operands in [-127,127])", M, N, K, Date().timeIntervalSince(t0), bounds.count))
    let nch = K / RANK
    let lanesTotal = (M / 64) * (N / 32) * 128 // max over tiles (64x32 has most threadgroups)
    let bufs = Bufs(a: dev.makeBuffer(bytes: A, length: A.count)!, b: dev.makeBuffer(bytes: B, length: B.count)!,
                    c: dev.makeBuffer(length: M * N * 4)!, tr: dev.makeBuffer(length: lanesTotal * 64)!,
                    dump: dev.makeBuffer(length: max(4, nch * M * N * 4))!)
    for v in vs {
      if v.v != 0 && K % v.rk != 0 { print("  \(v.name.padding(toLength: 22, withPad: " ", startingAt: 0)) skipped (k % rk != 0; kernel tail loop steps by rk)"); continue }
      memset(bufs.c.contents(), 0, M * N * 4); memset(bufs.tr.contents(), 0xA5, lanesTotal * 64); memset(bufs.dump.contents(), 0x5A, nch * M * N * 4)
      _ = runOnce(ps[v]!, v, bufs, M: M, N: N, K: K)
      var bad = 0, checked = 0, what = ""
      if v.v <= 1 {
        let C = bufs.c.contents().bindMemory(to: Int32.self, capacity: M * N)
        for x in 0..<(M * N) { checked += 1; if Int64(C[x]) != full[x] { bad += 1 } }
        what = "C (full k)"
      } else if v.dump {
        let D = bufs.dump.contents().bindMemory(to: Int32.self, capacity: nch * M * N)
        for ch in 0..<nch { for x in 0..<(M * N) { checked += 1; if D[ch * M * N + x] != bounds[ch][x] { bad += 1 } } }
        what = "accumulators @ every 128-K boundary"
      } else {
        let T = bufs.tr.contents().bindMemory(to: UInt32.self, capacity: lanesTotal * 16)
        let gx = N / v.bn, gy = M / v.bm
        let lanes = lanesFor["\(v.bm)x\(v.bn)x\(v.rk)"]!
        for lin in 0..<(gx * gy) {
          let (tx, ty) = tileXY(lin, gx: gx, gy: gy, morton: v.morton)
          let m0 = ty * v.bm, n0 = tx * v.bn
          for t in 0..<128 {
            var cells: [(Int, Int)] = []
            if v.v == 2 || v.v == 4 || v.v == 6 {
              let grp = v.pair ? [t & ~1, (t & ~1) | 1] : [t]
              for u in grp { cells += lanes[u].map { (m0 + $0.0, n0 + $0.1) } }
            } else {
              if t >= v.v3Active { continue }
              let cgs = v.bn / v.v3W, rp = t / cgs, cg = t % cgs
              for r in [2 * rp, 2 * rp + 1] { for c in 0..<v.v3W { cells.append((m0 + r, n0 + cg * v.v3W + c)) } }
            }
            let ref = transcript(cells, bounds, N: N)
            let g = lin * 128 + t
            for j in 0..<16 { checked += 1; if T[g * 16 + j] != ref[j] { bad += 1 } }
          }
        }
        what = "jackpot transcript words (\(v.v != 3 ? (v.pair ? "lane-pair fragment" : "lane fragment") : "threadgroup 2x\(v.v3W)"))"
      }
      if bad != 0 { allOK = false }
      print("  \(v.name.padding(toLength: 22, withPad: " ", startingAt: 0)) \(bad == 0 ? "EXACT   " : "MISMATCH") \(bad)/\(checked) bad  [\(what)]")
    }
  }
  print(allOK ? "ALL VARIANTS BIT-EXACT" : "SOME VARIANTS FAILED")
}

// MARK: - perf
func perfVariants() -> [Variant] {
  var vs: [Variant] = []
  for (bm, bn) in TILES {
    let pair = pairForTile["\(bm)x\(bn)"]!
    vs.append(Variant(v: 0, bm: bm, bn: bn, rk: 0))
    vs.append(Variant(v: 0, bm: bm, bn: bn, rk: 0, morton: true))
    for rk in RKS {
      vs.append(Variant(v: 1, bm: bm, bn: bn, rk: rk))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, morton: true, pair: pair))
      vs.append(Variant(v: 3, bm: bm, bn: bn, rk: rk))
    }
  }
  return vs
}
func makePerfBufs(M: Int, N: Int, K: Int, zero: Bool = false) -> Bufs {
  let A = zero ? [Int8](repeating: 0, count: M * K) : randInt8(M * K), B = zero ? [Int8](repeating: 0, count: K * N) : randInt8(K * N)
  let lanes = (M / 64) * (N / 32) * 128
  return Bufs(a: dev.makeBuffer(bytes: A, length: A.count)!, b: dev.makeBuffer(bytes: B, length: B.count)!,
              c: dev.makeBuffer(length: M * N * 4)!, tr: dev.makeBuffer(length: lanes * 64)!, dump: dev.makeBuffer(length: 4)!)
}
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count/2 - 1] + s[s.count/2]) / 2 }

func perf(_ shapes: [(Int, Int, Int)], reps: Int, variants: [Variant]? = nil, zero: Bool = false) {
  print("== perf (device \(dev.name), \(sh("sysctl -n machdep.cpu.brand_string")), macOS \(sh("sw_vers -productVersion"))) operands: \(zero ? "ALL ZERO" : "uniform random [-127,127]") ==")
  if variants == nil { prepareLayouts(verbose: false) }
  let vs = variants ?? perfVariants()
  let ps = buildLibrary(vs)
  for (M, N, K) in shapes {
    let bufs = makePerfBufs(M: M, N: N, K: K, zero: zero)
    let ops = 2.0 * Double(M) * Double(N) * Double(K)
    print("shape \(M)x\(N)x\(K)  (ops = 2mnk = \(String(format: "%.3e", ops)); \(reps) timed rounds, order reversed on odd rounds; GPU timestamps)")
    acquireLock()
    for v in vs { _ = runOnce(ps[v]!, v, bufs, M: M, N: N, K: K) }   // warm-up
    var times: [Variant: [Double]] = [:]
    for r in 0..<reps {
      let order = r % 2 == 0 ? vs : vs.reversed()
      for v in order { times[v, default: []].append(runOnce(ps[v]!, v, bufs, M: M, N: N, K: K)) }
    }
    releaseLock()
    let base: [String: Double] = Dictionary(uniqueKeysWithValues: TILES.map { t in
      ("\(t.0)x\(t.1)", ops / median(times[Variant(v: 0, bm: t.0, bn: t.1, rk: 0)]!) / 1e12) })
    print("  variant                 median TOPS   min TOPS   max TOPS   vs V0 same tile (median)")
    for v in vs {
      let t = times[v]!, med = ops / median(t) / 1e12, mn = ops / t.max()! / 1e12, mx = ops / t.min()! / 1e12
      let b = base["\(v.bm)x\(v.bn)"]!
      print(String(format: "  %@ %10.2f %10.2f %10.2f   %+6.1f%%", v.name.padding(toLength: 22, withPad: " ", startingAt: 0), med, mn, mx, (med / b - 1) * 100))
    }
  }
}

func sustain(seconds: Int, M: Int, N: Int, K: Int, v: Variant) {
  print("== sustained \(v.name) at \(M)x\(N)x\(K) for \(seconds) s (device \(dev.name)) ==")
  let ps = buildLibrary([v])[v]!
  let bufs = makePerfBufs(M: M, N: N, K: K)
  let ops = 2.0 * Double(M) * Double(N) * Double(K)
  acquireLock()
  let t0 = Date(); var winStart = Date(), winOps = 0.0, winGPU = 0.0, all: [Double] = []
  while Date().timeIntervalSince(t0) < Double(seconds) {
    // keep 2 command buffers in flight so the GPU never idles between dispatches
    let cb1 = queue.makeCommandBuffer()!; encode(cb1, ps, v, bufs, M: M, N: N, K: K); cb1.commit()
    let cb2 = queue.makeCommandBuffer()!; encode(cb2, ps, v, bufs, M: M, N: N, K: K); cb2.commit()
    cb2.waitUntilCompleted(); cb1.waitUntilCompleted()
    winOps += 2 * ops; winGPU += (cb1.gpuEndTime - cb1.gpuStartTime) + (cb2.gpuEndTime - cb2.gpuStartTime)
    let el = Date().timeIntervalSince(winStart)
    if el >= 10 {
      let wall = winOps / el / 1e12, gpu = winOps / winGPU / 1e12
      all.append(wall)
      print(String(format: "  t=%5.0f s  wall %.2f TOPS  (gpu-busy %.2f TOPS)", Date().timeIntervalSince(t0), wall, gpu))
      winStart = Date(); winOps = 0; winGPU = 0
    }
  }
  releaseLock()
  if !all.isEmpty { print(String(format: "  first window %.2f, last window %.2f, min %.2f, max %.2f TOPS", all.first!, all.last!, all.min()!, all.max()!)) }
}

// MARK: - main
let args = Array(CommandLine.arguments.dropFirst())
func parseShape(_ s: String) -> (Int, Int, Int) { let p = s.split(separator: "x").map { Int($0)! }; return (p[0], p[1], p[2]) }
switch args.first ?? "" {
case "probe":
  print("== cooperative tensor layout probe (device \(dev.name), macOS \(sh("sw_vers -productVersion"))) ==")
  print("descriptor: matmul2d_descriptor(BM, BN, RK, false, false, false, multiply_accumulate), execution_simdgroups<4>, int8 x int8 -> int32")
  prepareLayouts(verbose: true)
case "correct":
  correctness()
case "perf":
  let shapes = args.count > 1 ? args.dropFirst().map(parseShape) : [(4096, 4096, 4096), (4096, 4096, 2048), (4096, 4096, 8192), (8192, 8192, 2048), (8192, 8192, 8192)]
  perf(shapes, reps: 7)
case "diag":
  // readout-cost decomposition: V1 (no readout) vs V5 (touch 1 element) vs V2 (full fold) vs V4 (pipelined fold)
  let given = args.dropFirst().filter { $0 != "zero" }.map(parseShape)
  let shapes = given.isEmpty ? [(4096, 4096, 4096), (8192, 8192, 8192)] : given
  prepareLayouts(verbose: false)
  var vs: [Variant] = []
  for (bm, bn) in TILES {
    let pair = pairForTile["\(bm)x\(bn)"]!
    vs.append(Variant(v: 0, bm: bm, bn: bn, rk: 0))
    for rk in [64, 128] {
      vs.append(Variant(v: 1, bm: bm, bn: bn, rk: rk))
      vs.append(Variant(v: 5, bm: bm, bn: bn, rk: rk))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 4, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 6, bm: bm, bn: bn, rk: rk, pair: pair))
    }
  }
  perf(shapes, reps: 7, variants: vs, zero: args.contains("zero"))
case "final":
  // focused head-to-head of the decision-relevant variants, 15 alternating rounds
  let given = args.dropFirst().map(parseShape)
  let shapes = given.isEmpty ? [(4096, 4096, 4096), (4096, 4096, 2048), (4096, 4096, 8192), (8192, 8192, 2048), (8192, 8192, 8192)] : given
  prepareLayouts(verbose: false)
  var vs: [Variant] = []
  for (bm, bn) in TILES {
    let pair = pairForTile["\(bm)x\(bn)"]!
    vs.append(Variant(v: 0, bm: bm, bn: bn, rk: 0))
    vs.append(Variant(v: 0, bm: bm, bn: bn, rk: 0, morton: true))
    for rk in [64, 128] {
      vs.append(Variant(v: 1, bm: bm, bn: bn, rk: rk))
      vs.append(Variant(v: 2, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 6, bm: bm, bn: bn, rk: rk, pair: pair))
      vs.append(Variant(v: 6, bm: bm, bn: bn, rk: rk, morton: true, pair: pair))
    }
  }
  perf(shapes, reps: 15, variants: vs)
case "sustain":
  let secs = args.count > 1 ? Int(args[1])! : 180
  let (M, N, K) = args.count > 2 ? parseShape(args[2]) : (8192, 8192, 8192)
  let bm = args.count > 3 ? Int(args[3])! : 128, bn = args.count > 4 ? Int(args[4])! : 64, rk = args.count > 5 ? Int(args[5])! : 128
  let vnum = args.count > 6 ? Int(args[6])! : 2
  prepareLayouts(verbose: false)
  sustain(seconds: secs, M: M, N: N, K: K, v: Variant(v: vnum, bm: bm, bn: bn, rk: rk, morton: args.count > 7 && args[7] == "mort", pair: pairForTile["\(bm)x\(bn)"]!))
default:
  print("usage: r6bench probe | correct | perf [MxNxK ...] | diag [MxNxK ...] [zero] | sustain SECS [MxNxK] [BM BN RK V] [mort]")
}
