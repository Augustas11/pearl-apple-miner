// v4 emulation harness: runs the bit-exact B200 FP8 kernels (kernels.metal, compiled at runtime),
// checks them bit-for-bit against Pearl-oracle outputs, and times them.
//
//   v4emu verify <dir> <M> <N> <K> [a|b|all]          compare with <dir>/c_b200.bin
//   v4emu bench  <dir> <M> <N> <K> <reps> [a|b|all]   timing (GPU timestamps), takes the GPU lock
//
// <dir> holds a.bin (M x K E4M3 codes) and b.bin (N x K codes, B stored transposed) from v4-oracle.
import Foundation
import Metal

setvbuf(stdout, nil, _IOLBF, 0)
let args = CommandLine.arguments
func die(_ s: String) -> Never { FileHandle.standardError.write((s + "\n").data(using: .utf8)!); exit(1) }
guard args.count >= 6 else { die("usage: v4emu verify|bench <dir> <M> <N> <K> [reps] [a|b|all]") }
let mode = args[1], dir = args[2]
let M = Int(args[3])!, N = Int(args[4])!, K = Int(args[5])!
let reps = mode == "bench" ? Int(args[6])! : 0
let which = args.count > (mode == "bench" ? 7 : 6) ? args[mode == "bench" ? 7 : 6] : "all"
let NG = K / 32
let srcDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let kernelPath = ProcessInfo.processInfo.environment["V4EMU_KERNELS"] ?? (srcDir + "/kernels.metal")

func load(_ name: String) -> [UInt8] {
  guard let d = FileManager.default.contents(atPath: dir + "/" + name) else { die("cannot read \(dir)/\(name)") }
  return [UInt8](d)
}
let aCodes = load("a.bin"), bCodes = load("b.bin")
guard aCodes.count == M * K, bCodes.count == N * K else { die("operand size mismatch") }

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
print("device: \(dev.name)  apple10: \(dev.supportsFamily(.apple10))  shape \(M)x\(N)x\(K)  dir \(dir)")

func pipeline(_ defines: String, _ fn: String) -> MTLComputePipelineState {
  let extra = ProcessInfo.processInfo.environment["V4EMU_DEFINES"].map { $0.split(separator: ",").map { "#define \($0) 1\n" }.joined() } ?? ""
  let src = extra + defines + (try! String(contentsOfFile: kernelPath, encoding: .utf8))
  let opts = MTLCompileOptions()
  opts.languageVersion = .version4_0
  opts.mathMode = .safe
  do {
    let lib = try dev.makeLibrary(source: src, options: opts)
    let ps = try dev.makeComputePipelineState(function: lib.makeFunction(name: fn)!)
    print("pipeline \(fn): maxTotalThreadsPerThreadgroup \(ps.maxTotalThreadsPerThreadgroup), threadExecutionWidth \(ps.threadExecutionWidth), static tg mem \(ps.staticThreadgroupMemoryLength)")
    return ps
  } catch { die("compile \(fn) failed: \(error)") }
}

func buf(_ a: [UInt8]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: max(a.count, 1), options: .storageModeShared)! }
func buf32(_ a: [UInt32]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: max(a.count * 4, 4), options: .storageModeShared)! }

// ---------------------------------------------------------------------------
// Kernel B host preprocessing: per (row, group) limbs + metadata. O((M+N)K), outside the timed GEMM.
struct Prep { var cat: [Int8]; var meta: [UInt32] }
func prep(_ codes: [UInt8], rows: Int, transposedOut: Bool) -> Prep {
  // transposedOut=false: rows x 2K, per group [hi(32) lo(32)]
  // transposedOut=true : 2K x rows, per group rows [lo(32); hi(32)]
  var cat = [Int8](repeating: 0, count: rows * 2 * K)
  var meta = [UInt32](repeating: 0, count: rows * NG)
  for r in 0..<rows {
    for g in 0..<NG {
      var l1 = Int.max, p1 = 0, l2 = Int.max, h1 = Int.min, q1 = 0, h2 = Int.min
      var e = [Int](repeating: 0, count: 32), sg = [Int](repeating: 0, count: 32), neg = [Bool](repeating: false, count: 32)
      var any = false
      for t in 0..<32 {
        let c = Int(codes[r * K + g * 32 + t])
        let ef = (c >> 3) & 15, m = c & 7
        sg[t] = ef != 0 ? (m | 8) : m
        e[t] = max(ef, 1)
        neg[t] = c & 0x80 != 0
        if sg[t] == 0 { continue }
        any = true
        let l = e[t] + sg[t].trailingZeroBitCount
        if l < l1 { l2 = l1; l1 = l; p1 = t } else if l < l2 { l2 = l }
        if e[t] > h1 { h2 = h1; h1 = e[t]; q1 = t } else if e[t] > h2 { h2 = e[t] }
      }
      var ok = any
      var ia = [Int](repeating: 0, count: 32)
      if any {
        for t in 0..<32 where sg[t] != 0 {
          let tz = sg[t].trailingZeroBitCount; let v = (sg[t] >> tz) << (e[t] + tz - l1)
          // sig * 2^(e - l1) (integer since e + ctz(sig) >= l1)
          ia[t] = neg[t] ? -v : v
          if v > 32639 { ok = false }
        }
      }
      for t in 0..<32 {
        var hi = 0, lo = 0
        if ok { hi = (ia[t] + 128) >> 8; lo = ia[t] - hi * 256 }
        if transposedOut {
          cat[(g * 64 + t) * rows + r] = Int8(lo)
          cat[(g * 64 + 32 + t) * rows + r] = Int8(hi)
        } else {
          cat[r * 2 * K + g * 64 + t] = Int8(hi)
          cat[r * 2 * K + g * 64 + 32 + t] = Int8(lo)
        }
      }
      if any {
        let L2 = l2 == Int.max ? 31 : l2, H2 = h2 == Int.min ? 0 : h2
        precondition(l1 <= 31 && L2 <= 31)
        var w = UInt32(l1) | UInt32(p1) << 5 | UInt32(L2) << 10
        w |= UInt32(h1) << 15 | UInt32(q1) << 19 | UInt32(H2) << 24
        w |= (ok ? 1 : 0) << 28 | 1 << 29
        meta[r * NG + g] = w
      }
    }
  }
  return Prep(cat: cat, meta: meta)
}

// ---------------------------------------------------------------------------
struct Variant { let name: String; let run: (MTLComputeCommandEncoder) -> Void; let out: MTLBuffer; let stats: MTLBuffer }
var dims = [UInt32(M), UInt32(N), UInt32(K), 0]
let bA = buf(aCodes), bB = buf(bCodes)
var variants: [Variant] = []
var cWindows = 1.0

if which == "all" || which.split(separator: ",").contains("a") {
  let ps = pipeline("#define KERNEL_A 1\n", "kernel_a")
  let out = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let st = dev.makeBuffer(length: 16, options: .storageModeShared)!
  variants.append(Variant(name: "A(alu)", run: { e in
    e.setComputePipelineState(ps)
    e.setBuffer(bA, offset: 0, index: 0); e.setBuffer(bB, offset: 0, index: 1)
    e.setBuffer(out, offset: 0, index: 2); e.setBuffer(st, offset: 0, index: 3)
    e.setBytes(&dims, length: 16, index: 4)
    e.dispatchThreadgroups(MTLSize(width: N / 64, height: M / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  }, out: out, stats: st))
}
if which == "all" || which.split(separator: ",").contains(where: { $0.hasPrefix("b") }) {
  let t0 = Date()
  let pa = prep(aCodes, rows: M, transposedOut: false)
  let pb = prep(bCodes, rows: N, transposedOut: true)
  print(String(format: "kernel B host prep (limbs+meta, O((M+N)K)): %.2fs", Date().timeIntervalSince(t0)))
  let TM = Int(ProcessInfo.processInfo.environment["V4EMU_TM"] ?? "64")!, TN = Int(ProcessInfo.processInfo.environment["V4EMU_TN"] ?? "32")!
  let ps = pipeline("#define KERNEL_B 1\n#define TM \(TM)\n#define TN \(TN)\n#define MAXCAP \(TM * TN / 128)\n", "kernel_b")
  let acat = dev.makeBuffer(bytes: pa.cat, length: pa.cat.count, options: .storageModeShared)!
  let bcat = dev.makeBuffer(bytes: pb.cat, length: pb.cat.count, options: .storageModeShared)!
  let mA = buf32(pa.meta), mB = buf32(pb.meta)
  let out = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let st = dev.makeBuffer(length: 16, options: .storageModeShared)!
  let gmajor = { (meta: [UInt32], rows: Int) -> [UInt32] in
    var t = [UInt32](repeating: 0, count: meta.count)
    for r in 0..<rows { for g in 0..<NG { t[g * rows + r] = meta[r * NG + g] } }
    return t
  }
  let mA2 = buf32(gmajor(pa.meta, M)), mB2 = buf32(gmajor(pb.meta, N))
  let ps2 = pipeline("#define KERNEL_B2 1\n#define TM \(TM)\n#define TN \(TN)\n", "kernel_b2")
  let out2 = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let st2 = dev.makeBuffer(length: 64, options: .storageModeShared)!
  variants.append(Variant(name: "B2(NA int8x4,opt)", run: { e in
    e.setComputePipelineState(ps2)
    e.setBuffer(acat, offset: 0, index: 0); e.setBuffer(bcat, offset: 0, index: 1)
    e.setBuffer(mA2, offset: 0, index: 2); e.setBuffer(mB2, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(out2, offset: 0, index: 6); e.setBuffer(st2, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: ps2.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: out2, stats: st2))
  // B3: 16-bit group-major metadata
  let meta16 = { (meta: [UInt32], rows: Int) -> [UInt16] in
    var t = [UInt16](repeating: 0, count: meta.count)
    for r in 0..<rows { for g in 0..<NG {
      let w = meta[r * NG + g]
      t[g * rows + r] = UInt16(w & 31) | UInt16((w >> 15) & 15) << 5 | UInt16((w >> 28) & 1) << 9 | UInt16((w >> 29) & 1) << 10
    } }
    return t
  }
  let ma16 = meta16(pa.meta, M), mb16 = meta16(pb.meta, N)
  let mA3 = dev.makeBuffer(bytes: ma16, length: ma16.count * 2, options: .storageModeShared)!
  let mB3 = dev.makeBuffer(bytes: mb16, length: mb16.count * 2, options: .storageModeShared)!
  let sg = ProcessInfo.processInfo.environment["V4EMU_SG"] == "1"
  let ps3 = pipeline("#define KERNEL_B3 1\n#define TM \(TM)\n#define TN \(TN)\n" + (sg ? "#define SG_SCOPE 1\n" : ""), "kernel_b3")
  let out3 = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let st3 = dev.makeBuffer(length: 64, options: .storageModeShared)!
  variants.append(Variant(name: "B3(NA int8x4,lean)", run: { e in
    e.setComputePipelineState(ps3)
    e.setBuffer(acat, offset: 0, index: 0); e.setBuffer(bcat, offset: 0, index: 1)
    e.setBuffer(mA3, offset: 0, index: 2); e.setBuffer(mB3, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(out3, offset: 0, index: 6); e.setBuffer(st3, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / (sg ? 4 * TM : TM), depth: 1),
                           threadsPerThreadgroup: MTLSize(width: ps3.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: out3, stats: st3))
  do {
  variants.append(Variant(name: "B(NA int8x4)", run: { e in
    e.setComputePipelineState(ps)
    e.setBuffer(acat, offset: 0, index: 0); e.setBuffer(bcat, offset: 0, index: 1)
    e.setBuffer(mA, offset: 0, index: 2); e.setBuffer(mB, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(out, offset: 0, index: 6); e.setBuffer(st, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: ps.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: out, stats: st))
  }
}

// Kernel C host prep: balanced int8 limbs of 2a (grid 0.5) in window layout + per-(row, window) L2 norms
// (+inf when any element of the row-window is off the 0.5 grid).
if which == "all" || which.split(separator: ",").contains(where: { $0.hasPrefix("c") }) {
  let WG = Int(ProcessInfo.processInfo.environment["V4EMU_WG"] ?? "4")!
  let KW = 32 * WG, NW = K / KW
  let TM = Int(ProcessInfo.processInfo.environment["V4EMU_TM"] ?? "32")!, TN = Int(ProcessInfo.processInfo.environment["V4EMU_TN"] ?? "32")!
  func prepC(_ codes: [UInt8], rows: Int, transposedOut: Bool) -> ([Int8], [Float], Int) {
    var cat = [Int8](repeating: 0, count: rows * 2 * K)
    var norms = [Float](repeating: 0, count: rows * NW)
    var offgrid = 0
    for r in 0..<rows {
      for w in 0..<NW {
        var ss = 0.0, ok = true
        for t in 0..<KW {
          let c = Int(codes[r * K + w * KW + t])
          let ef = (c >> 3) & 15, m = c & 7
          let sig = ef != 0 ? (m | 8) : m
          let e = max(ef, 1)
          var ia = 0
          if sig != 0 {
            // 2a = sig * 2^(e - 9); integer iff e + ctz(sig) >= 9
            if e + sig.trailingZeroBitCount < 9 { ok = false } else { ia = (sig >> sig.trailingZeroBitCount) << (e - 9 + sig.trailingZeroBitCount) }
            if c & 0x80 != 0 { ia = -ia }
          }
          let v = Double(ia) / 2
          ss += v * v
          let hi = (ia + 64) >> 7, lo = ia - hi * 128
          if transposedOut {
            cat[(w * 2 * KW + t) * rows + r] = Int8(lo)
            cat[(w * 2 * KW + KW + t) * rows + r] = Int8(hi)
          } else {
            cat[r * 2 * K + w * 2 * KW + t] = Int8(hi)
            cat[r * 2 * K + w * 2 * KW + KW + t] = Int8(lo)
          }
        }
        if !ok { offgrid += 1 }
        norms[w * rows + r] = ok ? Float(ss.squareRoot()).nextUp : Float.infinity
      }
    }
    return (cat, norms, offgrid)
  }
  let t0 = Date()
  let (ca, na, oa) = prepC(aCodes, rows: M, transposedOut: false)
  let (cb, nb, ob) = prepC(bCodes, rows: N, transposedOut: true)
  print(String(format: "kernel C host prep (W=%d): %.2fs; off-grid row-windows A %.2f%% B %.2f%%", WG, Date().timeIntervalSince(t0),
               100.0 * Double(oa) / Double(M * NW), 100.0 * Double(ob) / Double(N * NW)))
  let psC = pipeline("#define KERNEL_C 1\n#define TM \(TM)\n#define TN \(TN)\n#define WG \(WG)\n", "kernel_c")
  let acat = dev.makeBuffer(bytes: ca, length: ca.count, options: .storageModeShared)!
  let bcat = dev.makeBuffer(bytes: cb, length: cb.count, options: .storageModeShared)!
  let nA = dev.makeBuffer(bytes: na, length: na.count * 4, options: .storageModeShared)!
  let nB = dev.makeBuffer(bytes: nb, length: nb.count * 4, options: .storageModeShared)!
  let outC = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let stC = dev.makeBuffer(length: 64, options: .storageModeShared)!
  variants.append(Variant(name: "C(NA grid-exact W=\(WG))", run: { e in
    e.setComputePipelineState(psC)
    e.setBuffer(acat, offset: 0, index: 0); e.setBuffer(bcat, offset: 0, index: 1)
    e.setBuffer(nA, offset: 0, index: 2); e.setBuffer(nB, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(outC, offset: 0, index: 6); e.setBuffer(stC, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: psC.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: outC, stats: stC))
  cWindows = Double(M) * Double(N) * Double(NW)
  if WG == 1 {
    // C2: whole-row grid flags (all row-windows finite) and the same limbs
    var ga = [UInt8](repeating: 1, count: M), gb = [UInt8](repeating: 1, count: N)
    for r in 0..<M { for w in 0..<NW where !na[w * M + r].isFinite { ga[r] = 0 } }
    for r in 0..<N { for w in 0..<NW where !nb[w * N + r].isFinite { gb[r] = 0 } }
    let gA = buf(ga), gB = buf(gb)
    let psC2 = pipeline("#define KERNEL_C2 1\n#define TM \(TM)\n#define TN \(TN)\n", "kernel_c2")
    let outC2 = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
    let stC2 = dev.makeBuffer(length: 64, options: .storageModeShared)!
    variants.append(Variant(name: "C2(NA grid-exact lean)", run: { e in
      e.setComputePipelineState(psC2)
      e.setBuffer(acat, offset: 0, index: 0); e.setBuffer(bcat, offset: 0, index: 1)
      e.setBuffer(gA, offset: 0, index: 2); e.setBuffer(gB, offset: 0, index: 3)
      e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
      e.setBuffer(outC2, offset: 0, index: 6); e.setBuffer(stC2, offset: 0, index: 7)
      e.setBytes(&dims, length: 16, index: 8)
      e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: psC2.threadExecutionWidth * 4, height: 1, depth: 1))
    }, out: outC2, stats: stC2))
  }
}

// Kernel D host prep: fp16 operands (E4M3 values are exact in fp16), A as M x K, B as K x N,
// per-(row, window) L2 norms rounded up, +inf when the row-window is off the 0.5 grid.
if which == "all" || which.split(separator: ",").contains(where: { $0.hasPrefix("d") }) {
  let WG = Int(ProcessInfo.processInfo.environment["V4EMU_WG"] ?? "1")!
  let KW = 32 * WG, NW = K / KW
  let TM = Int(ProcessInfo.processInfo.environment["V4EMU_TM"] ?? "64")!, TN = Int(ProcessInfo.processInfo.environment["V4EMU_TN"] ?? "32")!
  func e4m3(_ c: Int) -> (Double, Bool) {  // value, on 0.5 grid
    let ef = (c >> 3) & 15, m = c & 7
    let sig = ef != 0 ? (m | 8) : m
    let e = max(ef, 1)
    let v = Double(sig) * pow(2.0, Double(e - 10)) * (c & 0x80 != 0 ? -1 : 1)
    return (v, sig == 0 || e + sig.trailingZeroBitCount >= 9)
  }
  let lut = (0..<256).map { e4m3($0) }
  func prepD(_ codes: [UInt8], rows: Int, transposedOut: Bool) -> ([Float16], [Float]) {
    var h = [Float16](repeating: 0, count: rows * K)
    var norms = [Float](repeating: 0, count: rows * NW)
    for r in 0..<rows {
      for w in 0..<NW {
        var ss = 0.0, ok = true
        for t in 0..<KW {
          let (v, g) = lut[Int(codes[r * K + w * KW + t])]
          ss += v * v; ok = ok && g
          let hv = Float16(v); precondition(Double(hv) == v)
          if transposedOut { h[(w * KW + t) * rows + r] = hv } else { h[r * K + w * KW + t] = hv }
        }
        norms[w * rows + r] = ok ? Float(ss.squareRoot()).nextUp : Float.infinity
      }
    }
    return (h, norms)
  }
  let (ha, na) = prepD(aCodes, rows: M, transposedOut: false)
  let (hb, nb) = prepD(bCodes, rows: N, transposedOut: true)
  let psD = pipeline("#define KERNEL_D 1\n#define TM \(TM)\n#define TN \(TN)\n#define WG \(WG)\n", "kernel_d")
  let hA = dev.makeBuffer(bytes: ha, length: ha.count * 2, options: .storageModeShared)!
  let hB = dev.makeBuffer(bytes: hb, length: hb.count * 2, options: .storageModeShared)!
  let nA = dev.makeBuffer(bytes: na, length: na.count * 4, options: .storageModeShared)!
  let nB = dev.makeBuffer(bytes: nb, length: nb.count * 4, options: .storageModeShared)!
  let outD = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let stD = dev.makeBuffer(length: 64, options: .storageModeShared)!
  variants.append(Variant(name: "D(NA fp16 grid-exact W=\(WG))", run: { e in
    e.setComputePipelineState(psD)
    e.setBuffer(hA, offset: 0, index: 0); e.setBuffer(hB, offset: 0, index: 1)
    e.setBuffer(nA, offset: 0, index: 2); e.setBuffer(nB, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(outD, offset: 0, index: 6); e.setBuffer(stD, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: psD.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: outD, stats: stD))
  cWindows = Double(M) * Double(N) * Double(NW)
  // D2: per (row-block, window) and (col-block, window) maxima of the norms
  func tileMax(_ norms: [Float], rows: Int, block: Int) -> [Float] {
    var t = [Float](repeating: 0, count: (rows / block) * NW)
    for w in 0..<NW { for r in 0..<rows { let j = w * (rows / block) + r / block; t[j] = max(t[j], norms[w * rows + r]) } }
    return t
  }
  let ta = tileMax(na, rows: M, block: TM), tb = tileMax(nb, rows: N, block: TN)
  let tA = dev.makeBuffer(bytes: ta, length: ta.count * 4, options: .storageModeShared)!
  let tB = dev.makeBuffer(bytes: tb, length: tb.count * 4, options: .storageModeShared)!
  let psD2 = pipeline("#define KERNEL_D2 1\n#define TM \(TM)\n#define TN \(TN)\n#define WG \(WG)\n", "kernel_d2")
  let outD2 = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let stD2 = dev.makeBuffer(length: 64, options: .storageModeShared)!
  variants.append(Variant(name: "D2(NA fp16 grid-exact tile-bound W=\(WG))", run: { e in
    e.setComputePipelineState(psD2)
    e.setBuffer(hA, offset: 0, index: 0); e.setBuffer(hB, offset: 0, index: 1)
    e.setBuffer(tA, offset: 0, index: 2); e.setBuffer(tB, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(outD2, offset: 0, index: 6); e.setBuffer(stD2, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: psD2.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: outD2, stats: stD2))
}

// Kernel E host prep: fp32 operands (A: M x K, B: K x N), per (64-row block, group) norm maxima.
if which == "all" || which.split(separator: ",").contains("e") {
  var af = [Float](repeating: 0, count: M * K), bfT = [Float](repeating: 0, count: K * N)
  var na = [Float](repeating: 0, count: M * NG), nb = [Float](repeating: 0, count: N * NG)
  func val(_ c: Int) -> (Double, Bool) {
    let ef = (c >> 3) & 15, m = c & 7
    let sig = ef != 0 ? (m | 8) : m
    let e = max(ef, 1)
    return (Double(sig) * pow(2.0, Double(e - 10)) * (c & 0x80 != 0 ? -1 : 1), sig == 0 || e + sig.trailingZeroBitCount >= 9)
  }
  let lut = (0..<256).map { val($0) }
  for (codes, rows, isB) in [(aCodes, M, false), (bCodes, N, true)] {
    for r in 0..<rows { for g in 0..<NG {
      var ss = 0.0, ok = true
      for t in 0..<32 {
        let (v, gr) = lut[Int(codes[r * K + g * 32 + t])]
        ss += v * v; ok = ok && gr
        if isB { bfT[(g * 32 + t) * N + r] = Float(v) } else { af[r * K + g * 32 + t] = Float(v) }
      }
      let nv: Float = ok ? Float(ss.squareRoot()).nextUp : Float.infinity
      if isB { nb[g * N + r] = nv } else { na[g * M + r] = nv }
    } }
  }
  let EB = Int(ProcessInfo.processInfo.environment["V4EMU_EB"] ?? "2")!, ET = 16 * EB
  func tileMax(_ norms: [Float], rows: Int) -> [Float] {
    var t = [Float](repeating: 0, count: (rows / ET) * NG)
    for g in 0..<NG { for r in 0..<rows { let j = g * (rows / ET) + r / ET; t[j] = max(t[j], norms[g * rows + r]) } }
    return t
  }
  let ta = tileMax(na, rows: M), tb = tileMax(nb, rows: N)
  let fA = dev.makeBuffer(bytes: af, length: af.count * 4, options: .storageModeShared)!
  let fB = dev.makeBuffer(bytes: bfT, length: bfT.count * 4, options: .storageModeShared)!
  let tA = dev.makeBuffer(bytes: ta, length: ta.count * 4, options: .storageModeShared)!
  let tB = dev.makeBuffer(bytes: tb, length: tb.count * 4, options: .storageModeShared)!
  let EW = Int(ProcessInfo.processInfo.environment["V4EMU_EW"] ?? "1")!
  let psE = pipeline("#define KERNEL_E 1\n#define EB \(EB)\n#define EW \(EW)\n", "kernel_e")
  let outE = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  let stE = dev.makeBuffer(length: 64, options: .storageModeShared)!
  let noCheck = ProcessInfo.processInfo.environment["V4EMU_DEFINES"]?.contains("E_NO_CHECK") == true
  variants.append(Variant(name: noCheck ? "E(NO-CHECK plain fp32 simdgroup GEMM, not exact-guarded)" : "E(fp32 simdgroup grid-exact)", run: { e in
    e.setComputePipelineState(psE)
    e.setBuffer(fA, offset: 0, index: 0); e.setBuffer(fB, offset: 0, index: 1)
    e.setBuffer(tA, offset: 0, index: 2); e.setBuffer(tB, offset: 0, index: 3)
    e.setBuffer(bA, offset: 0, index: 4); e.setBuffer(bB, offset: 0, index: 5)
    e.setBuffer(outE, offset: 0, index: 6); e.setBuffer(stE, offset: 0, index: 7)
    e.setBytes(&dims, length: 16, index: 8)
    e.dispatchThreadgroups(MTLSize(width: N / ET, height: M / ET, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  }, out: outE, stats: stE))
  cWindows = Double(M) * Double(N) * Double(NG / EW)
}

// selection: comma list of a,b,b2,b3,c,c2,d,d2,e or all
if which != "all" {
  let want = Set(which.split(separator: ",").map { String($0).uppercased() + "(" })
  variants = variants.filter { v in want.contains(where: { v.name.hasPrefix($0) }) }
}

func runOnce(_ v: Variant) -> Double {
  memset(v.stats.contents(), 0, v.stats.length)
  let cb = queue.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  v.run(e)
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { die("GPU error in \(v.name): \(err)") }
  return cb.gpuEndTime - cb.gpuStartTime
}
func stats(_ v: Variant) -> (UInt32, UInt32) {
  let p = v.stats.contents().bindMemory(to: UInt32.self, capacity: 4)
  return (p[0], p[1])
}
let cellGroups = Double(M) * Double(N) * Double(NG)

if mode == "verify" {
  guard let refData = FileManager.default.contents(atPath: dir + "/c_b200.bin") else { die("no c_b200.bin") }
  let ref = refData.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
  for v in variants {
    memset(v.out.contents(), 0xFF, M * N * 4)
    let t = runOnce(v)
    let (s0, s1) = stats(v)
    let o = v.out.contents().bindMemory(to: UInt32.self, capacity: M * N)
    var bad = 0, first: [String] = []
    for i in 0..<(M * N) where o[i] != ref[i] {
      bad += 1
      if first.count < 5 { first.append(String(format: "cell(%d,%d) got %08x want %08x", i / N, i % N, o[i], ref[i])) }
    }
    let extra = v.name.hasPrefix("A") ? String(format: "pass1 thread-groups %u", s0)
                                      : (v.name.hasPrefix("C") || v.name.hasPrefix("D") || v.name.hasPrefix("E")) ? String(format: "fallback cell-windows %u = %.4f%% (layout violations %u)", s0, 100.0 * Double(s0) / cWindows, s1)
                                      : String(format: "fallback cell-groups %u = %.4f%% (layout/cap violations %u)", s0, 100.0 * Double(s0) / cellGroups, s1)
    print(String(format: "VERIFY %@ %@: %d cells, %d mismatches -> %@ | %@ | %.1f ms",
                 v.name, (dir as NSString).lastPathComponent, M * N, bad, bad == 0 ? "BIT-EXACT" : "INVALID", extra, t * 1e3))
    for f in first { print("   " + f) }
  }
} else if mode == "bench" {
  let lock = "/tmp/pmm-gpu-bench.lock"
  func acquire() {
    while true {
      if (try? FileManager.default.createDirectory(atPath: lock, withIntermediateDirectories: false)) != nil { return }
      print("  [lock busy, retry in 15 s]"); Thread.sleep(forTimeInterval: 15)
    }
  }
  let ops = 2.0 * Double(M) * Double(N) * Double(K)
  // Baseline (requested): the unmodified bench/int8bench.swift kernel, int8 x int8 -> int32
  // matmul2d at 4096^3 with tile 128x64, timed in every round right next to the kernels.
  let baseSrc = """
  #include <metal_stdlib>
  #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
  using namespace metal; using namespace mpp::tensor_ops;
  kernel void mm8(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int32_t* c [[buffer(2)]],
                      constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                      uint2 tgid [[threadgroup_position_in_grid]]) {
    tensor<device int8_t,  dextents<int32_t,2>, tensor_inline> A(a, dextents<int32_t,2>(K, M));
    tensor<device int8_t,  dextents<int32_t,2>, tensor_inline> B(b, dextents<int32_t,2>(N, K));
    tensor<device int32_t, dextents<int32_t,2>, tensor_inline> C(c, dextents<int32_t,2>(N, M));
    constexpr auto d = matmul2d_descriptor(128, 64, static_cast<int>(dynamic_extent));
    matmul2d<d, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y*128); auto tB = B.slice(tgid.x*64, 0); auto tC = C.slice(tgid.x*64, tgid.y*128);
    op.run(tA, tB, tC);
  }
  """
  let bopts = MTLCompileOptions(); bopts.languageVersion = .version4_0
  let baseLib = try! dev.makeLibrary(source: baseSrc, options: bopts)
  let basePs = try! dev.makeComputePipelineState(function: baseLib.makeFunction(name: "mm8")!)
  let BS = 4096
  let bA8 = dev.makeBuffer(length: BS * BS, options: .storageModeShared)!, bB8 = dev.makeBuffer(length: BS * BS, options: .storageModeShared)!
  let bC32 = dev.makeBuffer(length: BS * BS * 4, options: .storageModeShared)!
  var bsz = [UInt32(BS), UInt32(BS), UInt32(BS)]
  let baseline = Variant(name: "BASE(int8 mm 4096^3)", run: { e in
    e.setComputePipelineState(basePs)
    e.setBuffer(bA8, offset: 0, index: 0); e.setBuffer(bB8, offset: 0, index: 1); e.setBuffer(bC32, offset: 0, index: 2)
    e.setBytes(&bsz[0], length: 4, index: 3); e.setBytes(&bsz[1], length: 4, index: 4); e.setBytes(&bsz[2], length: 4, index: 5)
    e.dispatchThreadgroups(MTLSize(width: BS / 64, height: BS / 128, depth: 1), threadsPerThreadgroup: MTLSize(width: basePs.threadExecutionWidth * 4, height: 1, depth: 1))
  }, out: bC32, stats: dev.makeBuffer(length: 64, options: .storageModeShared)!)
  let baseOps = 2.0 * Double(BS) * Double(BS) * Double(BS)
  func loadavg() -> String { var l = [Double](repeating: 0, count: 3); getloadavg(&l, 3); return String(format: "%.2f %.2f %.2f", l[0], l[1], l[2]) }
  let cool = Double(ProcessInfo.processInfo.environment["V4EMU_COOL"] ?? "0") ?? 0
  let rounds = Int(ProcessInfo.processInfo.environment["V4EMU_ROUNDS"] ?? "2") ?? 2
  for round in 0..<rounds {  // alternate variants across rounds
    for v in variants {
      if cool > 0 { Thread.sleep(forTimeInterval: cool) }  // fanless MacBook Air: let the SoC cool between batches
      acquire()
      let la0 = loadavg()
      // baseline, kernel, baseline (bracketing) inside the same lock window
      _ = runOnce(baseline); _ = runOnce(baseline)
      var tb: [Double] = []
      for _ in 0..<3 { tb.append(runOnce(baseline)) }
      _ = runOnce(v); _ = runOnce(v)  // warm-up
      var ts: [Double] = []
      for _ in 0..<reps { ts.append(runOnce(v)) }
      for _ in 0..<3 { tb.append(runOnce(baseline)) }
      let la1 = loadavg()
      try? FileManager.default.removeItem(atPath: lock)
      tb.sort(); ts.sort()
      let bmed = tb[tb.count / 2]
      let baseT = baseOps / bmed / 1e12
      let med = ts[ts.count / 2], mn = ts[0]
      let ratio = (ops / med / 1e12) / baseT
      print(String(format: "  baseline int8 4096^3 (median of 6, bracketing): %.2f TOPS | loadavg before [%@] after [%@] | %@ median/baseline = %.4f -> normalized to idle 19 TOPS: %.3f TOPS-eq",
                   baseT, la0, la1, v.name, ratio, ratio * 19.0))
      let (s0, _) = stats(v)
      let fb = v.name.hasPrefix("B") ? String(format: " fallback %.4f%%", 100.0 * Double(s0) / cellGroups)
             : (v.name.hasPrefix("C") || v.name.hasPrefix("D") || v.name.hasPrefix("E")) ? String(format: " fallback-windows %.4f%%", 100.0 * Double(s0) / cWindows) : ""
      print(String(format: "BENCH r%d %@ %@ %dx%dx%d: median %.3f ms (%.3f TOPS-eq), min %.3f ms (%.3f TOPS-eq), reps %d%@",
                   round, v.name, (dir as NSString).lastPathComponent, M, N, K, med * 1e3, ops / med / 1e12, mn * 1e3, ops / mn / 1e12, reps, fb))
    }
  }
}
