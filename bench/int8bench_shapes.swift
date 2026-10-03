// Metal 4 matmul2d on the GPU Neural Accelerators: int8 x int8 -> int32 (exact) vs half x half -> float.
import Metal
import Foundation

func shader(_ tin: String, _ tout: String, _ name: String, _ TM: Int, _ TN: Int) -> String { """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void \(name)(device \(tin)* a [[buffer(0)]], device \(tin)* b [[buffer(1)]], device \(tout)* c [[buffer(2)]],
                    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                    uint2 tgid [[threadgroup_position_in_grid]]) {
  tensor<device \(tin),  dextents<int32_t,2>, tensor_inline> A(a, dextents<int32_t,2>(K, M));
  tensor<device \(tin),  dextents<int32_t,2>, tensor_inline> B(b, dextents<int32_t,2>(N, K));
  tensor<device \(tout), dextents<int32_t,2>, tensor_inline> C(c, dextents<int32_t,2>(N, M));
  constexpr auto d = matmul2d_descriptor(\(TM), \(TN), static_cast<int>(dynamic_extent));
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, tgid.y*\(TM)); auto tB = B.slice(tgid.x*\(TN), 0); auto tC = C.slice(tgid.x*\(TN), tgid.y*\(TM));
  op.run(tA, tB, tC);
}
""" }

let dev = MTLCreateSystemDefaultDevice()!
print("device:", dev.name)
let q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_0

func pipeline(_ src: String, _ fn: String) -> MTLComputePipelineState {
  let lib = try! dev.makeLibrary(source: src, options: opts)
  return try! dev.makeComputePipelineState(function: lib.makeFunction(name: fn)!)
}

func run(_ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], M: Int, N: Int, K: Int, TM: Int, TN: Int, iters: Int) -> Double {
  var m = UInt32(M), n = UInt32(N), k = UInt32(K)
  func once() {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&m, length: 4, index: 3); e.setBytes(&n, length: 4, index: 4); e.setBytes(&k, length: 4, index: 5)
    e.dispatchThreadgroups(MTLSize(width: (N+TN-1)/TN, height: (M+TM-1)/TM, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: ps.threadExecutionWidth*4, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  }
  once()
  let t = Date(); for _ in 0..<iters { once() }
  return Date().timeIntervalSince(t) / Double(iters)
}

// 1) exactness: int8 path vs CPU int64 reference, Pearl's real range [-128,127], K = 8192
do {
  let M = 64, N = 64, K = 8192, TM = 64, TN = 32
  var A = [Int8](repeating: 0, count: M*K), B = [Int8](repeating: 0, count: K*N)
  for i in 0..<A.count { A[i] = Int8.random(in: -128...127) }
  for i in 0..<B.count { B[i] = Int8.random(in: -128...127) }
  let ps = pipeline(shader("int8_t", "int32_t", "mm", TM, TN), "mm")
  let bA = dev.makeBuffer(bytes: A, length: A.count)!, bB = dev.makeBuffer(bytes: B, length: B.count)!
  let bC = dev.makeBuffer(length: M*N*4)!; memset(bC.contents(), 0, M*N*4)
  _ = run(ps, [bA, bB, bC], M: M, N: N, K: K, TM: TM, TN: TN, iters: 0)
  let C = bC.contents().bindMemory(to: Int32.self, capacity: M*N)
  var bad = 0
  for i in 0..<M { for j in 0..<N {
    var s: Int64 = 0; for l in 0..<K { s += Int64(A[i*K+l]) * Int64(B[l*N+j]) }
    if Int64(C[i*N+j]) != s { bad += 1 } } }
  print("int8->int32 exact vs CPU (K=8192, values -128..127): \(bad == 0 ? "IDENTICAL" : "\(bad) mismatches")")
}

// 2) same shapes as OpenJarvis benchmark, int8 only, tile 64x32 (fits small M/N)
for (M, N, K) in [(128, 128, 1024), (512, 512, 4096), (1024, 1024, 8192)] {
  let TM = 64, TN = 32
  let bA = dev.makeBuffer(length: M*K)!, bB = dev.makeBuffer(length: K*N)!, bC = dev.makeBuffer(length: M*N*4)!
  let p8 = pipeline(shader("int8_t", "int32_t", "mm8", TM, TN), "mm8")
  let t = run(p8, [bA, bB, bC], M: M, N: N, K: K, TM: TM, TN: TN, iters: 50)
  print(String(format: "Metal int8   m=%d n=%d k=%d: %.4fs -> %.1f GOPS", M, N, K, t, 2.0*Double(M*N*K)/t/1e9))
}
