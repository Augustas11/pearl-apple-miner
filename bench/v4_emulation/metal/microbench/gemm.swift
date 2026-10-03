import Metal
import Foundation
// Minimal fp32 simdgroup GEMM variants (2048x2048x4096) to locate kernel E's bottleneck.
let src = """
#include <metal_stdlib>
using namespace metal;
template <int TI, int TJ>
inline void body(device const float* A, device const float* B, device float* C, int N, int K, int m0, int n0) {
  simdgroup_float8x8 acc[TI][TJ];
  _Pragma("unroll") for (int i = 0; i < TI; ++i) _Pragma("unroll") for (int j = 0; j < TJ; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  for (int k = 0; k < K; k += 8) {
    simdgroup_float8x8 a[TI], b[TJ];
    _Pragma("unroll") for (int i = 0; i < TI; ++i) simdgroup_load(a[i], A + (m0 + 8 * i) * K + k, K);
    _Pragma("unroll") for (int j = 0; j < TJ; ++j) simdgroup_load(b[j], B + k * N + n0 + 8 * j, N);
    _Pragma("unroll") for (int i = 0; i < TI; ++i) _Pragma("unroll") for (int j = 0; j < TJ; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  _Pragma("unroll") for (int i = 0; i < TI; ++i) _Pragma("unroll") for (int j = 0; j < TJ; ++j) simdgroup_store(acc[i][j], C + (m0 + 8 * i) * N + n0 + 8 * j, N);
}
kernel void g44(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
                constant uint3& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  body<4, 4>(A, B, C, int(d.y), int(d.z), int(tg.y) * 64 + (sg / 2) * 32, int(tg.x) * 64 + (sg % 2) * 32);
}
kernel void g22(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
                constant uint3& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  body<2, 2>(A, B, C, int(d.y), int(d.z), int(tg.y) * 32 + (sg / 2) * 16, int(tg.x) * 32 + (sg % 2) * 16);
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_0
let lib = try! dev.makeLibrary(source: src, options: opts)
let q = dev.makeCommandQueue()!
let M = 2048, N = 2048, K = 4096
let a = dev.makeBuffer(length: M * K * 4)!, b = dev.makeBuffer(length: K * N * 4)!, c = dev.makeBuffer(length: M * N * 4)!
var d = [UInt32(M), UInt32(N), UInt32(K)]
let lock = "/tmp/pmm-gpu-bench.lock"
while (try? FileManager.default.createDirectory(atPath: lock, withIntermediateDirectories: false)) == nil { print("lock busy"); Thread.sleep(forTimeInterval: 15) }
defer { try? FileManager.default.removeItem(atPath: lock) }
for (name, tile) in [("g44", 64), ("g22", 32)] {
  let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
  var best = 1e9
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(ps); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(c, offset: 0, index: 2)
    e.setBytes(&d, length: 12, index: 3)
    e.dispatchThreadgroups(MTLSize(width: N / tile, height: M / tile, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  print(String(format: "%@: %.3f ms, %.2f TFLOP/s", name, best * 1e3, 2.0 * Double(M * N * K) / best / 1e12))
}
