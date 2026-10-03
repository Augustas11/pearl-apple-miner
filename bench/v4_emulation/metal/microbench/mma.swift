import Metal
import Foundation
// Peak-rate microbenchmark: simdgroup_matrix fp32/fp16 MMA and scalar fp32 FMA, register-only loops.
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma_f32(device float* o [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  simdgroup_float8x8 a(1.0f), b(0.5f), c[4];
  for (int i = 0; i < 4; ++i) c[i] = simdgroup_float8x8(float(tid & 1));
  for (int it = 0; it < 4096; ++it) {
    _Pragma("unroll") for (int i = 0; i < 4; ++i) simdgroup_multiply_accumulate(c[i], a, b, c[i]);
  }
  float s = 0; for (int i = 0; i < 4; ++i) s += c[i].thread_elements()[0];
  if (s == 12345.0f) o[tid] = s;
}
kernel void mma_f16(device float* o [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  simdgroup_half8x8 a(1.0h), b(0.5h); simdgroup_float8x8 c[4];
  for (int i = 0; i < 4; ++i) c[i] = simdgroup_float8x8(float(tid & 1));
  for (int it = 0; it < 4096; ++it) {
    _Pragma("unroll") for (int i = 0; i < 4; ++i) simdgroup_multiply_accumulate(c[i], a, b, c[i]);
  }
  float s = 0; for (int i = 0; i < 4; ++i) s += c[i].thread_elements()[0];
  if (s == 12345.0f) o[tid] = s;
}
kernel void fma_f32(device float* o [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float x[8]; for (int i = 0; i < 8; ++i) x[i] = float(tid + i);
  for (int it = 0; it < 4096; ++it) {
    _Pragma("unroll") for (int i = 0; i < 8; ++i) x[i] = fma(x[i], 0.999f, 0.5f);
  }
  float s = 0; for (int i = 0; i < 8; ++i) s += x[i];
  if (s == 12345.0f) o[tid] = s;
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_0
let lib = try! dev.makeLibrary(source: src, options: opts)
let q = dev.makeCommandQueue()!
let o = dev.makeBuffer(length: 1 << 24)!
let lock = "/tmp/pmm-gpu-bench.lock"
while (try? FileManager.default.createDirectory(atPath: lock, withIntermediateDirectories: false)) == nil { print("lock busy"); Thread.sleep(forTimeInterval: 15) }
defer { try? FileManager.default.removeItem(atPath: lock) }
for (name, flopsPerThreadIter) in [("mma_f32", 4.0 * 2 * 512 / 32), ("mma_f16", 4.0 * 2 * 512 / 32), ("fma_f32", 8.0 * 2)] {
  let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
  let threads = 10 * 1024 * 32
  var best = 1e9
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(ps); e.setBuffer(o, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let flops = Double(threads) * 4096 * flopsPerThreadIter
  print(String(format: "%@: %.2f TFLOP/s (best of 5)", name, flops / best / 1e12))
}
