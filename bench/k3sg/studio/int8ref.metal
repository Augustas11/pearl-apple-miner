// External reference only: the bench/int8bench.swift kernel (Metal 4 matmul2d int8 x int8 -> int32, 128x64 tile,
// execution_simdgroups<4>, one run() over the whole K, C stored). Same text as bench/f1_k3/k3.metal K3_VARIANT 3.
// On Apple7-9 Metal 4 TensorOps fall back to shader implementations; if this fails to compile or run there, the host
// prints the error verbatim and skips the column.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
typedef tensor<device int8_t, dextents<int32_t,2>, tensor_inline> TI8;
typedef tensor<device int, dextents<int32_t,2>, tensor_inline> TI32;

kernel void int8bench(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int* c [[buffer(2)]],
                      constant uint* MNK [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  const uint M = MNK[0], N = MNK[1], K = MNK[2];
  TI8 A(a, dextents<int32_t,2>(K, M));
  TI8 B(b, dextents<int32_t,2>(N, K));
  TI32 C(c, dextents<int32_t,2>(N, M));
  constexpr auto d = matmul2d_descriptor(128, 64, static_cast<int>(dynamic_extent));
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, tgid.y * 128); auto tB = B.slice(tgid.x * 64, 0); auto tC = C.slice(tgid.x * 64, tgid.y * 128);
  op.run(tA, tB, tC);
}
