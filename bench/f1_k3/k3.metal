// F1 K3-NA prototype: fused int8 matmul2d + V6 jackpot fold + keyed BLAKE3 + dual U256 compare + bounded found slots.
// Compiled at runtime by k3.swift with -D K3_VARIANT=<0|1|2|3>:
//   0  baseline: int8bench matmul2d (one run() over the whole K), C store DISABLED; destination is a cooperative
//      tensor and the only output is a conditional sink so the GEMM cannot be optimized away
//   1  fold-only: K loop (RK=128, multiply_accumulate) + V6 fold into the 16-word transcript; no hash/compare/atomics
//   2  full K3-NA: 1 + keyed BLAKE3 compress + U256 LE compare vs block and share bounds + atomic found slots
//   3  int8bench kernel unmodified (device-tensor destination, C stored); context only
// Pattern per lane (probe, docs/kb/r6-result.md): rows {r, r+8, r+64, r+72} x cols {c..c+3, +16, +32, +48};
// Pearl rows_pattern [0,8,64,72], cols_pattern [0..3,16..19,32..35,48..51].
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
typedef tensor<device int8_t, dextents<int32_t,2>, tensor_inline> TI8;
typedef tensor<device int, dextents<int32_t,2>, tensor_inline> TI32;

#define BM 128
#define BN 64
#define RANK 128
#define SLOT_WORDS 26   // t_rows, t_cols, transcript[16], hash[8]
#define SINK_MAGIC 0x9E3779B9u

struct K3Params {
  uint M, N, K, cap_block, cap_share, pad0, pad1, pad2;
  uint key[8];          // a_noise_seed as 8 LE u32
  uint bound_block[8];  // U256 LE words (word 7 most significant)
  uint bound_share[8];
};

// ---- BLAKE3 single-block keyed compress (flags CHUNK_START|CHUNK_END|ROOT|KEYED_HASH = 1|2|8|16) ----
#define ROTR(x, n) rotate((x), 32u - (n))
#define G(a, b, c, d, x, y) \
  a = a + b + (x); d = ROTR(d ^ a, 16u); c = c + d; b = ROTR(b ^ c, 12u); \
  a = a + b + (y); d = ROTR(d ^ a, 8u);  c = c + d; b = ROTR(b ^ c, 7u);
#define ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) \
  G(v0, v4, v8,  v12, m[s0],  m[s1]);  G(v1, v5, v9,  v13, m[s2],  m[s3]); \
  G(v2, v6, v10, v14, m[s4],  m[s5]);  G(v3, v7, v11, v15, m[s6],  m[s7]); \
  G(v0, v5, v10, v15, m[s8],  m[s9]);  G(v1, v6, v11, v12, m[s10], m[s11]); \
  G(v2, v7, v8,  v13, m[s12], m[s13]); G(v3, v4, v9,  v14, m[s14], m[s15]);

inline void blake3_keyed_block64(thread const uint* m, constant const uint* key, thread uint* out) {
  uint v0 = key[0], v1 = key[1], v2 = key[2], v3 = key[3], v4 = key[4], v5 = key[5], v6 = key[6], v7 = key[7];
  uint v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
  uint v12 = 0u, v13 = 0u, v14 = 64u, v15 = 27u;   // counter lo/hi = 0, block_len = 64, flags = 27
  ROUND(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
  ROUND(2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8)
  ROUND(3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1)
  ROUND(10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6)
  ROUND(12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4)
  ROUND(9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7)
  ROUND(11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13)
  out[0] = v0 ^ v8; out[1] = v1 ^ v9; out[2] = v2 ^ v10; out[3] = v3 ^ v11;
  out[4] = v4 ^ v12; out[5] = v5 ^ v13; out[6] = v6 ^ v14; out[7] = v7 ^ v15;
}

// U256 little-endian h <= bound (word 7 most significant).
inline bool u256_le(thread const uint* h, constant const uint* bnd) {
  int r = 0;
  #pragma unroll
  for (uint i = 0; i < 8; ++i) r = (h[i] < bnd[i]) ? -1 : ((h[i] > bnd[i]) ? 1 : r);
  return r <= 0;
}

inline void write_slot(device uint* arr, uint idx, uint tr, uint tc, thread const uint* jp, thread const uint* h) {
  device uint* s = arr + (ulong)idx * SLOT_WORDS;
  s[0] = tr; s[1] = tc;
  #pragma unroll
  for (uint j = 0; j < 16; ++j) s[2 + j] = jp[j];
  #pragma unroll
  for (uint j = 0; j < 8; ++j) s[18 + j] = h[j];
}

kernel void k3(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]],
               constant K3Params& P [[buffer(2)]], device atomic_uint* ctr [[buffer(3)]],
               device uint* blk [[buffer(4)]], device uint* shr [[buffer(5)]], device uint* sink [[buffer(6)]],
               device int* c [[buffer(7)]],
               uint2 tgid [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  uint m0 = tgid.y * BM, n0 = tgid.x * BN;
  TI8 A(a, dextents<int32_t,2>(P.K, P.M));   // m x k row-major (dim0 = k)
  TI8 B(b, dextents<int32_t,2>(P.N, P.K));   // k x n row-major (dim0 = n)
  (void)ctr; (void)blk; (void)shr; (void)sink; (void)c; (void)tid;
#if K3_VARIANT == 3
  TI32 C(c, dextents<int32_t,2>(P.N, P.M));
  constexpr auto d = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent));
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, m0); auto tB = B.slice(n0, 0); auto tC = C.slice(n0, m0);
  op.run(tA, tB, tC);
#elif K3_VARIANT == 0
  constexpr auto d = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent));
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, m0); auto tB = B.slice(n0, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int>();
  op.run(tA, tB, cT);
  thread const uint4* p4 = (thread const uint4*)&cT[0];
  uint4 v = p4[0];
  #pragma unroll
  for (ushort i = 1; i < BM * BN / 128 / 4; ++i) v ^= p4[i];
  uint x = v.x ^ v.y ^ v.z ^ v.w;
  if (x == SINK_MAGIC) sink[0] = x ^ tid;   // minimal sink: practically never taken, keeps the GEMM live
#else
  constexpr auto d = matmul2d_descriptor(BM, BN, RANK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA0 = A.slice(0, m0); auto tB0 = B.slice(n0, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), int>();
  #pragma unroll
  for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  uint jp[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  uint nfull = P.K / RANK;   // full rank chunks only; trailing k % r never enters the transcript (no C output)
  for (uint ch = 0; ch < nfull; ++ch) {
    auto tA = A.slice(ch * RANK, m0); auto tB = B.slice(n0, ch * RANK);
    op.run(tA, tB, cT);
    // V6 fold: probe-verified all elements valid and &cT[i] == &cT[0] + i (64 contiguous int32 per lane)
    thread const uint4* p4 = (thread const uint4*)&cT[0];
    uint4 v = p4[0];
    #pragma unroll
    for (ushort i = 1; i < BM * BN / 128 / 4; ++i) v ^= p4[i];
    uint xf = v.x ^ v.y ^ v.z ^ v.w;
    uint s = ch & 15u;   // slot (ll/r - 1) % 16
    #pragma unroll
    for (uint j = 0; j < 16; ++j) jp[j] = (j == s) ? (rotate(jp[j], 13u) ^ xf) : jp[j];
  }
#if K3_VARIANT == 1
  uint x = 0;
  #pragma unroll
  for (uint j = 0; j < 16; ++j) x ^= jp[j];
  if (x == SINK_MAGIC) sink[0] = x ^ tid;
#else
  uint h[8];
  blake3_keyed_block64(jp, P.key, h);
  bool fb = u256_le(h, P.bound_block), fs = u256_le(h, P.bound_share);
  if (fb || fs) {
    // GLOBAL pattern minima of this lane's element set (= Pearl t_rows / t_cols, offset_is_valid by construction)
    uint rmin = 0xFFFFFFFFu, cmin = 0xFFFFFFFFu;
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
      auto ix = cT.get_multidimensional_index(i);
      cmin = min(cmin, (uint)ix[0]); rmin = min(rmin, (uint)ix[1]);
    }
    uint tr = m0 + rmin, tc = n0 + cmin;
    if (fb) {
      uint idx = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
      if (idx < P.cap_block) write_slot(blk, idx, tr, tc, jp, h);
    }
    if (fs) {
      uint idx = atomic_fetch_add_explicit(&ctr[1], 1u, memory_order_relaxed);
      if (idx < P.cap_share) write_slot(shr, idx, tr, tc, jp, h);
    }
  }
#endif
#endif
}
