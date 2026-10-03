// K3-SG: Pearl v3 mining kernel for Apple7-9 GPUs (M1-M4) on fp32 simdgroup_matrix (SPEC §5.3).
// Compiled at runtime by k3sg.swift (MTLLanguageVersion 3.1, no Metal 4 / MPP) with -D macros:
//   VARIANT 0  base_i8 : same tiling, int8 operands converted to float while staging, ONE fp32 accumulator over the whole
//                        K (not exact; perf reference only), C store disabled (conditional sink)
//   VARIANT 1  base_f32: same tiling, fp32 operands from device memory ("plain fp32 simdgroup_matrix GEMM"), C store
//                        disabled (conditional sink). P3 baseline.
//   VARIANT 2  fold    : production K loop (fresh fp32 accumulator per 128-wide rank chunk, int32 running accumulator,
//                        XOR fold, rotl13 into 16-word transcript); sink only
//   VARIANT 3  k3      : 2 + keyed BLAKE3 compress + U256 LE compare vs block and share bounds + bounded atomic found slots
//   BM, BN   threadgroup tile; BK k-step staged in threadgroup memory (divides 128); WM x WN simdgroups, each 32x32;
//   PF       0 = global->threadgroup between two barriers, 1 = + register prefetch of the next k-step,
//            2 = register prefetch + double-buffered threadgroup tiles (one barrier per k-step).
// All variants share the loop structure (K/128 chunks x 128/BK k-steps); only the per-chunk epilogue differs.
//
// Exactness: operands in [-127,127] are exact in fp32; one rank chunk sums 128 products of magnitude <= 127^2, so
// |chunk sum| <= 128*16129 = 2,064,512 < 2^24 and every fp32 partial sum is an exactly representable integer. The chunk
// result is converted to int and added to an int32 running accumulator (|acc| <= 65536*16129 < 2^31).
//
// Per-lane element set (simdgroup_matrix 8x8 lane layout, MLX steel mma.h get_coord; VERIFIED at runtime by sg_probe):
//   lane holds row fm, cols {fn, fn+1} of each 8x8 fragment, fm = (qid&4) + ((lane>>1)&3), fn = (qid&2)*2 + (lane&1)*2,
//   qid = lane>>2. A 32x32 simdgroup tile (4x4 fragments) gives each lane
//   rows {fm + 0, 8, 16, 24} x cols {fn + 0, 1, 8, 9, 16, 17, 24, 25}  ->  Pearl rows_pattern [0,8,16,24] (h=4),
//   cols_pattern [0,1,8,9,16,17,24,25] (w=8), h*w = 32. Valid offsets = {o : o%32 < 8} x {o : o%32 < 8, o even}
//   = exactly the lane origins of 32-aligned simdgroup tiles, so lane tiles partition the matrix.
#include <metal_stdlib>
using namespace metal;

#ifndef VARIANT
#define VARIANT 3
#endif
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 16
#endif
#ifndef WM
#define WM 2
#endif
#ifndef WN
#define WN 2
#endif
#ifndef PF
#define PF 1              // 0: load global->threadgroup between two barriers; 1: + register prefetch of the next k-step;
#endif                    // 2: register prefetch + double-buffered threadgroup tiles (one barrier per k-step)

#define SGT 32            // simdgroup tile edge
#define TM 4              // 8x8 fragments per simdgroup tile, rows
#define TN 4              // ... cols
#define NT (32 * WM * WN) // threads per threadgroup
#define RANK 128
#define PADA 4
#define PADB 4
#define LDA (BK + PADA)
#define LDB (BN + PADB)
// #pragma unroll is NOT honoured for these loops by the Metal compiler (measured on M5: C[][] fragments spilled,
// ~10x slower); MLX steel uses the clang loop pragma.
#define UNROLL _Pragma("clang loop unroll(full)")
#define SLOT_WORDS 26     // t_rows, t_cols, transcript[16], hash[8]
#define SINK_MAGIC 0x9E3779B9u

static_assert(BM == WM * SGT && BN == WN * SGT, "threadgroup tile must be WM x WN simdgroup tiles of 32x32");
static_assert(RANK % BK == 0 && BK % 8 == 0, "BK must be a multiple of 8 dividing 128");
static_assert((BM * BK / 4) % NT == 0 && (BN * BK / 4) % NT == 0, "tile loads must split evenly over threads");
#define LA (BM * BK / 4 / NT)
#define LB (BN * BK / 4 / NT)

struct K3Params {
  uint M, N, K, cap_block, cap_share, pad0, pad1, pad2;
  uint key[8];          // a_noise_seed as 8 LE u32
  uint bound_block[8];  // U256 LE words (word 7 most significant)
  uint bound_share[8];
};

// ---- BLAKE3 single-block keyed compress (flags CHUNK_START|CHUNK_END|ROOT|KEYED_HASH = 27), same as bench/f1_k3 ----
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
  uint v12 = 0u, v13 = 0u, v14 = 64u, v15 = 27u;
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

inline bool u256_le(thread const uint* h, constant const uint* bnd) {
  int r = 0;
  UNROLL
  for (uint i = 0; i < 8; ++i) r = (h[i] < bnd[i]) ? -1 : ((h[i] > bnd[i]) ? 1 : r);
  return r <= 0;
}

inline void write_slot(device uint* arr, uint idx, uint tr, uint tc, thread const uint* jp, thread const uint* h) {
  device uint* s = arr + (ulong)idx * SLOT_WORDS;
  s[0] = tr; s[1] = tc;
  UNROLL
  for (uint j = 0; j < 16; ++j) s[2 + j] = jp[j];
  UNROLL
  for (uint j = 0; j < 8; ++j) s[18 + j] = h[j];
}

#if VARIANT == 1
typedef float in_t;
typedef float4 in4_t;
#else
typedef char in_t;
typedef char4 in4_t;
#endif

kernel void k3sg(device const in_t* a [[buffer(0)]],   // A' m x k row-major
                 device const in_t* b [[buffer(1)]],   // B' k x n row-major
                 constant K3Params& P [[buffer(2)]], device atomic_uint* ctr [[buffer(3)]],
                 device uint* blk [[buffer(4)]], device uint* shr [[buffer(5)]], device uint* sink [[buffer(6)]],
                 uint2 tgid [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                 ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
#if PF == 2
  threadgroup float As[2][BM * LDA];   // double-buffered: one barrier per k-step
  threadgroup float Bs[2][BK * LDB];
#else
  threadgroup float As[1][BM * LDA];
  threadgroup float Bs[1][BK * LDB];
#endif
  const uint m0 = tgid.y * BM, n0 = tgid.x * BN;
  const ushort sm = sgid / WN, sn = sgid % WN;
  const uint K = P.K, N = P.N;
  device const in_t* ap = a + (ulong)m0 * K;
  device const in_t* bp = b + n0;
  (void)ctr; (void)blk; (void)shr;

  simdgroup_float8x8 C[TM][TN];
  UNROLL
  for (ushort i = 0; i < TM; ++i)
    UNROLL
    for (ushort j = 0; j < TN; ++j) C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
#if VARIANT >= 2
  int acc[TM][TN][2];
  UNROLL
  for (ushort i = 0; i < TM; ++i)
    UNROLL
    for (ushort j = 0; j < TN; ++j) { acc[i][j][0] = 0; acc[i][j][1] = 0; }
  uint jp[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
#endif

  const uint nchunks = K / RANK;   // full rank chunks only; trailing k % r never runs (no C output)
  const uint nsteps = nchunks * (RANK / BK);
  // per-thread share of one k-step tile: LA float4 of A (BM x BK), LB float4 of B (BK x BN)
  uint aoff[LA], boff[LB];        // device element offsets relative to k0 = 0
  ushort ta[LA], tb[LB];          // threadgroup float offsets
  UNROLL
  for (ushort q = 0; q < LA; ++q) {
    const uint i = tid + q * NT, r = i / (BK / 4), c = (i % (BK / 4)) * 4;
    aoff[q] = r * K + c; ta[q] = r * LDA + c;
  }
  UNROLL
  for (ushort q = 0; q < LB; ++q) {
    const uint i = tid + q * NT, r = i / (BN / 4), c = (i % (BN / 4)) * 4;
    boff[q] = r * N + c; tb[q] = r * LDB + c;
  }
#if PF
  in4_t ra[LA], rb[LB];   // register prefetch of the next k-step (int8: 4 bytes per float4 of staging)
  UNROLL
  for (ushort q = 0; q < LA; ++q) ra[q] = *(device const in4_t*)(ap + aoff[q]);
  UNROLL
  for (ushort q = 0; q < LB; ++q) rb[q] = *(device const in4_t*)(bp + boff[q]);
#endif
  uint step = 0;
  for (uint ch = 0; ch < nchunks; ++ch) {
    for (uint kt = 0; kt < RANK / BK; ++kt, ++step) {
#if PF == 2
      const ushort buf = step & 1;
#else
      const ushort buf = 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
#if PF
      UNROLL
      for (ushort q = 0; q < LA; ++q) *(threadgroup float4*)(As[buf] + ta[q]) = float4(ra[q]);
      UNROLL
      for (ushort q = 0; q < LB; ++q) *(threadgroup float4*)(Bs[buf] + tb[q]) = float4(rb[q]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (step + 1 < nsteps) {
        const uint k1 = (step + 1) * BK;
        UNROLL
        for (ushort q = 0; q < LA; ++q) ra[q] = *(device const in4_t*)(ap + aoff[q] + k1);
        UNROLL
        for (ushort q = 0; q < LB; ++q) rb[q] = *(device const in4_t*)(bp + (ulong)k1 * N + boff[q]);
      }
#else
      const uint k0 = step * BK;
      UNROLL
      for (ushort q = 0; q < LA; ++q) *(threadgroup float4*)(As[0] + ta[q]) = float4(*(device const in4_t*)(ap + aoff[q] + k0));
      UNROLL
      for (ushort q = 0; q < LB; ++q) *(threadgroup float4*)(Bs[0] + tb[q]) = float4(*(device const in4_t*)(bp + (ulong)k0 * N + boff[q]));
      threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
      threadgroup const float* Asg = As[buf] + (sm * SGT) * LDA;
      threadgroup const float* Bsg = Bs[buf] + sn * SGT;
      UNROLL
      for (ushort kk = 0; kk < BK; kk += 8) {
        simdgroup_float8x8 Af[TM], Bf[TN];
        UNROLL
        for (ushort i = 0; i < TM; ++i) simdgroup_load(Af[i], Asg + (i * 8) * LDA + kk, LDA);
        UNROLL
        for (ushort j = 0; j < TN; ++j) simdgroup_load(Bf[j], Bsg + kk * LDB + j * 8, LDB);
        UNROLL
        for (ushort i = 0; i < TM; ++i)
          UNROLL
          for (ushort j = 0; j < TN; ++j) simdgroup_multiply_accumulate(C[i][j], Af[i], Bf[j], C[i][j]);
      }
    }
#if VARIANT >= 2
    // rank boundary: exact fp32 chunk sums -> int32 running accumulator -> XOR fold -> rotl13 into slot ch % 16
    uint x = 0;
    UNROLL
    for (ushort i = 0; i < TM; ++i)
      UNROLL
      for (ushort j = 0; j < TN; ++j) {
        auto te = C[i][j].thread_elements();
        acc[i][j][0] += int(te[0]);
        acc[i][j][1] += int(te[1]);
        x ^= uint(acc[i][j][0]) ^ uint(acc[i][j][1]);
        C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);   // fresh fp32 accumulator per rank chunk
      }
    const uint s = ch & 15u;   // slot (ll/r - 1) % 16
    UNROLL
    for (uint j = 0; j < 16; ++j) jp[j] = (j == s) ? (rotate(jp[j], 13u) ^ x) : jp[j];
#endif
  }

#if VARIANT <= 1
  uint x = 0;
  UNROLL
  for (ushort i = 0; i < TM; ++i)
    UNROLL
    for (ushort j = 0; j < TN; ++j) { auto te = C[i][j].thread_elements(); x ^= as_type<uint>(te[0]) ^ as_type<uint>(te[1]); }
  if (x == SINK_MAGIC) sink[0] = x ^ tid;   // practically never taken; keeps the GEMM live
#elif VARIANT == 2
  uint x = 0;
  UNROLL
  for (uint j = 0; j < 16; ++j) x ^= jp[j];
  if (x == SINK_MAGIC) sink[0] = x ^ tid;
#else
  uint h[8];
  blake3_keyed_block64(jp, P.key, h);
  const bool fb = u256_le(h, P.bound_block), fs = u256_le(h, P.bound_share);
  if (fb || fs) {
    // lane origin = global minimum of the lane's element set = Pearl t_rows / t_cols (a valid offset by construction)
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u), fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint tr = m0 + sm * SGT + fm, tc = n0 + sn * SGT + fn;
    if (fb) {
      const uint idx = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
      if (idx < P.cap_block) write_slot(blk, idx, tr, tc, jp, h);
    }
    if (fs) {
      const uint idx = atomic_fetch_add_explicit(&ctr[1], 1u, memory_order_relaxed);
      if (idx < P.cap_share) write_slot(shr, idx, tr, tc, jp, h);
    }
  }
#endif
}

// ---- layout probe: one simdgroup (32 threads). in[0..63] = r*8+c (row-major 8x8), in[64..127] = matrix Bq.
// out[0..63]    test 1 (load):    lane's thread_elements() after simdgroup_load of in   -> values r*8+c reveal coords
// out[64..127]  test 2 (inject):  thread_elements() := (1000+2*lane, 1001+2*lane), simdgroup_store -> inverse map
// out[128..191] test 3 (product): thread_elements() of in[0..63] x Bq (fp32 MMA) -> must equal CPU product at coords
// out[192..255] test 4 (int8 staging): char -> float conversion of in8[lane*8 .. +8) for all 256 int8 values
kernel void sg_probe(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                     device const char* in8 [[buffer(2)]], ushort lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 Mx;
  simdgroup_load(Mx, in, 8);
  auto te = Mx.thread_elements();
  out[lane * 2] = te[0]; out[lane * 2 + 1] = te[1];

  simdgroup_float8x8 X = make_filled_simdgroup_matrix<float, 8, 8>(-1.f);
  X.thread_elements()[0] = 1000.f + 2.f * lane;
  X.thread_elements()[1] = 1001.f + 2.f * lane;
  simdgroup_store(X, out + 64, 8);

  simdgroup_float8x8 Bq, Cq = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  simdgroup_load(Bq, in + 64, 8);
  simdgroup_multiply_accumulate(Cq, Mx, Bq, Cq);
  auto tc = Cq.thread_elements();
  out[128 + lane * 2] = tc[0]; out[128 + lane * 2 + 1] = tc[1];

  for (ushort i = 0; i < 8; ++i) out[192 + lane * 8 + i] = float(in8[lane * 8 + i]);
}
