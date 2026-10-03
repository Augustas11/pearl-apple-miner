// Bit-exact emulation of Pearl's B200 FP8 matmul (certificate v4) on Apple GPUs.
//
// Semantics being reproduced (Pearl zk-pow/src/api/fp8/utils.rs, fp8-scheme 2569546,
// matmul_fp8_windowed + windowed_group_sum with width 26):
//   per output cell, groups of 32 exact E4M3 products plus the FP32 carry C;
//   E = max(stored exponent of each non-zero term)   (product: ea+eb-14, carry: CE)
//   term_t = trunc_toward_zero(P_t * 2^19 / 2^(E - pe_t)),  carry term = trunc(4*Csig / 2^(E-CE))
//   S = exact integer sum;  C' = S * 2^(E-25) rounded toward zero to FP32 (24-bit significand),
//   S == 0 -> +0.
// Real-valued form used here: product p = a*b (a, b the exact E4M3 values),
//   term_t = trunc(p_t * 2^(25-E)), carry term = trunc(C * 2^(25-E)).
//
// Host passes M, N, K (all multiples of the tile sizes; K multiple of 32).

#include <metal_stdlib>
#if defined(KERNEL_B) || defined(KERNEL_B2) || defined(KERNEL_B3) || defined(KERNEL_C) || defined(KERNEL_C2) || defined(KERNEL_D) || defined(KERNEL_D2)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif
using namespace metal;

// 2^e as float for e in the normal range.
inline float pow2f(int e) { return as_type<float>(uint(e + 127) << 23); }

// Exact S * 2^e rounded toward zero to FP32 (S == 0 -> +0). Results are always normal here.
inline float rz_i64(long S, int e) {
  if (S == 0) return 0.0f;
  bool neg = S < 0;
  ulong mag = neg ? ulong(-S) : ulong(S);
  int w = 64 - int(clz(mag));
  if (w > 24) { mag >>= uint(w - 24); e += w - 24; }
  float f = float(uint(mag)) * pow2f(e);  // exact: mag < 2^24, power-of-two scale
  return neg ? -f : f;
}

// Exponent CE (GFloat convention, unbiased) of a non-zero normal float.
inline int f32_exp(float c) { return int((as_type<uint>(c) >> 23) & 0xFF) - 127; }

struct Dec { float v; int e; };  // exact value (0 for +-0) and effective exponent (-64 for zero)

inline Dec dec(uint c) {
  uint ef = (c >> 3) & 15u, m = c & 7u;
  uint sig = ef ? (m | 8u) : m;
  int e = int(max(ef, 1u));
  float v = float(sig) * pow2f(e - 10);  // value = sig * 2^(e-10)
  Dec d;
  d.v = (c & 0x80u) ? -v : v;
  d.e = sig ? e : -64;
  return d;
}

// ---------------------------------------------------------------------------
// Kernel A: pure-ALU bit-exact emulation. 64x64 cell tile per threadgroup,
// 256 threads, 4x4 cells per thread (rows ty+16i, cols tx+16j).
// Pass 1 (E = max(CE, max pe)) is skipped for a thread when, for all its cells,
// the carry dominates the group: C != 0 and rowHi + colHi - 14 <= CE.

#ifdef KERNEL_A
#define BM 64
#define BN 64

kernel void kernel_a(device const uchar* A [[buffer(0)]], device const uchar* B [[buffer(1)]],
                     device uint* out [[buffer(2)]], device atomic_uint* stats [[buffer(3)]],
                     constant uint4& dims [[buffer(4)]],
                     uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint N = dims.y, K = dims.z;
  threadgroup half Av[32][BM], Bv[32][BN];  // E4M3 values are exact in fp16
  threadgroup char Ae[32][BM], Be[32][BN];
  threadgroup int rowHi[BM], colHi[BN];
  const uint m0 = tgid.y * BM, n0 = tgid.x * BN;
  const uint tx = tid % 16, ty = tid / 16;
  // staging role: 4 threads per row, 8 codes each
  const uint sr = tid / 4, st = (tid % 4) * 8;

  float C[4][4];
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) C[i][j] = 0.0f;
  uint pass1_groups = 0;

  for (uint g = 0; g < K / 32; ++g) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      uint2 ca = *(device const uint2*)(A + (m0 + sr) * K + g * 32 + st);
      uint2 cb = *(device const uint2*)(B + (n0 + sr) * K + g * 32 + st);
      int ha = -64, hb = -64;
      for (uint u = 0; u < 8; ++u) {
        uint xa = ((u < 4 ? ca.x : ca.y) >> (8 * (u & 3))) & 0xFFu;
        uint xb = ((u < 4 ? cb.x : cb.y) >> (8 * (u & 3))) & 0xFFu;
        Dec da = dec(xa), db = dec(xb);
        Av[st + u][sr] = half(da.v); Ae[st + u][sr] = char(da.e); ha = max(ha, da.e);
        Bv[st + u][sr] = half(db.v); Be[st + u][sr] = char(db.e); hb = max(hb, db.e);
      }
      ha = max(ha, simd_shuffle_xor(ha, 1)); ha = max(ha, simd_shuffle_xor(ha, 2));
      hb = max(hb, simd_shuffle_xor(hb, 1)); hb = max(hb, simd_shuffle_xor(hb, 2));
      if ((tid & 3) == 0) { rowHi[sr] = ha; colHi[sr] = hb; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Anchor exponent per cell.
    int E[4][4];
    bool need = false;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
      float c = C[i][j];
      int ce = (c != 0.0f) ? f32_exp(c) : -1000;
      E[i][j] = ce;
      need |= (rowHi[ty + 16 * i] + colHi[tx + 16 * j] - 14 > ce);
    }
    if (need) {
      pass1_groups++;
      int mx[4][4];
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) mx[i][j] = -1000;
      for (uint t = 0; t < 32; ++t) {
        int ea[4], eb[4];
        for (int i = 0; i < 4; ++i) ea[i] = Ae[t][ty + 16 * i];
        for (int j = 0; j < 4; ++j) eb[j] = Be[t][tx + 16 * j];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) mx[i][j] = max(mx[i][j], ea[i] + eb[j]);
      }
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
        E[i][j] = max(E[i][j], mx[i][j] - 14);  // zero products give <= -142
    }
    float s[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
      if (E[i][j] < -200) E[i][j] = 0;  // no non-zero term at all: S = 0 below
      s[i][j] = pow2f(25 - E[i][j]);
    }
    // Pass 2: truncated aligned terms. Two int32 accumulators per cell (16 terms each,
    // |term| < 2^27) so no overflow; combined in int64 below.
    int acc0[4][4], acc1[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { acc0[i][j] = 0; acc1[i][j] = 0; }
    for (uint t = 0; t < 32; t += 2) {
      float a0[4], b0[4], a1[4], b1[4];
      for (int i = 0; i < 4; ++i) { a0[i] = float(Av[t][ty + 16 * i]); a1[i] = float(Av[t + 1][ty + 16 * i]); }
      for (int j = 0; j < 4; ++j) { b0[j] = float(Bv[t][tx + 16 * j]); b1[j] = float(Bv[t + 1][tx + 16 * j]); }
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        acc0[i][j] += int((a0[i] * b0[j]) * s[i][j]);
        acc1[i][j] += int((a1[i] * b1[j]) * s[i][j]);
      }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
      long S = long(acc0[i][j]) + long(acc1[i][j]) + long(int(C[i][j] * s[i][j]));
      C[i][j] = rz_i64(S, E[i][j] - 25);
    }
  }
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    out[(m0 + ty + 16 * i) * N + n0 + tx + 16 * j] = as_type<uint>(C[i][j]);
  atomic_fetch_add_explicit(&stats[0], pass1_groups, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel B: linearised path on the M5 Neural Accelerators.
//   Per (row, 32-group) the host stores balanced int8 limbs of
//   ia = sig * 2^(e - l1) (l1 = min over the group of e + ctz(sig)), ia = hi*256 + lo,
//   so the exact group dot product is  dot = HH*2^16 + (HL+LH)*2^8 + LL  (units 2^(l1a+l1b-20)).
//   Four int8 matmul2d runs (K=32) per group produce HH, HL+LH, LL exactly (int32).
//   If no term is truncated and the carry is not truncated (bound test from per-group
//   metadata), C' = RZ_f32(C + dot) exactly; otherwise the cell-group is recomputed
//   exactly by the whole SIMD-group cooperatively (one product per lane).
//
// Metadata word per (row|col, group):
//   bits 0-4 l1 (min e+ctz), 5-9 p1 (argmin), 10-14 l2 (2nd min, 31 = none),
//   15-18 h1 (max e), 19-23 q1 (argmax), 24-27 h2 (2nd max, 0 = none),
//   28 limbs_ok (|ia| <= 32639), 29 any non-zero.

#if defined(KERNEL_B) || defined(KERNEL_B2) || defined(KERNEL_B3) || defined(KERNEL_C) || defined(KERNEL_C2) || defined(KERNEL_D) || defined(KERNEL_D2) || defined(KERNEL_E)
#ifndef TM
#define TM 64
#endif
#ifndef TN
#define TN 32
#endif
#ifndef MAXCAP
#define MAXCAP 16
#endif

inline float exact_group(device const uchar* arow, device const uchar* bcol, float c, ushort lane) {
  // one product per lane
  uint xa = arow[lane], xb = bcol[lane];
  uint efa = (xa >> 3) & 15u, efb = (xb >> 3) & 15u;
  uint sa = efa ? ((xa & 7u) | 8u) : (xa & 7u);
  uint sb = efb ? ((xb & 7u) | 8u) : (xb & 7u);
  uint P = sa * sb;
  int pe = P ? int(max(efa, 1u) + max(efb, 1u)) - 14 : -1000;
  int pmax = simd_max(pe);
  int ce = (c != 0.0f) ? f32_exp(c) : -1000;
  int E = max(pmax, ce);
  if (E < -200) return 0.0f;  // no non-zero term
  uint sh = uint(E - pe);
  int term = (P && sh < 32u) ? int((P << 19) >> sh) : 0;
  if ((xa ^ xb) & 0x80u) term = -term;
  int hi = simd_sum(term >> 16), lo = simd_sum(term & 0xFFFF);
  long S = long(hi) * 65536 + long(lo) + long(int(c * pow2f(25 - E)));
  return rz_i64(S, E - 25);
}

#ifdef KERNEL_B
kernel void kernel_b(device const int8_t* acat [[buffer(0)]], device const int8_t* bcat [[buffer(1)]],
                     device const uint* metaA [[buffer(2)]], device const uint* metaB [[buffer(3)]],
                     device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                     device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                     constant uint4& dims [[buffer(8)]],
                     uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z), NG = K / 32;
  // acat: M x 2K, per group [hi(32) lo(32)];  bcat: 2K x N, per group rows [lo(32); hi(32)].
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> At((device int8_t*)acat, dextents<int32_t, 2>(2 * K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> Bt((device int8_t*)bcat, dextents<int32_t, 2>(N, 2 * K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 32, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
  auto sA = At.slice<32, TM>(0, m0);
  auto sB = Bt.slice<TN, 32>(n0, 0);
  auto cHH = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cX = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cLL = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  const ushort cap = cHH.get_capacity();

  float C[MAXCAP];
  int row[MAXCAP], col[MAXCAP];
  bool valid[MAXCAP];
  for (ushort i = 0; i < MAXCAP; ++i) {
    C[i] = 0.0f;
    valid[i] = (i < cap) && cHH.is_valid_element(i);
    row[i] = 0; col[i] = 0;
    if (valid[i]) {
      auto idx = cHH.get_multidimensional_index(i);
      col[i] = n0 + int(idx[0]);
      row[i] = m0 + int(idx[1]);
    }
  }
  uint fb_count = 0, bad_cap = (cap > MAXCAP) ? 1u : 0u;

  for (int g = 0; g < NG; ++g) {
    #pragma unroll
    for (ushort i = 0; i < MAXCAP; ++i) if (i < cap) { cHH[i] = 0; cX[i] = 0; cLL[i] = 0; }
    auto aHi = At.slice<32, TM>(64 * g, m0);
    auto aLo = At.slice<32, TM>(64 * g + 32, m0);
    auto bLo = Bt.slice<TN, 32>(n0, 64 * g);
    auto bHi = Bt.slice<TN, 32>(n0, 64 * g + 32);
#ifndef B_SKIP_MM
    op.run(aHi, bHi, cHH);
    op.run(aHi, bLo, cX);
    op.run(aLo, bHi, cX);
    op.run(aLo, bLo, cLL);
#endif
#ifdef B_SKIP_EPI
    #pragma unroll
    for (ushort i = 0; i < MAXCAP; ++i) if (i < cap) C[i] += float(cHH[i] + cX[i] + cLL[i]);
    continue;
#endif

    uint pend = 0;
    #pragma unroll
    for (ushort i = 0; i < MAXCAP; ++i) {
      if (!(i < cap) || !valid[i]) continue;
      uint ma = metaA[row[i] * NG + g], mb = metaB[col[i] * NG + g];
      if (!((ma >> 29) & 1u) || !((mb >> 29) & 1u)) continue;  // all products zero: C unchanged
      int l1a = int(ma & 31u), p1a = int((ma >> 5) & 31u), l2a = int((ma >> 10) & 31u);
      int h1a = int((ma >> 15) & 15u), q1a = int((ma >> 19) & 31u), h2a = int((ma >> 24) & 15u);
      int l1b = int(mb & 31u), p1b = int((mb >> 5) & 31u), l2b = int((mb >> 10) & 31u);
      int h1b = int((mb >> 15) & 15u), q1b = int((mb >> 19) & 31u), h2b = int((mb >> 24) & 15u);
      int pe_ub = ((q1a != q1b) ? max(h1a + h2b, h2a + h1b) : (h1a + h1b)) - 14;
      int lsb_pair = (p1a != p1b) ? min(l1a + l2b, l2a + l1b) : (l1a + l1b);
      float c = C[i];
      bool cz = (c == 0.0f);
      int ce = cz ? -1000 : f32_exp(c);
      uint csig = (as_type<uint>(c) & 0x7FFFFFu) | 0x800000u;
      bool carry_ok = cz || (pe_ub - ce <= 2 + int(ctz(csig)));
      int e_ub = cz ? pe_ub : max(pe_ub, ce);
      bool ok = carry_ok && (e_ub <= lsb_pair + 5) && ((ma >> 28) & 1u) && ((mb >> 28) & 1u);
      if (!ok) { pend |= (1u << i); continue; }
      long dot = long(cHH[i]) * 65536 + long(cX[i]) * 256 + long(cLL[i]);
      long S = dot << uint(l1a + l1b - 2);  // units of 2^-18
      if (!cz) {
        int sh = ce - 5;  // C = csig * 2^(ce-23) = (csig << (ce-5)) units; C is a multiple of 2^-18
        long cu = sh >= 0 ? (long(csig) << uint(sh)) : long(csig >> uint(-sh));
        S += (c < 0.0f) ? -cu : cu;
      }
      C[i] = rz_i64(S, -18);
    }
    // Cooperative exact fallback (whole SIMD-group, one product per lane).
    #pragma unroll
    for (ushort i = 0; i < MAXCAP; ++i) {
      if (!(i < cap)) continue;
      bool mine = (pend >> i) & 1u;
      simd_vote v = simd_ballot(mine);
      ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
      while (bits) {
        ushort L = ushort(ctz(bits));
        bits &= bits - 1;
        int r = simd_broadcast(row[i], L), cc = simd_broadcast(col[i], L);
        float c = simd_broadcast(C[i], L);
        float nc = exact_group(acode + r * K + 32 * g, bcode + cc * K + 32 * g, c, lane);
        if (lane == L) { C[i] = nc; fb_count++; }
      }
    }
  }
  #pragma unroll
  for (ushort i = 0; i < MAXCAP; ++i)
    if (i < cap && valid[i]) out[row[i] * N + col[i]] = as_type<uint>(C[i]);
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad_cap) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif
#endif

// ---------------------------------------------------------------------------
// Kernel B2: same arithmetic contract as kernel B, optimised epilogue.
//  * metadata stored group-major ([g][row]) and loaded once per distinct row / column of the
//    thread (the destination layout is derived from get_multidimensional_index at start and
//    checked to be a 4x4 row x column grid; stats[1] counts violations -> results invalid);
//  * cheap per-element predicate from (max e, min lsb) only; the pairing refinement and the
//    int64 general path run only when the cheap test fails;
//  * hot path: C != 0, CE >= 15, result keeps C's binade or grows:
//      Y = Csig + floor(sign(C) * dot * 2^(q - (CE-23)))  (int32), C' = RZ24(Y) * 2^(CE-23).
#ifdef KERNEL_B2

inline bool refined_ok(uint ma, uint mb, float c) {
  int l1a = int(ma & 31u), p1a = int((ma >> 5) & 31u), l2a = int((ma >> 10) & 31u);
  int h1a = int((ma >> 15) & 15u), q1a = int((ma >> 19) & 31u), h2a = int((ma >> 24) & 15u);
  int l1b = int(mb & 31u), p1b = int((mb >> 5) & 31u), l2b = int((mb >> 10) & 31u);
  int h1b = int((mb >> 15) & 15u), q1b = int((mb >> 19) & 31u), h2b = int((mb >> 24) & 15u);
  int pe_ub = ((q1a != q1b) ? max(h1a + h2b, h2a + h1b) : (h1a + h1b)) - 14;
  int lsb_pair = (p1a != p1b) ? min(l1a + l2b, l2a + l1b) : (l1a + l1b);
  bool cz = (c == 0.0f);
  int ce = cz ? -1000 : f32_exp(c);
  uint csig = (as_type<uint>(c) & 0x7FFFFFu) | 0x800000u;
  bool carry_ok = cz || (pe_ub - ce <= 2 + int(ctz(csig)));
  int e_ub = cz ? pe_ub : max(pe_ub, ce);
  return carry_ok && (e_ub <= lsb_pair + 5) && ((ma >> 28) & 1u) && ((mb >> 28) & 1u);
}

// General exact linear update (int64), valid when refined_ok() holds.
inline float linear_general(int hh, int x, int ll, int lsb, float c) {
  long dot = (long(hh) * 256 + long(x)) * 256 + long(ll);
  long S = dot << uint(lsb - 2);  // units of 2^-18
  if (c != 0.0f) {
    int ce = f32_exp(c);
    uint csig = (as_type<uint>(c) & 0x7FFFFFu) | 0x800000u;
    int sh = ce - 5;
    long cu = sh >= 0 ? (long(csig) << uint(sh)) : long(csig >> uint(-sh));
    S += (c < 0.0f) ? -cu : cu;
  }
  return rz_i64(S, -18);
}

kernel void kernel_b2(device const int8_t* acat [[buffer(0)]], device const int8_t* bcat [[buffer(1)]],
                      device const uint* metaA [[buffer(2)]], device const uint* metaB [[buffer(3)]],
                      device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                      device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                      constant uint4& dims [[buffer(8)]],
                      uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z), NG = K / 32;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> At((device int8_t*)acat, dextents<int32_t, 2>(2 * K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> Bt((device int8_t*)bcat, dextents<int32_t, 2>(N, 2 * K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 32, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
  auto sA = At.slice<32, TM>(0, m0);
  auto sB = Bt.slice<TN, 32>(n0, 0);
  auto cHH = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cX = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cLL = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();

  // Derive this thread's 4x4 (row x col) element grid and verify it.
  constexpr int CAP = TM * TN / 128, NR = CAP / 4;
  static_assert(CAP <= 32, "pend mask is 32 bits");
  int rows[NR], cols[4];
  uint layout_bad = (cHH.get_capacity() != CAP) ? 1u : 0u;
  for (ushort i = 0; i < CAP; ++i) {
    auto idx = cHH.get_multidimensional_index(i);
    int r = m0 + int(idx[1]), c = n0 + int(idx[0]);
    if (i % 4 == 0) rows[i / 4] = r;
    if (i < 4) cols[i] = c;
    if (!cHH.is_valid_element(i) || rows[i / 4] != r || cols[i % 4] != c) layout_bad = 1u;
  }
  float C[CAP];
  for (ushort i = 0; i < CAP; ++i) C[i] = 0.0f;
  uint fb_count = 0;

  for (int g = 0; g < NG; ++g) {
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) { cHH[i] = 0; cX[i] = 0; cLL[i] = 0; }
#ifndef B_SKIP_MM
    auto aHi = At.slice<32, TM>(64 * g, m0);
    auto aLo = At.slice<32, TM>(64 * g + 32, m0);
    auto bLo = Bt.slice<TN, 32>(n0, 64 * g);
    auto bHi = Bt.slice<TN, 32>(n0, 64 * g + 32);
    op.run(aHi, bHi, cHH);
    op.run(aHi, bLo, cX);
    op.run(aLo, bHi, cX);
    op.run(aLo, bLo, cLL);
#endif
#ifdef B_SKIP_EPI
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) C[i] += float(cHH[i] + cX[i] + cLL[i]);
    continue;
#endif
    uint ma[NR], mb[4];
    int hA[NR], lA[NR], hB[4], lB[4];
    bool okA[NR], okB[4];
    #pragma unroll
    for (int q = 0; q < NR; ++q) {
      ma[q] = metaA[g * M + rows[q]];
      hA[q] = int((ma[q] >> 15) & 15u); lA[q] = int(ma[q] & 31u); okA[q] = ((ma[q] >> 28) & 3u) == 3u;
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      mb[q] = metaB[g * N + cols[q]];
      hB[q] = int((mb[q] >> 15) & 15u); lB[q] = int(mb[q] & 31u); okB[q] = ((mb[q] >> 28) & 3u) == 3u;
    }
    uint pend = 0;
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      const int ri = i / 4, cj = i % 4;
      float c = C[i];
      uint cb = as_type<uint>(c);
      int ce = int((cb >> 23) & 0xFFu) - 127;
      uint csig = (cb & 0x7FFFFFu) | 0x800000u;
      int pe_ub = hA[ri] + hB[cj] - 14;
      int lsb = lA[ri] + lB[cj];
      bool fast = okA[ri] && okB[cj] && (cb & 0x7FFFFFFFu) != 0u && ce >= 15 &&
                  (pe_ub - ce <= 2 + int(ctz(csig))) && (max(pe_ub, ce) <= lsb + 5);
      if (fast) {
        long dot = long(cHH[i] * 256 + cX[i]) * 256 + long(cLL[i]);
        if (cb >> 31) dot = -dot;
        int sh = ce - 3 - lsb;  // (CE-23) - (lsb-20)
        long T = sh >= 0 ? (dot >> uint(min(sh, 63))) : (dot << uint(-sh));
        int Y = int(csig) + int(T);
        if (Y >= 0x800000) {
          uint w = 32u - clz(uint(Y));
          uint up = w > 24u ? w - 24u : 0u;
          uint ys = uint(Y) >> up;
          C[i] = as_type<float>((cb & 0x80000000u) | (uint(ce + 127 + int(up)) << 23) | (ys & 0x7FFFFFu));
          continue;
        }
      }
      // slow paths
      if (!((ma[ri] >> 29) & 1u) || !((mb[cj] >> 29) & 1u)) continue;  // no non-zero product
#ifdef B_NO_SLOW
      pend |= (1u << i);
      continue;
#endif
      if (refined_ok(ma[ri], mb[cj], c)) {
        C[i] = linear_general(cHH[i], cX[i], cLL[i], lsb, c);
      } else {
        pend |= (1u << i);
      }
    }
#ifndef B_NO_FB
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      bool mine = (pend >> i) & 1u;
      simd_vote v = simd_ballot(mine);
      ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
      while (bits) {
        ushort L = ushort(ctz(bits));
        bits &= bits - 1;
        int r = simd_broadcast(rows[i / 4], L), cc = simd_broadcast(cols[i % 4], L);
        float c = simd_broadcast(C[i], L);
        float nc = exact_group(acode + r * K + 32 * g, bcode + cc * K + 32 * g, c, lane);
        if (lane == L) { C[i] = nc; fb_count++; }
      }
    }
#else
    fb_count += popcount(pend);
#endif
  }
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) out[rows[i / 4] * N + cols[i % 4]] = as_type<uint>(C[i]);
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (layout_bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel B3: lean variant of kernel B.
//  * 16-bit metadata per (row|col, group), group-major: bits 0-4 l1 (min e+ctz), 5-8 h1 (max e),
//    9 limbs_ok, 10 any-non-zero;
//  * only the cheap (max e, min lsb) bound test; anything it cannot prove linear goes to the
//    cooperative exact path (higher fallback % on wide-binade data, smaller code);
//  * one int64 update path; coordinates re-derived from the cooperative tensor (no arrays).
//  SG_SCOPE=1: each SIMD-group owns its own TM x TN tile (execution_simdgroup);
//  otherwise 4 SIMD-groups share a TM x TN tile (execution_simdgroups<4>).
#ifdef KERNEL_B3
#ifdef SG_SCOPE
#define NSG 1
#define SCOPE execution_simdgroup
#else
#define NSG 4
#define SCOPE execution_simdgroups<4>
#endif

kernel void kernel_b3(device const int8_t* acat [[buffer(0)]], device const int8_t* bcat [[buffer(1)]],
                      device const ushort* metaA [[buffer(2)]], device const ushort* metaB [[buffer(3)]],
                      device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                      device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                      constant uint4& dims [[buffer(8)]],
                      uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                      ushort sgid [[simdgroup_index_in_threadgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z), NG = K / 32;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> At((device int8_t*)acat, dextents<int32_t, 2>(2 * K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> Bt((device int8_t*)bcat, dextents<int32_t, 2>(N, 2 * K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 32, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, SCOPE> op;
#ifdef SG_SCOPE
  // threadgroup = 4 SIMD-groups stacked along M
  const int m0 = (int(tgid.y) * 4 + int(sgid)) * TM, n0 = int(tgid.x) * TN;
#else
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
#endif
  auto sA = At.slice<32, TM>(0, m0);
  auto sB = Bt.slice<TN, 32>(n0, 0);
  auto cHH = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cX = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cLL = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  constexpr int CAP = TM * TN / (32 * NSG);
  static_assert(CAP <= 32, "pend mask is 32 bits");
  uint bad = (cHH.get_capacity() != CAP) ? 1u : 0u;
  float C[CAP];
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) { C[i] = 0.0f; bad |= cHH.is_valid_element(i) ? 0u : 1u; }
  uint fb_count = 0;

  for (int g = 0; g < NG; ++g) {
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) { cHH[i] = 0; cX[i] = 0; cLL[i] = 0; }
#ifndef B_SKIP_MM
    auto aHi = At.slice<32, TM>(64 * g, m0);
    auto aLo = At.slice<32, TM>(64 * g + 32, m0);
    auto bLo = Bt.slice<TN, 32>(n0, 64 * g);
    auto bHi = Bt.slice<TN, 32>(n0, 64 * g + 32);
    op.run(aHi, bHi, cHH);
    op.run(aHi, bLo, cX);
    op.run(aLo, bHi, cX);
    op.run(aLo, bLo, cLL);
#endif
    uint pend = 0;
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      auto idx = cHH.get_multidimensional_index(i);
      const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
      uint ma = metaA[g * M + r], mb = metaB[g * N + cc];
      if (((ma & mb) >> 10) == 0u) continue;  // no non-zero product: C unchanged
      int lsb = int(ma & 31u) + int(mb & 31u);
      int pe_ub = int((ma >> 5) & 15u) + int((mb >> 5) & 15u) - 14;
      float c = C[i];
      uint cb = as_type<uint>(c);
      bool cz = (cb & 0x7FFFFFFFu) == 0u;
      int ce = int((cb >> 23) & 0xFFu) - 127;
      uint csig = (cb & 0x7FFFFFu) | 0x800000u;
      bool ok = ((ma & mb) & 0x200u) && (cz ? (pe_ub <= lsb + 5)
                                            : ((pe_ub - ce <= 2 + int(ctz(csig))) && (max(pe_ub, ce) <= lsb + 5)));
      if (!ok) { pend |= (1u << i); continue; }
      long S = ((long(cHH[i] * 256 + cX[i]) << 8) + long(cLL[i])) << uint(lsb - 2);  // units 2^-18
      int sh = ce - 5;
      long cu = cz ? 0 : (sh >= 0 ? (long(csig) << uint(sh)) : long(csig >> uint(-sh)));
      S += (cb >> 31) ? -cu : cu;
      C[i] = rz_i64(S, -18);
    }
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      simd_vote v = simd_ballot(bool((pend >> i) & 1u));
      ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
      if (bits == 0) continue;
      auto idx = cHH.get_multidimensional_index(i);
      const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
      while (bits) {
        ushort L = ushort(ctz(bits));
        bits &= bits - 1;
        int rr = simd_broadcast(r, L), ccc = simd_broadcast(cc, L);
        float c = simd_broadcast(C[i], L);
        float nc = exact_group(acode + rr * K + 32 * g, bcode + ccc * K + 32 * g, c, lane);
        if (lane == L) { C[i] = nc; fb_count++; }
      }
    }
  }
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    auto idx = cHH.get_multidimensional_index(i);
    out[(m0 + int(idx[1])) * N + n0 + int(idx[0])] = as_type<uint>(C[i]);
  }
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel C: "grid-exact" path. If every element of an A row and a B column is a multiple of 0.5
// (true for Pearl-policy-passing constant-magnitude operands, see docs), then 2a, 2b are integers
// with |2a| <= 896 and every product is a multiple of 0.25. While all group-boundary partial sums
// stay below 2^22 in magnitude, B200 never truncates a term or the carry and its RZ is the identity
// (|C| < 2^22 on a 0.25 grid fits 24 bits; E <= 21 so the 25-bit window keeps 2^-4), hence each
// group result is exactly C + dot. Per window of W groups:
//   window_dot = HH*2^14 + (HL+LH)*2^7 + LL   (balanced int8 limbs of 2a: hi in [-7,7], lo in [-64,63])
//   condition: C on the 0.25 grid and |C_window_start| + ||a_w||_2 * ||b_w||_2 < 2^22 (Cauchy-Schwarz bound on
//   every prefix);
//   ||.|| = +inf for rows/cols not on the 0.5 grid.
// Otherwise the window is recomputed group by group with the cooperative exact path.
// WG == 1: no Cauchy-Schwarz bound needed; the exact post-group sum is checked (|C+dot| < 2^22).
#ifdef KERNEL_C
#ifndef WG
#define WG 4
#endif
kernel void kernel_c(device const int8_t* acat [[buffer(0)]], device const int8_t* bcat [[buffer(1)]],
                     device const float* normA [[buffer(2)]], device const float* normB [[buffer(3)]],
                     device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                     device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                     constant uint4& dims [[buffer(8)]],
                     uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z), NW = K / (32 * WG);
  constexpr int KW = 32 * WG;
  // acat: M x 2K, per window [hi(KW) lo(KW)];  bcat: 2K x N, per window rows [lo(KW); hi(KW)].
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> At((device int8_t*)acat, dextents<int32_t, 2>(2 * K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> Bt((device int8_t*)bcat, dextents<int32_t, 2>(N, 2 * K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, KW, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
  auto sA = At.slice<KW, TM>(0, m0);
  auto sB = Bt.slice<TN, KW>(n0, 0);
  auto cHH = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cX = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cLL = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  constexpr int CAP = TM * TN / 128;
  static_assert(CAP <= 32, "pend mask is 32 bits");
  uint bad = (cHH.get_capacity() != CAP) ? 1u : 0u;
  float C[CAP];
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) { C[i] = 0.0f; bad |= cHH.is_valid_element(i) ? 0u : 1u; }
  uint fb_count = 0;

  for (int w = 0; w < NW; ++w) {
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) { cHH[i] = 0; cX[i] = 0; cLL[i] = 0; }
#ifndef B_SKIP_MM
    auto aHi = At.slice<KW, TM>(2 * KW * w, m0);
    auto aLo = At.slice<KW, TM>(2 * KW * w + KW, m0);
    auto bLo = Bt.slice<TN, KW>(n0, 2 * KW * w);
    auto bHi = Bt.slice<TN, KW>(n0, 2 * KW * w + KW);
    op.run(aHi, bHi, cHH);
    op.run(aHi, bLo, cX);
    op.run(aLo, bHi, cX);
    op.run(aLo, bLo, cLL);
#endif
    uint pend = 0;
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      auto idx = cHH.get_multidimensional_index(i);
      const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
#ifdef B_SKIP_EPI
      C[i] += float(cHH[i] + cX[i] + cLL[i]);
      continue;
#endif
#if WG == 1
      // Exact per-group check: C on the 0.25 grid, row/col group on the 0.5 grid, |C + dot| < 2^22.
      float c4 = C[i] * 4.0f;            // exact (power-of-two scale)
      int ci = int(c4);                  // exact when c4 is an integer below 2^24
      int S = ci + (cHH[i] * 16384 + cX[i] * 128 + cLL[i]);  // units of 2^-2
      bool ok = (float(ci) == c4) && (abs(ci) < (1 << 24)) && (abs(S) < (1 << 24)) &&
                (normA[w * M + r] < INFINITY) && (normB[w * N + cc] < INFINITY);
      if (ok) C[i] = float(S) * 0.25f; else pend |= (1u << i);
#else
      float bound = fabs(C[i]) + normA[w * M + r] * normB[w * N + cc];
      // C itself must be on the 0.25 grid (it may not be after an off-grid fallback window).
      if (bound < 4194000.0f && C[i] * 4.0f == rint(C[i] * 4.0f)) {  // < 2^22 with margin for the bound's rounding
        int dot = cHH[i] * 16384 + cX[i] * 128 + cLL[i];  // units of 2^-2
        C[i] = C[i] + float(dot) * 0.25f;                 // exact: |result| < 2^22 on a 0.25 grid
      } else {
        pend |= (1u << i);
      }
#endif
    }
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      simd_vote v = simd_ballot(bool((pend >> i) & 1u));
      ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
      if (bits == 0) continue;
      auto idx = cHH.get_multidimensional_index(i);
      const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
      while (bits) {
        ushort L = ushort(ctz(bits));
        bits &= bits - 1;
        int rr = simd_broadcast(r, L), ccc = simd_broadcast(cc, L);
        float c = simd_broadcast(C[i], L);
        for (int g = w * WG; g < (w + 1) * WG; ++g)
          c = exact_group(acode + rr * K + 32 * g, bcode + ccc * K + 32 * g, c, lane);
        if (lane == L) { C[i] = c; fb_count++; }
      }
    }
  }
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    auto idx = cHH.get_multidimensional_index(i);
    out[(m0 + int(idx[1])) * N + n0 + int(idx[0])] = as_type<uint>(C[i]);
  }
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel C2: kernel C (W = 1) with a minimal epilogue. Same arithmetic contract.
//  * per row / column a whole-K flag "every element on the 0.5 grid" (gridA/gridB, 1 byte), hoisted;
//  * the accumulator is kept as an int32 in units of 2^-2 (exact mode) and only checked |S| < 2^24;
//  * an element that leaves exact mode (|S| >= 2^24 or a fallback result off the grid) switches to
//    float mode for the rest of K (its bits are kept in the same register) and is always recomputed
//    by the cooperative exact path;
//  * one SIMD-wide ballot per group decides whether the fallback loop runs at all.
#ifdef KERNEL_C2
kernel void kernel_c2(device const int8_t* acat [[buffer(0)]], device const int8_t* bcat [[buffer(1)]],
                      device const uchar* gridA [[buffer(2)]], device const uchar* gridB [[buffer(3)]],
                      device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                      device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                      constant uint4& dims [[buffer(8)]],
                      uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int N = int(dims.y), K = int(dims.z), NG = K / 32;
  const int M = int(dims.x);
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> At((device int8_t*)acat, dextents<int32_t, 2>(2 * K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> Bt((device int8_t*)bcat, dextents<int32_t, 2>(N, 2 * K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 32, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
  auto sA = At.slice<32, TM>(0, m0);
  auto sB = Bt.slice<TN, 32>(n0, 0);
  auto cHH = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cX = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  auto cLL = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  constexpr int CAP = TM * TN / 128;
  static_assert(CAP <= 32, "pend mask is 32 bits");
  uint bad = (cHH.get_capacity() != CAP) ? 1u : 0u;
  int Ci[CAP];          // exact mode: value * 4 ; float mode: float bits
  uint fmode = 0;       // bit i: element i in float mode
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    Ci[i] = 0;
    bad |= cHH.is_valid_element(i) ? 0u : 1u;
    auto idx = cHH.get_multidimensional_index(i);
    if (!(gridA[m0 + int(idx[1])] && gridB[n0 + int(idx[0])])) fmode |= (1u << i);  // off-grid: float mode
  }
  uint fb_count = 0;

  for (int g = 0; g < NG; ++g) {
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) { cHH[i] = 0; cX[i] = 0; cLL[i] = 0; }
#ifndef B_SKIP_MM
    auto aHi = At.slice<32, TM>(64 * g, m0);
    auto aLo = At.slice<32, TM>(64 * g + 32, m0);
    auto bLo = Bt.slice<TN, 32>(n0, 64 * g);
    auto bHi = Bt.slice<TN, 32>(n0, 64 * g + 32);
    op.run(aHi, bHi, cHH);
    op.run(aHi, bLo, cX);
    op.run(aLo, bHi, cX);
    op.run(aLo, bLo, cLL);
#endif
    uint pend = fmode;
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      int S = Ci[i] + cHH[i] * 16384 + cX[i] * 128 + cLL[i];
      bool ok = abs(S) < (1 << 24);
      if (ok && !((fmode >> i) & 1u)) Ci[i] = S; else pend |= (1u << i);
    }
    if (simd_any(pend != 0u)) {
      #pragma unroll
      for (ushort i = 0; i < CAP; ++i) {
        simd_vote v = simd_ballot(bool((pend >> i) & 1u));
        ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
        if (bits == 0) continue;
        auto idx = cHH.get_multidimensional_index(i);
        const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
        float cv = ((fmode >> i) & 1u) ? as_type<float>(Ci[i]) : float(Ci[i]) * 0.25f;
        while (bits) {
          ushort L = ushort(ctz(bits));
          bits &= bits - 1;
          int rr = simd_broadcast(r, L), ccc = simd_broadcast(cc, L);
          float c = simd_broadcast(cv, L);
          float nc = exact_group(acode + rr * K + 32 * g, bcode + ccc * K + 32 * g, c, lane);
          if (lane == L) {
            fb_count++;
            float n4 = nc * 4.0f;
            if (!((fmode >> i) & 1u) && fabs(nc) < 4194304.0f && n4 == rint(n4)) {
              Ci[i] = int(n4);
            } else {
              fmode |= (1u << i);
              Ci[i] = as_type<int>(nc);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    auto idx = cHH.get_multidimensional_index(i);
    float cv = ((fmode >> i) & 1u) ? as_type<float>(Ci[i]) : float(Ci[i]) * 0.25f;
    out[(m0 + int(idx[1])) * N + n0 + int(idx[0])] = as_type<uint>(cv);
  }
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel D: grid-exact path on the fp16 Neural-Accelerator matmul, carry kept in the fp32 accumulator.
// For rows/cols on the 0.5 grid every product is a multiple of 0.25, and while every partial sum (in
// ANY accumulation order) stays below 2^22 it is an exactly representable fp32 value, so an fp32
// accumulator that rounds correctly (any order) computes it exactly. Before each window of W groups:
//   bound = |C| + ||a_w||_2 ||b_w||_2  (Cauchy-Schwarz: bounds every partial sum inside the window)
// bound < 2^22 and C on the 0.25 grid => the B200 never truncates/rounds in the window (E <= 21, RZ identity) and the NA's
// fp32 result is exact, so B200 == NA. Otherwise the cell's C is saved and the window is recomputed
// group by group with the cooperative exact path (the NA's value for that cell is discarded).
// Assumes the NA accumulates fp16 x fp16 products in fp32 (or wider) with correct rounding — this is
// NOT documented by Apple; the bit-exact verification against Pearl's oracle is the evidence.
#ifdef KERNEL_D
#ifndef WG
#define WG 1
#endif
kernel void kernel_d(device const half* ah [[buffer(0)]], device const half* bh [[buffer(1)]],
                     device const float* normA [[buffer(2)]], device const float* normB [[buffer(3)]],
                     device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                     device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                     constant uint4& dims [[buffer(8)]],
                     uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z);
  constexpr int KW = 32 * WG;
  const int NW = K / KW;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> At((device half*)ah, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> Bt((device half*)bh, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, KW, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
  auto sA = At.slice<KW, TM>(0, m0);
  auto sB = Bt.slice<TN, KW>(n0, 0);
  auto cC = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();
  constexpr int CAP = TM * TN / 128;
  uint bad = (cC.get_capacity() != CAP) ? 1u : 0u;
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) { cC[i] = 0.0f; bad |= cC.is_valid_element(i) ? 0u : 1u; }
  float saved[CAP];
  uint fb_count = 0;

  for (int w = 0; w < NW; ++w) {
    ulong pend = 0;
#ifndef D_NO_CHECK
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      auto idx = cC.get_multidimensional_index(i);
      float bound = fabs(cC[i]) + normA[w * M + m0 + int(idx[1])] * normB[w * N + n0 + int(idx[0])];
      saved[i] = cC[i];
      // C must itself be on the 0.25 grid (not guaranteed after an off-grid fallback window).
      if (!(bound < 4194000.0f) || cC[i] * 4.0f != rint(cC[i] * 4.0f)) pend |= (1ul << i);  // 2^22 minus margin
    }
#endif
    auto a = At.slice<KW, TM>(KW * w, m0);
    auto b = Bt.slice<TN, KW>(n0, KW * w);
    op.run(a, b, cC);
    if (simd_any(pend != 0ul)) {
      #pragma unroll
      for (ushort i = 0; i < CAP; ++i) {
        simd_vote v = simd_ballot(bool((pend >> i) & 1ul));
        ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
        if (bits == 0) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
        while (bits) {
          ushort L = ushort(ctz(bits));
          bits &= bits - 1;
          int rr = simd_broadcast(r, L), ccc = simd_broadcast(cc, L);
          float c = simd_broadcast(saved[i], L);
          for (int g = w * WG; g < (w + 1) * WG; ++g)
            c = exact_group(acode + rr * K + 32 * g, bcode + ccc * K + 32 * g, c, lane);
          if (lane == L) { cC[i] = c; fb_count++; }
        }
      }
    }
  }
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    auto idx = cC.get_multidimensional_index(i);
    float c = cC[i];
    out[(m0 + int(idx[1])) * N + n0 + int(idx[0])] = as_type<uint>(c == 0.0f ? 0.0f : c);  // B200 zero is +0
  }
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel D2: kernel D with a per-tile bound (one threshold per threadgroup and window):
//   |C| < 2^22 - max_{rows of tile} ||a_w|| * max_{cols of tile} ||b_w||   (=> per-cell CS bound holds)
// plus a per-element 0.25-grid check only once a fallback has happened in the thread. The saved
// carry of a pending cell is parked in the output buffer (device memory, rare) instead of registers.
#ifdef KERNEL_D2
#ifndef WG
#define WG 1
#endif
kernel void kernel_d2(device const half* ah [[buffer(0)]], device const half* bh [[buffer(1)]],
                      device const float* tmaxA [[buffer(2)]], device const float* tmaxB [[buffer(3)]],
                      device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                      device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                      constant uint4& dims [[buffer(8)]],
                      uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z);
  constexpr int KW = 32 * WG;
  const int NW = K / KW, MB = M / TM, NB = N / TN;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> At((device half*)ah, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> Bt((device half*)bh, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(TM, TN, KW, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int m0 = int(tgid.y) * TM, n0 = int(tgid.x) * TN;
  auto sA = At.slice<KW, TM>(0, m0);
  auto sB = Bt.slice<TN, KW>(n0, 0);
  auto cC = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();
  constexpr int CAP = TM * TN / 128;
  uint bad = (cC.get_capacity() != CAP) ? 1u : 0u;
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) { cC[i] = 0.0f; bad |= cC.is_valid_element(i) ? 0u : 1u; }
  uint fb_count = 0;
  bool dirty = false;  // a fallback happened in this thread: C may be off the 0.25 grid

  for (int w = 0; w < NW; ++w) {
    const float thr = 4194000.0f - tmaxA[w * MB + int(tgid.y)] * tmaxB[w * NB + int(tgid.x)];  // -inf if off-grid
    ulong pend = 0;
    #pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      float c = cC[i];
      bool p = !(fabs(c) < thr) || (dirty && c * 4.0f != rint(c * 4.0f));
      if (p) {
        pend |= (1ul << i);
        auto idx = cC.get_multidimensional_index(i);
        out[(m0 + int(idx[1])) * N + n0 + int(idx[0])] = as_type<uint>(c);  // park the carry
      }
    }
    auto a = At.slice<KW, TM>(KW * w, m0);
    auto b = Bt.slice<TN, KW>(n0, KW * w);
    op.run(a, b, cC);
    if (simd_any(pend != 0ul)) {
      dirty = true;
      #pragma unroll
      for (ushort i = 0; i < CAP; ++i) {
        simd_vote v = simd_ballot(bool((pend >> i) & 1ull));
        ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
        if (bits == 0) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = m0 + int(idx[1]), cc = n0 + int(idx[0]);
        float parked = ((pend >> i) & 1ull) ? as_type<float>(out[r * N + cc]) : 0.0f;
        while (bits) {
          ushort L = ushort(ctz(bits));
          bits &= bits - 1;
          int rr = simd_broadcast(r, L), ccc = simd_broadcast(cc, L);
          float c = simd_broadcast(parked, L);
          for (int g = w * WG; g < (w + 1) * WG; ++g)
            c = exact_group(acode + rr * K + 32 * g, bcode + ccc * K + 32 * g, c, lane);
          if (lane == L) { cC[i] = c; fb_count++; }
        }
      }
    }
  }
  #pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    auto idx = cC.get_multidimensional_index(i);
    float c = cC[i];
    out[(m0 + int(idx[1])) * N + n0 + int(idx[0])] = as_type<uint>(c == 0.0f ? 0.0f : c);  // B200 zero is +0
  }
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------------
// Kernel E: grid-exact path on the shader-core fp32 simdgroup_matrix units (no Neural Accelerator) —
// the path an M1-M4 / M3 Ultra GPU would use. Same arithmetic contract as kernel D2 (W = 1):
// per 32-K group and threadgroup tile, |C| < 2^22 - max||a_g|| * max||b_g|| and C on the 0.25 grid =>
// the B200 group result is exactly C + dot and fp32 MMA accumulation is exact; otherwise the cell's
// group is recomputed with the cooperative exact path (parked carry in the output buffer).
// E_NO_CHECK: plain fp32 GEMM with the same structure (reference for the check's overhead).
// Tile ETILE x ETILE per threadgroup, 4 SIMD-groups of (ETILE/2)^2 = EB x EB blocks of 8x8, fragments
// loaded straight from device memory (a 4x4-block-per-SIMD-group variant with threadgroup staging ran
// at only ~0.25 TOPS on M5: too many live accumulators).
#ifdef KERNEL_E
#ifndef EB
#define EB 2
#endif
#define ETILE (16 * EB)
#ifndef EW
#define EW 1
#endif
kernel void kernel_e(device const float* af [[buffer(0)]], device const float* bf [[buffer(1)]],
                     device const float* tmaxA [[buffer(2)]], device const float* tmaxB [[buffer(3)]],
                     device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                     device uint* out [[buffer(6)]], device atomic_uint* stats [[buffer(7)]],
                     constant uint4& dims [[buffer(8)]],
                     uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
  const int M = int(dims.x), N = int(dims.y), K = int(dims.z), NG = K / 32, MB = M / ETILE, NB = N / ETILE;
  const int m0 = int(tgid.y) * ETILE, n0 = int(tgid.x) * ETILE;
  const int sm = (sg / 2) * (8 * EB), sn = (sg % 2) * (8 * EB);   // SIMD-group sub-tile origin
  // lane -> (row, col) of thread_elements()[0] in an 8x8 block (MLX steel layout), checked below
  const ushort qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  uint bad = 0;
  {
    threadgroup float probe[64];
    if (tid < 64) probe[tid] = float(tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 pm;
    simdgroup_load(pm, probe, 8);
    if (pm.thread_elements()[0] != float(fm * 8 + fn) || pm.thread_elements()[1] != float(fm * 8 + fn + 1)) bad = 1;
  }
  simdgroup_float8x8 acc[EB][EB];
  _Pragma("unroll") for (int i = 0; i < EB; ++i) _Pragma("unroll") for (int j = 0; j < EB; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  uint fb_count = 0;
  bool dirty = false;

  for (int g = 0; g < NG; g += EW) {
    ulong pend = 0;
#ifndef E_NO_CHECK
    {  // one check per EW groups; the bound covers all EW groups of the window
    float bsum = 0.0f;
    _Pragma("unroll") for (int q = 0; q < EW; ++q) bsum += tmaxA[(g + q) * MB + int(tgid.y)] * tmaxB[(g + q) * NB + int(tgid.x)];
    const float thr = 4194000.0f - bsum * 1.0001f;  // margin for the rounding of the sum
    _Pragma("unroll") for (int i = 0; i < EB; ++i) _Pragma("unroll") for (int j = 0; j < EB; ++j) _Pragma("unroll") for (int h = 0; h < 2; ++h) {
      float c = acc[i][j].thread_elements()[h];
      bool p = !(fabs(c) < thr) || (dirty && c * 4.0f != rint(c * 4.0f));
      if (p) {
        pend |= (1ul << ((i * EB + j) * 2 + h));
        out[(m0 + sm + 8 * i + fm) * N + n0 + sn + 8 * j + fn + h] = as_type<uint>(c);  // park
      }
    }
    }
#endif
    _Pragma("unroll") for (int kk = 0; kk < 4 * EW; ++kk) {
      simdgroup_float8x8 a[EB], b[EB];
      _Pragma("unroll") for (int i = 0; i < EB; ++i) simdgroup_load(a[i], af + (m0 + sm + 8 * i) * K + g * 32 + 8 * kk, K);
      _Pragma("unroll") for (int j = 0; j < EB; ++j) simdgroup_load(b[j], bf + (g * 32 + 8 * kk) * N + n0 + sn + 8 * j, N);
      _Pragma("unroll") for (int i = 0; i < EB; ++i) _Pragma("unroll") for (int j = 0; j < EB; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
#ifndef E_NO_CHECK
    if (simd_any(pend != 0ul)) {
      dirty = true;
      _Pragma("unroll") for (int i = 0; i < EB; ++i) _Pragma("unroll") for (int j = 0; j < EB; ++j) _Pragma("unroll") for (int h = 0; h < 2; ++h) {
        const uint bit = uint((i * EB + j) * 2 + h);
        simd_vote v = simd_ballot(bool((pend >> bit) & 1ul));
        ulong bits = ulong(static_cast<simd_vote::vote_t>(v));
        if (bits == 0) continue;
        const int r = m0 + sm + 8 * i + fm, cc = n0 + sn + 8 * j + fn + h;
        float parked = ((pend >> bit) & 1ul) ? as_type<float>(out[r * N + cc]) : 0.0f;
        while (bits) {
          ushort L = ushort(ctz(bits));
          bits &= bits - 1;
          int rr = simd_broadcast(r, L), ccc = simd_broadcast(cc, L);
          float c = simd_broadcast(parked, L);
          for (int q = 0; q < EW; ++q)
            c = exact_group(acode + rr * K + 32 * (g + q), bcode + ccc * K + 32 * (g + q), c, lane);
          if (lane == L) { acc[i][j].thread_elements()[h] = c; fb_count++; }
        }
      }
    }
#endif
  }
  _Pragma("unroll") for (int i = 0; i < EB; ++i) _Pragma("unroll") for (int j = 0; j < EB; ++j) _Pragma("unroll") for (int h = 0; h < 2; ++h) {
    float c = acc[i][j].thread_elements()[h];
    out[(m0 + sm + 8 * i + fm) * N + n0 + sn + 8 * j + fn + h] = as_type<uint>(c == 0.0f ? 0.0f : c);
  }
  atomic_fetch_add_explicit(&stats[0], fb_count, memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
}
#endif
