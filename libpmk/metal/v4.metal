// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.

// Pearl certificate-v4 B200 emulation for libpmk.
// Production path: fused BF16 clean/noise quantization -> guarded Kernel E -> fixed 16x16 lottery.
#include <metal_stdlib>
using namespace metal;

#ifndef V4_EB
#define V4_EB 2
#endif
#define V4_ETILE (16 * V4_EB)
#ifndef V4_EW
#define V4_EW 1
#endif
#define V4_SLOT_WORDS 26
#define ROTR(x, n) rotate((x), 32u - (n))
#define G(a, b, c, d, x, y) \
  a = a + b + (x); d = ROTR(d ^ a, 16u); c = c + d; b = ROTR(b ^ c, 12u); \
  a = a + b + (y); d = ROTR(d ^ a, 8u);  c = c + d; b = ROTR(b ^ c, 7u);
#define ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) \
  G(v0, v4, v8,  v12, m[s0],  m[s1]);  G(v1, v5, v9,  v13, m[s2],  m[s3]); \
  G(v2, v6, v10, v14, m[s4],  m[s5]);  G(v3, v7, v11, v15, m[s6],  m[s7]); \
  G(v0, v5, v10, v15, m[s8],  m[s9]);  G(v1, v6, v11, v12, m[s10], m[s11]); \
  G(v2, v7, v8,  v13, m[s12], m[s13]); G(v3, v4, v9,  v14, m[s14], m[s15]);

struct V4Params {
  uint M, N, K, R, cap_block, cap_share, mode, pad;
  uint key[8];
  uint bound_block[8];
  uint bound_share[8];
};

inline float pow2f(int e) { return as_type<float>(uint(e + 127) << 23); }

inline float bf16_to_f32(ushort x) { return as_type<float>(uint(x) << 16); }

inline ushort f32_to_bf16_rne(float x) {
  uint b = as_type<uint>(x);
  uint rb = (b >> 16) & 1u;
  return ushort((b + 0x7fffu + rb) >> 16);
}

inline float fp8_to_f32(uchar bits) {
  uint sign = uint(bits & 0x80u) << 24;
  uint exp = (bits >> 3) & 0x0fu;
  uint man = bits & 7u;
  uint mag = 0;
  if (exp == 0u) {
    if (man != 0u) {
      uint lz = clz(man) - 28u;
      mag = ((121u - lz) << 23) | (((man << lz) & 7u) << 20);
    }
  } else {
    mag = ((exp + 120u) << 23) | (man << 20);
  }
  return as_type<float>(sign | mag);
}

inline bool fp8_on_half_grid(uchar bits) {
  uint exp = (bits >> 3) & 15u;
  uint man = bits & 7u;
  uint sig = exp ? (man | 8u) : man;
  if (sig == 0u) return true;
  uint e = max(exp, 1u);
  return e + ctz(sig) >= 9u;
}

inline uchar f32_to_fp8_e4m3(float x, thread uint &sat, thread uint &nan) {
  uint xb = as_type<uint>(x);
  bool neg = (xb >> 31) != 0u;
  float m = fabs(x);
  if (!isfinite(m)) { nan++; return neg ? 0xfe : 0x7e; }
  uint sign = neg ? 0x80u : 0u;
  uint bits = as_type<uint>(m);
  int e = int(bits >> 23) - 127;
  uint mag;
  if (e < -6) {
    mag = uint(rint(m * 512.0f));
  } else {
    uint exp = uint(e + 7);
    uint fm = bits & 0x7fffffu;
    uint mant = fm >> 20;
    uint rem = fm & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (mant & 1u))) {
      mant++;
      if (mant == 8u) { mant = 0u; exp++; }
    }
    if (exp > 15u || (exp == 15u && mant >= 7u)) { sat++; mag = 0x7eu; }
    else { mag = (exp << 3) | mant; }
  }
  if (mag > 0x7eu) { sat++; mag = 0x7eu; }
  return uchar(sign | mag);
}

inline float rz_i64(long S, int e) {
  if (S == 0) return 0.0f;
  bool neg = S < 0;
  ulong mag = neg ? ulong(-S) : ulong(S);
  int w = 64 - int(clz(mag));
  if (w > 24) { mag >>= uint(w - 24); e += w - 24; }
  float f = float(uint(mag)) * pow2f(e);
  return neg ? -f : f;
}

inline int f32_exp(float c) { return int((as_type<uint>(c) >> 23) & 0xffu) - 127; }

inline float exact_group(device const uchar* arow, device const uchar* bcol, float c, ushort lane) {
  uint xa = arow[lane], xb = bcol[lane];
  uint efa = (xa >> 3) & 15u, efb = (xb >> 3) & 15u;
  uint sa = efa ? ((xa & 7u) | 8u) : (xa & 7u);
  uint sb = efb ? ((xb & 7u) | 8u) : (xb & 7u);
  uint P = sa * sb;
  int pe = P ? int(max(efa, 1u) + max(efb, 1u)) - 14 : -1000;
  int pmax = simd_max(pe);
  int ce = (c != 0.0f) ? f32_exp(c) : -1000;
  int E = max(pmax, ce);
  if (E < -200) return 0.0f;
  uint sh = uint(E - pe);
  int term = (P && sh < 32u) ? int((P << 19) >> sh) : 0;
  if ((xa ^ xb) & 0x80u) term = -term;
  int hi = simd_sum(term >> 16), lo = simd_sum(term & 0xffff);
  long S = long(hi) * 65536 + long(lo) + long(int(c * pow2f(25 - E)));
  return rz_i64(S, E - 25);
}

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
  for (uint i = 0; i < 8; ++i) r = (h[i] < bnd[i]) ? -1 : ((h[i] > bnd[i]) ? 1 : r);
  return r <= 0;
}

inline void write_slot(device uint* arr, uint idx, uint tr, uint tc, thread const uint* jp, thread const uint* h) {
  device uint* s = arr + (ulong)idx * V4_SLOT_WORDS;
  s[0] = tr; s[1] = tc;
  for (uint j = 0; j < 16; ++j) s[2 + j] = jp[j];
  for (uint j = 0; j < 8; ++j) s[18 + j] = h[j];
}

kernel void v4_quantize_operand(device const char* clean [[buffer(0)]],
                                device const uchar* ecodes [[buffer(1)]],
                                device const uchar* fcodes [[buffer(2)]],
                                device const ushort* alpha [[buffer(3)]],
                                device const ushort* beta [[buffer(4)]],
                                device uchar* out_codes [[buffer(5)]],
                                device float* out_floats [[buffer(6)]],
                                device atomic_uint* qstats [[buffer(7)]],
                                constant V4Params& P [[buffer(8)]],
                                uint2 tgid [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  uint col = tgid.x, row = tgid.y;
  if (row >= P.M || col >= P.K) return;
  float c = 0.0f;
  for (uint g = 0; g < P.R / 32; ++g) {
    c = exact_group(ecodes + row * P.R + 32 * g, fcodes + col * P.R + 32 * g, c, lane);
  }
  ushort nb = f32_to_bf16_rne(c);
  ushort prod = f32_to_bf16_rne(bf16_to_f32(beta[row]) * bf16_to_f32(nb));
  ushort clean_bf16 = f32_to_bf16_rne(float(clean[row * P.K + col]));
  ushort sum = f32_to_bf16_rne(fma(bf16_to_f32(alpha[row]), bf16_to_f32(clean_bf16), bf16_to_f32(prod)));
  float clamped = clamp(bf16_to_f32(sum), -448.0f, 448.0f);
  uint sat = (clamped != bf16_to_f32(sum)) ? 1u : 0u, nan = 0u;
  uchar code = f32_to_fp8_e4m3(clamped, sat, nan);
  if (lane == 0) {
    out_codes[row * P.K + col] = code;
    out_floats[row * P.K + col] = fp8_to_f32(code);
    atomic_fetch_add_explicit(&qstats[0], 1u, memory_order_relaxed);
    if (sat) atomic_fetch_add_explicit(&qstats[1], sat, memory_order_relaxed);
    if (nan) atomic_fetch_add_explicit(&qstats[2], nan, memory_order_relaxed);
  }
}

kernel void v4_transpose_b(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                           constant V4Params& P [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  uint k = gid.x, n = gid.y;
  if (n < P.N && k < P.K) out[k * P.N + n] = in[n * P.K + k];
}

kernel void v4_decode_codes(device const uchar* codes [[buffer(0)]], device float* out [[buffer(1)]],
                            constant uint2& dims [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  uint col = gid.x, row = gid.y;
  if (row < dims.x && col < dims.y) out[row * dims.y + col] = fp8_to_f32(codes[row * dims.y + col]);
}

kernel void v4_tilemax(device const uchar* codes [[buffer(0)]], device float* tmax [[buffer(1)]],
                       constant uint4& dims [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  uint tiles = dims.x, rows = dims.y, K = dims.z, NG = K / 32;
  uint tile = gid.x, g = gid.y;
  if (tile >= tiles || g >= NG) return;
  uint r0 = tile * V4_ETILE, r1 = min(r0 + uint(V4_ETILE), rows);
  float ssmax = 0.0f;
  for (uint r = r0; r < r1; ++r) {
    float ss = 0.0f;
    bool ok = true;
    for (uint t = 0; t < 32; ++t) {
      uchar code = codes[r * K + g * 32 + t];
      float v = fp8_to_f32(code);
      ss += v * v;
      ok = ok && fp8_on_half_grid(code);
    }
    ssmax = max(ssmax, ok ? sqrt(ss) : INFINITY);
  }
  tmax[g * tiles + tile] = isinf(ssmax) ? INFINITY : nextafter(ssmax, INFINITY);
}

kernel void v4_kernel_e(device const float* af [[buffer(0)]], device const float* bf [[buffer(1)]],
                        device const float* tmaxA [[buffer(2)]], device const float* tmaxB [[buffer(3)]],
                        device const uchar* acode [[buffer(4)]], device const uchar* bcode [[buffer(5)]],
                        device uint* out [[buffer(6)]], device uint* stats [[buffer(7)]],
                        constant V4Params& P [[buffer(8)]],
                        uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
  const int M = int(P.M), N = int(P.N), K = int(P.K), NG = K / 32, MB = M / V4_ETILE, NB = N / V4_ETILE;
  const int m0 = int(tgid.y) * V4_ETILE, n0 = int(tgid.x) * V4_ETILE;
  const int sm = (sg / 2) * (8 * V4_EB), sn = (sg % 2) * (8 * V4_EB);
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
  simdgroup_float8x8 acc[V4_EB][V4_EB];
  for (int i = 0; i < V4_EB; ++i) for (int j = 0; j < V4_EB; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  ulong fb_count = 0;
  bool dirty = false;
  for (int g = 0; g < NG; g += V4_EW) {
    ulong pend = 0;
    float bsum = 0.0f;
    for (int q = 0; q < V4_EW; ++q) bsum += tmaxA[(g + q) * MB + int(tgid.y)] * tmaxB[(g + q) * NB + int(tgid.x)];
    const float thr = 4194000.0f - bsum * 1.0001f;
    for (int i = 0; i < V4_EB; ++i) for (int j = 0; j < V4_EB; ++j) for (int h = 0; h < 2; ++h) {
      float c = acc[i][j].thread_elements()[h];
      bool p = !(fabs(c) < thr) || (dirty && c * 4.0f != rint(c * 4.0f));
      if (p) {
        pend |= (1ul << ((i * V4_EB + j) * 2 + h));
        out[(m0 + sm + 8 * i + fm) * N + n0 + sn + 8 * j + fn + h] = as_type<uint>(c);
      }
    }
    for (int kk = 0; kk < 4 * V4_EW; ++kk) {
      simdgroup_float8x8 a[V4_EB], b[V4_EB];
      for (int i = 0; i < V4_EB; ++i) simdgroup_load(a[i], af + (m0 + sm + 8 * i) * K + g * 32 + 8 * kk, K);
      for (int j = 0; j < V4_EB; ++j) simdgroup_load(b[j], bf + (g * 32 + 8 * kk) * N + n0 + sn + 8 * j, N);
      for (int i = 0; i < V4_EB; ++i) for (int j = 0; j < V4_EB; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    if (simd_any(pend != 0ul)) {
      dirty = true;
      for (int i = 0; i < V4_EB; ++i) for (int j = 0; j < V4_EB; ++j) for (int h = 0; h < 2; ++h) {
        const uint bit = uint((i * V4_EB + j) * 2 + h);
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
          for (int q = 0; q < V4_EW; ++q) c = exact_group(acode + rr * K + 32 * (g + q), bcode + ccc * K + 32 * (g + q), c, lane);
          if (lane == L) { acc[i][j].thread_elements()[h] = c; fb_count++; }
        }
      }
    }
  }
  for (int i = 0; i < V4_EB; ++i) for (int j = 0; j < V4_EB; ++j) for (int h = 0; h < 2; ++h) {
    float c = acc[i][j].thread_elements()[h];
    out[(m0 + sm + 8 * i + fm) * N + n0 + sn + 8 * j + fn + h] = as_type<uint>(c == 0.0f ? 0.0f : c);
  }
  uint tileIndex = tgid.y * uint(NB) + tgid.x;
  uint sgFb = simd_sum(uint(fb_count));
  uint sgBad = simd_any(bad != 0u) ? 1u : 0u;
  if (lane == 0) {
    stats[tileIndex * 8u + uint(sg)] = sgFb;
    stats[tileIndex * 8u + 4u + uint(sg)] = sgBad;
  }
}

kernel void v4_lottery(device const uint* c_bits [[buffer(0)]], device atomic_uint* ctr [[buffer(1)]],
                       device uint* blk [[buffer(2)]], device uint* shr [[buffer(3)]],
                       constant V4Params& P [[buffer(4)]],
                       uint2 tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint tr = tgid.y * 16u, tc = tgid.x * 16u;
  uint v = 0;
  if (lane < 16) {
    const uint sr = (uint(lane) / 4u) * 4u, sc = (uint(lane) % 4u) * 4u;
    for (uint r = 0; r < 4; ++r) {
      for (uint c = 0; c < 4; ++c) {
        uint bits = c_bits[(tr + sr + r) * P.N + tc + sc + c];
        v = rotate(v * 0x9E3779B1u + bits, 13u);
      }
    }
  }
  uint jp[16];
  for (uint i = 0; i < 16; ++i) jp[i] = simd_shuffle(v, ushort(i));
  if (lane != 0) return;
  uint h[8];
  blake3_keyed_block64(jp, P.key, h);
  bool fb = u256_le(h, P.bound_block), fs = u256_le(h, P.bound_share);
  if (fb) {
    uint idx = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
    if (idx < P.cap_block) write_slot(blk, idx, tr, tc, jp, h);
  }
  if (fs) {
    uint idx = atomic_fetch_add_explicit(&ctr[1], 1u, memory_order_relaxed);
    if (idx < P.cap_share) write_slot(shr, idx, tr, tc, jp, h);
  }
}


kernel void v4_fp8_roundtrip(device const uchar* codes [[buffer(0)]],
                              device uint* decoded [[buffer(1)]],
                              device uchar* recoded [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= 256u) return;
  uchar code = codes[gid];
  float v = fp8_to_f32(code);
  decoded[gid] = as_type<uint>(v);
  uint sat = 0u, nan = 0u;
  recoded[gid] = f32_to_fp8_e4m3(v, sat, nan);
}
