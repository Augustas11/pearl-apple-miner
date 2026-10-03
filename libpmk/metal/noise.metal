#include <metal_stdlib>
using namespace metal;

struct NoiseParams {
  uint M, N, K, R;
  uint a_key[8];
  uint b_key[8];
};

// Uses the shared keyed compression from k3sg.metal.
inline uint label_word(bool isA, uint i) {
  if (i == 0) return isA ? 0x65745f41u : 0x65745f42u; // "A_te" / "B_te"
  if (i == 1) return 0x726f736eu;                     // "nsor"
  return 0u;
}

inline void noise_msg(uint block, bool sparse, bool isA, thread uint *m) {
  for (uint i = 0; i < 16; ++i) m[i] = 0u;
  m[sparse ? 1 : 0] = block + 1u;
  for (uint i = 0; i < 8; ++i) m[8 + i] = label_word(isA, i);
}

inline int dense_byte(uint linear, bool isA, constant NoiseParams& P) {
  uint m[16], h[8];
  noise_msg(linear >> 5, false, isA, m);
  blake3_keyed_block64(m, isA ? P.a_key : P.b_key, h);
  uint w = h[(linear & 31u) >> 2];
  uint b = (w >> ((linear & 3u) * 8u)) & 255u;
  return int(b & 63u) - 32;
}

inline uint2 sparse_pair(uint row, bool isA, constant NoiseParams& P) {
  uint m[16], h[8];
  noise_msg(row >> 3, true, isA, m);
  blake3_keyed_block64(m, isA ? P.a_key : P.b_key, h);
  uint u = h[row & 7u];
  uint i0 = u & (P.R - 1u);
  uint i1 = i0 ^ (1u + uint(((ulong)(P.R - 1u) * (ulong)u) >> 32));
  return uint2(i0, i1);
}

inline void dense_block(device char *out, uint block, bool isA, constant NoiseParams& P) {
  uint m[16], h[8];
  noise_msg(block, false, isA, m);
  blake3_keyed_block64(m, isA ? P.a_key : P.b_key, h);
  const uint base = block << 5;
  for (uint i = 0; i < 8; ++i) {
    const uint w = h[i];
    out[base + i * 4u + 0u] = char(int(w & 63u) - 32);
    out[base + i * 4u + 1u] = char(int((w >> 8) & 63u) - 32);
    out[base + i * 4u + 2u] = char(int((w >> 16) & 63u) - 32);
    out[base + i * 4u + 3u] = char(int((w >> 24) & 63u) - 32);
  }
}

inline void sparse_block(device uint2 *out, uint block, bool isA, constant NoiseParams& P) {
  uint m[16], h[8];
  noise_msg(block, true, isA, m);
  blake3_keyed_block64(m, isA ? P.a_key : P.b_key, h);
  const uint base = block << 3;
  for (uint i = 0; i < 8; ++i) {
    const uint u = h[i];
    const uint i0 = u & (P.R - 1u);
    const uint i1 = i0 ^ (1u + uint(((ulong)(P.R - 1u) * (ulong)u) >> 32));
    out[base + i] = uint2(i0, i1);
  }
}

kernel void noise_dense_a(device char *eal [[buffer(0)]],
                          constant NoiseParams& P [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
  uint total = P.M * P.R;
  uint base = gid << 5;
  if (base >= total) return;
  dense_block(eal, gid, true, P);
}

kernel void noise_dense_b(device char *ebr [[buffer(0)]],
                          constant NoiseParams& P [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
  uint total = P.N * P.R;
  uint base = gid << 5;
  if (base >= total) return;
  dense_block(ebr, gid, false, P);
}

kernel void noise_sparse_a(device uint2 *ear [[buffer(0)]],
                           constant NoiseParams& P [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
  uint base = gid << 3;
  if (base < P.K) sparse_block(ear, gid, true, P);
}

kernel void noise_sparse_b(device uint2 *ebl [[buffer(0)]],
                           constant NoiseParams& P [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
  uint base = gid << 3;
  if (base < P.K) sparse_block(ebl, gid, false, P);
}

kernel void noise_apply_a(device const char *a [[buffer(0)]],
                          device const char *eal [[buffer(1)]],
                          device const uint2 *ear [[buffer(2)]],
                          device char *ap [[buffer(3)]],
                          constant NoiseParams& P [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  uint total = P.M * P.K;
  if (gid >= total) return;
  uint row = gid / P.K;
  uint l = gid - row * P.K;
  uint2 p = ear[l];
  int noise = int(eal[row * P.R + p.x]) - int(eal[row * P.R + p.y]);
  ap[gid] = char(int(a[gid]) + noise);
}

kernel void noise_apply_b(device const char *bt [[buffer(0)]],
                          device const char *ebr [[buffer(1)]],
                          device const uint2 *ebl [[buffer(2)]],
                          device char *bp [[buffer(3)]],
                          constant NoiseParams& P [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  uint total = P.K * P.N;
  if (gid >= total) return;
  uint l = gid / P.N;
  uint col = gid - l * P.N;
  uint2 p = ebl[l];
  int noise = int(ebr[col * P.R + p.x]) - int(ebr[col * P.R + p.y]);
  bp[gid] = char(int(bt[col * P.K + l]) + noise);
}

kernel void noise_apply_b_tiled(device const char *bt [[buffer(0)]],
                                device const char *ebr [[buffer(1)]],
                                device const uint2 *ebl [[buffer(2)]],
                                device char *bp [[buffer(3)]],
                                constant NoiseParams& P [[buffer(4)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 tgid [[threadgroup_position_in_grid]]) {
  threadgroup char tile[32][33];
  const uint col0 = tgid.x * 32u;
  const uint l0 = tgid.y * 32u;
  const uint tx = tid.x;
  const uint ty = tid.y;

  for (uint dy = 0; dy < 4u; ++dy) {
    const uint l = l0 + tx;
    const uint col = col0 + ty * 4u + dy;
    if (l < P.K && col < P.N) {
      tile[ty * 4u + dy][tx] = bt[col * P.K + l];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint dy = 0; dy < 4u; ++dy) {
    const uint l = l0 + ty * 4u + dy;
    const uint col = col0 + tx;
    if (l < P.K && col < P.N) {
      uint2 p = ebl[l];
      int noise = int(ebr[col * P.R + p.x]) - int(ebr[col * P.R + p.y]);
      bp[l * P.N + col] = char(int(tile[tx][ty * 4u + dy]) + noise);
    }
  }
}

// Validation entry point for the exact keyed compression used by K1 and K3.
kernel void pmk_hash_vector(constant uint *message [[buffer(0)]],
                            constant uint *key [[buffer(1)]],
                            device uint *digest [[buffer(2)]]) {
  uint m[16], h[8];
  for (uint i = 0; i < 16; ++i) m[i] = message[i];
  blake3_keyed_block64(m, key, h);
  for (uint i = 0; i < 8; ++i) digest[i] = h[i];
}
