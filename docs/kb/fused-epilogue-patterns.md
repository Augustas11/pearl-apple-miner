# Fusing the per-rank XOR-fold jackpot into a Metal GEMM

## 1. What CUDA does (reference)
- **Where the fold happens:** in the mainloop, not the epilogue. After each `gemm()` k_block, `accumulate(tCrC, k_block)` runs (`collective_mainloop.hpp:294-306`). When `k_block_count % (R/32) == 0` it does three things:
  1. `warpgroup_wait<0>()` to drain the async WGMMA.
  2. `xor_reduction` of the thread's whole int32 fragment through a `lop3` tree.
  3. `m_tile_transcript[idx] = rotl13(m_tile_transcript[idx]) ^ hash` (`pow_utils.hpp:166-182`).
- **No cross-thread communication.** The pattern is the fragment (`settings.py:20-26`).
- **PoW check:** after the K loop, each thread runs one keyed BLAKE3 compress on its 16-word transcript and compares the result against the target (`pearl_gemm_kernel.h:262-273`).
- **On a find:** lock + write header (`pow_utils.hpp:267-306`).
- **Output traffic per tile is 0 bytes unless a block is found.** The CUDA kernel still writes C because it serves inference; a pure miner need not.

## 2. Metal analog A (preferred): pattern = per-lane fragment, all in registers
```metal
// sketch only, UNVERIFIED that int32 coop-tensor layout matches the probe on every OS/GPU
constexpr auto d = matmul2d_descriptor(BM, BN, RK /*multiple of 16, divides r*/,
                    false, true /*B^T K-major*/, false /*relaxed*/,
                    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<d, execution_simdgroups<SG>> op;
auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.get_mask(i)) cT[i] = 0;
uint jp[16] = {0};
for (int kc = 0; kc < dot_len; kc += r) {          // one rank chunk
  for (int kk = 0; kk < r; kk += RK) op.run(A.slice(kc+kk, m0), B.slice(n0, kc+kk), cT);
  uint x = 0;
  #pragma unroll
  for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.get_mask(i)) x ^= as_type<uint>(cT[i]);
  uint t = (kc / r) & 15;
  jp[t] = rotate(jp[t], 13u) ^ x;                  // MSL rotate() = rotl
}
// trailing k % r: run matmul only if C needed; never enters transcript
uint h[8] = key;  blake3_compress_keyed_single_block(jp, h);  if (le256(h) <= target) report(lane_tile_origin);
```

### Validity condition
For each lane, the set `{get_multidimensional_index(i) | mask(i)}` must equal `origin + rows_pattern × cols_pattern`, with periodic patterns that tile the matrix.

### Example: MLX NAX layout
- A 16×16 fragment, where each lane holds rows {fm, fm+8} × cols {fn..fn+3}, with `fm ∈ 0..7` and `fn ∈ {0,4,8,12}`.
- A simdgroup computing a 32×64 tile as 2×4 fragments gives each lane:
  - rows {0,8,16,24}: `PeriodicPattern.from_list([0,8,16,24])`, shape (8,4);
  - cols {0..3, 16..19, 32..35, 48..51}: shape (1,4),(16,4).
- That is h·w = 64, with h and w even and 2 dims. Valid per `sanity_checks.rs:38-43`.
- The valid offsets reproduce exactly the 32 lane origins.
- **This is derived from MLX's assumed layout and must be checked by the probe.**

### Example: simdgroup_matrix 8×8 (M1–M4)
- Each lane holds 1 row × 2 adjacent cols per fragment.
- A simdgroup holding a 32×32 tile as 4×4 fragments gives each lane:
  - rows {0,8,16,24}: shape (8,4);
  - cols {0,1,8,9,16,17,24,25}: shape (1,2),(8,4).
- That is h·w = 32, the minimum allowed.

### Cost model
- Per rank chunk there is one XOR per accumulator element, against r MACs (2r ops) per element. At r=128 that is 1 ALU op per 256 MMA ops.
- On M5 the int8 Neural Accelerator rate is ~1900–2048 ops/clk/core (https://tzakharko.github.io/apple-neural-accelerators-benchmark/). That means ~8 output elements per clock per core need folding, which is ~8 of 128 ALU lanes (~6%).
- **UNKNOWN:** whether reading `cT[i]` between `run()` calls forces the Neural Accelerator to sync or copy registers. This is the single most important experiment (feasibility-risks R6).

## 3. Metal analog B: pattern ≠ fragment, reduce across lanes
- **Cross-lane XOR.**
  - MSL has `simd_xor(T)` (reduction across all active lanes) and `simd_shuffle_xor(data, mask)` (butterfly) for integers (MSL spec §SIMD-group functions; summary: https://gist.github.com/rgov/9139d725841670e8cbdf1593d5f369da).
  - If a hash tile spans a known subset of lanes (e.g. lanes differing in bit b), fold locally first, then `x ^= simd_shuffle_xor(x, 1<<b)` for each lane bit in the subset.
  - XOR is associative and commutative, so order never matters. **Only the element set matters.**
- **Threadgroup fallback.** At each rank boundary: `cT.store(tg_tensor)`, barrier, each lane XORs its pattern's elements from threadgroup memory, barrier.
  - Cost: one threadgroup write + read per accumulator element per r MACs.
  - This is the same fallback shape Apple shows for incompatible layouts (WWDC26-330).
- **No atomics for the fold.**
  - The only atomic is the found-signal. CUDA uses an `atomicCAS` spin-lock (`pow_utils.hpp:267-306`).
  - On Metal, use instead an `atomic_fetch_add_explicit` on a device `atomic_uint` counter, writing into a slot array, with no lock.

## 4. Built-ins that do NOT help
`reduce_rows` / `reduce_columns` support only sum, max, and min (`MPPTensorOpsMatMul2d.h:342-347, 588-609`). Sum-based reductions also lose XOR semantics. Do not use them for the fold.

## 5. BLAKE3 in-kernel
- **Cost:** one compression per hash tile per GEMM. In analog A, each lane runs one BLAKE3 compress after the K loop: 7 rounds × 8 G functions, about 600 ALU ops. That is negligible against 2·h·w·k MMA ops (≥ 2·32·2048).
- **Porting:** port `blake3/blake3.cuh` (214 lines) directly. Metal `rotate()` covers the `rotr`s.
- **Open-source Metal BLAKE3 references** (neither audited; correctness UNVERIFIED):
  - https://github.com/carni-ships/zkMetal (`Sources/Shaders/hash/blake3.metal`)
  - https://github.com/teddyjfpender/stwo-zig (`src/backends/metal/shaders/include/blake3.metal`)
- Test against the `blake3` crate's test vectors.

## 6. M1–M4 exactness trick (no int8 MMA)
- Use float `simdgroup_matrix`. int7 values are exact in fp32, and products ≤ 127² = 16129 are exact.
- A per-chunk sum is at most r·16129, which is < 2^24 for every legal r ≤ 1024 (1024·16129 = 16,516,096 < 16,777,216). So **a fresh fp32 accumulator per rank chunk is exact**.
- At each rank boundary, convert to int32, add to the int32 running accumulator, then fold. The rank boundary is exactly where the fold happens anyway.
- **Half-precision inputs:** it is UNVERIFIED whether a half×half product is formed exactly before fp32 accumulation (16129 needs 14 mantissa bits; half has 11). Use float inputs unless an exhaustive test passes.
