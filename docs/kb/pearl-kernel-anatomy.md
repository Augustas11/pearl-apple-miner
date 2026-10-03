# Pearl CUDA miner: how it is built, and what a Metal port must match bit-for-bit

Paths are relative to `vendor/pearl/` (Pearl HEAD 7039e66f) unless they start with `/`.

## 0. Which code to trust (read first)
- **Source of truth for the proof:** the Rust verifier and miner in `zk-pow/`.
  - Verifier jackpot: `zk-pow/src/circuit/chip/jackpot/helper.rs:9-35`.
  - Reference miner loop: `zk-pow/src/ffi/mine.rs:19-128`.
  - The CUDA kernel is one implementation of this contract, not the contract itself.
- **Do not use `miner/miner-base/src/miner_base/noisy_gemm.py` as an oracle for general patterns.**
  - It hashes contiguous `hash_tile_h x hash_tile_w` blocks (`noisy_gemm.py:298-307`, `inner_hash.py:41-65`).
  - It ignores `rows_pattern`/`cols_pattern`.
  - It only matches the protocol when the patterns are contiguous ranges, e.g. `range(16)` × `range(16)`, which is what OpenJarvis uses (`vendor/openjarvis/src/openjarvis/mining/_mps_miner_loop_main.py:147-148`).
- Test against `zk-pow` `compute_jackpot` / `try_mine_one` and `verify_plain_proof`. `mine.rs:665` shows `verify_plain_proof` used on mined proofs.

## 1. The pipeline, stage by stage

| # | Stage | CUDA location | What it computes | Types | Needed for mining only? |
|---|---|---|---|---|---|
| 1 | Job key | `mine.rs:431-436` | `job_key = blake3(header_bytes ‖ mining_config_bytes)` | bytes | yes (protocol) |
| 2 | Commitment (tensor hash) | `pearl-gemm/csrc/tensor_hash/*` (GPU Merkle, 128 threads/block, 512 leaves/block: `tensor_hash_api.hpp:20-22`); Rust reference `mine.rs:442-479` | `hash_a = blake3(pad1024(A row-major), key=job_key)`, `hash_b = blake3(pad1024(Bᵀ row-major), key=job_key)`. Optional salting with (m,n) (`zk-pow/src/api/seed.rs:11-26`). `b_seed = blake3(job_key ‖ hash_b)`, `a_seed = blake3(b_seed ‖ hash_a_or_activations)` | u8 → 32B | yes (protocol) |
| 3 | Noise generation | `gemm/noise_generation_kernel.h` (doc header lines 1-62) | `E_AL (m×r)`, `E_BR (n×r)` dense in [-32,31]. `E_AR`, `E_BL (k×r)` sparse: one +1 and one -1 per row of k. One BLAKE3 compression per 32 output bytes | int8 / u8 indices | yes (protocol) |
| 4 | Noising A / B | `gemm/pearl_noisingA_kernel.h`, `pearl_noisingB_kernel.h` (1 producer + 2 consumer warpgroups: `pearl_noisingA_kernel.h:50-56`) | `A' = A + E_AL·E_AR`, `B' = B + E_BL·E_BR`. Also the denoise factors `A·E_BL (m×r)` and `E_AR·B' (r×n)` | int8 in; int32 / fp16 factors | **noised A', B' yes; denoise factors no** |
| 5 | Denoise converter | `gemm/denoise_converter_kernel.h` | int32 factors → fp16 scaled by 2^12 / 2^14 (`pearl_gemm_constants.hpp`) | int32 → fp16 | **no** (inference only) |
| 6 | NoisyGEMM mainloop + jackpot | `gemm/collective_mainloop.hpp:245-324`, `pow_utils.hpp:128-200` | int8 WGMMA into int32 accumulators. Every `R` K-steps, XOR-fold each thread's accumulator fragment, then `rotl13`-XOR it into a 16-word register transcript | int8 → int32 | **yes (core)** |
| 7 | Denoise epilogue | `collective_epilogue.hpp:361-457` | `C -= A·E_BL·E_BR + E_AL·E_AR·B'` via fp16 WGMMA with fp32 accumulation and 2^12 scaling | fp32 | **no**: `SkipDenoising` template flag (`kernel_traits.hpp:35`, `pearl_gemm_kernel.h:174-186, 249-254`) |
| 8 | Scale + store C | `collective_epilogue.hpp:460-594` (TMA store) | per-row/col scales, cast, TMA store | fp32 → out | **no** (a pure miner can skip writing C) |
| 9 | PoW check | `pow_utils.hpp:205-234`, called at `pearl_gemm_kernel.h:262-273` | keyed BLAKE3 single-block compress of the 64-byte transcript with `key = a_noise_seed`, then `hash <= target` as uint256 | u32 | yes |
| 10 | Opened-block signal | `pow_utils.hpp:244-307`, `host_signal_header.hpp:22-47` | spin-lock with `atomicCAS`. The first finder writes the tile coordinate, thread index, and per-register (row, col) lists to pinned host memory | — | yes (the format is an implementation choice) |
| 11 | Proof construction | Rust `mine.rs:111-122, 381-391` | Merkle multi-proofs of the opened rows of A and cols of Bᵀ (`pearl_blake3::MerkleTree`) | — | yes; CPU, rare |

`inner_hash_kernel.cu:10-58` is only a single-thread micro-benchmark and correctness probe for `xor_reduction`. It is not in the mining path.

## 2. CUDA implementation choices (free to change on Metal)
- **Hardware target:** SM90 (Hopper) WGMMA, TMA, cluster multicast (`kernel_traits.hpp:63-78`).
- **Threads:** one 64×bN slab per consumer warpgroup (`kernel_traits.hpp:51-53`). Accumulator type `int32_t` (`kernel_traits.hpp:24`).
- **Default tiles:** `tile_size_m=128, tile_size_n=256, tile_size_k=128, noise_rank=128` (`miner/miner-base/src/miner_base/settings.py:10-16`).
- **Default patterns:** `rows_pattern=[0,8]`, `cols_pattern=[0,1,8,9,…,248,249]` (64 entries) (`settings.py:20-26`).
  - That is exactly the set of (row, col) a single Hopper WGMMA thread holds in a 64×256 int32 accumulator: 2 rows × 64 cols = 128 registers.
  - **The hash tile is "whatever one thread holds".** Pearl picked its patterns to match its register fragment, so the XOR needs no cross-thread communication.
- **Hash accumulator:**
  - XOR tree with 3-input `lop3` (`pow_utils.hpp:17-115`).
  - Rotate with `shf.l.wrap` (`pow_utils.hpp:26-34`).
  - `warpgroup_wait<0>()` before each fold. This deliberately drains the async MMA pipe at every rank boundary (`pow_utils.hpp:169-181`).
- **Transcript:** kept in registers per tile, preloaded and written back per k_tile (`pow_utils.hpp:156-199`).
- **Scheduling:** persistent tile scheduler with L2 swizzle (`tile_scheduler.hpp:1-60`).
- **Noise generation:** R=64/128 only in CUDA (`noise_generation_kernel.h:24`). Dense output can also be written as fp16 for denoising (`:574-580`).

## 3. Protocol-mandated: the bit-exact contract a Metal port must meet

### 3.1 Mining configuration (committed in `job_key`)
`MiningConfiguration { common_dim k, rank r, mma_type, rows_pattern, cols_pattern, moe }` (`zk-pow/src/api/proof.rs:63-72`). The miner chooses it.

Verifier constraints (`zk-pow/src/api/sanity_checks.rs:27-58`):
- r is a power of 2 in [32, 1024], and r % 16 == 0.
- k ≤ 2^16, k % 64 == 0, 16r ≤ k ≤ 4r², and k ≥ 1024.
- h = |rows_pattern| and w = |cols_pattern| are both even (TILE_H=2, `circuit/pearl_program.rs:22`), with 32 ≤ h·w ≤ 256.
- (h+w)·dot_len ≤ 4 MiB.
- m, n ≤ 2^24. `t_rows + max(rows_pattern) < m`, and likewise for cols.

**Rank-penalty rule:** `rank >= PENALTY_BASE_RANK = 128` (`sanity_checks.rs:13, 164-180`). The miner rejects ranks below 128 with "blocks would be rejected by consensus" (`settings.py:40-48`). OpenJarvis defaults to `--rank 64` (`_mps_miner_loop_main.py:289`), which is non-compliant.

With r=128, k must be in [2048, 65536].

### 3.2 PeriodicPattern semantics
- `to_list` is at `zk-pow/src/api/proof_utils.rs:158-171`, `offset_is_valid` at `:223-232`, and `period` at `:242-245`.
- A pattern has at most 3 (stride, length) dims. It must start at 0 and be sorted.
- The jackpot tiles for a GEMM are `{offset_r + p : p ∈ rows_pattern} × {offset_c + q : q ∈ cols_pattern}`, for every `offset_r ∈ [0,m)` with `offset_is_valid`, and the same for c.
  - m and n must be multiples of the period (`mine.rs:485-497`).
  - The tiles exactly partition the matrix.
- Difficulty is normalized by work: the target is scaled by `h·w·dot_len` (`sanity_checks.rs:183-192, 229-231`). Under the penalty rule it is scaled by `h·w·(dot_len/r)·128` (`:194-196`).
  - Smaller hash tiles therefore get an easier target per tile, and the expected number of blocks per op stays the same.
- **So the Metal port may choose any valid pattern, including one that matches the Metal accumulator fragment layout.**

### 3.3 Noise (bit-exact)
Source: `zk-pow/src/circuit/pearl_noise.rs`.
- **Random hash.** `get_random_hash(index, seed_label, key, slot)` = `blake3(msg64, key=noise_seed)` (`:46-57`).
  - `msg[slot*4..+4] = (1+index) as i32 LE`.
  - `msg[32..64]` = the 32-byte label `"A_tensor"` or `"B_tensor"`, zero padded (`:19-30`).
  - Slot 0 is used for dense matrices and slot 1 for sparse ones.
- **Dense** E_AL/E_BRᵀ (`:61-80`): row-major linear byte index → block `idx/32`; value = `(byte & 63) - 32`, in [-32, 31].
- **Sparse** E_ARᵀ/E_BL (`:90-116`): 8 lines per hash.
  - `u` = LE u32, `i0 = u & (r-1)`, `i1 = i0 ^ (1 + mulhi(r-1, u))`.
  - The row has +1 at i0 and -1 at i1.
- **Noise values.** `noise_A[row][l] = E_AL[row][i0(l)] - E_AL[row][i1(l)]` (sparse gather-subtract, `:35-43, 149`). The B side is analogous with `b_noise_seed` (`:142-153`).
- **Ranges.**
  - The signal range is [-64, 64] (`mine.rs:15-16`), so noised values are in [-127, 127]. That fits int8 and is "Int7xInt7ToInt32" (`gpu_matmul_config.py:30`).
  - Noised operands are computed in i32 (`mine.rs:65-82`). Keep A and B in [-64, 64] so int8 never wraps.

### 3.4 Jackpot accumulation (the kernel's real job)
Source: `mine.rs:87-107`. The verifier is identical (`circuit/chip/jackpot/helper.rs:19-34`).

```
acc[u][v] : i32 = 0                        // over the tile's h×w (row,col) set
jackpot[16] : u32 = 0
for ll in (r ..= k).step_by(r):            // only FULL rank chunks; trailing k % r ignored
    acc[u][v] += Σ_{l in ll-r .. ll} A'[row_u][l] * B'[l][col_v]   // cumulative, never reset
    x = XOR over all h·w of (acc as u32)   // order-independent
    t = (ll/r - 1) % 16
    jackpot[t] = rotl32(jackpot[t], 13) ^ x
```

- **Constants:** `JACKPOT_SIZE=16`, `LROT_PER_TILE=13` (`zk-pow/src/circuit/pearl_program.rs:23,25`). CUDA uses `HASH_ACCUMULATE_ROTATION=13` (`pow_utils.hpp:15`).
- **What gets folded:** the noised product A'·B', not the denoised one. The epilogue denoise does not touch the transcript (`pearl_gemm_kernel.h:237-265`).
- **Overflow:** |acc| ≤ 65536·127·127 ≈ 1.06e9 < 2^31, so valid k needs no wraparound semantics.
- **CUDA mapping:** `reduce_every_k = R / MMAAtom_K` (`collective_mainloop.hpp:276`). The transcript index cycles via `m_reduction_count` (`pow_utils.hpp:192-198`).

### 3.5 PoW hash
- **Hash.** `jackpot_hash = blake3(le_bytes(jackpot[0..16]) /*64B*/, key = a_noise_seed)` (`zk-pow/src/api/proof_utils.rs:1502-1505`).
  - On the GPU this is one keyed BLAKE3 compress with flags CHUNK_START|CHUNK_END|ROOT|KEYED_HASH (`pow_utils.hpp:208-215`).
- **Success test.** `U256::from_little_endian(hash) <= bound` (`mine.rs:108-110`). CUDA compares word 7 down to word 0 (`pow_utils.hpp:217-231`).
- **Bound.** Derived from nbits and scaled: `extract_difficulty_bound`, or `penalized_target_bound` for pool shares (`sanity_checks.rs:229-246`).

### 3.6 Opened block
- The proof needs:
  - `t_rows` and `t_cols`, the minimum index of the winning tile (`proof.rs:101-102`);
  - Merkle multi-proofs of the winning rows of A and cols of Bᵀ (`mine.rs:111-112, 381-391`).
- The GPU only has to report the tile's (offset_r, offset_c), or enough to reconstruct it. CUDA's per-register row/col dump (`host_signal_header.hpp:29-33`) is an implementation choice.

### 3.7 Summary of what must be bit-exact
1. Noise bytes (BLAKE3 + mask + mulhi).
2. Noised A' and B'.
3. int32 cumulative accumulator values at every full-r boundary.
4. The tile element set (offset + pattern).
5. The XOR fold, `rotl13`, and slot `(chunk-1) % 16`.
6. The keyed BLAKE3 of the 64-byte LE transcript with `a_noise_seed`.
7. The uint256 LE comparison.

Everything else is free: tile shapes, thread layout, denoising, C output, and scheduling.

### 3.8 MoE variant
`try_mine_one_moe` (`mine.rs:131-313`) uses the same jackpot over routed rows. It is out of scope for a v1 dense kernel.
