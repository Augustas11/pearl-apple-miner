# K3-SG design: Pearl v3 mining on fp32 simdgroup_matrix (Apple7–9, primary target M3 Ultra)

Date 2026-10-02 · code `bench/k3sg/` · M5 evidence `bench/evidence/k3sg_m5_correct.txt` · SPEC §5.3, §5.4, §8.

## Status
- **Correctness: bit-exact on M5 (Apple10), all checks.** 981 case-runs over 9 tile configs, 747,837 found slots
  compared, 0 failures. `pearl_mining` 0.3.1 cross-check passes 4/4 with the SG `MiningConfiguration`. Four
  deliberately broken kernels are all caught.
- **Performance: unmeasured on the target.** The M5 is fanless and throttles, and its MLX fp32 path uses the Neural
  Accelerators, so no M5 number says anything about the M3 Ultra. P3 and absolute TOPS need an M3 Ultra window.
- **Layout: probed on M5 only.** It matches MLX `get_coord`. The M3 Ultra must re-probe; `sg_window.sh` step 1 does this.

## 1. Pattern (the MiningConfiguration must use exactly this)
| | value | `from_list` shape | notes |
|---|---|---|---|
| `rows_pattern` | `[0, 8, 16, 24]` | (8,4),(32,1),(32,1) | h = 4, period 32 |
| `cols_pattern` | `[0, 1, 8, 9, 16, 17, 24, 25]` | (1,2),(8,4),(32,1) | w = 8, period 32 |
| h·w | 32 | | the legal minimum (32 ≤ h·w ≤ 256), h and w even, ≤ 3 dims, start 0 |

- **Config:** r = 128, `Int7xInt7ToInt32`. Serialized bytes, checked by `crosscheck_sg.py`:
  `00080000800000000703000000000001030300000000…`.
- **Shape limits:** m % 32 = 0 and n % 32 = 0 (Pearl). The kernel also needs m % BM = 0 and n % BN = 0.
- **Proof size:** (h+w)·k = 12k ≤ 786 KB at k = 65536, well under 4 MiB. A find opens 4 rows of A and 8 rows of Bᵀ.
- **Difficulty:** the bound scales with h·w·dot_len = 32k. Each tile has half the work of the K3-NA tile (h·w = 64),
  so its target is half as hard. Expected blocks per op are unchanged (`sanity_checks.rs:183-196`).

## 2. Lane layout and why the pattern is legal
- **Layout per 8×8 fragment (M5 probe; MLX `mma.h` `get_coord`).** Lane l holds row fm and cols {fn, fn+1}:
  - qid = l>>2;
  - fm = (qid&4) + ((l>>1)&3);
  - fn = (qid&2)·2 + (l&1)·2.

  So fm ∈ 0..7 and fn ∈ {0,2,4,6}. All 64 elements are covered once.
- **The probe (`k3sg probe`, kernel `sg_probe`) checks the layout four ways:**
  1. `simdgroup_load` of a known matrix, then each lane reads `thread_elements()`.
  2. It injects known values through `thread_elements()` and checks where `simdgroup_store` puts them (the inverse map).
  3. It checks that the fp32 MMA accumulator follows the same layout, against an exact CPU product.
  4. It checks that char→float is exact for all 256 int8 values.
- **Simdgroup tile = 32×32, i.e. 4×4 fragments.** Each lane owns rows {fm + 0,8,16,24} × cols {fn + 0,1,8,9,16,17,24,25}.
  That is the pattern above shifted by the lane origin (fm, fn).
- **The probe then checks legality from the measured layout,** with no assumptions:
  - every lane set is a product rows × cols of 32 elements;
  - all 32 lanes share one normalized pattern;
  - the 32 lane sets partition the 32×32 tile;
  - the `from_list` port is legal;
  - lane origins = Pearl valid offsets in [0,32)²: rows 0..7, cols {0,2,4,6};
  - the measured pattern = the committed constants.

  Any mismatch prints `PROBE: FAIL` and the M3 Ultra verdict becomes "DO NOT MINE" (fail closed, §5.4).
- **Partition proof.** `offset_is_valid` for this pattern means row offset % 32 < 8, and col offset % 32 < 8 and even.
  Simdgroup tiles are 32-aligned, so the lane origins are exactly the valid offsets. Pearl's
  `threads_partition` tiles are therefore exactly the per-lane sets, and every element of the m×n matrix is in one tile.
- **Why not a bigger tile per lane?** h·w = 32 is already legal, so no cross-lane `simd_shuffle_xor` is needed. A 64×32
  simdgroup tile (h·w = 64) would double the accumulator registers (64 fp32 + 64 int32 per lane). The BLAKE3 cost per
  tile is already negligible: one compress per lane per job vs 2·32·k MMA ops.

## 3. Threadgroup → tile mapping
- **Grid:** (n/BN, m/BM) threadgroups of 32·WM·WN threads, with BM = 32·WM and BN = 32·WN.
- **Threadgroup (x, y):** rows [y·BM, +BM), cols [x·BN, +BN).
- **Simdgroup s:** sm = s / WN, sn = s % WN. Its tile origin is (y·BM + 32·sm, x·BN + 32·sn).
- **Lane l:** t_rows = y·BM + 32·sm + fm(l), t_cols = x·BN + 32·sn + fn(l). These are the global pattern minima, which
  is what goes into the found slot and the proof.
- **Each lane owns one Pearl tile for the whole K:** one transcript and one hash. There are m·n/32 tiles per job.
- **Same mapping for every cfg.** The sweep cfgs differ only in threadgroup shape (WM×WN ∈ {2×2, 4×2, 2×4, 4×4}),
  BK ∈ {8,16,32} and the prefetch mode, so every cfg uses the same pattern. All 9 cfgs are verified bit-exact.

## 4. Kernel (variant 3)
- **Staging.** Per k-step BK, each thread loads `char4` from A' (m×k row-major) and B' (k×n row-major) and converts to
  `float4`. It stores into threadgroup tiles As[BM][BK+4] and Bs[BK][BN+4]; the +4 pad is MLX's 16-byte pad.
  - PF = 1 prefetches the next k-step into registers during the MMAs. int8 makes this cheap: 4 B per float4 staged.
  - PF = 2 also double-buffers the threadgroup tiles, so there is one barrier per k-step.
  - Half inputs are not used.
- **MMA.** `simdgroup_load` of 4 A and 4 B fragments per 8-wide k slice, then 16 `simdgroup_multiply_accumulate` into
  C[4][4] (fp32).
- **Rank boundary (every 128 k).** For each of the 16 fragments, `thread_elements()` → `int` → add to the int32 running
  accumulator acc[4][4][2] → XOR all 32 into x. Then reset the fragment to 0 (a fresh fp32 accumulator per chunk), and
  set `jp[ch % 16] = rotl(jp[ch % 16], 13) ^ x`. A trailing k % 128 never runs; there is no C output.
- **Exactness.** |operand| ≤ 127 and |product| ≤ 16129. Every fp32 partial sum within a chunk is an integer with
  magnitude ≤ 128·16129 = 2,064,512 < 2^24, so it is exact whatever the hardware summation order. Across k ≤ 2^16,
  |acc| ≤ 2^16·16129 < 2^31.
- **Epilogue (same as K3-NA / F1).**
  - One keyed BLAKE3 compress of the 64-byte LE transcript: flags 27, key = a_noise_seed.
  - U256 LE compare against the block bound and against the share bound.
  - On a find: `atomic_fetch_add` on a per-array counter, and the slot is written only if idx < cap. The slot holds
    t_rows, t_cols, jp[16] and hash[8]. Counters keep counting past capacity, so the host sees overflow (§5.1).
- **Compiler finding (M5, measured).** `#pragma unroll` was not honoured on these loops. The C[4][4] fragments
  spilled, and the GEMM ran ~10× slower: 0.25 vs 2.6 TOPS at 2048³. `_Pragma("clang loop unroll(full)")`, as MLX steel
  uses, fixes it. Re-check on the M3 compiler: the sweep shows it.

## 5. Baselines and references (SPEC P3, measured in the M3 Ultra window)
| name | what | role |
|---|---|---|
| `base_f32` | same tiling and loop structure, fp32 operands from device memory, one fp32 accumulator over K, C store disabled (conditional sink) | **P3 baseline: K3-SG ≥ 0.80× same-round median** |
| `base_i8` | same, int8 operands converted while staging | isolates fold + hash cost (k3/base_i8, fold/base_i8) |
| `fold` | variant 2 (no hash/compare/atomics) | k3/fold = epilogue cost |
| `mlx` | MLX `mx.matmul` fp32 (mlx 0.31.2), persistent subprocess in the same rounds, wall time | external: is our GEMM competitive with MLX steel? |
| `int8bench` | Metal 4 `matmul2d` int8, C stored | external; on Apple7–9 TensorOps fall back to shaders, may fail (column dropped, error verbatim) |

**Caveat: P3 can pass against a weak baseline.** The baseline is our own GEMM, so P3 passing says nothing about absolute
speed. Read k3/mlx alongside it. On M5, MLX fp32 runs on the Neural Accelerators, so k3/mlx there means nothing. On M3
Ultra, MLX uses steel `simdgroup_matrix`, which is the honest yardstick.

## 6. What the M3 Ultra window must establish (`bench/k3sg/studio/sg_window.sh`)
1. **Probe:** the M3 Ultra (Apple9, macOS 26.4.1) lane layout = MLX `get_coord`. Then pattern legality and pattern =
   committed constants. Fail → do not mine; the M3 layout would need its own pattern (Rigel reported a different 8×8
   layout on M4, so this is the main open risk).
2. **Correctness on the M3 Ultra GPU and compiler:**
   - the 4 pre-generated oracle vector jobs × 9 cfgs, checked bit-exact in Swift against `tiles.bin`;
   - 256×256×4096 uniform and ±127 with the boundary vectors hash−1/hash/hash+1;
   - 128×128×65536 with the boundary vectors;
   - Pearl's own c=64 job with Pearl's bound, where the block array must equal the oracle's winners, including Pearl's
     first winning tile (0,36).
3. **Best cfg:** a k3-only sweep at 4096³.
4. **P3:** k3/base_f32 median ≥ 0.80 at 4096²×4096 and at 8192²×4096 (31 paired rounds), plus k3/mlx and absolute TOPS.
5. **Sustained:** 240 s at 8192²×4096, TOPS per 10 s window. This is information for P4/P5 on a fan-cooled Mac.

Exit codes: 0 all pass, 1 step failure, 2 criterion failure (probe, correctness or P3).

## 7. Not done here / open
- The production C-ABI integration (libpmk) and the §5.4 startup-probe cache.
- K1/K2 noise on Apple7–9. They are plain ALU kernels, but have not been run there.
- A Metal 3 precompiled metallib (SPEC §5.3 "separate artifact"). Today the source compiles at runtime as Metal 3.1,
  which needs macOS ≥ 14.
- Perf tuning beyond the 9-cfg sweep, chosen after the M3 Ultra numbers.
- M1/M2 (Apple7/8): the same code and probe, untested.
