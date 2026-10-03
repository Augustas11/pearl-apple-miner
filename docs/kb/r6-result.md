# R6 result: reading int32 accumulators every 128 K-steps on the M5 Neural Accelerator path

Date 2026-10-02 · Apple M5 (10-core GPU) · macOS 26.5 · Metal 4.0 runtime compile · `matmul2d` int8×int8→int32,
`execution_simdgroups<4>`, `multiply_accumulate`, `relaxed_precision=false`.
Code: `bench/v3_readout/` (`r6bench.swift`, README has the exact commands).
Raw output: `bench/evidence/v3_readout_m5_{probe,correct,perf,diag,final,sustain}.txt`.

## Verdict: **MANAGEABLE**
- **The Neural Accelerator does not stall on readout.** Reading one accumulator element every 128 K (V5) costs about the same as not reading at all (V1). The cost scales with *how* the elements are read, not with the fact that they are read.
- **The obvious implementation is a trap.** Folding through the documented per-element API (`is_valid_element(i)` + `cT[i]`, V2) costs 33–53% vs V0.
- **The best valid variant costs about 8% median.** V6 does the same fold through a raw `uint4` pointer over the cooperative-tensor storage. Over five shapes (128x64 tile, RK=128, 15 alternating rounds) its median overhead vs V0 is **−7.7%** (range +3.8% to −24.2%). Vs V1, the same K-loop without readout, it is −12% to +8%, which is inside run-to-run noise.
- **Most of what remains is the explicit K loop (V1 vs V0), not the readout.**
- **Caveat:** V6 depends on two facts from the probe:
  - every element is valid;
  - `&cT[i] == &cT[0] + i`, with 256 B of contiguous int32 per lane.

  Both are implementation-defined. A production kernel must re-probe per OS and GPU, then fail closed to V2/V3, or to a pointer-free fold if Apple's accessors get cheaper.

## Layout probe (destination int32 cooperative tensor)
`get_multidimensional_index` returns (col, row) (dim0 = N). The layout is identical for RK ∈ {16,32,64,128}.

| Tile (BM×BN) | Elements per lane (all valid) | Lane set | Pearl PeriodicPattern | Result |
|---|---|---|---|---|
| 128×64 | 64 | rows {r, r+8, r+64, r+72} × cols {c..c+3, +16, +32, +48} | rows (8,2)(64,2) h=4; cols (1,4)(16,4) w=16; h·w=64 | **VALID per lane**. Lane origins equal Pearl's valid offsets exactly (row period 128, col period 64), so no cross-lane traffic |
| 64×32 | 16 | rows {r, r+8, r+32, r+40} × cols {c..c+3} | h·w=16 < 32 | **INVALID alone**. Lanes t and t^1 together give rows (8,2)(32,2) × cols (1,8), h·w=32: valid with one `simd_shuffle_xor(x,1)` per 128 K |

The checker ports zk-pow `PeriodicPattern::from_list`/`from_bytes`/`offset_is_valid` and checks h, w even, 32 ≤ h·w ≤ 256, and the exact partition.

## Correctness (CPU int64 oracle, operands uniform in [-127,127])
Shapes: 256×256×8192, 128×128×65536, and 256×128×2112 (2112 has a trailing k%128 = 64 that must not enter the transcript). Every variant, tile, and RK is **bit-exact**:
- C for V0/V1;
- every accumulator at every 128-K boundary (`_dump` builds; 8.4M values at k=65536);
- every jackpot transcript word for V2/V3/V4/V6, row-major and Morton.

161 checked variant×shape lines, 0 mismatches (`v3_readout_m5_correct.txt`). RK=128 is skipped only on the 2112 shape, a host limit: K must be a multiple of RK.

## Throughput (`final` run: random operands, GPU timestamps, 15 rounds, order reversed on odd rounds)
Median TOPS / slowest-run TOPS. Δ is the median vs V0 of the same tile. Exact means bit-exact in the correctness run.

| Shape (m×n×k) | V0 128×64 | V1 rk128 (no readout) | V2 rk128 (accessor fold) | **V6 rk128 (pointer fold)** | V6 Δ vs V0 | V0 64×32 | V6 64×32 rk128 (pair) | Δ |
|---|---|---|---|---|---|---|---|---|
| 4096³ | 10.59 / 5.94 | 10.67 / 7.94 | 6.42 / 4.18 (−39%) | 9.77 / 6.71 | −7.7% | 4.89 / 3.28 | 4.66 / 1.72 | −4.7% |
| 4096²×2048 | 12.36 / 9.69 | 11.15 / 9.30 | 6.82 / 5.58 (−45%) | 10.70 / 8.37 | −13.4% | 5.97 / 4.57 | 5.61 / 4.30 | −6.1% |
| 4096²×8192 | 8.08 / 4.63 | 7.74 / 5.01 | 4.42 / 2.82 (−45%) | 8.38 / 4.72 | +3.8% | 3.52 / 2.69 | 3.51 / 2.19 | −0.1% |
| 8192²×2048 | 7.43 / 6.51 | 6.37 / 6.07 | 3.53 / 3.37 (−53%) | 5.63 / 5.44 | −24.2% | 3.40 / 2.65 | 3.23 / 3.00 | −4.9% |
| 8192³ | 8.08 / 5.12 | 7.18 / 3.85 | 4.76 / 3.49 (−41%) | 7.76 / 5.41 | −4.0% | 3.86 / 3.08 | 3.52 / 2.98 | −8.7% |
| Exact | Y | Y | Y | Y | | Y | Y | |

Notes:
- **Readout decomposition** (`diag`, 128×64, rk128; 4096³ and 8192³, random and all-zero operands; same-round ratios vs V1):
  - V5 (read one element) is **−5%…+6%** vs V1, i.e. no sync penalty.
  - V2 (accessor fold) is **−41%…−47%** vs V1.
  - V6 (pointer fold) is **−8%…+6%** vs V1.
- **V3 (threadgroup fallback) ≈ V2** in the full `perf` matrix: −21%…−69% for 128×64. It works and is exact, but it is not needed for 128×64.
- **V4 (pipelined, second cooperative tensor) is −78%…−89%.** Do not double the accumulator registers.
- **RK:** RK=16 costs 30–50% even without readout (V1). RK=64/128 are best. Use RK=128, one `run()` per rank chunk.
- **Morton order:** ±5% in most cases, within noise. Not a lever at these sizes.
- **64×32 tile:** it runs at less than half the 128×64 rate, so use 128×64.

## Sustained run (`v3_readout_m5_sustain.txt`)
- **V6 128×64 rk128 at 8192³ for 180 s:** windows 4.93–5.51 TOPS, first window 4.98, last 5.17. **Flat, no thermal decline** over 3 minutes.
- **V0 at 8192³ for 60 s right after:** 5.81–6.28 TOPS.
- **V6 at 4096³ right after that:** 5.80–6.12 TOPS. So once the machine is in that state, the 4096³ rate is just as low as the 8192³ rate.

## Measurement environment (read before quoting absolute TOPS)
- **The machine was heavily loaded during every timing run.** No other GPU user was present: the GPU lock was free, and `ioreg` Device Utilization read 0% between runs.
- **CPU load was heavy:**
  - `qemu-system-x86_64` used ~100–140% CPU;
  - a concurrent `rustc` build used ~520%;
  - `vm.loadavg` rose from 5 at the start to 34 (`final`) and 56–97 (`sustain`).
- **Absolute numbers are depressed and noisy.** The unmodified `bench/int8bench.swift`, run in the same window, gave 8.7 TOPS at 4096³ and 10.7 TOPS at 8192³, versus the 19 TOPS on record (`v3_readout_m5_final.txt` header).
- **What this changes:**
  - Only the **ratios** between variants measured in the same alternating rounds are trustworthy here.
  - The earlier 19 → 11 TOPS drop at 8192³ could not be attributed to thermals or to cache on this run. The sustained curve is flat, and 4096³ was equally low in the same state, which points at SoC power/CPU contention.
  - The sustained run needs repeating on an idle machine to get absolute numbers.

## What this means for the kernel
1. Use the 128×64 tile, `execution_simdgroups<4>`, RK=128, and `multiply_accumulate` into one int32 cooperative tensor. Hash pattern: rows [0,8,64,72], cols [0..3,16..19,32..35,48..51] (h·w=64). The fold needs no cross-lane or threadgroup traffic.
2. Fold via direct storage access (V6), gated by a startup probe that checks:
   - all elements are valid;
   - the storage is contiguous;
   - the `get_multidimensional_index` set matches the expected pattern.

   If any check fails, fall back to the V2 accessor fold. V2 is exact but about 40% slower.
3. Expect about 0–10% from the explicit K loop plus readout, vs the single-`run()` GEMM. R6 is not a blocker.
