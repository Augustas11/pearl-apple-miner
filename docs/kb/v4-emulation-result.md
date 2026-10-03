# v4 (FP8 certificate) emulation on Apple GPUs: measured result

Experiment of 2026-10-02 on this Mac (MacBook Air Mac17,3, Apple M5, 10-core GPU, macOS 26.5). Plan: `fp8-cert-v4-risk.md` §7.
Code and exact commands: `bench/v4_emulation/README.md`. Raw outputs: `bench/evidence/v4_emulation_m5_*.txt`.
"TOPS-eq" = 2·m·n·k / time for a B200-bit-exact result. "Ratio" = kernel TOPS-eq / plain int8 `matmul2d` (the unmodified
`bench/int8bench.swift` kernel, 4096³) timed in the same lock window; "normalized" = ratio × 19 (idle v3 baseline).

## Verdict: marginal

- **v4 can be mined bit-exactly on an M5 at ~5 TOPS-eq, but only with miner-chosen "grid" operands.**
  - Constant-magnitude X (the KB's miner-friendly family) passes Pearl's jackpot policy and puts every E4M3 operand on a 0.5 grid. Then B200 never truncates or rounds at k = 4096, so its output is the exact dot product.
  - Kernel D does that dot product on the fp16 Neural Accelerators, guarded by a per-group Cauchy-Schwarz bound. It measured 4.9–5.4 TOPS-eq (ratio 0.25–0.28, normalized 4.8–5.4).
  - That is above the ~2 TOPS-eq decision line, but only **~¼ of v3 on the same Mac**. fp16 NA throughput is half the int8 rate, and the fp16 matmul needs a 32-K readout.
- **Generic operands are dead on Apple.** Uniform or LLM-like (Gaussian, log-normal block scales) X, still policy-passing, measured:
  - Best NA kernel: 0.2–0.6 TOPS-eq absolute, ≤ 1.0 normalized.
  - Pure ALU: 0.2–0.35 TOPS-eq.
- **The grid path is fragile.** It depends on:
  - k: fallback is 0.01% at k = 4096, ~3% at 16384, 21–32% at 65536.
  - Pearl continuing to allow constant-magnitude pure-mining operands.
  - If either goes, v4 on Apple falls to ≤ 1 TOPS-eq.
- **Extrapolations (not measured):**
  - M5 Ultra: ~40 TOPS-eq grid path vs ~150 TOPS v3.
  - M3 Ultra: grid path ~8 TOPS-eq (fp32 shader MMA, 0.45× its fp32 GEMM), generic ~2 TOPS-eq. M3 Ultra loses less than M5, because its v3 already ran fp32 GEMM.
  - The gap to NVIDIA FP8 (H100 ~1,979 dense TFLOPS) grows from ~100× under v3 to ~400× (M5, grid) or ~2,000× (generic).

## Correctness (bit-exact vs Pearl's own code)

- **Oracle: Pearl's own code.** It compiles Pearl's source files verbatim: `utils.rs` B200/H100 `matmul_fp8`, `quantization.rs`, `jackpot_policy.rs`, `dtype.rs`, `compute.rs`, `layout.rs`, `unpredictability.rs` (vendor 2569546).
  - Pearl's own unit tests pass, including `b200_matmul_fp8_matches_python_reference_vectors` (`v4_emulation_m5_pearl_unit_tests.txt`).
  - An independent re-derivation matched Pearl's output bits on every analysed cell.
- **Every kernel is bit-exact on 107,282,432 output cells (k = 4096), with 0 mismatches** (`v4_emulation_m5_verify.txt`).
  - The cells cover 17 sets: const / uniform / gauss at 256², 2048², and 2×4096² (const, gauss) or 1×4096² (uniform), plus adv_uniform / adv_edge at 256², 1024² and 2048².
  - Adversarial content: ±0-heavy groups, ±448 with subnormals, far-binade groups, cancelling pairs, constant groups.
  - Kernels A, C2, D and E are also exact at k = 16384 and 65536 (256² cells each, `v4_emulation_m5_kdep.txt`).
- **Policy:** every policy family passes Pearl's `JackpotPolicy::evaluate` (B200, 4×64 tiles, all checks) on every generated set. That covers k = 1024…65536 for const and k = 4096 for the others.
- **Invalid variants caught during development** (the oracle check found each):
  - Kernel C's first version accepted a carry that was off the 0.25 grid after an earlier fallback: 453 mismatches on uniform_256.
  - Kernel D had the same bug.
  - D2 / D at 128×64 overflowed a 32-bit pending mask (CAP = 64): 172 mismatches on const_256.
  - All three were fixed before the runs above.

| Kernel | Hardware path | Exact? (107.3 M cells) | Fallback: const / uniform / gauss / adv_edge |
|---|---|---|---|
| A pure ALU | shader ALU | Y | n/a (data-dependent pass 1 only) |
| B linearized | NA int8 ×4 + ALU epilogue | Y | 0% / 1.6–1.7% / 4.2–4.3% / 77.5% |
| B3 lean | NA int8 ×4 | Y | 0% / 3.9–4.0% / 7.3–7.5% / 77.6% |
| C2 grid-exact | NA int8 ×4, int32 carry | Y | 0.002% / 100% / 100% / 100% |
| D grid-exact | NA fp16, fp32 carry | Y | 0.011% / 88.9% / 99.3% / 98.1% |
| E grid-exact | shader fp32 `simdgroup_matrix` | Y | 0.020% / 100% / 100% / 100% |

The fallback is exact, so a high rate costs speed but never correctness. The SIMD-group recomputes the cell-group with one product per lane.

## Throughput on M5 (median of 2 alternated rounds; best = min-time rep)

The int8 baseline ran at 17–20.9 TOPS in most batches. It fell to 4–10 TOPS in batches next to long kernels (A, B on gauss/uniform, 4096³), from thermal throttling on this fanless Air plus other load (loadavg 3–17). So ratios for those batches overstate the kernel. Rely on absolute numbers there.

| Kernel | Family | Shape | TOPS-eq median r1 / r2 | best | ratio | normalized | fallback |
|---|---|---|---|---|---|---|---|
| A | const | 2048²×4096 | 0.233 / 0.213 | 0.319 | 0.046* | 0.87* | – |
| A | uniform | 2048²×4096 | 0.209 / 0.233 | 0.326 | 0.043* | 0.82* | – |
| A | gauss | 2048²×4096 | 0.219 / 0.203 | 0.313 | 0.045* | 0.85* | – |
| A | const | 256²×4096 | 0.353 / 0.265 | 0.354 | 0.017 | 0.33 | – |
| B | const | 2048²×4096 | 0.801 / 0.834 | 1.037 | 0.044 | 0.83 | 0% |
| B | uniform | 2048²×4096 | 0.470 / 0.452 | 0.572 | 0.031 | 0.58 | 1.63% |
| B | gauss | 2048²×4096 | 0.222 / 0.273 | 0.349 | 0.054* | 1.03* | 4.30% |
| B | uniform | 256²×4096 | 0.423 / 0.541 | 0.545 | 0.028 | 0.53 | 1.70% |
| B3 | const | 2048²×4096 | 0.859 / 0.884 | 1.087 | 0.046 | 0.88 | 0% |
| C2 | const | 2048²×4096 | 4.516 / 4.247 | 4.599 | 0.227 | 4.32 | 0.002% |
| C2 | const | 4096³ | 3.609 / 3.389 | 4.436 | 0.210 | 3.99 | 0.002% |
| **D** | const | 2048²×4096 | **5.295 / 4.867** | 5.346 | **0.255** | **4.84** | 0.011% |
| **D** | const | 4096³ | **4.933 / 4.866** | 5.378 | **0.283** | **5.38** | 0.011% |
| D | const | 256²×4096 | 4.333 / 4.365 | 4.387 | 0.231 | 4.39 | 0.011% |
| D | uniform / gauss | 2048²×4096 | 0.017 / 0.015 | 0.019 | 0.001 | 0.03 | 89% / 99% |
| E | const | 2048²×4096 | 1.368 / 1.216 | 1.580 | 0.068–0.077 | 1.3–1.5 | 0.020% |
| E no-check (plain fp32 GEMM) | const | 2048²×4096 | 2.942 / 2.852 | 2.971 | 0.151–0.153 | 2.9 | – |

\* The baseline in that batch was throttled (4.7–5.8 TOPS), so the ratio is inflated.

Full tables, including 256² for all families and 4096³: `v4_emulation_m5_bench.txt` (rounds 1–2) and `v4_emulation_m5_bench_e.txt` (kernel E). Regenerate with `bench/v4_emulation/summarize.py`.

### Where the time goes (ablations, `v4_emulation_m5_microbench.txt`)

- **Generic path (B3, const_2048):**
  - Full kernel: 35.8–41 ms.
  - Epilogue only: 20.7–21.7 ms.
  - Four int8 `matmul2d` runs per 32-K group alone (kernel B, epilogue compiled out): 8.6–9.0 ms (≈15–16 TOPS int8).
  - The per-32-K exact epilogue costs ~75 lane-ops per cell-group and does not overlap with the NA. That ALU cost, not the NA, caps the generic path.
- **Grid path (D, const_2048):**
  - Full kernel: 6.38 ms. Without the guard: 6.47 ms. The guard is free.
  - fp16 `matmul2d` at a 64×32 tile with a 32-K chunk tops out at ~5.3–5.8 TOPS.
  - The same fp16 NA at 128×64 over the full K reaches 11.3–11.6 TOPS. But with 64 accumulators per thread, the per-group guard made 128×64 slower (D2: 2.41–2.47 TOPS-eq).
  - The ceiling for this approach on M5 is ≈ 11 TOPS-eq (½ × int8). Measured: 5.4.
- **Shader cores (M5):**
  - fp32 `simdgroup_matrix` peak: 3.9 TFLOP/s.
  - fp16-input `simdgroup_matrix`: 2.9 TFLOP/s.
  - Scalar fp32 FMA: 2.3 TFLOP/s.
  - Plain fp32 simdgroup GEMM: E0 (2×2 8×8 blocks per SIMD-group, direct loads) reached 2.9 TOPS. A minimal GEMM microbenchmark reached 1.2 TFLOP/s with 2×2 blocks and 0.34–0.38 with 4×4 (register pressure). The first kernel-E design used 4×4 and ran at ~0.2 TOPS-eq.

## Why the grid path is exact (and when it stops being)

- **Every operand is a multiple of 0.5.** With X = ±c and the fused quantization `α·X + β·noise` in BF16:
  - α·c ≥ 64 is a multiple of 0.5.
  - Every BF16 sum of magnitude ≥ 64 has ulp ≥ 0.5.
  - E4M3 rounding of anything ≥ 4 keeps ulp ≥ 0.5.
  - Measured off-grid fraction: 0.00% of row-groups. Exponent fields span 6–15 (0.5 … 448), so the span is ~10 binades, not "a few". The grid is what matters, not the span.
- **Then B200 cannot truncate or round.** Products are multiples of 0.25. While every group-boundary partial sum satisfies |C| < 2^22:
  - The anchor is E ≤ 21, so the 25-bit window keeps 2^-4. No term and no carry is truncated.
  - The 24-bit RZ is the identity.
  - So the group result is exactly C + dot.
- **Any correctly rounded fp32 accumulator then gives the exact result in any order.** All partial sums are representable. The kernels prove the precondition per 32-K group: C on the 0.25 grid and |C| + ‖a_g‖·‖b_g‖ < 2^22. Otherwise they fall back.
- **Fallback grows with k** (|C| ~ √k):
  - B200 itself starts rounding in 0.01% of groups at k = 16384 and 0.06% at k = 65536.
  - The kernels' sufficient test falls back on 2.7–3.6% (16384) and 21–32% (65536) of groups.
  - An int32-carry variant of C2 that applies the 24-bit RZ for 2^22 ≤ |C| < 2^24 would recover large k. It is not built.

## Extrapolation (labelled; nothing below was measured on those machines)

| Machine | Grid operands (best path) | Generic operands | v3 on same machine | Basis |
|---|---|---|---|---|
| M5 (measured) | 4.9–5.4 (D, NA fp16) | ≤ 0.6 abs (B), ≤ 1.0 normalized | 19 TOPS int8 | this run |
| M5 Ultra (80 cores) | ~40 | ~4 | ~150 | ×8 cores, same core and NA design; clocks unknown |
| M3 Ultra (80 cores, no NA) | ~7.5 (0.45 × 17); ≤ 17 with an MLX-grade guarded GEMM | ~1.5–2.5 (kernel A × 8 cores) | 17 TOPS (MLX fp32, bit-exact) | E vs E0 on M5 = 0.43–0.47; per-core ALU parity with M5 assumed |
| H100 / B200 native | 1,979 / ~4,500 FP8 dense TFLOPS (NVIDIA spec; B200 figure UNVERIFIED) | same | – | native FP8 |

## Corrections to `fp8-cert-v4-risk.md`

1. **§3 "~5% fallback" is wrong in both directions.**
   - True non-linear cell-groups (a term or the carry actually truncated): const 0%, uniform 0.2%, gauss 0.4%, adv_edge 59–61%.
   - Bound-based predicates fall back on 0% / 1.6–4% / 4.2–7.5% (`v4_emulation_m5_oracle.txt`).
2. **§3 "fp16×fp16→fp32 exact if the binade span ≤ 11" is not the binding condition.** Exactness needs every partial sum to be representable.
   - On a 0.5 grid that means |partial| < 2^22, whatever the span.
   - Generic data fails it for a different reason: tiny-lsb elements.
3. **§3 "whether NA fp16 accumulation into fp32 is exact is UNVERIFIED" is now answered for the regime that matters.** M5 `matmul2d` half×half→float was bit-exact on ~38 M guarded const-family cells, where all partials are representable.
   - This says nothing about the NA's rounding mode outside that regime.
4. **§4 linearized B200 "~2 TOPS-eq" was optimistic for generic data.**
   - Measured ≤ 1.04 TOPS-eq with a 0% fallback, 0.2–0.6 on uniform and gauss.
   - The ALU epilogue every 32 K, ~75 lane-ops per cell-group, is the bottleneck, not the NA.
   - Pure ALU "0.3–0.4 TOPS-eq" was about right: 0.2–0.35 measured.
5. **§4 the M5 ALU budget "2.0e12 ops/s" holds roughly.**
   - Scalar FMA microbench: 2.3 TFLOP/s.
   - fp32 `simdgroup_matrix`: 3.9 TFLOP/s.
   - The fp16 NA ran at 11.3–11.6 TOPS, half the 19–21 TOPS int8 rate.
6. **§2 "-0 vs +0 changes the hash" is unreachable on B200 for E4M3 inputs with zero carry-in.**
   - An exact-zero group sum returns +0.
   - The signed-zero underflow branch needs magnitudes below 2^-149, but the smallest product is 2^-18.
7. **§3 "Constant-magnitude X keeps entries within a few binades" is wrong.** Entries span 0.5 … 448. The useful property is the 0.5 grid. Liveness (|X̄| ≈ 2σ < 8σ) passes as stated.
8. **§4 M3 Ultra.** The KB has pure ALU at "2–3 TOPS-eq" and linearized at "4–6". For grid operands, M3 Ultra's v4 path is an fp32 GEMM plus a cheap guard, so it lands at roughly 0.45–1.0× of its v3 speed (~8–17 TOPS-eq), not 4–6. Generic data stays ~2.
9. **Confirmed as written:**
   - B200 semantics: 32 products plus the carry, anchor = max stored exponent (ea+eb−14 for products, ignoring the significand), per-term truncation toward zero, 26-bit window, RZ to fp32.
   - Products are exact (P ≤ 225).
   - k is a multiple of 32.
   - Rank r = 32 (`public_params.rs:221`).
   - δ = 0.5 for B200.

## Risks and open items

- **Policy dependence.** Grid operands are a pure-mining artifact. A consensus or policy change could remove them, for example requiring committed real weights or penalising constant-magnitude X. Without them v4 on Apple is ≤ 1 TOPS-eq.
- **k dependence.** k is a job parameter in [1024, 65536]. Past ~16 K the grid path needs the RZ-aware int32 variant.
- **Timing noise.** This is a fanless MacBook Air, run under other load. Repeat on a cooled machine before using absolute numbers for economics.
- **H100 device mode was not measured.** The KB analysis favours B200 for emulation, and the grid argument applies only to B200's 25-bit window.
