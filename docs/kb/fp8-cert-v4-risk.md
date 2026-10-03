# FP8 / certificate-v4: what it means for Apple Silicon mining

Short paths are relative to `vendor/pearl-fp8/` (branch `fp8-scheme`, HEAD 2569546). "wp" = https://pearlresearch.ai/Pearl_Whitepaper.pdf ("Pearl Floating Point Scheme Specification", Sept 2026). Estimates marked UNVERIFIED are not yet measured.

## 1. Status and timeline
- **Fork rule.** At and after `Fp8ForkHeight`, a block must carry a V4 certificate. A value of 0 disables the fork (`node/chaincfg/params.go:283-289, 334-338`).
- **Strict cutover, no overlap window.**
  - `RequiredCertVersion` returns exactly one version (`params.go:340-356`), and validation rejects any other (`node/blockchain/validate.go:537-542`).
  - v3 int7 proofs are invalid after activation.
- **Where the fork is set.**
  - `Fp8ForkHeight: 1` appears only on regtest and simnet (`params.go:489, 783`).
  - Mainnet, testnet and testnet2 don't set it, so it defaults to 0 (disabled).
  - The latest mainnet fork is `SaltedSeedForkHeight` 99000 (`params.go:388`).
- **PR #311** "feat(fp8): certificate-v4 + mining":
  - Status: open, created 2026-09-09, updated 2026-10-01, merge state BLOCKED.
  - Size: +102,134 / -29,244 lines across 523 files.
  - It replaces the old int7 / noisy-GEMM host path in miner-base.
  - CI moves to B200 (SM100).
  - No activation date appears in the PR or in releases v1.4.6–v1.4.11.
- **Cadence signal only.** Recent mainnet forks came weeks apart: MoE at 71935, dense-only at 91630, rank-penalty at 96251 (v1.3.0, 2026-08-05), salted-seed at 99000 (v1.4.1, 2026-08-11). A v4 mainnet height within 1–3 months of merge is plausible but UNVERIFIED.

## 2. What a valid v4 proof requires

### Scheme
`Quant::Fp8E4M3Prequant`: int8 values, BF16 block scales, FP8 E4M3 matmul. The matmul AIR is device-specific, H100 or B200 (`zk-pow/src/circuit/fp8/mod.rs:11-12, 28-31`).

### The device is miner-chosen and committed
- It is a committed byte, `Device{H100=0, B200=1}` (`zk-pow/src/api/fp8/public_params.rs:144-147`). Per wp §3, Device is chosen from a consensus-allowed list and bound into the seeds.
- The reference miner just maps the local GPU to a Device (`miner/miner-base/src/miner_base/devices.py:1-6, 20-34`). The verifier cannot check what hardware actually ran.
- So an Apple miner may claim B200 (or H100), but must reproduce that device's arithmetic bit for bit. wp §7 explicitly allows any faster bit-exact `MatMul_Device` implementation.

### Operands
- Noise lines come from keyed BLAKE3 XOF, L2-normalized to 256 (`api/fp8/noise.rs:1-29`).
- Rows are scaled and rounded to E4M3, max 448 (`api/fp8/quantization.rs:36-39`; wp App. C.4).
- The noise-to-signal ratio δ is 1 for H100 and 0.5 for B200 (`public_params.rs:153-166`).

### Matmul semantics (the hard part)
- Products are exact: 4-bit significands, P ≤ 225 (`api/fp8/utils.rs:115-145`).
- Accumulation runs in groups of 32 products plus the carry (`utils.rs:9-12`). Within each group:
  1. Align to the group's max exponent.
  2. Truncate every term toward zero.
  3. Sum exactly.
  4. Truncate the sum to the device width (`windowed_group_sum`, `utils.rs:147-231`).
- **B200:** 26-bit significand. Each term is ±floor(P·2^19 / 2^REL), and the carry is floor(4C / 2^(E−CE)). Groups chain over all of k (`utils.rs:14-17, 234-341`; `circuit/fp8/matmul_b200_stark/stark.rs:1-48`).
- **H100:** 14-bit significand. Each term is ±floor(P·2^7 / 2^shift). Each 128-product window starts a fresh +0 local accumulator, then adds into an FP32 total with round-to-nearest-even (`utils.rs:18-20, 425-523`; `circuit/fp8/matmul_h100/stark.rs:1-33`).

### Lottery
- The v4 lottery folds the bits of the **final** FP32 outputs of A'B'ᵀ. This differs from v3, which folds the per-rank cumulative int32 accumulators.
- The fold is order-dependent: `lane = rotl13(lane·0x9E3779B1 + bits)` over 16 committed subtiles (`utils.rs:552-570`; layout in `api/layout.rs:399-422`).
- Tile limits: h ≥ 4, w ≥ 16, 256 ≤ h·w ≤ 2048, and at least 16 elements per subtile (`layout.rs:25-66`).
- -0 vs +0 changes the hash (`utils.rs:155-162`).

### Jackpot policy
The winning tile must also pass four checks (`api/fp8/jackpot_policy.rs:19-36`; wp §5):
1. Entry liveness: at most 1/64 dead entries with |X̄| ≥ 8σ.
2. A noise floor of σ ≥ 1.
3. Unpredictable summands: at most k·h·w/20 skippable.
4. Tamed products.

These checks target shortcuts that let the device truncate the noise away (wp §5, §7 "Precision shortcuts").

## 3. Can Apple matrix hardware do any of it?
- **No native path.**
  - Apple has no FP8 MMA. The M5 Neural Accelerator does fp16 and int8. FP8 types arrive in OS 27, and Rigel measured them as emulated on M4.
  - Even a native FP8 unit would not match NVIDIA's group-truncation semantics.
- **Linearization idea.** Write x = mag·2^s, so a B200 term is floor(x_a·x_b / D) with D a power of two. Then group sum = Σp/D − Σ(p mod D)/D.
  - **Σp is linear, so a matmul can compute it exactly.** Either use int8 limbs (x needs ~18 bits, so 9 int8 matmuls), or use fp16×fp16→fp32 if the product binade span within the group is ≤ ~11 (fp32 partial sums stay exact).
  - **The correction Σ(p mod D) is per-term and nonlinear.** It is nonzero only for truncated terms: REL > 19 on B200, shift > 7 + v2(P) on H100.
  - **The anchor E is data-dependent.** It is a per-cell, per-group max-plus product with no hardware support. Per-row and per-column group bounds can stand in where they prove E equals the carry exponent.
- **B200 mode is the better target.**
  - Its 25-bit window keeps terms exact unless REL > 19.
  - A pure miner picks its own operands. Constant-magnitude random-sign X with δ=0.5 noise keeps entries within a few binades. That data also passes liveness, since |X̄| ≈ 2σ is under the 8σ cap.
  - Estimate: ~5% of cell-groups need the exact fallback (UNVERIFIED).
  - Carry alignment and 24-bit truncation cost O(1) int64 ALU work per cell per 32 MACs.
- **H100 mode is much worse.** Its 14-bit carry is re-truncated every group, and terms are exact only for shift ≤ 7. Estimate: ~10–20% of terms truncated, forcing per-term work (UNVERIFIED). If emulating, choose B200.
- **The linearized path needs a Neural Accelerator readout every 32 K steps**, 4x more often than v3's 128. Whether Neural Accelerator fp16 accumulation into fp32 is exact is UNVERIFIED (Rigel: M4 accumulates at ≥ fp32).

## 4. Throughput (order of magnitude, UNVERIFIED)

| Path | M5 (10 cores) | M3 Ultra (80 cores) |
|---|---|---|
| Pure-ALU B200 emulation | ~0.15–0.2 T MAC/s = ~0.3–0.4 TOPS-eq | ~1–1.4 T MAC/s = ~2–3 TOPS-eq |
| Linearized B200 | ALU-bound at ~1 T MAC/s = ~2 TOPS-eq | ~4–6 TOPS-eq (MMA shares the ALUs) |
| Today's v3 (measured) | 19 TOPS | 17 TOPS |

- **Pure-ALU cost model:** ~10–14 int ops per MAC (multiply mags, add exponents, max pass, REL, shift, sign-correct truncation, int64 add, two passes). The ALU budget is 128 lanes/core/clk.
  - M5: 10 cores × ~1.55 GHz ≈ 2.0e12 ops/s.
  - M3 Ultra: 80 cores × ~1.4 GHz ≈ 14e12 ops/s (clock UNVERIFIED).
- **Linearized cost model:** ~1.5–2.5 ALU ops/MAC on top of the matmul.
- **NVIDIA reference:** H100 FP8 dense is ~1,979 TFLOPS (3,958 with sparsity: https://www.nvidia.com/en-us/data-center/h100/). B200 is higher (figure UNVERIFIED).
- **Gap:** Apple vs NVIDIA grows from ~100x under v3 to ~1,000–6,000x (M5) or ~350–1,000x (M3 Ultra) under v4.

## 5. Consumer NVIDIA
- **RTX 50 mines natively.** SM120 maps to `Device.BLACKWELL`, because "SM120's warp-level FP8 MMA reproduces the Blackwell atom arithmetic bit for bit" (`devices.py:16-17, 30-33`).
- **RTX 4090 is excluded.** SM89 (RTX 4090) and SM86 raise `ValueError` (`devices.py:34`). They must emulate, the same position Apple is in, but with far more ALU.

## 6. Verdict
- **v4 badly wounds the Apple bet, and kills it if B200 linearization fails.** Even if linearization works, v4 on Apple runs ~3–10x below v3 on the same Mac.
- **A v3 int7 Metal kernel lasts until mainnet sets `Fp8ForkHeight`.** That is unset today; plan for weeks to a few months.
- **Signals to watch:**
  - PR #311 merging.
  - A release that adds `Fp8ForkHeight` to `MainNetParams`.
  - The `getblocktemplate` field `requiredcertversion` (`node/btcjson/chainsvrresults.go:303-307`).
- **The only native-speed hedge** would be a consensus change adding an Apple Device. No evidence of one was found.

## 7. Microbenchmark that settles it (~1 day)
1. **Oracle:** `zk-pow` `B200::matmul_fp8` (`utils.rs:329-331`) on 256×256×4096 policy-passing operands. Build them from constant-magnitude ±X plus noise from `api/fp8/noise.rs`.
2. **Kernel A:** pure-ALU bit-exact Metal B200 emulation. Measure MAC/s on M5 and M3 Ultra.
3. **Kernel B:** fp16 `matmul2d` (M5) or `simdgroup_matrix` (M3), with K=32 chunks into fp32.
   - The ALU applies the carry and 24-bit truncation.
   - It flags groups outside span ≤ 11 or E=CE; those fall back to kernel A.
   - Measure throughput, the fallback fraction, and bit-exactness over ≥ 1e8 cells.
4. **Decision rule:** if kernel B on M5 is under ~2 TOPS-eq, v4 is uneconomic on Apple and the horizon is the v3 window only.
