# F1 result: full K3-NA prototype on M5 (SPEC v0.2 §9 F1)

Date 2026-10-02 · Apple M5 MacBook Air (Mac17,3, fanless) · macOS 26.5 (25F71) · on AC power, battery 100% charged.
Code: `bench/f1_k3/` (README has the exact commands). Raw output: `bench/evidence/f1_k3_m5_{correct,perf}.txt`.

## Verdict: **F1 FAIL** (bit-exact: **Y**; P1: **N** at both shapes)
- **Correctness passes.** The production kernel is bit-exact against the CPU oracle on every check, and the oracle
  reproduces Pearl's own miner and verifier.
- **P1 fails narrowly.** Full K3 runs at **0.832×** (4096²×4096) and **0.821×** (8192²×4096) of the baseline median. The
  target is 0.85. Both CIs are tight enough (half-width ≤ 0.05). At 8192 the whole 90% CI is below 0.85; at 4096 the upper
  bound is 0.856.
- **The hash is not the problem.** BLAKE3 + two U256 compares + atomics cost **+1.3–1.4%** over fold-only. The gap is
  the K loop + V6 fold vs one `run()` over the whole K: fold-only is already at 0.84× of the baseline.
- **Caveat: a loaded machine.** Both batches ran with loadavg ~8–12 and a mid-batch throughput collapse (per-run GPU
  time rose ~10× at both shapes). Ratios come from paired, alternating rounds, so they partly cancel this, but they are
  not idle numbers. An idle re-run is the cheapest way to tell whether ~2–3 points are measurement or kernel.

## Kernel (as built)
- `matmul2d` int8×int8→int32, `execution_simdgroups<4>`, tile 128×64, RK = 128 (one `run()` per rank chunk),
  `multiply_accumulate`, `relaxed_precision = false`, A' m×k and B' k×n row-major.
- V6 fold (`uint4` view of the 64 contiguous int32 per lane) → `rotl13` into `jp[(chunk) % 16]`. Trailing k % 128 is never
  run (no C output).
- Per lane: one keyed BLAKE3 compress of the 64-byte LE transcript (flags 27, key = a_noise_seed as 8 u32). Then a U256 LE
  compare against the block bound B and the share bound S.
- Finds: a device `atomic_uint` counter per array. Block capacity is ≥ 4 and share capacity ≥ 64; the arrays are separate.
  - Slot = (t_rows, t_cols, transcript[16], hash[8]).
  - t_rows / t_cols = threadgroup origin + the lane's minimum row / column from `get_multidimensional_index`. That is the
    global pattern minimum, and therefore a Pearl valid offset.
  - Classification is independent per array: a hash ≤ B lands in block, and a hash ≤ S lands in share.
  - Counters keep counting past capacity; only `idx < cap` is written.

## Correctness (`f1_k3_m5_correct.txt`)

| Check | Result |
|---|---|
| Oracle vs `pearl_mining` 0.3.1 (`mine()` at m=128, n=64, k=2048, r=128, this pattern, cert v3; signals pinned with `signal_range=(c,c)`, c ∈ {0, 1, −64, 64}) | **PASS 4/4.** The oracle predicted the winning tile before mining (tile #3, #1, #9, #0). Opened rows/cols, raw A/Bᵀ roots, m/n/k/r all match. `verify_plain_proof_for_cert_version(3, …)` accepts. With `nbits_override` just below/above the oracle hash, the verifier rejects/accepts exactly as the oracle predicts. The GPU kernel on the same noised job matches the oracle and has Pearl's winning tile in its block array |
| (a) all transcripts + hashes, full jobs: 256×256×4096 and 128×64×65536, plus 256×256×2048, 384×192×4096, and a ±127 extreme-magnitude 128×64×65536 job | **PASS.** Every tile appears exactly once, bit-exact (transcript[16] + hash[8] + t_rows/t_cols) |
| (b) boundary vectors: bound = hash−1 / hash / hash+1 for the min, median and max tiles, plus ±2^(32i), i = 1..7 | **PASS.** Min tile: (0, 1, 1) finds as expected. All counters and sets equal the oracle's |
| (c) non-saturated deterministic wins/losses (S = 2^256/16, B = 2^256/200), run twice | **PASS.** Win/loss sets equal the oracle's and are identical across runs |
| (d) slot overflow (bound = max, capacities 4 / 64) | **PASS.** Counter = tile count (e.g. 1152 / 1152), only `cap` slots are written, the guard canaries are intact |
| (e) simultaneous block + share finds (large and production capacities, exact fit 4 / 64) | **PASS.** Each array holds exactly its oracle set |
| Total | 109 GPU cases, 35,332 slots compared, **0 failures** |

## Performance (`f1_k3_m5_perf.txt`)

Method:
- 31 paired rounds per shape over int8bench / base / fold / k3, with the order reversed on odd rounds.
- Timing is GPU timestamps on identical random operands in [−127, 127].
- **base** = the int8bench matmul2d 128×64 kernel with the C store disabled (a cooperative-tensor destination plus a
  conditional sink).
- **k3** uses a share bound giving ≈ 4 finds per job and a block bound of 2^200 − 1, with capacities 4 / 64.
- Ratio = t_base / t_variant per round. The CI is a 90% percentile bootstrap of the median.

| Shape | P1 k3/base median [90% CI] (half-width) | fold/base | k3/fold | Hash+compare+atomics overhead | base/int8bench | Median TOPS base / fold / k3 | Loadavg at start → end |
|---|---|---|---|---|---|---|---|
| 4096²×4096 | **0.832** [0.814, 0.856] (0.021) | 0.844 | 0.986 | **+1.40%** [−0.11, +4.89] | 1.042 | 8.64 / 6.89 / 6.56 | 11.42 → 12.35 |
| 8192²×4096 | **0.821** [0.812, 0.839] (0.013) | 0.840 | 0.988 | **+1.26%** [+0.84, +2.62] | 1.016 | 7.12 / 5.90 / 5.87 | 7.91 → 7.07 |

P1 gate (median ≥ 0.85 and CI half-width ≤ 0.05):
- 4096²×4096: **FAIL** (CI half-width OK).
- 8192²×4096: **FAIL** (CI half-width OK).

Notes:
- **Absolute TOPS are depressed.** Peaks were ~20 TOPS, but per-run time grew up to ~10× mid-batch at both shapes, and
  the medians are far below the 19 TOPS on record. Only the paired ratios are meaningful.
- **Share finds per job were constant** (8 at 4096, 3 at 8192). The operands are fixed per batch, so the same tiles win
  every run. That also confirms the compare and atomic paths ran in every timed K3 run.
- **Load gate.** The harness checks loadavg ≤ 8 before waiting for the lock. The 4096 batch passed that check, then waited
  1680 s for the lock (another agent held it) and started at loadavg 11.4. Flagged here: it started above 8.

## What this means
1. **The fused epilogue is cheap.** BLAKE3 + dual compare + bounded atomics adds ~1.3%. K3-NA's design (per-lane pattern,
   no cross-lane traffic, lock-free slots) is sound and exact.
2. **P1 is lost in the explicit RK = 128 K loop + fold.** Fold-only is ≈ 0.84× of the single-`run()` baseline.
   - This matches R6, where V6 vs V0 had a −7.7% median but a range down to −24%.
   - Here the baseline is also ~2–4% faster than int8bench, because it does not store C.
3. **Before invoking the §1.2 stop:** re-run `k3 perf` on an idle machine (loadavg < 3). The margin is 2–3 points and both
   batches were loaded. If it still fails, the levers are the K-loop structure and scheduling (e.g. threadgroup
   traversal order, a persistent-threadgroup schedule), not the hash. None of these is measured here.

## Idle rerun (2026-10-02 08:25Z, loadavg 4.0–4.6, AC power, 31 rounds; `bench/evidence/f1_k3_m5_perf_idle.txt`)

| Shape | P1 k3/base median | 90% CI | fold/base | hash+compare+atomics |
|---|---|---|---|---|
| 4096²×4096 | **0.868 (passes 0.85)** | [0.833, 0.905] | 0.849 | −0.4% (noise) |
| 8192²×4096 | **0.818 (misses)** | [0.792, 0.828] | 0.831 | +1.4% |

**Thermal throttling is the dominant effect on this fanless MacBook Air.** Raw per-round GPU times climb within seconds of starting:
- **4096³:** base goes from 6.0 ms to a 53 ms peak, then settles around 17–20 ms.
- **8192³:** base goes from 24 ms to about 60 ms and stays there.

**In the cool first rounds K3 nearly matches the baseline:** 4096 at 6.27 vs 6.04 ms (0.96), 8192 at 31.8 vs 24.4 ms (0.77). Once throttled, the fold/K-loop share grows.

**Conclusions:**
- The P1 verdict is **pass at 4096, near-miss at 8192, both measured while throttling**. A fan-cooled Apple10 Mac is needed for an authoritative P1/P5.
- Sustained mining on a MacBook Air settles at roughly **8–9 TOPS**, not 19. Plan per-device duty cycles and the fan-cooled requirement accordingly (SPEC P4).
