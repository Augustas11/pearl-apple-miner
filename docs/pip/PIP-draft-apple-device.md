---
pip: 9999
title: INT8_EXACT, an Exact-Integer Device Profile
description: Proposes a later-fork INT8 quantizer and exact-integer Device profile for the FP protocol; motivated and measured on Apple Silicon, implementable on any conformant INT8 hardware.
author: Augustas (Malibu) (@Augustas11)
status: Draft
type: Standards Track
category: Consensus
created: 2026-10-02
requires: 3
discussions-to: https://github.com/pearl-research-labs/pips/issues/14
---

## Abstract

The FP protocol ([PIP-3](./pip-0003.md) [1], whitepaper [2]) commits a `Device`
whose FP8 E4M3 datapath the verifier replays bit for bit. The reference
implementation admits two such profiles, H100 and B200. Hardware that cannot
reproduce either datapath natively must emulate it.

This draft proposes one additional profile for a fork after the FP fork. It has
two parts:

- **A `Quant` value `Int8RowPrequant`.** After noising, it rounds each row to
  signed INT8 in `[-127, 127]` using one derived per-row scale.
- **A `Device` value `INT8_EXACT`.** Its matmul is the exact integer dot
  product, which fits a signed 32-bit integer for every admissible `k`.

Apple Silicon motivated the profile and supplied the preliminary component
measurements. Nothing in the profile is Apple-specific.

The draft specifies the arithmetic precisely. It does **not** establish three
things, and lists each as an open requirement:

- the hardness premise for the new quantizer;
- the sufficiency of the candidate jackpot policy;
- useful-work quality or end-to-end throughput.

## Motivation

`Device` and `Quant` are enumerations "chosen from a consensus-allowed list"
[2, §4.2]. PIP-3 describes hardware dependence as "a necessary evil" and the FP
protocol as "a first step toward a protocol supporting further useful
workloads" [1, Rationale].

The reference miner admits SM90, SM100 and SM120, mapping SM120 to the B200
profile because "SM120's warp-level FP8 MMA reproduces the Blackwell atom
arithmetic bit for bit" [3, `devices.py` L16, L20–L34]. A profile is thus
admitted by bit-exact reproduction of its arithmetic, not by vendor.

The tested Apple path does not provide native H100- or B200-compatible
arithmetic. On one Apple M5, software reproduction of the pinned
`B200::matmul_fp8` reached 0.2–0.6 TOPS-equivalent on the tested ordinary
synthetic operands. A native INT8 GEMM microbenchmark on the same machine
reached about 19 TOPS (§Evidence). These are component measurements. They do
not measure the profile proposed here.

The goal is a profile whose dominant cost is an INT8 GEMM with fully specified,
order-independent results. It must keep the FP protocol's structure:
commit-then-noise, post-noise non-linear quantization, a lottery on the output
tile, and a jackpot policy.

## Specification

The key words MUST, MUST NOT, SHOULD and MAY are to be interpreted as in
RFC 2119.

### Normative baseline

This PIP extends the FP protocol as implemented on branch `fp8-scheme` at
commit `f696760b259500ecb608469ea3953aeabbe78948` [4]. That commit includes
pearl PR #355 (ancestor-header chain, `public_data` of 244 bytes). Where the
whitepaper and that implementation differ, the implementation governs, as the
whitepaper itself states for the policy [2, §5.1]. The differences relevant
here are:

| Topic | Whitepaper [2] | Baseline implementation [4] |
|---|---|---|
| Jackpot policy | four checks: liveness, noise floor, tamed products, unpredictable summands (`eps_pred = 1/16`) | three checks: liveness, noise floor, consolidated unpredictable-summands census with budget `floor(k*|I_A|*|I_B|/20)`; no separate tamed-products gate (`jackpot_policy.rs` L16–L35; `unpredictability.rs` L349–L352, L570–L577) |
| `F` noise bases | `F_X` from `noise_seed_X` | both `F_A` and `F_B` from `noise_seed_B`, with distinct addresses (`noise.rs` L25–L29) |
| `delta` | `1/2` | H100 `1`, B200 `1/2` (`public_params.rs` L160–L165) |
| Certificate number | — | version 4; PIP-3 is being renumbered from 3 to 4 by an open pips PR [1] |

This PIP adds enumeration values within certificate version 4. It does not
add a certificate version.

### Enumerations

```text
Quant  := Fp8E4M3Prequant = 0 | Int8RowPrequant = 1
Device := H100 = 0 | B200 = 1 | INT8_EXACT = 2

Pairs_v4   := { (0, 0), (0, 1) }
Pairs_int8 := { (1, 2) }
```

Both bytes keep their positions in `pB` [4, `public_params.rs` L396–L403]. They
are bound into the noise seeds as in [2, eq. (10)].

### Activation contract

Let `H_fp8 := Fp8ForkHeight` and `H_i8 := Int8ExactForkHeight`, where a value
of `0` means disabled.

```text
Int8ExactActive(h) := H_fp8 != 0  and  H_i8 != 0  and  h >= H_i8
AllowedPairs(h)    := Pairs_v4 ∪ (Pairs_int8 if Int8ExactActive(h) else ∅)
```

- **Parameter constraint.** If both parameters are non-zero, `H_i8` MUST be
  greater than `H_fp8`. If `H_fp8 = 0`, the profile is disabled whatever the
  value of `H_i8`.
- **Block rule.** A block at height `h` carrying a version-4 certificate MUST be
  rejected if the `(Quant, Device)` pair in its `pB` is not in
  `AllowedPairs(h)`. This rejects the pair at `H_i8 - 1` and accepts it at
  `H_i8` and `H_i8 + 1`.
- **Where the rule is enforced.** It MUST be enforced at the height-aware
  certificate-rule layer, alongside the existing version gate
  [4, `validate.go` L508–L541]. Context-free proof verification MUST NOT be the
  only gate.
- **Reorganisations.** Acceptance is a function of block height and certificate
  only, so blocks are re-evaluated by height across a reorganisation that
  crosses `H_i8`.
- **Verifier selection.** A node MUST verify an `INT8_EXACT` proof against its
  own trusted verifier setup for that device. It MUST NOT use setup material
  supplied by the proof or by a peer.

### Noise atom

For `Device = INT8_EXACT`, each entry of the noise product `E F^T` [2, App. C.4]
uses the atom below. The noise lines `e`, `f` come from the unchanged sampler
[2, App. C.3; 4, `noise.rs`].

```text
NoiseAtom(e, f) := RNE_FP32( Σ_{u=0}^{31} e_u · f_u )   // e, f ∈ E4M3^32; exact real sum; zero ↦ +0
N := RNE_BF16(NoiseAtom(E, F))
```

### Noisy quantization (Int8RowPrequant)

The commit type (FP10), openings and row norms are unchanged [2, App. C.6].
The scales `alpha_t` and `beta_t` are derived BF16 values, not serialized
fields. They are computed as in [2, App. C.4] with the BF16 operations of
[2, App. C.1], substituting:

```text
Q_I := 127
delta_I := 1                                   // lg2_delta = 0
q_tu  := MulBF16(beta_t, N_tu)                 // unchanged from v4
y_tu  := RNE_FP32(alpha_t · X_tu + q_tu)       // CHANGED: v4 uses FMA_BF16 here
A~_tu := RNE_Z(clamp(y_tu, −127, 127))         // ties to even; A~ ∈ {−127, …, 127}
```

This quantizer changes two things relative to v4:

1. The output type is INT8 instead of E4M3.
2. The noised value `y` is rounded once to FP32, where v4 rounds it to BF16
   (`quantization.rs` L179–L184).

The second change is deliberate. In `[64, 128)` BF16 spacing is 0.5, so
rounding to BF16 and then to an integer would produce exact ties on about half
the codes. Implementations and the AIR MUST implement the FP32 path. Any
non-finite intermediate value MUST cause rejection [2, App. C.1].

### Matrix multiplication

```text
MatMul_INT8_EXACT(U, V)_ij := Σ_{u=0}^{k−1} U_iu · V_ju        // exact over Z
|MatMul_INT8_EXACT(U, V)_ij| ≤ 2^16 · 127^2 = 1,057,030,144 < 2^30
```

### Lottery extractor

This replaces the FP32 bit pattern used by the baseline
[4, `utils.rs` L540–L570]. The integers MUST NOT be converted numerically to
FP32.

```text
lanes[j] := 0                                   for j = 0..15
for j in 0..15:
  for idx in lane_assignment(P_A, P_B)[j]:      // committed order [4, layout.rs]
    word     := T[idx] mod 2^32                 // two's complement
    lanes[j] := rotl32((lanes[j] · 0x9E3779B1 + word) mod 2^32, 13)
z := LE32(lanes[0]) ‖ … ‖ LE32(lanes[15])       // 64 bytes
J := H_"jackpot"(z; noise_seedA)                // unchanged
```

The target rule and work model `T_tile = |I_A|·|I_B|·k` are unchanged
[2, eq. (12)].

### Candidate jackpot policy (new, heuristic)

The baseline census decides skippable summands from the device's accumulation
grid. Under exact accumulation that grid is empty, so the baseline census
would never count a skip. This PIP therefore proposes a **new** policy. Its
constants are heuristic and unmeasured, and they MUST be fixed by analysis and
measurement before this PIP leaves Draft.

1. **Liveness.** As in the baseline, with `delta_I`.
2. **Noise floor.** `sigma_i ≥ sigma_min_I` for every opened row, with
   candidate `sigma_min_I = 8`.
3. **Zero-product census.**

   ```text
   S_0 := { (i, j, u) : A~_iu = 0 or B~_ju = 0 }
   |S_0| = k·|I_A|·|I_B| − Σ_u nzA_u · nzB_u
   require |S_0| ≤ floor(k·|I_A|·|I_B| / 20)    // candidate; baseline budget
   ```

   Here `nzA_u` counts the rows of `I_A` whose entry `u` is non-zero, and
   `nzB_u` likewise for `I_B`.

A tamed-products check is not part of the baseline. Adding one would be a
further new requirement and is not proposed here.

### Zero-knowledge verifier

`Device = INT8_EXACT` requires the following proof tables:

- a matmul table proving `MatMul_INT8_EXACT` and the extractor word mapping;
- an `InputQuantStark` variant proving the quantizer above, including the FP32
  path;
- `ScaleStark` constants for `Q_I` and `delta_I`;
- census tables for the candidate policy.

In the baseline, setup is per device: one preprocessed LUT commitment and one
compiled wrapper circuit set per `Device` [4, `zk.rs` L150–L177;
`wrapper.rs` L60–L68]. The AIR, LUT inventory and envelope are deferred to the
reference implementation.

## Rationale

- **Why a new `Quant`, not only a new `Device`.** `Quant` fixes the quantized
  type and algorithm, and `Device` fixes the remaining arithmetic [2, §4.6]. An
  exact path native to INT8 matrix units needs INT8 operands. E4M3 values
  scaled to integers need up to 18 bits.
- **Why per-row scales.** The baseline already derives per-row scales. With one
  scale per row, the whole length-`k` dot product is one GEMM with no per-block
  epilogue or mid-`K` readout, since the lottery uses final outputs only.
- **Why exact accumulation.** It is the only candidate whose result does not
  depend on accumulation order, tiling or hardware generation. Hardware
  providing conformant signed INT8×INT8→INT32 arithmetic can implement the GEMM
  directly. Other paths, such as chunked FP32 on GPUs without an INT8 matrix
  unit, need separate validation.
- **Why `delta_I = 1` and `sigma_min_I = 8`.** These are heuristics.
  - Under ideal real arithmetic, `sigma = delta·127 / (rho + delta·√32)` with
    `rho = linf/l2`. So `sigma ≥ 8` admits `rho ≤ 10.2` at `delta = 1`, but only
    `rho ≤ 5.1` at `delta = 1/2`. BF16 boundaries shift these cut-offs slightly.
  - For i.i.d. Gaussian rows, `rho` is about 4.2 at `k = 4096` and about 4.9 at
    `k = 65536`. Both are admitted under either `delta`, but at `delta = 1/2`
    the margin is small. Real rows are not Gaussian.
  - The noise-entropy (`log2 sigma + 2.05` bits) and zero-rate (`≈ 0.40/sigma`)
    figures behind `sigma_min_I = 8` are Gaussian approximations. They are not
    guarantees for Pearl's discrete, normalized sampler.
  - **Tuning warning.** At `sigma = 8`, an independent near-zero Gaussian
    approximation gives about 4.98% zero entries per operand and 9.72% zero
    products. That exceeds the candidate budget of 1/20. Tiles near the noise
    floor would therefore be rejected or would need different constants. Joint
    acceptance must be measured with the exact sampler and BF16 arithmetic.

### Alternatives considered

Apple figures are for one M5. Unless marked measured, they are estimates.

| | Semantics | Apple estimate | Main issue |
|---|---|---|---|
| A1 (this PIP) | INT8 per-row, exact INT32 | bounded by INT8 GEMM rate; end-to-end unmeasured | new quantizer, hardness premise unestablished |
| A2 | INT8 with 1×32 block scales, ordered FP32 accumulation | ~8–12 TOPS-eq (estimate) | per-block epilogue; more accurate for outlier-heavy rows |
| B | E4M3 operands with Apple's fp16 `matmul2d` accumulation | — | vendor does not specify accumulation order or rounding; rejected |
| C | E4M3 operands, exact sum, one FP32 rounding | ~2 TOPS-eq (estimate, about 9 INT8 limb GEMMs) | native on no tested hardware |
| D | status quo: B200 emulation | 0.2–0.6 TOPS-eq ordinary operands (measured) | — |

A1 is recommended. A2 is the fallback if the useful-work evaluation shows
per-row INT8 to be too lossy.

### Naming

`Device` values name arithmetic profiles: SM120 commits `B200`. `INT8_EXACT`
names the arithmetic. Apple is the motivating platform only.

## Drawbacks

- **A second quantizer.** The hardness premise "depends on the data type and
  quantizer" [2, §2], so this adds a second premise, policy and AIR surface to
  review and to rely on.
- **Not FP8.** Useful-work compatibility is an acceptance requirement, not an
  established result. Outlier-heavy rows fail the candidate noise floor and
  would need smoothing or rotation [5, 6].
- **Economic impact.** The `Device` byte is claimed by the miner and cannot be
  checked [2, §4.2]. Any conformant INT8 hardware could mine this profile.
  Existing profile semantics are unchanged, but miner composition and relative
  profitability may change materially, possibly favouring non-Apple
  accelerators.
- **No order constraint on the arithmetic.** Exact semantics permit any exact
  algorithm, including reassociation and fast matrix multiplication. The FP8
  profiles' truncation forbids this. The practical gain is unmeasured.
- **Unvalidated constants.** `delta_I`, `sigma_min_I` and the census budget are
  candidates, not tuned values.

## Backwards Compatibility

The profile is additive and becomes valid only at `H_i8`, after the FP fork.
H100 and B200 arithmetic, quantizer, policy and wire encoding are unchanged.

Nodes without this PIP reject `Quant = 1` and `Device = 2`, because both
discriminants fail closed [4, `public_params.rs` L133–L142, L199–L210].
Activation is therefore a hard fork.

- **Validating nodes** need the fork upgrade.
- **Pool and gateway software** that parses or verifies version-4 public data
  needs the fork upgrade.
- **Existing FP8 arithmetic** remains valid.

Whether existing verifier identities are preserved depends on keeping the H100
and B200 AIRs, LUT commitments, envelopes, public-input layouts and regression
fixtures unchanged. This must be shown by regression tests. It is not assumed.

## Security Considerations

Exact accumulation removes device truncation. That removal is what the PIP-3
"floating-point attack" exploits. It does **not** establish quantized-subspace
hardness [2, §2] for `Int8RowPrequant`.

The zero-product census addresses only one form of skipped work, zero factors.
It does not address non-zero structured shortcuts. The rounding error of a
uniform integer grid is periodic in the noised value, unlike E4M3's. Whether
that is exploitable is open.

No concrete attack is known to the author. Before this PIP can leave Draft, the
following analyses are required:

1. **Residual-product algorithms.** For low-rank or structured operands, can
   `Σ_u T^A_iu T^B_ju` be computed or approximated below generic cost?
   (`T` is the rounding residual.)
2. **Adaptive input selection.** Can the miner choose operands that pass the
   policy while concentrating outputs or reducing the effective alphabet?
3. **Approximate candidate generation.** Can a miner find likely-winning tiles
   from approximations and check them exactly only afterwards?
4. **Grinding and accepted tickets per total cost.** This must count seed and
   commitment grinding, cross-profile commitment choices, preprocessing,
   rejected candidates and structured operands.
5. **Policy acceptance on honest data.** Measure with the exact sampler and BF16
   arithmetic, together with the false-acceptance behaviour of adversarial
   families.

Implementation risk is concentrated in a few places:

- the FP32 rounding path;
- `RNE_Z` ties and the clamp;
- the two's-complement word mapping;
- the census;
- the height-aware activation gate.

All of these require differential testing against an independent
implementation.

## Privacy Considerations

None beyond PIP-3.

## Reference Implementation

Not yet available. The plan is:

1. **Plaintext reference.** `zk-pow` arms for the new enumerations: noise atom,
   quantizer, matmul, extractor, candidate policy and activation gate. The
   author can provide this.
2. **Kernels.** INT8 GEMM kernels (Metal), bit-exact against (1) on at least
   10^8 cells including adversarial cases. The author can provide these.
3. **Proof tables.** AIR, LUT and envelope work. This needs review by the FP8
   circuit maintainers.

## Test Vectors

These are to be produced with (1), byte-encoded:

- **Matmul.**
  - `k = 65536`, all `+127`: `1,057,030,144`.
  - All `−127` against `+127`: word `0xC0FF0000`.
  - Exact cancellation to 0.
- **Extractor.**
  - A lane fed `−1,057,030,144` then `−1` gives `0xE000181F`, then `0x0A8DC784`.
  - Complete 64-byte lottery messages for mixed-sign tiles.
- **Quantizer.**
  - An exact `alpha·X + q` in `(64.5, 64.75)` yields `64` on the v4 BF16 path
    and `65` on the FP32 path.
  - `.5` ties in FP32.
  - Clamp at `±127`.
  - Rows at the norm floor.
  - Rows at the `sigma_min_I` boundary.
- **Noise atom.** Subnormal inputs, exact cancellation to `+0`, and FP32 ties.
- **Policy.** Tiles at the census budget and at budget + 1.
- **Activation.**
  - Heights `H_i8 − 1`, `H_i8`, `H_i8 + 1`.
  - `H_i8 = 0` (disabled).
  - `H_fp8 = 0` (disabled).
  - Pairs outside `AllowedPairs`.

## Evidence

### Provenance

| Item | Revision | Role |
|---|---|---|
| Benchmarks, emulation oracle, unit tests | `fp8-scheme` @ `2569546` (before PR #355) | evidence only |
| Proposed normative baseline | `fp8-scheme` @ `f696760` (includes PR #355) | specification |
| Files cited for arithmetic, quantizer, policy, noise, wrapper, devices, fork rules | identical at both revisions | — |

PR #355 changes the ancestor-header encoding and the `public_data` size only.
No claim in this PIP depends on it.

### Component measurements

"TOPS" here means `2 × MAC/s`. All figures come from one Apple M5 (10-core
GPU, macOS 26.5) and are preliminary component microbenchmarks. Logs,
commands and revisions will be packaged with the PIP after discussion.

- **INT8 GEMM microbenchmark.** Metal 4 `matmul2d` `int8×int8→int32`, 4096³:
  about 19 TOPS on an idle machine; 8.7–12.4 under heavy CPU load.
- **INT8 GEMM exactness.** 161 variant-by-shape checks of accumulators against
  a CPU int64 oracle, 0 mismatches. This covers the GEMM and readout component
  only.
- **B200 emulation.** Six Metal kernels matched `B200::matmul_fp8` on
  107,282,432 cells at `k = 4096`, with 0 mismatches. Speeds: 0.2–0.6
  TOPS-equivalent on ordinary operands; 4.9–5.4 on the tested
  constant-magnitude corpus.
  - At `k = 4096`, the grid and accumulator bounds of that corpus made replay
    exact.
  - At `k = 65536`, the emulator's sufficient guards fell back on 21–32% of
    groups, while B200 rounding itself occurred in about 0.06%.
- **Certificate-v3 results.** A regtest node accepted proofs mined by an Apple
  GPU under the INT protocol. This does not validate this profile.
- **Hardware.** The tested OS exposes no FP8 tensor type. Apple announced FP8
  tensor types for OS 27, measured as emulated on M4. Support at the API level
  would not by itself provide B200-compatible arithmetic.

### Unmeasured

- `INT8_EXACT` end-to-end throughput.
- Useful-model quality through the full pipeline: FP10 commitment, noising,
  rounding, descaling, peeling and policy rejection, compared with the original
  model and an ordinary INT8 baseline.
- Proof cost.
- Matched non-Apple INT8 throughput.

The relative figure of an M5 against an H100 compares a local microbenchmark
with an advertised dense peak [7]. It is not a matched mining comparison.

## References

1. [PIP-3: FP PoUW Certificates](./pip-0003.md), with the open promotion PR
   renumbering it to certificate version 4:
   <https://github.com/pearl-research-labs/pips/pull/13>.
2. [Pearl Floating Point Scheme Specification](https://pearlresearch.ai/Pearl_Whitepaper.pdf),
   September 2026.
3. `miner/miner-base/src/miner_base/devices.py` at [4].
4. `pearl-research-labs/pearl` branch `fp8-scheme` @
   `f696760b259500ecb608469ea3953aeabbe78948` (PR
   <https://github.com/pearl-research-labs/pearl/pull/311>; includes
   <https://github.com/pearl-research-labs/pearl/pull/355>).
5. [SmoothQuant](https://arxiv.org/abs/2211.10438).
6. [QuaRot](https://arxiv.org/abs/2404.00456).
7. [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/).

## Copyright

Copyright and related rights waived via
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).
