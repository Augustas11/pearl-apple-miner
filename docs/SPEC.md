# SPEC: Pearl v3 Metal miner ("pmk") — v0.3

Status: v0.3 (reviewed; fixes applied) — **v1 = K3-SG on the M3 Ultra first; K3-NA mines after F1 passes on a fan-cooled M5** — feasibility results folded in (F1–F3, M3 Ultra K3-SG window) · 2026-10-02 · owner: Augustas (Malibu)
Evidence base: `docs/kb/*.md`, `bench/evidence/*`. Every number cites a measured file or is marked TARGET / UNVERIFIED.

## 1. Goal, scope, window

### 1.1 Goal
Build the fastest correct Pearl v3 miner for Apple Silicon. v3 means cert version 3, int7×int7→int32.

The targets are **Apple10 GPUs (M5 family) and Apple7–9 GPUs (M1–M4, primary: a fan-cooled M3 Ultra), on operator-owned Macs**. Why both: the fanless M5 MacBook Air throttles to about 8–9 TOPS sustained (`kb/f1-result.md`, idle rerun), while the M3 Ultra mines with K3-SG at **16.6 TOPS sustained, flat over 4 min** (`bench/evidence/k3sg_m3ultra.txt`), and most installed Macs are M1–M4. The miner produces proofs that a certificate-verifying Pearl node accepts. It mines mainnet through the operator's node (solo) or, after G5b, an existing public pool (§7.6).

Why: the only existing Apple GPU miner (OpenJarvis `apple-mps-pearl`) runs at 4.5 GOPS on M5 and 8.1 GOPS on M3 Ultra (`bench/evidence/oj_upgraded_m{5,3ultra}.txt`). A raw Metal int8 matmul is about 1,900–4,200× faster.

### 1.2 Decision record: yield and stop budget (operator decision, not a technical gate)
- **Expected blocks.** E[blocks/day/device] = ops_per_s × 86400 × target / 2^257.
  - `target` comes from live `getblocktemplate.bits`.
  - With r = 128 and k % 128 = 0, the penalized bound equals the block bound (`sanity_checks.rs:190-196`).
- **What this means for one device.** One M3 Ultra at ~16.6 TOPS against bits 0x177fd82e (2026-10-02) ≈ **one block per ~36 years** solo (E ≈ 7.6e-5 blocks/day by the formula above). Even the easiest mainnet target is ~5.6e14 ops per block. Pool or fleet aggregation is the only way to get regular payouts.
- **Why build anyway.** The operator's thesis is positioning: become the Apple-Silicon Pearl miner and pool, and support a PIP adding an Apple device. Per-device yield is not the thesis. Recompute E[blocks] weekly from live bits and log it.
- **Stop budget.** Stop v3 work, keeping only what is built, if any of these happens:
  1. A feasibility gate F1–F3 fails (§9). F1 is scoped per kernel family; see the §9 F1 status.
  2. Mainnet `Fp8ForkHeight` is announced before G5 passes.
  3. Effort passes **4 engineer-weeks** without G5. This number is a TARGET that the operator can change.

### 1.3 Window
- v3 stays valid until mainnet sets `Fp8ForkHeight`. After that it is rejected outright (`docs/kb/fp8-cert-v4-risk.md` §1).
- **The timeline is unknown.** PR #311 is open and BLOCKED; PR #355 (v4 ancestor-header encoding) merged into `fp8-scheme` 2026-10-02. Estimate (PLAN §1, low confidence): v4 mainnet late Oct–late Nov 2026. Past forks activated within **hours** of release.
- Watch `Augustas11/pearl-watch` and `getblocktemplate.requiredcertversion`.

### 1.4 Non-goals (v1)
- v4 / FP8 mining. That is a separate track (§11).
- Mining inside LLM inference.
- MoE jobs.
- Running a pool server.
- **Third-party fleets of Macs.** A separate spec, gated on owner consent and authorization (§11).
- Concurrent mining while another GPU workload (for example local inference serving) runs on the same Mac (§7.3).

## 2. Definitions
- **Job:** one (header, config, A, B) attempt.
- **Find:** a tile whose jackpot hash ≤ the threshold.
- **Block threshold:** from header `nbits`.
- **Share threshold:** a local diagnostic threshold S, expressed as compact nbits and used only with `nbits_override`.
- **dot_len** = k − (k mod r).

## 3. Protocol contract

### 3.1 Consensus (bit-exact; source of truth = `zk-pow` @ 7039e66f, not `noisy_gemm.py`)

| # | Must match | Reference |
|---|---|---|
| C1 | `job_key = blake3(header_bytes ‖ config_bytes)`, using upstream serialization (76-byte incomplete header, upstream `MiningConfiguration::to_bytes`) | `mine.rs:431-436`, `proof.rs` |
| C2 | Commitment: padded raw row-major A and padded raw row-major Bᵀ, each a keyed BLAKE3 under `job_key`, giving **raw** Merkle roots. Seeds use the **salted** derivation for cert v3: A's root salted with m, B's with n, **for seed derivation only**. Proofs carry raw roots | `mine.rs:442-479`, `seed.rs:11-59`, `proof_utils.rs:268-277, 519` |
| C3 | Noise: keyed BLAKE3 with dense `(b&63)-32`, sparse `i0`/`i1` via the mulhi rule, gather-subtract. Noise ∈ [-63,63] | `circuit/pearl_noise.rs:19-153` |
| C4 | A' = A + E_A and B' = B + E_B, in [-127,127]. Signal ∈ [-64,64] (the verifier range-checks it) | `mine.rs:15-16, 65-82`, `verify.rs:101-133` |
| C5 | Cumulative int32 accumulator. Fold only at full r-chunks; trailing k mod r is excluded | `mine.rs:87-107`, `jackpot/helper.rs:19-34` |
| C6 | `x = XOR(tile acc as u32)`. Slot `(ll/r − 1) % 16`. `jackpot[t] = rotl32(jackpot[t],13) ^ x` | `pearl_program.rs:23,25` |
| C7 | Tile element set = (t_rows + rows_pattern) × (t_cols + cols_pattern). Offsets per `offset_is_valid`, so tiles partition the matrix. t_rows + max(rows_pattern) < m, and likewise for cols | `proof_utils.rs:158-245` |
| C8 | `blake3(le_bytes(jackpot[16]), key=a_noise_seed)`, read as U256 little-endian, must be ≤ bound | `proof_utils.rs:1502-1505`, `sanity_checks.rs:190-246` |
| C9 | r is a power of 2 in [32,1024]. k ≤ 2^16, k % 64 = 0, 16r ≤ k ≤ 4r². h and w even, 32 ≤ h·w ≤ 256. (h+w)·dot_len ≤ 4 MiB. Patterns: ≤ 3 dims, start at 0, sorted. m,n ≤ 2^24. Rank-penalty rule for r < 128. **STARK degree_bits ≤ 19** (ZK only; `verify_plain_proof` does not check it) | `sanity_checks.rs:13-58, 164-196`, `pearl_program.rs:46-77`, `pearl_circuit.rs:154-162` |
| C10 | Network `requiredcertversion` = 3 | `validate.go:537-542` |

### 3.2 v1 policy (stricter than consensus)
- **Rank:** r = 128, so the rank penalty is satisfied by construction. Assert this when building the config.
- **k:** k % 128 = 0, so dot_len = k. **Default k = 4096** (TARGET: the P6 objective, GPU rate vs k, is not yet evaluated; F3 alone would favour 2048) (F3: degree_bits 14, ZK proving ≈ 9.8 s on M5 10-core, ≈ 4.9% orphan risk). Allowed k ∈ [2048, 8192]; 16384 is borderline (≈ 8.9% orphans). Source: `docs/kb/f3-result.md`.
- **Patterns (per kernel family; each committed in its own MiningConfiguration):**

  | Kernel | rows_pattern | cols_pattern | h·w | v1 shape rule | Evidence |
  |---|---|---|---|---|---|
  | K3-NA (Apple10) | `[0,8,64,72]` | `[0..3,16..19,32..35,48..51]` | 64 | m % 128, n % 64 | `kb/r6-result.md`, `kb/f1-result.md` |
  | K3-SG (Apple7–9) | `[0,8,16,24]` | `[0,1,8,9,16,17,24,25]` | 32 | m % 64, n % 64 (production cfg 64×64; pattern period 32) | `kb/k3sg-design.md`, M3 Ultra probe in `bench/evidence/k3sg_m3ultra.txt` |

  - (h+w)·k ≤ 163,840 B (NA) / 98,304 B (SG) at k = 8192; the consensus cap is 4 MiB. Pearl itself requires only t + max(pattern) < m/n and `offset_is_valid`; the shape rules are kernel constraints.
  - Build patterns with `PeriodicPattern.from_list`, and assert that serialization round-trips.
- **Attempts:** the header is immutable within a job.
  - **A is fixed per template** (hashed once). **B is fresh per job**, from a CSPRNG seeded from OS entropy.
  - This is legal: the verifier checks only the range and Merkle consistency (`verify.rs:101-133`). Seeds chain through hash_b (`proof_utils.rs:268-277`), so a fresh B re-randomizes both noise seeds.
  - Recorded in ADR-001. It is policy, not a consensus requirement.

## 4. Architecture

```
operator Mac (Apple10 or Apple7–9)
  pmk-miner (Python): orchestration + gateway glue only (metadata, never matrix bytes)
    ├─ pmkcore (Rust, C ABI): job builder — CSPRNG A/B into shared buffers, commitment (blake3+rayon), seeds,
    │                          proof construction (zk-pow PlainProof, Merkle multiproofs), jackpot oracle (G1)
    ├─ libpmk  (Swift, C ABI): Metal — probe, K1 noise_gen, K2 noise_apply, K3-NA / K3-SG mine_gemm; one command buffer per job
    └─ verifier gate: verify_plain_proof_for_cert_version(3, header, proof[, nbits_override])
  pearl-gateway (operator-controlled, local UDS/loopback) → operator's pearld (remote or local)
```

Rules:
- **R-A1. Native hot path.**
  - Job build (RNG, commitment, seeds) and GPU dispatch are native. Python touches only job metadata.
  - Python per-job overhead must be < 2% of job wall time. Both native libraries release the GIL.
  - Why: OpenJarvis's per-tile Python is ~2,000× slow, and Python-side hashing alone fails P2 by ~3× (reviews #3).
- **R-A2. Double-buffering** follows the job lifecycle in §4.1. Use `addCompletedHandler`, never `waitUntilCompleted`, on the hot path.
- **R-A3. Never submit an unverified proof.** Every proof must first pass the local verifier gate. A gate failure is P0 for that device: stop mining there.
- **R-A4. Memory limits.** Every buffer < 2 GB (llama.cpp #28748). Total pmk memory ≤ the §6.1 budget.
- **R-A5. Credentials.**
  - Node RPC credentials and the gateway stay on operator-controlled hosts.
  - Patch out the gateway's credential logging at source (`pearl_client.py:20`), and test the error paths for leaks.
  - Miner↔gateway traffic goes over a protected Unix socket or loopback. The gateway's TCP RPC is never exposed off-host.
- **R-A6. Cert-version guard.** On every job, `cert_version == 3`, or the job is rejected, pending work is cancelled, and an alert fires.

### 4.1 Job lifecycle (buffer ownership)
- **States:** `building → gpu → scanned → (proving → submitted) → released`.
- **Slot release:** a buffer slot is released only after:
  - its found slots are processed;
  - every proof built from it is submitted or abandoned.
- **Each job record keeps:** job_id, template id (prev-hash + bits + header bytes), config bytes, raw A/Bᵀ (or the retained A plus Bᵀ), and seeds.
- **Proof construction:**
  - rows = sorted(t_rows + rows_pattern), cols = sorted(t_cols + cols_pattern), using the **global** pattern minima, not threadgroup origins.
  - Open the raw A and Bᵀ.
  - Assert the reconstructed config equals the job config.

## 5. Kernels

### 5.1 K3-NA: fused GEMM + fold + PoW (Apple10)
- **Matmul:** Metal 4 `matmul2d` int8×int8→int32, `execution_simdgroups<4>`, tile 128×64, `RK = 128` (one `run()` per rank chunk), `mode::multiply_accumulate`, `relaxed_precision = false`.
  - Operand layout: K2 writes A' as m×k row-major and B' as k×n row-major, for `matmul2d(…, transpose_left=false, transpose_right=false)`, which is the measured configuration.
  - The commitment is over raw Bᵀ (n×k). K2 reads Bᵀ.
- **Fold (V6):** a `uint4` view of the int32 cooperative-tensor storage, XOR, then rotl13 into a per-lane 16-word transcript.
  - **Measured (F1, `kb/f1-result.md`):** the full K3 (fold + BLAKE3 + compare + atomics) is bit-exact on 109 cases / 35,332 slots. BLAKE3 + compare + atomics add ≈ 1.3%.
  - **Speed vs baseline on the fanless M5 Air (idle rerun):** 0.868 at 4096²×4096 (passes 0.85) and 0.818 at 8192²×4096. The Air throttles within seconds; in the cool first rounds K3/base was 0.96 at 4096² and 0.77 at 8192². An authoritative P1 needs a fan-cooled Apple10 Mac.
- **Fallback (V2):** the per-element accessor fold. It is exact and ~40% slower, and is used when §5.4 rejects V6.
- **PoW:** one keyed BLAKE3 compress per lane, compared U256 LE against **two** thresholds: share S and block B.
- **Found slots:**
  - A device `atomic_uint` counter for each of two arrays:
    - block finds: ≥ 4 reserved slots, never shared with shares;
    - share finds: ≥ 64 slots.
  - Every write is bounds-checked.
  - **Overflow** (counter > capacity) makes the host recompute the full job's jackpots on the CPU from the retained job data with the G1 oracle, which is deterministic recovery. Overflow is expected only in tests.
- **No C output.**

### 5.2 K1/K2: noise
- **K1:** BLAKE3 keyed single-block, ported from `pearl-gemm/csrc/blake3/blake3.cuh`. It must pass `blake3` crate test vectors.
- **K2:** gather-subtract; writes A' and B' in the §5.1 layouts.
- **Budget:** K1 + K2 ≤ 10% of K3 time at the default shape (TARGET; measured in F2).

### 5.3 K3-SG: M1–M4 (incl. M3 Ultra) — **v1 (reprioritized 2026-10-02)**
- **Math:** fp32 `simdgroup_matrix<float,8,8>` with int8→float conversion while staging, and a fresh fp32 accumulator per 128-chunk added into an int32 running accumulator. This is provably exact, since 128·127² < 2^24.
- **Lane ownership:** each lane owns exactly one Pearl tile of a 32×32 simdgroup tile (pattern in §3.2). No cross-lane traffic.
- **Production config:** `64x64x16x2x2x2` (the window's "BEST" from a 5-round sweep at 4096², `bench/evidence/k3sg_m3ultra.txt`; the README default `64x64x16x2x2x1` is superseded).
- **Measured on the M3 Ultra (2026-10-02 measurement window, `bench/evidence/k3sg_m3ultra.txt`):**
  - Layout probe PASS: it matches MLX `get_coord`, every lane set is a legal pattern, and lane origins equal Pearl's valid offsets.
  - Correctness: 846 case-runs (4 jobs × 9 cfgs) plus the layout probe, 0 failures, including boundary, overflow and simultaneous-find cases.
  - P3: K3-SG / same-round fp32 GEMM = **0.912 (4096²×4096) and 0.917 (8192²×4096)**, passing the 0.80 target. Versus MLX fp32: 0.99 and 0.95.
  - Absolute: **16.35–16.58 TOPS**. A 240 s sustained run was flat (−0.0%).
- **Not yet tested:** M1, M2, M4. The probe fails closed ("DO NOT MINE") on any layout mismatch.
- **Artifact:** a separate artifact. The window ran on macOS 26.4.1 with runtime-compiled Metal source; the minimum OS for older Macs is settled per machine by the probe.

### 5.4 Startup probes (fail closed)

#### K3-NA probe
- **Cache key:** (GPU name, OS build, Metal compiler version, libpmk + metallib hash, descriptor/specialization).
- **Checks:**
  1. Per-lane `get_multidimensional_index` sets, normalized, equal the committed pattern and exactly partition the tile.
  2. Every element is valid (`get_mask`).
  3. The storage is contiguous and 16-byte aligned, and the V6 view equals the accessor values element for element.
  4. **The production pipeline itself, not a separate probe kernel,** runs a known-answer job at 256×256×**4096** (exercises the 16-slot wrap) and matches the embedded oracle transcripts and finds exactly.
- **Actions:**
  - 1, 2, or 4 fails → refuse to mine on this device.
  - Only 3 fails → V2. Report it as the "V2 performance class".

#### K3-SG probe (as run on the M3 Ultra; `kb/k3sg-design.md` §2)
1. `simdgroup_load` → the `thread_elements()` layout covers all 64 fragment elements exactly once.
2. Injecting values via `thread_elements()` lands them where `simdgroup_store` places them.
3. The fp32 MMA accumulator follows the same layout (checked against an exact CPU product).
4. char→float conversion is exact for all 256 int8 values.
5. Pattern and partition checks:
   - the layout equals the hard-coded `get_coord` map;
   - every lane set is a rows × cols product;
   - all lanes share one normalized pattern equal to the committed constants;
   - lane sets partition the 32×32 tile;
   - lane origins equal Pearl's valid offsets.
6. A known-answer job at 256×256×4096 on the production pipeline.

Any failure → refuse to mine ("DO NOT MINE"). There is no V2 fallback for K3-SG. The cache key uses the source hash plus compile options in place of the metallib hash.

## 6. Device gating and resources

| Family | v1 | Min OS | Artifact |
|---|---|---|---|
| Apple10 (M5 family) | K3-NA | macOS 26.4 (int8 tensors; the probe settles it per machine) | Metal 4 metallib (`-std=metal4.0 -mmacosx-version-min=26.2`) + runtime-compile fallback with identical flags |
| Apple7–9 (M1–M4, incl. M3 Ultra) | K3-SG (v1; measured on M3 Ultra) | TARGET macOS 14+ (binary minos 14.0, runtime MSL 3.1); tested on 26.4.1 only | separate artifact (fp32 simdgroup_matrix) |

### 6.1 Supported shapes and memory
- **Default job:** m = n = 8192, k = 4096 (F2: build ≈ 12 ms vs GPU ≈ 85 ms per job; P2 ≈ 0.99–1.09). 4096² is marginal (first-round drop, unexplained). The probed device rate sizes each command buffer to ≤ 400 ms predicted GPU time (8192²×4096 ≈ 33 ms on M3 Ultra, ≈ 85 ms on M5 Air).
- **Total pmk memory:** ≤ 25% of physical RAM. That covers rotating slots (≤ 3), the retained A, and Merkle state. Proving memory is accounted separately on the gateway host.
- **Protocol extremes** (e.g. m = 2^24) are rejected by allocation-free validation, not allocated.

## 7. Mining operation

### 7.1 Job loop
1. Get the template. Its identity is prev-hash + bits + header bytes, not height alone.
2. Build A (once per template) and B (per job).
3. Run on the GPU.
4. Scan the found slots.
5. Verifier gate.
6. Submit.

A new template invalidates in-flight jobs and their commitments.

### 7.2 Submission tracking
`submitPlainProof` returns "submitted" before validation (`server.py:214-220`), and stale or duplicate proofs are dropped silently. So:
- **Confirm acceptance on chain.** Within 120 s, check `getblock`/`getbestblockhash` for the submitted header, and tail the gateway log for `Block rejected` / `Error submitting block`.
- **Classify every outcome:** accepted, consensus-invalid, stale, duplicate, transport error, proving error.
- **Only consensus-invalid** (a node rejection after the local gate passed) is a P0. It stops that device and alerts.
- **Stale and duplicate** outcomes are counted and expected.

### 7.3 Coexistence (v1)
- v1 does **not** mine while another GPU workload (for example inference serving) is running on the same Mac. Run pmk on an otherwise idle machine; pause other GPU workloads first and confirm they are paused.
  - Why: a pause hook with no caller on the other side can't guarantee yielding.
- **Long runs:** G5b (≥ 1 h) and G6 (24 h) need a dedicated machine, or a window in which other GPU workloads stay paused for the whole run.
- **Resume reporting:** when an outer wrapper pauses other workloads, resume can fail silently after a GPU-heavy window. The optional lock/resume-hook features (`--lab-owner-token`, `--resume-hook`) exist so v1 tooling reports the resume outcome and fails the run if resume is unconfirmed.
- Coexistence with other workloads (pause acknowledgement, queue depth, CPU-worker suspension, inference-latency acceptance) is Phase 2.

### 7.6 Pool mode (spec to be completed after the §13.6 research)
- **Template source:** the pool job. The header is used exactly as given, never modified.
- **Share target:** the pool difficulty converted to compact nbits, used **only** as `nbits_override` in the verifier gate and as the kernel's share bound.
- **Checks that change in pool mode:** the §7.5 payout check and the §7.2 on-chain check do not apply, because the pool's coinbase pays the pool. Pool acceptance replaces them, plus an occasional cross-check of the pool's dashboard for the operator wallet.
- **Untrusted input:** all pool messages are size-limited and validated. Use TLS where offered. No node credentials are involved.
- **Compatibility prerequisite:** the pool must accept our MiningConfiguration: rank 128, k = 4096, and both patterns, including the h·w = 32 SG pattern. Otherwise K3-SG cannot pool-mine.
- **Specified by research (`docs/kb/pearl-pool-protocol.md` §6):** JSON object dialect only; authorize `{wallet, worker, pass, agent:"pmk/<ver>"}`; submit `{job_id, plain_proof: base64(bincode PlainProof)}` (≈60–100 KB), always under the job_id the work was built from. The pool does the ZK proving. HeroMiners has a fixed difficulty of 2^21 and no TLS. Apply the miner-side validation rules in §5 of that doc (64 KiB line cap, strict notify checks, cert_version == 3, D_floor, rate limits, wallet logged only as prefix…suffix).
- **Units:** 1 pool hash = 1 MAC = 2 of our ops. M3 Ultra ≈ 8.3 TH/s pool ≈ one share per 18 min at HeroMiners ≈ 0.18 PRL/day.

### 7.4 Thresholds
- **Block threshold:** from header `nbits`.
- **Share threshold:** a local diagnostic S, as compact nbits, used only via `verify_plain_proof_for_cert_version(3, …, nbits_override=S)`.
- **Never modify header `nbits`.** The Pearl gateway is not a pool, so shares are not submitted to it in v1.
- Choose S for about 1 share/min/device. The observed share rate feeds the lost-find monitor (§7.5).

### 7.5 Health monitors
- **Lost-find monitor:** alarm if the observed shares over a window fall outside the Poisson 99.9% interval of E = completed_ops × target(S)/2^257.
- **Daily E[blocks] log** (§1.2).
- **Payout check:** at startup and in G5, the coinbase outputs pay the operator-approved script on the expected network (HRP / network checked). Why: the gateway builds the coinbase from `PEARLD_MINING_ADDRESS` and its parser accepts any HRP (`blockchain_utils.py:124`).

## 8. Performance targets and measurement

Ratios reduce load sensitivity but **do not make results load-proof**. Every result reports load, power source, machine model, and variance.

**P3 baseline:** `base_f32` (same tiling, fp32 operands, C store disabled). **P1 baseline:** the `int8bench` matmul2d kernel, 128×64 tile, **with the C store disabled**, GPU timestamps, identical inputs, and paired alternating rounds (≥ 15).

| ID | Metric | Target | Gate |
|---|---|---|---|
| P1 | Full K3-NA (fold + BLAKE3 + compare + atomics) vs baseline at 4096²×4096 and 8192²×4096 | median ≥ 0.85, with the 90% CI half-width ≤ 0.05 | F1, G4 |
| P2 | Wall-clock completed work: Σ2mnk over ≥ 200 jobs ÷ elapsed, vs K3-alone wall rate | ≥ 0.85 | F2, G4 |
| P3 | K3-SG vs same-round fp32 simdgroup GEMM (same tiling, C store disabled) | ≥ 0.80 — **PASS on M3 Ultra (0.91–0.92)** | G4 |
| P4 | Sustained 30 min, fan-cooled Mac (M3 Ultra for K3-SG; an Apple10 Mac for K3-NA) | ≤ 10% decline first→last 5-min window, and absolute ≥ 0.8× idle P5 | G4 |
| P5 | Idle absolute (loadavg < 3, on power, Low Power off) | record; this is the reference for P4 | G4 |
| P6 | Find→accept latency on the proving host at the production (k, pattern, m, n), plus peak proving memory | record; choose k to maximize ops/s × e^(−latency/194 s) subject to degree_bits ≤ 19 | F3 |

Fanless Macs (MacBook Air) count for correctness and P1/P2 ratios only.
- **K3-SG P4/P5 run on the M3 Ultra.** A 4-minute flat run is measured; the 30-minute run is still to do.
- **K3-NA P4/P5 need a fan-cooled Apple10 Mac** (M5 Pro/Max/Ultra), which the operator is acquiring.

On the M5, take `/tmp/pmm-gpu-bench.lock` for GPU benchmarks. For long runs on any Mac, run on an otherwise idle machine and pause other GPU workloads.

## 9. Gates

**Feasibility gates (before the full build; a failure triggers §1.2 stop or redesign):**

| Gate | Pass condition | Status (2026-10-02) |
|---|---|---|
| F1 Full K3 prototype | K3-NA with BLAKE3, compare and slot atomics is bit-exact on the G3 vector subset, and meets P1 at both shapes | **FAIL at 8192²** (0.818, CI [0.792, 0.828]); **PASS at 4096²** (0.868, CI [0.833, 0.905]); bit-exact. Measured on a throttling fanless Air. **Decision (operator-confirmed 2026-10-02):** no §1.2 stop, because K3-SG passed P3 on the M3 Ultra. K3-NA does not mine in v1 until F1 passes on a fan-cooled Apple10 Mac |
| F2 Native job prep | pmkcore build of A/B, commitment and seeds keeps up: P2 ≥ 0.85 at the default shape, measured end to end | **PASS on the M5 Air** (0.99–1.09 at 8192², `kb/f2-result.md`). K3-SG P2 on the M3 Ultra is not yet measured (GPU job ≈ 33 ms vs build median ≈ 12.7 ms) |
| F3 Certificate generation | On the intended proving host, at k ∈ {2048, 4096, 8192, max}: measured proving latency, peak memory and degree_bits. Proving moves off the gateway event loop (or the latency is accepted with a bounded queue) | **PASS:** degree_bits 13–18 for k 2048–65536; 7.3–13 s for k ≤ 8192 (`kb/f3-result.md`). The off-event-loop change is still a build item |

**Release gates:**

| Gate | Pass condition |
|---|---|
| G1 Oracle | A Rust harness on zk-pow computes per-tile transcripts, hashes and finds for any (A, B, config). Every intermediate byte string is pinned: header bytes, config bytes, padded A/Bᵀ, raw roots, salted seed inputs, seeds, noise. Self-tested against `pearl_mining.mine()` + `verify_plain_proof` |
| G2 Noise | K1/K2 outputs are byte-identical to `pearl_noise.rs` on ≥ 20 jobs, including k ∈ {2048, 65536} and the min/max supported m, n |
| G3 Kernel exact | Transcripts, hashes and find classification are bit-identical to G1 on ≥ 50 jobs **for each kernel family** (K3-NA on Apple10; K3-SG on each Apple7–9 device class before it mines), k ∈ {2048, 4096, 8192, 65536}, full-range operands. Plus:<br>(a) **boundary vectors**: bound = hash−1, hash, hash+1;<br>(b) non-saturated deterministic wins and losses;<br>(c) negative jobs: wrong seed derivation (legacy), wrong header, wrong config, rank 64, out-of-range signal, illegal offsets, malformed Merkle proofs, all rejected by the verifier gate;<br>(d) slot-overflow and simultaneous-find tests |
| G4 Perf (per kernel family) | **K3-SG:** P3 met; P2 ≥ 0.85 measured on the M3 Ultra with K3-SG; P5 recorded; P4 (30 min) on the M3 Ultra. **K3-NA:** P1 and P2 met and P5 recorded on a fan-cooled Apple10 Mac; P4 deferred until one is available. v1 ships per family: a family that has not passed G4 does not mine |
| G5 Consensus | On **regtest** (certificate verification on; MoE, rank-penalty and salted-seed forks at height 1, `requiredcertversion = 3`), pmk at the production config mines ≥ 3 blocks accepted by pearld. Coinbase pays the configured script. A **deliberately corrupted certificate is rejected** by pearld. Simnet does not count (`BFNoPoWCheck`, `process.go:38-46`) |
| G5b Pool shares (v1 step 1.3; after G5 **and G7**) | pmk in pool mode (§7.6, object dialect, `docs/kb/pearl-pool-protocol.md`) at HeroMiners (`stratum+tcp://sg.pearl.herominers.com:1200`; operator's existing wallet, one worker per Mac; share nbits 0x1a07fff8). **T2 compatibility probe:** the first share is accepted (otherwise retry at Kryptex TLS :8048; if both reject the SG pattern, a pool-standard-pattern kernel is needed). **T3:** ≥ 20 submitted shares (≈ 6 h on the M3 Ultra at ~18 min/share; long-run window), ≥ 95% accepted, 0 gate failures, stale < 2%, share count inside the Poisson 99.9% interval, worker visible under the operator wallet. Every share first passes `verify_plain_proof_for_cert_version(3, pool_header, proof, nbits_override=share_nbits)`. Payout credit is recorded when it lands (1 PRL minimum ≈ 6 days at ~0.18 PRL/day) |
| G6 Mainnet health | 24 h against the operator's node: jobs track templates; 0 verifier-gate failures; 0 consensus-invalid outcomes; share rate inside the §7.5 Poisson interval; daily E[blocks] logged. A real block find is not required |
| G7 Review | Three-lane audit (code / security / architect) via `omc ask codex`: 0 CRITICAL / 0 HIGH / 0 MEDIUM |

**Fork scheduling never waives a gate.**

## 10. Risks and kill switches

| Risk | Trigger | Action |
|---|---|---|
| v4 fork scheduled | `Fp8ForkHeight` in a release or in `MainNetParams` (pearl-watch, `requiredcertversion`) | Per §1.2. R-A6 rejects v4 jobs automatically. No gate waivers |
| Layout/probe drift after an OS update | §5.4 fails | Automatic V2 or refusal; open a P1 |
| Consensus-invalid outcome | §7.2 classification | Stop that device, P0 investigation |
| Consensus rule change (rank, patterns, seeds) | pearl-watch release notes | Re-run G1–G5 before resuming |
| Proving latency / orphans | P6 above the budget | Lower k, move proving to a faster host, or accept with the budget documented |
| Thermal/power | P4 fails | Per-device duty cycle; exclude from headline numbers |

## 11. Related tracks (not in v1)
- **T-v4 (smarter kernel):** a separate, unproven pipeline. It needs its own commitments, noise, lottery, certificate and node-acceptance gates.
  - Measured arithmetic so far (`docs/kb/v4-emulation-result.md`): about 5 TOPS-eq on M5 with constant-magnitude operands that pass the policy; ≤ ~1 TOPS-eq with ordinary operands.
  - v1's oracle, probe, verifier gate and job lifecycle must be scheme-pluggable.
- **T-PIP:** the `INT8_EXACT` profile (later fork after v4).
  - Discussion is open: https://github.com/pearl-research-labs/pips/issues/14 (posted 2026-10-02).
  - The draft is in `docs/pip/`.
  - Every further post needs the operator's explicit OK.
- **T-fleet (separate spec):** a third-party fleet of Macs. Prerequisites:
  - documented owner consent and opt-in/revocation;
  - an agreement review (power and resource use);
  - an authenticated, encrypted, revocable miner↔pool protocol with quotas;
  - no node credentials on miner hosts;
  - coexistence with other workloads (§7.3);
  - payout accounting.
- **T-pool (Phase 2):** a pool built on Pearl's pool support (`zk-pow/bindings/go/src/plain.rs`). Before that, v1 mines on an existing public pool (G5b).
- **Product concept (operator):** combine inference serving and pmk mining on one Mac. It serves inference when requests arrive and mines when idle; later it mines while serving.
- **T-mlx-models (parallel research track):** make Pearl-format models (`pearl-ai/*-pearl` on Hugging Face) run on Macs in MLX, first for inference, then for mining-while-serving. Work items:
  - **Feasibility:**
    - Map Pearl's quantization scheme into MLX. Example: Qwen3-30B-A3B uses W7A7 int (per-channel weights, dynamic per-token activations) on attention and expert gate/up, and FP8 128×128 block weights with FP8 group-128 activations on expert down.
    - Determine what MLX supports natively today.
  - **Conversion:** a converter (reuse or extend OpenJarvis `scripts/pearl/model_converter.py`) from Pearl safetensors to MLX format, preserving the quantized values exactly where mining needs them.
  - **Inference quality:** compare perplexity and eval parity of the MLX port vs the reference (vLLM / dequantized bf16).
  - **Inference speed:** tokens/s on M5 and M3 Ultra vs current 4-bit models. Pearl's 7-bit weights roughly double bytes per token.
  - **Mining-while-serving:** a custom MLX linear layer covering int7 activation quantization (+ Hadamard / smooth scale), the noisy int8 GEMM with jackpot fold (the K3 kernel), and denoising so outputs stay correct.
  - **Fit with Swift serving stacks:** a Swift host such as `mlx-swift-lm` would need the same linear layer.

  This feeds Phase 4 (mining inside AI serving), and it is also a strong PIP argument: useful-work mining on Apple hardware.

  **Test host:** model work needs a large-memory Mac (M3 Ultra, 256 GB): downloads, conversion, inference/quality/speed tests, and end-to-end mining-while-serving tests. Why: the models are tens of GB. Run these on an otherwise idle machine with other GPU workloads paused, and take `/tmp/pmm-gpu-bench.lock` for benchmarks.

  Work that doesn't load the GPU (reading configs, writing code, converter unit tests on tiny tensors) may run elsewhere. Note that M3 Ultra has no Neural Accelerators, so the mining-while-serving path there uses the K3-SG kernel (v1).

## 12. Repository layout (v1)
```
pmkcore/        Rust: job builder (exists: F2, AES-CTR CSPRNG, byte-identical seeds/roots), commitment, proofs, G1 oracle
libpmk/         Swift package: Metal host, C ABI (pmk_init, pmk_probe, pmk_run_job, pmk_poll)
libpmk/metal/   K1, K2, K3-NA (from bench/f1_k3), K3-SG (from bench/k3sg), BLAKE3, probes (.metal)
miner/          Python orchestrator, gateway glue, verifier gate, submission tracking, monitors
bench/          benchmarks + evidence/
docs/           SPEC.md, kb/, adr/ (ADR-001 fixed-A policy, ADR-002 gateway patches)
```

## 13. Open questions
1. **Fan-cooled Apple10 access** for K3-NA P1/P4/P5 (M5 Pro/Max/Ultra). The operator is acquiring one; it is not blocking.
2. **Proving host** (decided 2026-10-02):
   - **v1 solo:** proving runs on the mining Mac itself. Finds are rare, and the node credentials stay on operator Macs.
   - **Pool (Phase 2):** a dedicated cloud host runs pool + pearld + proving. Shares are cheap to verify (`verify_plain_proof`); ZK proving happens only for block-qualified shares. Use neither a small 2-vCPU VPS (slower proving, more orphans) nor a home Mac (uptime, public endpoint).
   - **Sizing:** F3 measures proving cost per core so the pool host can be sized from it.
3. **Is Pearl testnet live on cert v3?** If so, add a testnet run to G5 as optional extra evidence.
4. **`MTLGPUFamily.apple10`:** confirm the exact SDK name, and the int8-tensor minimum OS per machine.
5. **Fleet chip mix:** unknown. Hardware profiles of a prospective fleet are not known.
6. **Pool protocol (G5b):** Pearl stratum as used by HeroMiners (job id format `<seq>_<n>`, e.g. `000001fd_2097152`; share difficulty in PH). Research the share message, the vardiff floor (the M3 Ultra at ~16.6 needs ~9 min per share at 9 PH unless the pool lowers difficulty), custom-miner policy, and whether pool "TH/s" equals our 2·m·n·k/s. Resolved: pool H = MAC = 2 ops, so the M3 Ultra ≈ 8.3 TH/s pool ≈ 0.18 PRL/day (~$0.21).
7. **Long-run host for G5b/G6 (decided 2026-10-02):** an M3 Ultra Mac Studio, in a one-off long run (e.g. overnight) with other GPU workloads paused for the whole run. Lock and confirmed-resume steps stay as in §7.3.
8. **K3-NA on Apple9?** int8 `matmul2d` ran at 18.1–19.6 TOPS on the M3 Ultra (`k3sg_m3ultra.txt`). The likely gain over K3-SG (0.85–0.90 of int8bench) is small.
9. **P6 at the SG config:** F3 used the NA pattern. SG has h+w = 12 vs 20, so degree_bits ≤ NA. Record it during G5.
10. **v4 proof format:** PR #355 changed the v4 public data from 212 to 244 bytes. This matters only for T-v4.
