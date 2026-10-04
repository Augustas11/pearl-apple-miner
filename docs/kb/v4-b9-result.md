<!-- SPDX-License-Identifier: Apache-2.0 -->

# B9 production FP8 result

Measured on the M5 Air (`Mac17,3`, Apple M5 / Apple10), macOS 26.5 build
`25F71`, against Pearl `f696760b259500ecb608469ea3953aeabbe78948`.
The original B9 gate counts below are historical. B9-fix verification and the
entry-point correction are recorded separately below.

## B9-fix root cause and entry-point correction

Running the unmodified `bash scripts/pmk_regtest_e2e_v4.sh` without admission
reproduced the operator-facing exit 1. The complete traceback is in
`bench/evidence/b9_fix_valueerror_traceback.txt`:
`mine → poll_templates → GatewayClient.get_job → GatewayJob.from_gateway_dict
→ GatewayJob.__post_init__ → validate_v4_g3_admission_file`, ending with
`ValueError: missing v4 G3 admission file`.

The gateway supplied a valid v4 template with complete ancestry. This was a
missing local admission setting, not a v4 wire-format or dispatch defect. The
CLI incorrectly classified this configuration failure as `malformed_job` and
logged only `ValueError`, hiding the actionable cause. B9-fix adds a persistent
`[v4] admission_file` setting to the real CLI, with config-relative paths and
explicit environment precedence. Both Python and Metal use the same selected
path. Admission failures now identify the `v4_admission` gate and both ways to
configure the record. All existing hardware, library hash, freshness, ancestry,
coinbase, and proof checks remain active.

The old `b9_v4_regtest_e2e_summary.txt` used the special entry script and did
not prove the normal entry point. That evidence is preserved as
`b9_fix_baseline_v4_regtest_e2e_summary.txt`. The historical final report omitted
the required admission environment, so its entry-point claim was not a
reproducible bare-command claim. The special entry script was already absent
when B9-fix began; no alias or alternate mining logic has been added.

The captured gateway response is retained in
`miner/tests/fixtures/v4_regtest_gateway_job.json`. The real CLI/GPU fixture
regression failed before the fix (`b9_fix_fixture_before.txt`) and passed after
(`b9_fix_fixture_after.txt`). It tests admission, coinbase authorization, v4 GPU
dispatch, and graceful signal handling. Full node acceptance is a separate E2E
gate. Missing admission remains a deliberate fail-closed error.

## Consensus and CPU gates

- The [source delta](v4-delta-2569546-f696760.md) preserves the constant-magnitude
  grid argument. PR #355 requires the complete 108-byte selected ancestor and
  parent-first intermediate ancestry; public parameters are now 244 bytes.
- Current-pin emulation oracle: **98 passed**, zero failures or ignored tests.
- Upstream Rust library suite: **436 passed**, zero failures, six pre-existing
  ignored fixture/heavy tests (442 listed tests covered by complementary runs).
- Existing v3 pmkcore: **16 passed**. New v4 core integration: **6 passed**.
  Every v4 Rust test target also compiled; doc tests passed (zero examples).
  The 436 upstream test executions above cover the vendored code path-included
  by v4; the full duplicate debug-mode upstream suite was not rerun.
- Actual certificate construction through `pmkcore_v4_build_certificate`:
  244-byte public data, 59,554-byte proof, accepted by Pearl's v4 verifier.
  Mutations to public data and proof bytes were rejected. Construction took
  553.372 seconds including setup; honest verification took 44.060 seconds.

Evidence: `bench/evidence/b9_v0_*.txt`, `b9_v3_cargo.txt`,
`b9_v1_cert_tests.txt`, `b9_v1_cargo_compile.txt`, `b9_v1_cargo_docs.txt`,
and `b9_v1_certificate_roundtrip.txt`.

## G1 and G3-v4

The startup probe checks fused quantization, every C cell, lottery messages and
hashes, counters, and overlapping block/share predicates. Its 16 tiles produce
8 block slots and 16 share slots. Both streams use independent predicates, as
v3 does. Context initialization and periodic refresh run this probe.

The complete G3 comparison passed **105,185,280 cells with zero mismatches**:

| Coverage | Cells | Result |
| --- | ---: | --- |
| Constant-magnitude grid family | 105,054,208 | Exact |
| Two adversarial families | 131,072 | Exact; 100% fallback |

The constant cases include 256², 2048² and 4096² at K=4096, plus 256² at
K=1024 and K=16384. Six independently seeded 4096² cases supply the bulk of
the count; duplicate inputs cannot inflate admission.

Fused GPU/CPU integration also passed **4,194,304 quantized values**, **52
unique lottery tiles**, **7 accepted plain proofs**, and **28 rejected corrupt
proofs**, with zero mismatches. It covers both rectangular orientations, all
three supported K values, and equal block/share bounds.

| Constant-family K | Fallback rate |
| --- | ---: |
| 1024 | 0% |
| 4096 | 0.01990–0.02080% |
| 16384 | 3.61233% |

The adversarial cases also verify the >1% alert at K≤4096. These are cell-group
rates, not the fraction of whole jobs retried. Admission binds the GPU, OS,
shader/probe inputs, upstream pin, and exact loaded host binary SHA256.

Evidence: `bench/evidence/b9_g3_v4.txt`, `b9_v2_startup_probe.txt`, and
`b9_v2_integration.txt`. Admission is in
`bench/v4_emulation/vectors/b9_g3/admission.json`.

## Version switch and pool wire

A loopback test used one Native/Pipeline instance at 2048²×4096. It produced
and locally verified a real v3 GPU proof, switched to a v4 notify, classified
that late v3 proof as **stale**, and submitted **4 accepted v4 shares** from a
real v4 GPU job. An invalid v4 proof was rejected. The isolated upstream Python
extension also decoded and re-encoded the fixture's PlainProofV4 bytes exactly.

Evidence: `bench/evidence/b9_v3_v4_switch.txt`, `b9_v5_pool_mock_e2e.txt`, and
`b9_v4_python_plainproof.txt`. No public pool was contacted.

## Node regtest (B9-fix)

From the repository root, with no miner-command override and no pre-held GPU
lock:

```sh
export PMK_V4_G3_ADMISSION_FILE="$PWD/bench/v4_emulation/vectors/b9_g3/admission.json"
bash scripts/pmk_regtest_e2e_v4.sh
```

**Exit 0.** The pinned node accepted **3 v4 blocks** through the permanent
gateway adapter and normal `.venv/bin/python -m pmk_miner --mode solo` entry
point. The harness checked coinbase payouts at heights **1, 2, and 3** and the
node explicitly rejected both an altered proof and altered public data. The
miner log records three payout verifications. The owned run directory was
removed. Both the shell wrapper and its Python harness are byte-identical to
the files at the start of B9-fix (as are the v3 harness files).

This proves the unchanged command with the required admission environment.
It does not claim that an unconfigured bare-shell invocation can mine v4;
omitting admission still fails closed, now with an actionable diagnostic.

The adapter adds the existing pmk receipt/queue/coinbase contract to the v4
gateway while preserving upstream `PlainProofV4` and `ProofPool`. Solo v4
confirmation allows 30 minutes for cold circuit setup; v3 retains two minutes.

Fresh evidence: `bench/evidence/b9_fix_v4_verify.txt`, `b9_fix_v4_summary.json`,
`b9_fix_v4_miner_redacted.txt`, `b9_fix_v4_tap_redacted.txt`, and
`b9_fix_harness_sha256_{before,after}.json`. The formerly stale
`b9_v4_regtest_e2e_summary.txt` now mirrors this completed normal-entry run.
The prior special-entry summary is retained under the baseline filename above.

## Regression gates (B9-fix)

Commands below run from the repository root. The v3 evidence-prefix setting
only keeps generated evidence inside the authorized `bench/evidence/b9*` scope.

- `.venv/bin/python -m pytest -q miner/tests`: **exit 0, 302 passed**, zero failed
  or skipped, in **432.64 seconds**. This includes the captured-fixture real solo
  tests for config admission, environment admission, and fail-closed missing
  admission, plus seven config/precedence/restoration cases.
- `PMK_EVIDENCE_PREFIX=b9_fix_v3 bash scripts/pmk_regtest_e2e.sh`: **exit 0,
  4 accepted blocks**, coinbase checked at heights **1–4**, corrupted certificate
  rejected, and graceful miner shutdown verified. The script and Python harness
  were not modified.
- `bash libpmk/tests/run.sh`: **exit 0, 20 Swift tests**, **94 v3 G3 cases**, and
  **50 full-slot jobs** passed, including boundary, overflow, callback, and
  GPU-to-proof checks.
- The [exact combined Rust command](v4-rust-tests.md) exited **0**: **16 v3 +
  436 upstream v4 + 6 v4 integration = 458 passed**, zero failures or filtered
  tests. Six pre-existing upstream ignores remain; no skips or filters were
  added. Both packages' binary/doc test harnesses passed with zero cases defined.
- Python compilation checks passed for both changed production modules and both
  added test modules. No Ruff, mypy, or Pyright executable is installed in the
  project environment; no dependency was added.

Fresh evidence: `bench/evidence/b9_fix_miner_pytest.txt`,
`b9_fix_v3_regtest_e2e_summary.json`, `b9_fix_libpmk_tests.txt`,
`b9_fix_rust_v3_v4_release.txt`, and `b9_fix_rust_summary.txt`. Earlier B9 counts
(292 miner tests and 3 v3 blocks) are historical and superseded by this rerun.

Final cleanup and identity checks are recorded in
`bench/evidence/b9_fix_final_validation.json`: all owned regtest run roots were
removed, no processes reference those roots, the GPU lock is absent, the special
entry is absent, and the loaded library still matches admission. No commits,
SSH, or real pool connections were made.

## Paired Air throughput

At 8192²×4096, the same GPU lock window contained three warmups and ten measured
iterations per path, interleaved v3/v4:

| Path | Median TOPS-eq | Median GPU time |
| --- | ---: | ---: |
| V3 K3-SG command buffer | 1.13991 | 482.322 ms |
| V4 Kernel E matmul | 0.48788 | 1126.953 ms |

**V4/V3 ratio: 0.42800×.** V4 fallback was 1,768,030 of 8,589,934,592
cell-groups per measured job: **0.0205826%**.

V4 timing excludes quantization, decode/transpose staging, and lottery readout;
v3 uses the existing `pmk_run_job` command-buffer timing, including its noise
and lottery stages. This is a kernel-path comparison, not end-to-end mining
throughput. Raw samples and binary identity are recorded in
`bench/evidence/b9_v5_paired_bench.txt`.

## Operating limits

- The measured 8192² v4 command exceeds the miner's 400 ms prediction budget.
  Use a smaller configured shape for mining; 2048²×4096 was exercised through
  the real version-switch/pool path. The 8192² measurement is a benchmark.
- M1–M4 hardware admission is pending. Portable Metal code and the admission
  gate are present, but this M5 result does not admit another device.
- Public pool support for complete v4 ancestry remains unverified. Missing
  ancestry or a missing/mismatched admission record fails closed.
- The initial v4 production policy supports B200 rank 32, dimensions divisible
  by 32 up to 8192, and K in {1024,4096,16384}, within memory limits.
- Regtest activates v4 at height 1; this upstream pin has no activation-height
  CLI override. The actual cross-version test uses the local pool harness.
