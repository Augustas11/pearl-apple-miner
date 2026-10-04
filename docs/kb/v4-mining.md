<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cert-v4 mining path (B9)

The v4 path uses Pearl `fp8-scheme` at
`f696760b259500ecb608469ea3953aeabbe78948`. The separate cert-v3 vendor and C ABI
remain in place. See [the source delta](v4-delta-2569546-f696760.md) for the
108-byte ancestor-header change and proof serialization.

## Supported jobs

The initial production policy is dense B200, rank 32, E4M3, unit-scale clean
operands with independent random signs and magnitude 64. Supported contraction
dimensions are **1024, 4096, and 16384**. GPU matrix dimensions must be multiples
of 32 and no larger than 8192. The miner additionally checks the aggregate memory
budget against 25% of physical RAM. A valid upstream shape outside this initial
policy is rejected locally.

For a process that must run on both sides of the fork, use `k = 4096` and
matrix dimensions divisible by 64 and at most 8192; these satisfy both scheme
policies. A v4-only process can also use 1024 or 16384 and dimensions divisible
by 32. Templates select the scheme before scheme-specific shape validation.
An incompatible configured shape fails closed at a version transition.

The Air's no-restart pool test used this configuration:

```toml
m = 2048
n = 2048
k = 4096
slots = 2
```

The 8192² v4 benchmark exceeds the miner's 400 ms command prediction limit on
this Air. Set a suitable shape before the fork; admission does not override
the command-duration guard or resize a configured job.

Each job owns its v4 operands, commitments, upstream noise factors, and selected
ancestor. No v3 operand or commitment is reused as v4. The job builder uses
Pearl's transcript, Merkle codecs, quantization/policy, B200 emulator, and plain
proof verifier. Plain proofs are serialized by `PlainProofV4::to_bytes` before
base64 encoding for the existing `plain_proof` submission field.

Gateway jobs must include complete, parent-first `ancestor_headers`; pool v4
notifications must provide equivalent authenticated data. This repository has
not established that public pools implement that extension. Missing ancestry
rejects the job. No public pool connection is part of B9 verification.

## Build and admission

From the repository root:

```sh
scripts/fetch_vendor.sh --fp8
cargo build --release --manifest-path pmkcore/v4/Cargo.toml --lib --bins
swift build --package-path libpmk -c release
python3 scripts/pmk_v4_make_g3_vectors.py --threads 4
```

The reference generator calls Pearl's CPU B200 emulator for every output cell.
It records input, output, executable, and source hashes. It resumes completed
cases only when their provenance still matches. Small diagnostic campaigns
cannot earn admission.

The fused integration harness must also check GPU quantized codes, rectangular
shapes, all supported k values, lottery messages/hashes, accepted plain proofs,
and rejected corruptions. Its JSON result is supplied to the exactness gate:

```sh
python3 scripts/pmk_v4_integration.py
python3 scripts/pmk_v4_g3.py \
  bench/v4_emulation/vectors/b9_g3/manifest.json \
  --integration bench/v4_emulation/vectors/b9_integration.json \
  --admission bench/v4_emulation/vectors/b9_g3/admission.json
export PMK_V4_G3_ADMISSION_FILE="$PWD/bench/v4_emulation/vectors/b9_g3/admission.json"
```

The gate requires at least 100 million distinct constant-family output cells,
the required shapes/k coverage, zero mismatches, and adversarial inputs that
exercise the exact fallback. Admission is specific to the GPU, OS build,
loaded `libpmk.dylib` SHA256, compiled shader settings/source, and upstream pin;
it expires after seven days. Rebuilding the library requires fresh integration
and G3 evidence, even when only host code changed.
The native context validates it before v4 mining. An admission earned on the
M5 Air does not admit an M1–M4 device.

The normal `python -m pmk_miner` CLI also accepts a persistent admission path
in its TOML config, for both solo and pool mode:

```toml
[v4]
admission_file = "/absolute/path/to/admission.json"
```

Relative paths resolve against the config file's directory. An explicitly set
`PMK_V4_G3_ADMISSION_FILE` overrides this setting, including an empty or invalid
override (which fails closed). The CLI exposes the selected path to both Python
validation and the native Metal initializer for the duration of the run. It
does not discover, generate, renew, or bypass an admission record. Missing or
invalid admission on solo template polling is reported as `v4_admission`,
separately from malformed jobs.

GPU diagnostics and benchmarks take `/tmp/pmm-gpu-bench.lock`. When an outer
harness owns it, it passes `PMK_GPU_LOCK_HELD=1` to child commands. Do not nest
two lock acquisitions or remove a lock owned by another active process.

## Solo gateway

Build the pinned node and isolated v4 Python extension with
`scripts/pmk_build_pearld_v4.sh` and
`scripts/pmk_build_gateway_python_v4.sh`. Keep the v3 `.venv` intact.
The v4 gateway requires `miner/gateway_patches/0002-b9-v4-safe-async-proving.patch`
to export coinbase authorization data and honor pmk's submission receipt and
queue contract. The patch preserves upstream `PlainProofV4` decoding and
`ProofPool`; the raw upstream gateway does not echo pmk's submission IDs.

With the normal loopback node RPC and payout environment configured, launch a
patched temporary copy using the existing gateway launcher:

```sh
PYTHONPATH="$PWD/miner:$PWD/vendor/pearl-fp8/miner/miner-base/src:$PWD/vendor/pearl-fp8/miner/miner-utils/src" \
  bench/v4_emulation/pearl-build/gateway-python/bin/python \
  -m pmk_miner.gateway_launcher \
  --source vendor/pearl-fp8/miner/pearl-gateway \
  --patch miner/gateway_patches/0002-b9-v4-safe-async-proving.patch \
  -- start
```

The launcher patches its copy, leaving the pinned vendor source unchanged.
`scripts/pmk_v4_gateway_adapter_check.py` checks patch application separately.

The unmodified v4 E2E harness uses the normal miner entry point. Its generated
config does not specify admission, so supply the existing admission explicitly:

```sh
export PMK_V4_G3_ADMISSION_FILE="$PWD/bench/v4_emulation/vectors/b9_g3/admission.json"
bash scripts/pmk_regtest_e2e_v4.sh
```

The script takes the GPU lock itself. Do not pre-take it or set a miner-command
override. Omitting admission intentionally fails closed; a gateway template
cannot authorize hardware admission. See [B9 results](v4-b9-result.md) for the
recorded command and evidence, and [Rust tests](v4-rust-tests.md) for the command
covering both core crates.

## Dispatch and monitoring

`cert_version` selects the scheme for each job. Only versions 3 and 4 are
recognized; v4 additionally requires admission and ancestry. A version change
drains or drops work from the previous template before allocating the new
scheme's operands. A completed share from an old template/version is stale.
Persisted submission records carry the cert version and ancestry so recovery
does not reinterpret a v4 proof as v3.

Solo confirmation allows 30 minutes for a v4 submission, including the first
upstream circuit setup (measured at more than nine minutes on this Air).
V3 retains its two-minute confirmation deadline. A deadline does not permit
blind resubmission: unresolved ledger entries still fail closed.

K3-V4 uses portable shader FP32 `simdgroup_matrix` with four 8×8 fragments per
SIMD group. Each 32-K group must satisfy the grid and accumulator bound before
using the fast path. Other groups use the exact B200 fallback. The job result
reports fallback groups and total cell-groups; a rate above 1% at k ≤ 4096
raises an alert. A high rate reduces throughput without relaxing exactness.

Recorded B9 gate outcomes, counts, fallbacks, regtest results, and paired Air
throughput belong in `bench/evidence/b9_*.txt`. Historical prototype results are
not production admission evidence.
