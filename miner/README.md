# pmk_miner (B3)

Python 3.12 solo orchestration for the existing pmkcore and libpmk C ABIs.
Run from the repository with its `.venv` and `PYTHONPATH=miner`; no CUDA, Torch,
Python matrix generation, or Python matrix hashing is used.

```sh
uv pip install --python .venv/bin/python --require-hashes -r miner/requirements.lock
PYTHONPATH=miner .venv/bin/python -m pmk_miner.gateway_launcher -- start --debug
PYTHONPATH=miner .venv/bin/python -m pmk_miner \
  --mode solo --gateway 127.0.0.1:8337 --config miner/local.toml
```

The gateway launcher inherits `PEARLD_*` and `MINER_RPC_*` settings. It copies
vendor source and applies `gateway_patches/0001-b3-safe-async-proving.patch` to
that copy. The source checkout remains untouched. Gateway proving runs in a
bounded process worker, with bounded admission and correlated outcome logs.
Only loopback TCP or a protected Unix socket is supported. Node confirmation
uses only local read-only RPC. Pool mode is B6; `transport.JobSource` is its
future job/submission boundary.

`miner/requirements.lock` pins the third-party macOS arm64 Python 3.12 runtime
dependencies with hashes for `uv pip install --require-hashes`. Local Pearl
packages (`py-pearl-mining`, `miner-utils`, `miner-base`, `pearl-gateway`) still
come from `vendor/pearl` at the audited revision and are installed separately by
the setup flow; the lock covers the PyPI wheel set those local packages require. The reviewed
constraints are in `miner/requirements.in`; regenerate with the command recorded
at the top of the lock file. Install local packages with `--no-deps` after the
hash-checked dependency install so their broad dependency ranges do not resolve
a different environment.

Minimal config (substitute an operator-approved address/script and private env
file; do not commit secrets):

```toml
m = 8192
n = 8192
k = 4096
slots = 2

[gateway]
env_file = "/absolute/path/to/private/gateway.env"
log_file = "/absolute/path/to/gateway.log"

[payout]
hrp = "rprl"
script = "5120<approved 32-byte taproot program>"

[run]
template_poll_seconds = 0.25
state_dir = "/absolute/path/to/private/pmk-state"
```

The env file contains `PEARLD_RPC_URL`, `PEARLD_RPC_USER`,
`PEARLD_RPC_PASSWORD`, `PEARLD_MINING_ADDRESS` and must match the gateway's
configuration. Startup checks the approved script and address checksum/HRP;
each accepted block's actual coinbase is checked again. The miner logs explicit
JSON metadata rather than configuration, proof payloads, credentials or wallets.
SIGINT/SIGTERM stop new dispatches and drain owned native work before release.
Run state defaults to `~/.local/state/pmk`; regtest and window wrappers must set
`[run].state_dir` to a private temp or run-specific directory so pending
submissions, ledgers and monitor counts do not mix with real runs.

Long runs can be owned by an optional outer wrapper that pauses other GPU workloads. When such a wrapper has
created `~/.lab-window.lock` (or a directory containing `pmk-owner.json`), invoke
pmk with `--lab-owner-token`, `--resume-hook` and `--resume-report`. The lock
JSON must have a matching `token` and `pause_confirmed: true`; otherwise pmk
refuses to start. The resume hook must print JSON with `outcome` equal to
`resumed` or `fallback-restarted` and `confirmed: true`; pmk writes the resume
report and fails the run if the resume is unconfirmed.

A is generated and committed once per immutable template. B is generated and
committed for each job by pmkcore directly into `pmk_buffer_alloc` storage.
There are 2–3 rotating B slots and one retained A. A slot follows
`building -> gpu -> scanned -> proving -> submitted -> released`, omitting the
proof states for empty/abandoned jobs. Only completion callbacks wake the
pipeline. A release owner retains the native job until all finds are submitted
or abandoned. Cert != 3 cancels eligibility and stops; Metal has no cancellation
ABI, so already-dispatched work drains without submission. Native probe, verifier
gate, rank/bound, or consensus rejection failures fail closed.

Share targets are diagnostic and are never submitted to the solo gateway.
Their verifier uses `nbits_override` without changing header bytes. The Poisson
monitor uses actual completed operations and that job's compact share target.
`submitted` acknowledges admission only: a correlated rejection is classified,
and acceptance requires matching header fields on the node's chain within
120 seconds. Stale, duplicate, transport and proving outcomes are separate from
consensus-invalid P0 failures.

The default is SG 8192²×4096, rank 128, production patterns. Shapes and a
conservative combined native/GPU/Merkle budget are validated before allocation;
every buffer is below 2 GB and the total estimate is below 25% of RAM. The
measured GPU rate refuses further dispatch when the selected shape predicts
more than 400 ms. On a throttled Air, reduce m/n to 4096 if this guard fires.
The full-size G5/G4 and G6 health gates on the M3 Ultra remain separate tasks.

## Verification

```sh
PYTHONPATH=miner .venv/bin/python -m pytest -q miner/tests
scripts/pmk_regtest_e2e.sh 3 600
PYTHONPATH=miner .venv/bin/python scripts/pmk_native_check.py --jobs 2
PYTHONPATH=miner .venv/bin/python scripts/pmk_native_check.py --m 4096 --n 4096 --jobs 20
```

The pytest integration launches a fresh local certificate-verifying regtest
node, patched gateway and real K3-SG miner, mines at least three blocks, checks
their coinbase outputs, and injects corrupted ZK proof/public-data bytes. It
uses 128²×4096 for bounded regtest find counts, with the production SG pattern
and rank. Every GPU path takes `/tmp/pmm-gpu-bench.lock`, retries every 15 seconds,
and releases only a lock it acquired. Runtime regtest files are private under a
secure temporary directory and are removed in `finally`; `--keep-run` preserves a
sanitized copy for debugging. Evidence is in `bench/evidence/b3_*.txt`.

Per-job overhead reports active Python orchestration CPU divided by job wall
time, excluding native C-ABI work and native proof verification. This is an
orchestration measurement, not the P2 completed-work throughput ratio. Tiny
regtest shapes and easy-target proof bursts are not production performance
measurements. No mainnet, pool, M3 Ultra performance, or 24-hour-health claim is
made here.

## B3 evidence on the M5 Air

`bench/evidence/b3_pytest.txt`: **74 passed**, one upstream aiohttp deprecation
warning. The real regtest test confirmed **4 accepted blocks**, approved
coinbases, both corrupted-certificate rejections, and a clean CLI exit after its
configured on-chain confirmation threshold (`b3_regtest_e2e.txt`).

Final ordinary-job orchestration measurements (no finds):

| Shape | Jobs | Aggregate Python overhead | Per-job range |
| --- | ---: | ---: | ---: |
| 8192²×4096 | 2 | 0.070% | 0.053–0.086% |
| 4096²×4096 | 20 | 0.165% | 0.127–0.242% |

These include active job-coroutine slices, native-worker Python glue, callback
notification, scan, lifecycle and release; C-ABI/native verifier CPU is excluded.
Per-job build/dispatch/GPU-wait/scan/proof/submit stage wall times are logged.
The outer RPC/health loop is not included in this microbenchmark. The very small
128² regtest shape, with frequent proof construction, can exceed 2% overhead and
is correctness evidence only. A longer 8192² run hit the 400 ms safety guard on
this Air (`b3_dispatch_guard.txt`); the guard remained enabled. These measurements
are not a P2 throughput pass or a full-size G5 M3 Ultra pass.

The F2 path commits A once per template. On rare finds/overflow, the current
pmkcore oracle ABI reconstructs native Merkle/noise state from retained raw
inputs; it does not expose a reusable A-tree handle. That extra native cold-path
work is included in proof stage timing.

## B6 object-dialect pool mode

Pool mode requires the pmk py-pearl-mining patch that exposes
`extract_difficulty_bound(nbits, cfg)` and `nbits_to_difficulty(nbits)` from the
same Rust verifier helpers used by `nbits_override`. Build it from an isolated
Pearl copy and install it into the local venv with an interpreter that already
has the pinned build requirements installed:

```sh
uv pip install --python .venv/bin/python --require-hashes -r scripts/studio_b4/build-requirements.lock
scripts/pmk_build_pool_binding.sh --install
```

The bundle build must apply
`miner/native_patches/0001-expose-pool-bound-helpers.patch` to its copied
`vendor/pearl` tree before running `maturin`; the helper only accepts isolated
Pearl roots under `dist/`, such as
`--pearl-root dist/studio_b4/.build-work/vendor/pearl`.

```sh
PYTHONPATH=miner .venv/bin/python -m pmk_miner \
  --mode pool --pool-url stratum+ssl://prl.kryptex.network:8048 \
  --wallet-file /private/operator/wallet.txt \
  --wallet-allowlist /private/operator/pool-wallets.txt \
  --worker studio --config /private/operator/pool.toml
```

The wallet file holds one wallet. The separately managed allowlist has one
approved wallet per line (blank lines and `#` comments are ignored). Wallets
are masked in logs. Pool mode does not read gateway env files or node RPC
credentials. Use a dedicated state directory, without the solo `[gateway]`
or `[payout]` sections:

```toml
m = 4096
n = 4096
k = 4096
slots = 2
[pool]
difficulty_floor = 10000
[run]
state_dir = "/private/operator/pmk-pool-state"
max_submitted = 20
max_seconds = 43200
```

`stratum+tcp`, `stratum+ssl`, `stratum+tls`, and `tls` URLs require an explicit
port. TLS uses system certificate validation. Only object-dialect authorize,
notify, and full bincode PlainProof submission are supported. There is no gzip,
compact submission, pool-supplied configuration, or AlphaPool challenge dialect.
The inbound frame cap is 64 KiB; outbound proof messages can be larger.

A notify's header is immutable and determines A; B is fresh for every attempt.
Changing the pool job ID or target under the same header updates the immutable
job metadata without regenerating A. Old jobs may submit under their original
ID within the same connection session; reconnect invalidates every old job.
Every submitted proof, including a block candidate, passes the cert-v3 local
gate with the rounded-down share nbits. The ledger records the originating
session, pool job ID, target, nbits and config and never changes a terminal
outcome. Unacknowledged pre-crash submissions become transport outcomes and
are not retransmitted.

Invalid/low-difficulty rejection before any acceptance stops with
`pool rejected SG config; try next pool`; the same verdict after an acceptance is P0.
Stale outcomes above 5% stop the run. A durable safety halt requires operator
investigation before a new run. A time limit may finish below the requested
share count; check the summary, not just process exit. `pool_summary` includes
accepted/stale/rejected counts, all completed GPU work, observed finds, the
99.9% Poisson interval, and wall-clock completed-work throughput. Pool H/s is
MAC/s (half ops/s). The optional remote dashboard fetch is not enabled;
its unverified API remains an operator check for T2/T3.

Tests use loopback mock pools only. Real SG policy acceptance, dashboard worker
visibility, and payout credit remain T2/T3 checks performed by the operator in
a long-run window, after G5 and G7.

### Pool gate diagnostics and bounded local dispatch checks

`FatalDeviceError` and P0 device/verifier alerts include `gate`,
`native_function`, `native_code`, and a redacted `error_message`. Native ABI
failures retain their actual function and return code. Python device and
verifier policy failures use reserved codes `-2001` and `-2002`; these are
Python gate codes, not Metal return codes. Bound-helper failures identify
`Scheme.nbits_bound`; they can happen before any GPU work is dispatched.

A pool run can set `[run] max_jobs = 2` to finish after two completed GPU jobs.
The limit reserves work across slots and drains proof/submission handling
before stopping. This is useful for production-shape correctness checks on a
slower test GPU; the production command-buffer timing guard remains enabled.
The loopback harness supports `--block-nbits 0x177fd82e --difficulty 2097152
--dispatch-only --target-completed-jobs 2 --m 8192 --n 8192 --k 4096 --slots 2`.
Use the separate easier-difficulty mainnet-bits test to exercise real proof
finding and verification without waiting for a production-difficulty share.

The pool window checks the installed local wheels against shipped,
hash-locked wheel contents and checks native bound-helper semantics before
connecting. It repairs stale same-version packages using the offline wheels;
a successfully imported older `pearl_mining` is insufficient for pool mining.
