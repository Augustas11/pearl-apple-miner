# Long-run window harness (offline bundle)

Build on the arm64 development Mac with Rust, Go, Xcode command-line tools, `uv`,
and the repository's Python 3.12 `.venv` available. All compilation occurs in
`dist/studio_b4/.build-work`; existing source directories are not changed.
No commits or remote actions are performed by these scripts.

```bash
cd <repo root>
scripts/studio_b4/build_bundle.sh
```

The builder compiles release libpmkcore, release libpmk, pearld and prlctl using
`scripts/build_pearld.sh`, and the cp312 abi3 `pearl_mining` extension using
maturin. It builds local wheels using a dedicated build venv containing maturin, setuptools,
wheel and uv-build installed from `build-requirements.lock` with pre-approved
hashes; local package paths are absolute. Runtime wheels are downloaded
with `--require-hashes` against the current `miner/requirements.lock`. Its torch
hashes are for the official PyTorch CPU index, not PyPI’s different same-version
wheel; a build-only direct-URL mapping selects that artifact without changing any
approved hashes or the shipped miner lock. The window
installs that unchanged lock offline first, then installs `local-wheels.lock`
(local Pearl packages and pytest, with pytest wheels resolved through
`test-requirements.lock`) with `--no-deps --require-hashes`. The root
`requirements.lock` inventories the entire wheel set with hashes. `binary-audit.json` records deployment targets and
linked libraries. The native audit includes every Mach-O member inside wheels. The dylibs,
G3 checker and Python extension target macOS 14.0; the Go node targets 26.0
to match its upstream Rust FFI configuration. Both are below the macOS 26.4.1 used for the recorded run. SwiftPM's resource lookup is adapted **in the build copy only**
to use `PMK_RESOURCE_BUNDLE`, set by the window script. The bundle includes both
an unmodified gateway source copy for patch tests and the prepatched runtime
copy, plus the Metal sources and probe test vectors.

Copy the payload (not build scratch or prior run state) to the target Mac, then run the window there. These commands
are instructions only; the builder/window never runs SSH or manages other machines:

```bash
rsync -a --delete \
  --exclude '/.build-work/' --exclude '/.venv-b4/' --exclude '/.b4_run/' \
  --exclude '/bench/' --exclude '/results/' --exclude '/build.log' --exclude '/b4_summary.json' --exclude '__pycache__/' \
  dist/studio_b4/ <target-host>:~/pmk-b4/
# then, on the target Mac:
cd ~/pmk-b4 && ./b4_window.sh
```

Run the window on an otherwise idle machine: pause other GPU workloads first (the window itself never pauses or
resumes anything).

### Optional lock-owner token and confirmed resume

If you pause other GPU workloads with your own wrapper, the window can take part in a fail-closed hand-off. An empty or
foreign `~/.lab-window.lock` always fails closed; the bundle never adopts a lock by reading out its token. To use the hand-off:

1. Have your wrapper create `~/.lab-window.lock/pmk-owner.json` with an unpredictable
   `token` and `pause_confirmed: true`, only after the other workloads are actually paused.
2. In the process invoking the window, export `B4_LAB_OWNER_TOKEN` and
   `B4_LAB_RESUME_HOOK` (absolute path to an executable that resumes the paused workloads). Optional
   `B4_LAB_RESUME_REPORT` defaults to `b4/.b4_run/lab-resume-report.json`.
3. Keep the token in `B4_LAB_OWNER_TOKEN`; do not put it on the command line. The hook and report may be
   supplied as `--resume-hook "$HOOK" --resume-report "$REPORT"`, and the window forwards only those paths to the
   miner. The hook must exit zero and print one JSON object with `confirmed:
   true` and `outcome: "resumed"` or `"fallback-restarted"` after confirming the paused workloads are back. A placeholder
   successful hook is not sufficient.
4. The controller defers the miner's resume until **all** GPU stages and owned
   children have finished. It calls `LabSession.finish()` once, including on failures, and fails the window if resume
   is unconfirmed. Your wrapper should keep its own independent failure/signal fallback and remove only its own marker/lock.

No pause/resume command is guessed or installed by this bundle. Without a token and hook, the window assumes the machine is already idle.

The window uses the interpreter in `PMK_PYTHON` (or `B4_STUDIO_PYTHON`; default: the one running the controller) and creates its own
offline venv. No cargo, Rust, package index, remote node, pool, or Internet is
used in the window. Package installation uses `--no-index`; runtime networking is restricted to fresh
localhost regtest/gateway ports. Preflight observes sysctl, memory, disk and (optionally, via `PMK_PREFLIGHT_PS_PATTERN`) RSS of other processes;
it does not change services or settings. The controller takes
`/tmp/pmm-gpu-bench.lock` for every GPU stage and propagates verified inherited
ownership to the miner. It never removes a foreign lock. The window deadline is at most
2400 seconds including installation and cleanup; 615 seconds are reserved for
child cleanup and B7’s at-most-600-second confirmed resume hook. Owned subprocesses are
terminated on failure, signal or deadline.

The window log is `~/pmk-b4-work/logs/b4-<UTC>.log`. Detailed service logs and P6 events
are in `<bundle>/.b4_run/`; the summary is `<bundle>/b4_summary.json`.
The summary distinguishes probe, fast pytest, G5, P6, P2 and B4 sustained-smoke
criteria. Exit codes: **0** all selected checks pass (quick mode reports `PASS_SELECTED`), **1** execution failure,
**2** criterion failure (probe failure says **DO NOT MINE**). A quick run is
correctness evidence only and never counts as production G5/P2/P6.

### Pool-mode window for G5b T2/T3

`pool_window.sh` is the pool-mode entrypoint. It is copied into the
bundle root beside `b4_window.sh` and is meant to be invoked through an outer
wrapper that owns pausing other GPU workloads and `/tmp/pmm-gpu-bench.lock`; the pool window verifies inherited
`PMK_GPU_LOCK_HELD=1`, requires the lock owner token via environment for real
runs, runs `pmk_miner --mode pool` through a no-resume shim, and writes
`pool_summary.json`. It never passes the owner token on argv and never calls
the resume hook itself.

T2 compatibility probe (`--max-accepted 1`) passes only after a pool-accepted
share. A time limit or transport/reply timeout without a pool verdict is
`INCONCLUSIVE` (exit 3); a rejection is `FAIL` (exit 2, configuration
incompatibility). Stale shares and safety/health failures also fail. `PASS` is
exit 0, and preflight/execution errors use exit 1. Both the top-level result and
pool criterion preserve `INCONCLUSIVE`. T3 requires its accepted-share target
as well as the ratio/health checks.

T2 runs until the first accepted share or a time limit:

```bash
PMK_GPU_LOCK_HELD=1 ./pool_window.sh \
  --pool-url stratum+tcp://<region>.pearl.herominers.com:1200 \
  --wallet-file /path/to/wallet.txt \
  --wallet-allowlist /path/to/operator-wallet-allowlist.txt \
  --worker pool-t2 \
  --max-accepted 1 \
  --max-submitted 1 \
  --max-seconds 7200
```

T3 runs in a long-run window until the target submitted/accepted
share budget or the overnight time limit:

```bash
PMK_GPU_LOCK_HELD=1 ./pool_window.sh \
  --pool-url stratum+tcp://<region>.pearl.herominers.com:1200 \
  --wallet-file /path/to/wallet.txt \
  --wallet-allowlist /path/to/operator-wallet-allowlist.txt \
  --worker pool-t3 \
  --max-accepted 20 \
  --max-submitted 20 \
  --max-seconds 28800
```

Only root shape keys (`m`, `n`, `k`, `slots`) are copied from `--config`; pool
policy such as difficulty floor remains owned by `pmk_miner` and the explicit
pool CLI inputs. Each invocation writes a unique `.pool_run/<run-id>/` state
directory so durable halt/count state is never reused across overnight attempts.

The summary records accepted, stale and rejected counts from the miner's
`pool_summary` event, plus the Poisson check and throughput fields
(`ops_per_second` and `pool_hashrate`). There is no K3-alone comparison in this
window, so the summary labels the P2 ratio as absent and applies no ratio gate.
It does not contact a pool during tests and does not take or remove locks.

For a local self-test on a development Mac, supply local Python and use `--quick`:

```bash
PYTHON="$PWD/.venv/bin/python" dist/studio_b4/b4_window.sh --quick
```

The recorded local validation is `bench/evidence/b4_bundle_selftest.txt`. The
quick path runs preflight, an offline venv install, the full G3 vector suite, the actual fail-closed K3-SG
probe, an 8192×8192×4096 two-slot startup/one-job smoke with a verified proof,
fast miner tests and a small 128×128×4096 certificate-verifying regtest.
The production smoke runs under the same GPU lock and fails on overflow or
truncated results. Run it separately with `.venv/bin/python scripts/pmk_production_smoke.py --slots 3` to exercise three-slot allocation.
Production measurements are taken separately on the target Mac.

P6 starts when the miner observes a completed GPU result and ends at the node
acceptance response. Proof digests connect these events; full header fields
match each accepted response to an on-chain block. Memory is a 250 ms sampled
peak of the entire gateway/prover process tree (KiB), including the gateway
parent; it is not an allocator high-water mark. The 10-minute check is a B4
smoke/Poisson check named `b4_sustained_smoke`, reports
`satisfies_spec_p4: false`, and is not the separate 30-minute P4 gate.

Admission is generated on the execution device, never baked into the bundle.
`bin/g3-admit` is compiled from the current PMK sources and actual G3 vector
suite, with fail-closed assertions and relocatable resources. It checks all four
vector sets, slot contents, counts, overflow subsets and guard canaries. Only
after success does the window write `.b4_run/g3-admission.json`, binding the G3
pass to the actual Metal name/family, probe cache key and OS build for six hours.
The subsequent production dylib initialization enforces that record on Apple7–9,
including M3 Ultra. A pass on one Mac does not certify another chip.

Each regtest gets a private, run-specific state directory for the B7 ledger and
monitor. P6 retains the gateway's submission identity and matches proof digests
and headers through node acceptance. The quick self-test does not claim M3 Ultra
production performance or sustained-health acceptance.
