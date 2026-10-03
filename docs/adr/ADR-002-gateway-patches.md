# ADR-002: Runtime Patch pearl-gateway for B3 Solo Mining

## Status

Accepted for B3; hardened by B7-fix.

## Context

The upstream `pearl-gateway` is vendored under `vendor/` and must remain read-only. B3 still needs two safety changes before a local miner can use the gateway in solo mode:

- The gateway logs the Pearl RPC password and full payout address during startup.
- `submitPlainProof` acknowledges quickly, but the scheduled task performs ZK proof generation in the asyncio process, which can make the admission server unresponsive while proving.

The miner also needs gateway logs that can be correlated with a mining job header so it can classify submission outcomes after the JSON-RPC method has acknowledged admission.

## Decision

`pmk_miner.gateway_launcher` copies `vendor/pearl/miner/pearl-gateway` to a temporary runtime directory, applies `miner/gateway_patches/0001-b3-safe-async-proving.patch` to that copy with `patch -p1` from inside the copy, and starts the copied source with its `src/` path first in `PYTHONPATH`. The vendor source is never edited.

The patch:

- Fully redacts RPC users, RPC passwords, and mining addresses before logging.
- Removes request-parameter trace logging and sanitizes RPC URL/error/result log paths that could carry credentials.
- Refuses non-loopback Miner RPC TCP hosts and symlink UDS socket paths; the socket remains mode `0600`.
- Adds a bounded `PMK_GATEWAY_PROVING_QUEUE_SIZE` admission semaphore in `MinerRpcServer`.
- Moves block/ZK proof generation into a `ProcessPoolExecutor` controlled by `PMK_GATEWAY_PROVING_WORKERS`.
- Requires a caller-persisted `mining_job.submission_id` and echoes it in `{"status":"submitted","submission_id":"..."}` before validation; over-capacity submissions return JSON-RPC error `-32010`. Admission precedes proof/job decoding.
- Exposes serialized coinbase and its index-zero Merkle branch in `getMiningInfo` so miners authorize payout before dispatch.
- Emits JSON log payloads with `event`, `submission_id`, `proof_hash`, header identity, template metadata, and distinct outcome classes.
- Emits `block_submission_error` with `stage`/`phase` (`proving` or `node`), `classification`, and `error_type`; the error log omits raw exception text and tracebacks.
- Supports the existing regtest tap by launching a tap script with the patched gateway source first in `PYTHONPATH`.

## Consequences

The event loop remains responsible for socket admission and Pearl node RPC I/O. CPU-heavy proving runs in another process so PyO3 or native proving code cannot pin the event loop. The bounded queue makes overload visible at the miner boundary instead of accumulating unbounded background proving work.

The launcher interface is:

```bash
python -m pmk_miner.gateway_launcher [--source PATH] [--patch PATH] [--copy-parent PATH] [--tap-script PATH] [--] start --debug
```

When used as a library, `patch_gateway_copy(...)` returns the patched copy paths, `build_gateway_command(...)` returns the subprocess command/env, and `launch_patched_gateway(...)` starts the process.

Structured outcome logs are JSON strings carried by the existing gateway logger. The primary correlation fields are:

```json
{"event":"block_submission_error","submission_id":"...","header_hash":"...","stage":"node","phase":"node","classification":"transport","error_type":"ValueError"}
```

## Evidence

`miner/tests/test_gateway_patch.py` applies the patch to a copy of the vendor gateway and checks that credential logs are removed, proving is routed through a bounded process worker, admission returns before background processing, structured outcome fields are present, and tap launch places the patched source before other `PYTHONPATH` entries.

Runtime coverage applies the patch in a fresh external temporary directory, verifies the copied source is actually patched there, exercises bounded queue-full admission and permit release after success and background exceptions, captures client logs to ensure raw credentials are absent, checks loopback/UDS guards, verifies real `PlainProof`/header primitive serialization round trips, and uses a test-only picklable worker helper to prove a process-worker delay does not block an asyncio heartbeat.

Generic node exceptions are transport errors. Only explicit node rejection results
are consensus-invalid. See ADR-003 for durable correlation, restart and unknown
submission handling; a gateway acknowledgement is not chain acceptance.
