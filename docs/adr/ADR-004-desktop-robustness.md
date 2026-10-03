# ADR-004: Desktop-safe mining controls and offline benchmarking

Status: accepted for B11 implementation.

The design borrows only product ideas identified in the Breakwater comparison:
prevent App Nap, respond conservatively to battery state, bound pool silence,
offer a duty-cycle control, and report an offline benchmark. No Breakwater code
is copied.

## Activity and power boundary

`libpmk` owns the macOS APIs behind a small C ABI. `pmk_activity_begin(reason)`
returns a retained opaque handle and `pmk_activity_end(handle)` releases it;
callers end each non-null handle exactly once, while ending null is a no-op. The
activity uses `ProcessInfo` options `userInitiated` and
`idleSystemSleepDisabled`. It deliberately omits `latencyCritical` and
`idleDisplaySleepDisabled`: mining may prevent idle system sleep, but it must
not keep the display awake or request latency-sensitive scheduling.

`pmk_power_source()` reports stable integer values: unknown `0`, AC `1`,
battery `2`, and desktop `3`. IOKit identifies the providing source; external
AC or UPS power with an internal battery is `AC`, while external power with no
internal battery is `desktop`.
Unknown and unsupported sources fail conservatively to `unknown`.

The default `--on-battery pause` policy permits AC and desktop operation and
pauses battery or unknown sources. A power transition gates the next GPU
dispatch. It does not cancel a submitted command buffer: in-flight work drains,
is accounted and released normally, then subsequent dispatch waits until an
allowed source returns. `--on-battery run` is the explicit opt-in to continue.
Desktops are therefore unaffected by the default laptop safety policy.

## Intensity and telemetry

Intensity is coordinated across all pipeline slots. Below 100%, one shared gate
serializes GPU bursts so another slot cannot fill the intended idle period.
After a completed burst of actual Metal duration `t`, the next dispatch waits
`t * (100 / intensity - 1)` from the Metal completion timestamp, counting any
host-observation delay toward the idle gap. CPU preparation and proof handling may continue
during that interval. At 100%, the extra serialization and delay are disabled.

Requested intensity is policy, not a utilization measurement. Routine telemetry
reports actual GPU busy time as the union of completed Metal timestamp
intervals, avoiding double counting if observations overlap, divided by actual
elapsed wall time. Power source, paused state, requested intensity, busy seconds
and busy percentage travel together so operators can interpret achieved duty
cycle rather than infer it from the flag.

## Pool liveness

Pool TCP sockets enable kernel keepalive when the platform exposes it. Keepalive
helps detect a dead peer but cannot prove application progress, so a separate
silence watchdog measures time since the last valid inbound pool message. A
silent connection becomes a transport failure and follows the existing
reconnect path instead of continuing indefinitely on stale work. Optional
platform keepalive settings are best effort; the application deadline remains
authoritative.

The public flag contracts are bounded and reject non-finite values:

- `--pool-silence-timeout`: 30–1800 seconds, default 180.
- `--on-battery`: `pause` or `run`, default `pause`.
- `--intensity`: integer 10–100 percent, default 100.
- `--benchmark [SECONDS]`: 10–600 seconds, default 60 when present.
- `--difficulty`: 1–2^64, default 2^21.

## Offline benchmark

`--benchmark` runs without a node, gateway or pool and rejects mining connection
and configuration flags. It uses the real production pipeline at
8192×8192×4096 with two slots, an exact synthetic target that should produce no
candidates, and the same desktop controls as mining. It reports completed work,
MAC/s, TOPS, jobs/s, actual GPU busy percentage, power source and an expected
share interval for the selected difficulty.

A full device G3 admission runs before the benchmark timer starts. Admission
time is excluded from the timed window, but admission is still mandatory; a
cached probe-only result is insufficient. The benchmark never submits work and
does not mine or earn anything. Its expected-share interval is a throughput
projection, not an earnings claim.

## Validation boundary

Targeted tests cover ABI lifecycle and enum mapping, power-transition draining,
coordinated slot throttling, measured interval merging, flag bounds, pool
silence/keepalive behavior, and benchmark accounting. Final build, full-suite,
G3 and hardware evidence are recorded separately by the B11 validation run.
