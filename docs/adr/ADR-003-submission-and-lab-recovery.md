# ADR-003: Preserve submission finality across process restarts

A proof is assigned a submission ID and durably recorded before network send.
The JSONL ledger binds that ID to the immutable template and proof hash. A lost
acknowledgement can mean that the gateway admitted the proof; it therefore keeps
the reservation and resumes confirmation instead of retransmitting. Startup
resolves outstanding admissions before dispatching new work. Explicit node
verdicts, proving errors, transport errors, stale work and duplicates are distinct.
An unresolved submission is `unknown-submission`, stops mining, and persists a
safety halt. Consensus-invalid also persists a halt. Investigate either before
archiving/resetting the run state; restarting the process is not an override.

Acceptance is confirmed against the chain, followed by the existing payout check.
Every new template must also supply a coinbase and Merkle branch that commit to
the approved script before work starts. One verified same-template backup is
retained, with the total bounded by the slot count, and used only after an
explicit recoverable proving/transport result. Repeated proofs are never resent.

`run.state_dir` defaults to `~/.local/state/pmk`; use a separate directory for each
independent node/network deployment. Regtest uses an isolated temporary directory.
The run record retains completed operations, elapsed mining time, diagnostic
share counters, retry deadlines and the daily expectation checkpoint. Diagnostic
targets are recomputed within a template; monitoring accounts for each job's
actual target.

## GPU and external lab ownership

The miner interoperates with `/tmp/pmm-gpu-bench.lock` directory users. PMK writes
its owner PID and serializes ownership/recovery using an OS file lock. It only
reclaims a directory bearing its own marker and a demonstrably dead PID. Unknown
owners are never removed automatically.

A pre-existing `~/.lab-window.lock` blocks startup unless `--lab-owner-token`
matches the JSON lock's `token` (or `pmk-owner.json` inside a directory lock), and
`pause_confirmed` is true. An owned window additionally requires `--resume-hook`
and `--resume-report`. The executable hook is invoked after mining drains, even
on failure; the outer wrapper owns pause/resume operations and fallback
restart. It must emit a JSON object with `confirmed: true` and `outcome: resumed`
or `fallback-restarted`. A failed command, malformed output or unconfirmed
resume is recorded as failure and makes the CLI fail. Arbitrary hook output is
never copied into logs/reports. PMK does not remove the external lab lock.

The actual pause/resume, host verification, long-window authorization and
production hardware evidence remain the outer wrapper's and operator's responsibility.
These hooks do not assert that a long run has happened.
