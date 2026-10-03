# ADR-003: Object pool jobs retain their originating session and share target

Status: implemented for B6; public-pool SG acceptance remains a T2/T3 gate.

A pool header is the template identity. Notify metadata (pool job ID, target,
compact share bits and session) is immutable for an attempt. A repeated header
can reuse A while adopting new notify metadata for subsequent attempts. B is
fresh per attempt, preserving ADR-001. Work completing after a new header may
submit only under its original ID and only in its original connection session.
Disconnect invalidates all work eligibility immediately.

The pool target is decoded as a big-endian integer and compacted downward.
The kernel uses the upstream `extract_difficulty_bound` called by the verifier.
The Python binding patch is stored under `miner/native_patches/` and is applied
only to an isolated build copy; it exports the existing native functions without
changing consensus or the repository's vendor source. Missing exports fail
closed. Rank 128, production SG patterns and non-saturated bounds remain local
policy. Every submitted find, including a block candidate, is verified with
cert version 3 and its own compact share override against the unchanged header.
If a pool sends a target harder than the block target, that condition is logged;
a block-only find is counted but is not submitted as a share.

Pool submissions use the append-only, fsynced ledger. Prepared metadata cannot
be rewritten, and a terminal verdict cannot change. Unfinished entries from a
previous process become transport outcomes rather than being retransmitted in
a new session. A policy rejection before any accepted share is a compatibility
failure; a policy rejection after acceptance is P0. Both halt durably. Transport
ambiguity is never treated as an accepted share or a pool stale verdict.

Lost-find expectation sums actual completed GPU MACs multiplied by each job's
wire target divided by 2^256. Completion is accounted before proof construction
or submission, including work later abandoned. The summary also exposes the
compact-target expectation so rounding is visible. Pool H/s is MAC/s; completed
ops/s alone is not a P2 ratio pass without a paired K3-alone baseline.

Long-run lifecycle remains owned by an optional outer wrapper. The pool window requires an inherited GPU
lock and an environment-only ownership token. It validates the existing
pause/ownership contract, creates isolated run state, and defers resume of
paused workloads to the wrapper. It never creates/removes locks or invokes launchd control.
