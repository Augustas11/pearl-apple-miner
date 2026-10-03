# ADR-001: A fixed per template, fresh B per job

Status: accepted for B3 cert-v3 solo mining.

The template key is the previous-block hash, compact bits, and all 76 immutable
header bytes. A height alone cannot identify work. The SG configuration is fixed
for a pipeline: rows `[0,8,16,24]`, columns `[0,1,8,9,16,17,24,25]`, rank 128,
and k divisible by 128 in [2048,8192].

`pmkcore_template_init` fills a shared A buffer using native OS-entropy-backed
CSPRNG generation and hashes it once when the template is installed. Each
`pmkcore_build_job` fills a rotating B-transpose buffer with fresh native CSPRNG
output, commits it, and derives both salted v3 noise seeds. Python sees buffer
pointers, roots, seeds, headers, configurations and proofs; it never reads or
writes matrix bytes.

A fresh B changes the seed chain for both A and B. The verifier requires signal
range and Merkle consistency, not independently fresh A. This policy reduces
preparation cost while retaining a fresh work attempt. It is policy, not a
consensus change. Reusing A across different headers is rejected by ownership:
the pipeline must drain all retained jobs before replacing its template.

Find processing retains raw A and B until the pmkcore oracle has constructed all
proofs and each is submitted or explicitly abandoned. Proof row/column indices
are sorted global pattern minima. Every block proof passes the v3 plain verifier
against its original header; diagnostic shares use only `nbits_override`.

The plain verifier at Pearl 7039e66f does not enforce the rank penalty. Therefore
both the native builder and Python boundary enforce rank 128 and k%128==0, and
Python checks `penalized_target_bound(target, config) == target * 32 * k` before
dispatch. Overflow/saturated bounds are refused.

Evidence: `bench/evidence/b3_native_finds.txt` exercises actual shared-buffer job
preparation, fixed-A/fresh-B roots, callback dispatch and locally verified block
proofs. `bench/evidence/b3_overhead_default.txt` and
`bench/evidence/b3_overhead_4096.txt` record overhead measurements. These do not
replace the M3 Ultra throughput/release gates or the G6 24-hour health run.
