# libpmk

Swift package producing `libpmk/.build/release/libpmk.dylib`, with C declarations
in `include/libpmk.h`. Requires macOS 14+, an Apple7-or-newer GPU and unified
memory. B2 tests on the Apple M5 establish correctness only; they make no
performance claim or claim of passing the Studio release gates.

```sh
swift build --package-path libpmk -c release
libpmk/tests/run.sh
```

Ship the adjacent `libpmk_PMK.bundle` with the dylib. It contains runtime Metal
source and the embedded startup oracle. The kernel is copied byte-for-byte from
`bench/k3sg/k3sg.metal`; compile macros select production `64x64x16x2x2x2`,
MSL 3.1, with fast math disabled.

## C ABI

```c
int32_t pmk_init(pmk_context *out, char *error, uint64_t error_capacity);
int32_t pmk_init_diagnostic(pmk_context *out, char *error, uint64_t error_capacity);
int32_t pmk_probe(pmk_context ctx, char *cache_key, uint64_t capacity);
void pmk_destroy(pmk_context ctx);
int32_t pmk_buffer_alloc(pmk_context ctx, uint64_t bytes, void **out);
int32_t pmk_buffer_release(pmk_context ctx, void *buffer);
int32_t pmk_run_job(pmk_context ctx, const pmk_job_desc *desc,
                    pmk_completion callback, void *user, pmk_job *out);
int32_t pmk_poll(pmk_job job, pmk_result *out);
int32_t pmk_job_wait_callback(pmk_job job);
int32_t pmk_job_release(pmk_job job);
```

`pmk_init` synchronously runs all six fail-closed probe steps: fragment load,
inject/store, exact MMA layout, all int8-to-float conversions, pattern/partition
legality, and 31 known-answer cases at 256×256×4096 on the actual production
pipeline. Failure returns a null context and a “DO NOT MINE” diagnostic.
`pmk_probe` returns the cached successful verdict's SHA-256 key, covering Metal
source, compile options, GPU name and OS build. Caching is context-local; each
new context probes, so an on-disk cache cannot bypass the startup check.
A command-buffer or CPU recovery mismatch poisons the context.

`pmk_init` also enforces the Apple7-9 G3 admission file. It refuses to mine
unless a record keyed by exact GPU name, device class, OS build and probe cache
key already exists. A record without `valid_hours` is a durable class
certification and remains valid until the OS build or probe cache key changes;
periodic probe refresh is a separate sentinel and does not update or age the G3
record. Operators may set a positive finite `valid_hours` for an expiring lab
admission. `pmk_init_diagnostic` is the only bootstrap path for the Studio G3
exactness campaign and test harnesses: it runs the same probe and kernel code
but skips the admission-file check so an unapproved device can earn evidence.
The G3 runner must write the approval file only after the diagnostic campaign
passes. libpmk never auto-creates approved entries.

Allocate raw A and Bᵀ with `pmk_buffer_alloc`, fill the shared memory directly
(e.g. via pmkcore), then submit their base pointers in `pmk_job_desc`. A is m×k
row-major, Bᵀ is n×k row-major, and every signed signal byte must be in [-64,64].
The descriptor carries ABI version 1, cert version 3, rank 128, seeds and inclusive
block/share U256 bounds as eight LE uint32 words. Shapes require m,n positive
multiples of 64, and production `pmk_run_job` requires 2048≤k≤8192 with
k%128=0. `pmk_run_job_diagnostic` is the explicit test/vector path for
non-production k values such as 65536.

`pmk_run_job` validates before allocating job buffers, then enqueues exactly one
command buffer K1→K2→K3. It never waits for GPU completion. An optional
`pmk_completion(job,user)` callback runs after `addCompletedHandler` and any CPU
recovery; otherwise poll until `PMK_PENDING` changes to success or an error.
The callback may poll and release its job. Choose one release owner and never
concurrently release a handle from callback and polling threads. An external
release owner must call `pmk_job_wait_callback` before release; it returns only
after the foreign callback has fully returned. Calling the wait from inside the
callback returns `PMK_BUSY`, so callback-owned release remains nonblocking.

Keep raw inputs immutable and retain the job until all finds have been processed
and all proofs submitted or abandoned. `pmk_poll` does not consume results.
Result pointers remain valid until successful `pmk_job_release`. There are at
most three unreleased jobs per context. Input-buffer release returns `PMK_BUSY`
while jobs exist. Release jobs, then buffers, then destroy the context; never use
handles or input pointers after their owning objects are released/destroyed.

Each 104-byte `pmk_slot` contains global pattern minima `t_rows,t_cols`, 16
transcript words, and 8 hash words. Atomic append order is unspecified. Arrays
have independent counters and capacities (block≥4, share≥64), bounds-checked
writes, and checked guard regions. Overflow sets bit 0 for blocks / bit 1 for
shares. Recovery independently regenerates Pearl noise from retained raw inputs
and seeds, checks it against GPU K2 output, and recomputes all jackpots on CPU.
It checks recovered counts against GPU totals and returns complete arrays with
`recovered=1`. `*_count` are original GPU totals; `*_stored` are returned lengths.
GPU timestamps are Metal clock seconds and exclude CPU recovery.

Every Metal buffer is strictly below 2,000,000,000 bytes and the device maximum.
Allocation-free shape/budget validation rejects protocol extremes. A context
reserves at most 25% of physical RAM for inputs, job buffers, scratch and worst-case
recovery. Use one context, and include pmkcore/Merkle allocations in the caller's
overall 25%-RAM budget; libpmk cannot account for allocations made by other libraries.

## Tests and evidence

`tests/run.sh` atomically takes `/tmp/pmm-gpu-bench.lock`, retries every 15 seconds,
and removes only its own lock. It runs XCTest and the public C-ABI Python runner
using the repository `.venv`. Evidence is captured in
`libpmk/tests/evidence/b2_libpmk_tests.txt`; the B2 delivery also includes the
requested copy at `bench/evidence/b2_libpmk_tests.txt`.

Validated on macOS 26.5 (25F71), Apple M5: 8 native tests, all 94 bench
cases and 50 public-ABI jobs passed, including cert-v3 proof acceptance.

The PMK library is release-optimized. The XCTest target uses `-Onone` because
the optimized `@testable` Job-construction harness crashed under Swift 6.3.3.
This workaround is limited to the harness; the public C-ABI tests exercise the
optimized dylib.
No tests are skipped. Plain `swift test` also builds the library in debug mode.

G2 uses 20 deterministic cases produced by vendored `zk_pow::circuit::pearl_noise`,
including k=2048 and k=65536. Full-buffer SHA-256 values pin every output byte;
seed, input and prefix checks also pin reference regeneration. Sixteen keyed
single-block BLAKE3 vectors cross-check the Rust `blake3` crate, Pearl's hash,
the GPU compressor and the CPU oracle. Regenerate resources with:

```sh
cd libpmk/tests/reference
CARGO_TARGET_DIR=target cargo run --release -- vectors
```

G3 compares all slots/counters with four bench oracle jobs, including
256×256×4096, full-range ±127, and 128×128×65536; checks hash−1/hash/hash+1,
word-carry boundaries, simultaneous finds and overflow; and independently checks
the full-job CPU oracle. The Python runner adds 50 raw-input K1→K2→K3 jobs,
callbacks, invalid inputs, three retained jobs, CPU overflow recovery, and a
PlainProof built for a GPU find and accepted by `verify_plain_proof_for_cert_version(3)`.
Deliberately broken layout/hash kernels must fail the startup probe.
