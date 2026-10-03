# pmkcore

Native Pearl cert-v3 job preparation, G1 CPU oracle, and PlainProof construction.
Reference: vendored `zk-pow` / `pearl-blake3` at Pearl `7039e66f`, SPEC v0.3.
The existing F2 Rust/C API and `PmkTemplate`/`PmkJob` layouts are preserved.

## Build and test

Run from this directory so Cargo reads `.cargo/config.toml` (ARM AES configuration):

```sh
cd pmkcore
cargo build --offline --release -j 2
RAYON_NUM_THREADS=2 ../.venv/bin/python -B tests/ffi_smoke.py
RAYON_NUM_THREADS=2 cargo test --offline -j 2 -- --nocapture --test-threads=1
cargo fmt --check
cargo clippy --offline --all-targets --no-deps -j 2 -- -D warnings
cc -std=c11 -Wall -Wextra -Werror -fsyntax-only tests/abi_layout.c
```

`cargo test` includes `tests/verify_b1.py`, executed with the repository's
`.venv/bin/python`, where `pearl_mining`, NumPy and Python `blake3` must already be
installed. A missing Python interpreter/package fails the test; nothing is skipped.
No network, GPU, SSH, or node is used. Test inputs are reproducible seeded random
jobs; production F2 generation still uses OS entropy. The Rust tests exercise
reference mining, retained-data proof construction, both patterns, stored K3-SG
vectors, malformed inputs, and the C ABI. Python verifies the exported proofs
using `verify_plain_proof_for_cert_version(3, header, proof, nbits_override=...)`.

The local test evidence is `tests/b1_pmkcore_tests.txt`. These are correctness
checks, not GPU performance, ZK proving, or node acceptance gates.

## Rust API

```rust,no_run
use pmkcore::oracle::{build_config, OracleJob, Pattern};
# fn example(header: &[u8; 76], a: &[u8], bt: &[u8]) -> Result<(), pmkcore::PmkError> {
let config = build_config(Pattern::Sg, 4096, 64, 64)?;
let job = OracleJob::new(header, &config.to_bytes(), 64, 64, a, bt)?;
let share_bound = job.bound(0x1e100000);
let block_bound = job.bound(job.header().nbits);
let tile = job.tile(0, 0, share_bound, block_bound)?;
if tile.is_share != 0 || tile.is_block != 0 {
    let proof_bytes = job.build_plain_proof(tile.t_rows, tile.t_cols)?;
    // Verify with Pearl against the immutable header and the relevant nbits.
    assert!(!proof_bytes.is_empty());
}
let pinned_intermediates = job.export_vectors();
# Ok(()) }
```

- `build_config(Pattern::{Na,Sg}, k, m, n)` constructs patterns through upstream
  `PeriodicPattern::from_list`, checks semantic/byte serialization round trips,
  and enforces consensus C9 plus v1 policy: rank 128, dense int7, `k % 128 == 0`,
  `2048 <= k <= 8192`. NA requires `m % 128 == 0`, `n % 64 == 0`; SG requires
  `m % 64 == 0`, `n % 64 == 0`. Dimensions must be nonzero and at most `2^24`.
- `OracleJob::new` copies exact `m*k` raw A bytes and `n*k` raw Bᵀ bytes, both
  row-major signed int8 in `[-64,64]`. Inputs may be freed or overwritten after
  creation. Only the two specified patterns are supported. The diagnostic oracle
  allows consensus `k` through 65536 for G2/G3; use the builder for production v1
  policy. The historical F2 configuration validation remains unchanged.
- `intermediates()` returns an immutable snapshot of every commitment/seed/noise
  stage. Raw commitments use zero padding to a 1024-byte boundary. With supported
  dimensions and k, raw matrix lengths already land on that boundary.
- `tile(tr,tc,share,block)` checks global pattern minima, computes the cumulative
  int32 transcript with rank 128, then the keyed BLAKE3 hash. `scan` returns every
  tile in ascending row-offset then column-offset order. `tile_count()` is cheap.
- Bounds are **32-byte little-endian scaled U256 values**, compared inclusively.
  Flags are independent: a block may also be a share. `bound(nbits)` uses upstream
  saturating `extract_difficulty_bound`; rank 128 makes the penalty neutral. A
  saturated target is useful for tests; callers choose appropriate pool targets.
- `build_plain_proof` opens sorted global rows/columns from the retained **raw**
  trees, checks the reconstructed configuration bytes, legal offsets and STARK
  degree bound. It serializes upstream `PlainProof` with upstream's bincode 1.3
  wire format (fixed-width little-endian). It deliberately does not claim that
  an arbitrary requested tile meets a target: the caller must verify every find.
  Python accepts `PlainProof.from_base64(base64.b64encode(proof_bytes).decode())`.
- `transcript` is a low-level Rust helper for already-noised A and Bᵀ, including
  stored K3-SG vectors. It does not generate noise or validate a production job.

NA pattern: rows `[0,8,64,72]`, columns
`[0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51]`.
SG pattern: rows `[0,8,16,24]`, columns `[0,1,8,9,16,17,24,25]`.
Offsets are defined by upstream `offset_is_valid`, not GPU threadgroup origins.

**Upstream verification limit:** at `7039e66f`, Python's cert-v3 plain verifier
does not call the separate `check_rank_penalty`. A correctly committed/mined
rank-64 proof can pass that verifier. Both the B1 builder and oracle reject rank
64 before accepting a job. Tests distinguish an altered-rank proof (rejected by
the verifier because its commitment no longer matches) from an independently
mined rank-64 proof (rejected by pmkcore policy and upstream's separate rank
check). Callers accepting external proofs must enforce rank 128 as well as
checking the verifier's returned `(ok, message)` tuple; absence of an exception
does not mean success.

## C ABI

Include `include/pmkcore.h`; link `target/release/libpmkcore.dylib` on macOS,
`libpmkcore.so` on Linux, or the static library. All new fallible calls return
zero on success or a negative `PMK_E_*` code; `pmkcore_strerror` describes errors.
New errors are illegal offset (-10), proof failure (-11), caught panic (-12),
and resource limit (-13). The Rust panic boundary cannot recover from process
OOM or invalid caller pointers.

```c
int32_t pmkcore_build_config(uint32_t pattern, uint32_t k, uint32_t m,
                            uint32_t n, uint8_t out_config[52]);
int32_t pmkcore_oracle_job_create(const uint8_t header[76], const uint8_t config[52],
    uint32_t m, uint32_t n, const uint8_t *a, uint64_t a_len,
    const uint8_t *bt, uint64_t bt_len, PmkOracleJob **out_job);
void pmkcore_oracle_job_free(PmkOracleJob *job);
int32_t pmkcore_oracle_tile(const PmkOracleJob *job, uint32_t t_rows, uint32_t t_cols,
    const uint8_t share_bound[32], const uint8_t block_bound[32], PmkTileResult *out);
int32_t pmkcore_oracle_scan(const PmkOracleJob *job, const uint8_t share_bound[32],
    const uint8_t block_bound[32], PmkTileResult *out, uint64_t out_cap, uint64_t *out_len);
int32_t pmkcore_oracle_build_plain_proof(const PmkOracleJob *job, uint32_t t_rows,
    uint32_t t_cols, uint8_t *out, uint64_t out_cap, uint64_t *out_len);
int32_t pmkcore_oracle_export_vectors(const PmkOracleJob *job,
    uint8_t *out, uint64_t out_cap, uint64_t *out_len);
```

Pattern IDs are `0=NA`, `1=SG`. `PmkTileResult` holds `t_rows`, `t_cols`, 16 native
`uint32_t` transcript words, 32 hash bytes, and `uint32_t is_share,is_block`.
For variable output, call with `out=NULL,out_cap=0` to query length, allocate,
then call again. Capacity/length is measured in tile records for `scan`, bytes
for proofs/vectors. Too-small buffers receive no partial data and report the
required length. No buffer is allocated for the caller to free other than the
opaque job, which must be released with `pmkcore_oracle_job_free` exactly once.
All pointers must be valid and suitably aligned for their stated sizes, and
output ranges must not alias inputs. Concurrent immutable reads of a job are
allowed; freeing it while any call is active is not. `free(NULL)` is harmless.

## Test-vector format: PMKVEC01

`export_vectors` / `pmkcore_oracle_export_vectors` produces a deterministic binary
container, without Rust layouts or native-endian integers:

1. Eight ASCII bytes `PMKVEC01`.
2. Field count (`u32 LE`).
3. For each field: name length (`u16 LE`), ASCII name, payload length (`u64 LE`),
   then exactly that many payload bytes.

Fields, in serialization order:

| Field | Payload |
| --- | --- |
| `dimensions` | m, n, k, rank as four u32 LE |
| `header`, `config` | 76, 52 upstream serialized bytes |
| `padded_a`, `padded_bt` | Raw signed-int8 two's-complement bytes, zero chunk pad |
| `job_key`, `raw_root_a`, `raw_root_b` | 32 bytes each |
| `salt_input_a`, `salt_input_b` | root + dimension u32 LE + 28 zero bytes (64 each) |
| `salted_root_a`, `salted_root_b` | 32 bytes each |
| `seed_input_a`, `seed_input_b` | 64-byte concatenations hashed to each seed |
| `a_noise_seed`, `b_noise_seed` | 32 bytes each |
| `salt_key_a`, `salt_key_b` | Cert-v3 domain keys, 32 bytes each |
| `e_al`, `e_br_t` | Dense factors, m×128 and n×128 row-major int8 |
| `noise_a`, `noise_bt` | Gather-subtract noise, m×k and n×k row-major int8 |
| `noised_a`, `noised_bt` | Raw + noise, m×k and n×k row-major int8 |
| `e_ar_t`, `e_bl` | k pairs of (+1 index, -1 index), each index u32 LE |

B-side arrays are **transposed** throughout. A GPU expecting k×n B' must transpose
`noised_bt`. Use `tile`/`scan` for transcripts, hashes and classifications; the
intermediate container is independent of chosen bounds.

`tests/intermediates.rs` pins reproducible k=4096 exports, after independently
checking every section against `bench/f1_k3/oracle.py`:

- NA BLAKE3: `08f2e0c3d6a8c44440499a337526a7afd3bc10a355e2fb0168805124a635ffbe`
- SG BLAKE3: `127d5b12c565edeec6cade3b474eb6e3b914dc58ab163ce76bfb210215dd3317`

## Resource scope

This CPU implementation is a correctness oracle and overflow-recovery path.
It retains raw bytes, Merkle trees, dense/sparse noise, full noise and noised
matrices. Creation rejects a conservative retained/construction estimate over
1 GiB before copying matrix inputs, so protocol-extreme shapes fail closed.
Export and complete scans allocate additional memory; prefer individual tiles
when checking large jobs. The application must enforce the combined 25%-of-RAM
budget across jobs, GPU buffers and proving. The F2 preparation functions remain
the throughput path for ordinary mining jobs.
