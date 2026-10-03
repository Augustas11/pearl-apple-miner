# f2_jobprep: gate F2 (native job prep keeps the GPU fed)

Question (SPEC §9 F2, §8 P2): can `pmkcore` build Pearl v3 jobs fast enough that the K3 kernel never waits?
Result and verdict: `docs/kb/f2-result.md`. Raw output: `bench/evidence/f2_jobprep_m5.txt`.

## What a job build is (pmkcore, `pmkcore/src/lib.rs`)
- Per template, once: `job_key = blake3(header76 ‖ config52)`; A (m×k) from the CSPRNG; raw keyed-BLAKE3
  Merkle root of padded A; salted with m (cert v3). A is fixed per template (SPEC §3.2, ADR-001).
- Per job (`pmkcore_build_job`): fresh Bᵀ (n×k row-major, int8 in [-64,64]) written straight into the
  caller's buffer (a shared `MTLBuffer` in `p2bench`), its raw root, then
  `b_noise_seed = blake3(job_key ‖ bind_root_b(root_b, n))`, `a_noise_seed = blake3(b_noise_seed ‖ bind_root_a(root_a, m))`.
- CSPRNG: AES-128-CTR on the ARMv8 AES unit, fresh 128-bit key from OS entropy (`getrandom`) per job.
  2 keystream bytes per element, multiply-shift to [-64,64] (relative bias ≤ 1/508; consensus checks
  only the range).
- Generation and hashing are fused per 64 KiB segment (generate, then hash while it is in cache) on a
  rayon pool; segment chaining values are merged with `blake3::hazmat` into the tree root.
  `pmkcore/tests/reference.rs` proves the root equals `pearl_blake3::MerkleTree::new(..).root()`.

## Correctness
```bash
cd pmkcore && cargo test --release
```
- `reference_miner_jobs_match`: 24 jobs whose A and B come from zk-pow's own `try_mine_one`
  (ChaCha8-seeded, salted cert-v3 derivation). The raw roots in its `PlainProof`, the job key, and the
  seeds from zk-pow's verifier path (`parse_proof` → `PublicProofParams::commitment_hash`) equal pmkcore's.
- `generated_jobs_fixed_a_fresh_b`: 26 pmkcore-generated jobs over 2 templates: range, zero pad, root
  equals pearl_blake3's tree root of the bytes written, seeds equal zk-pow's, and no seed repeats.
- `tree_root_matches_pearl_blake3`: 17 sizes (1 … 4097 chunks, unaligned payload).
- `signal_distribution`, `c_abi_and_errors`.

## CPU benchmark (`src/main.rs`)
```bash
cd bench/f2_jobprep && cargo build --release && ./target/release/f2_jobprep
```
Sections: BLAKE3 alone (64 MiB; Pearl's `blake3_digest` single-thread, `MerkleTree` rayon, pmkcore
hash-only), template build, `pmkcore_build_job` via the C ABI for n ∈ {1024…8192}, k ∈ {2048, 4096, 8192},
T ∈ {1, 9, 10} rayon threads, the build/GPU ratio, and the smallest shape that keeps build ≤ 0.8× GPU.

## End-to-end P2 proxy (`p2bench.swift`)
```bash
cd pmkcore && cargo build --release            # builds target/release/libpmkcore.a
cd ../bench/f2_jobprep
swiftc -O -target arm64-apple-macosx26.5 p2bench.swift \
  -import-objc-header ../../pmkcore/include/pmkcore.h ../../pmkcore/target/release/libpmkcore.a -o /tmp/p2bench
/tmp/p2bench                                   # default 6 shapes; or e.g. /tmp/p2bench 8192x8192x4096 --jobs 70 --rounds 3
```
- GPU kernel: the R6 V6 kernel (`matmul2d` int8→int32, 128×64, RK=128, pointer XOR-fold + rotl13
  transcript, no C store). No K1/K2 noise and no final BLAKE3/compare (F1 work), so it stands in for K3.
- `gpu` mode: K3-alone wall rate (same buffers, 2 command buffers in flight).
- `pipe` mode: 3 rotating Bᵀ slots; the main thread calls `pmkcore_build_job` (9 threads) into
  `slot.contents()` and commits; slots are released from `addCompletedHandler`.
- P2 = pipe rate / gpu rate over ≥ 200 jobs per mode (default 3 rounds × 70), rounds alternate order.

Both binaries wait for 1-min loadavg ≤ 8 (≤ 30 min, then proceed and print a FLAG), then take
`mkdir /tmp/pmm-gpu-bench.lock` (retry every 15 s), log loadavg before/after, and `rmdir` it.
The crate's `.cargo/config.toml` sets `--cfg aes_armv8`; without it `aes` falls back to software AES
(~0.16 GB/s) and pmkcore refuses to compile.
