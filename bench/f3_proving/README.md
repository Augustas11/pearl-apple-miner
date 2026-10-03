# F3 proving benchmark

Measures Pearl ZK certificate generation (`pearl_mining.generate_proof_for_cert_version(3, ...)`) at the v1 production pattern
(rank 128, Int7xInt7ToInt32, rows [0,8,64,72], cols [0-3,16-19,32-35,48-51], cert v3).

- `prove_one.py`: one process, one k: mine a PlainProof (m=128,n=64, nbits 0x1e010000), verify it, prove, verify the ZK proof, print JSON. Rep 0 = cold circuit cache, later reps warm (in-process cache).
- `run_f3.py`: wrapper. Takes `/tmp/pmm-gpu-bench.lock`, waits for loadavg<=8, runs under `/usr/bin/time -l`, appends a JSON line to `bench/evidence/f3_proving_m5.txt`.
- `grid.sh` (k sweep), `scaling.sh` (RAYON_NUM_THREADS 1/2/4/6/10 + large-k rerun), `quiet.sh` (clean second pass); `summarize.py` prints the evidence file.
- `degbits/`: tiny Rust crate (path dep on vendor/pearl/zk-pow, not modified) that calls the same `PublicProofParams::compile().degree_bits()` the prover uses (`prove.rs:62`). Build: `cd degbits && cargo build --release`; run `target/release/degbits m n k`.

Results: `docs/kb/f3-result.md`.
