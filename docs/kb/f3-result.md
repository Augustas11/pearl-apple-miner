# F3 result: ZK certificate generation (Apple M5 MacBook Air, 10 cores = 4P+6E, 32 GB)

**Verdict: F3 PASS.** Every k in {2048 ... 65536} has degree_bits <= 19 (max 18) and proves successfully. k=2048..16384 prove in 7-18 s on the M5 (orphan loss 3.7-8.9% < 10%).

Setup: pearl_mining 0.3.1, production pattern (rank 128, rows [0,8,64,72], 16-col pattern), cert v3, PlainProof from `pearl_mining.mine` (m=128, n=64, nbits 0x1e010000), `verify_plain_proof_for_cert_version(3)` passed for every proof, ZK proof verified ("Verified"). Code: `bench/f3_proving/`; raw: `bench/evidence/f3_proving_m5.txt`.

## Caveats
- The machine was shared. First-sweep numbers (load 7-8 at start, spikes to 70 from other agents) are inflated; the k=32768/65536 first-sweep reps (65-281 s) were clearly disturbed and are discarded; they were rerun (1 rep, load ~7.5, still somewhat loaded). Pass-2 numbers for k<=16384 started at load 2.6-7.6 (my own earlier runs raise it). Treat all values as +-15%, loaded upper bounds.
- Latency was measured at m=128, n=64. degree_bits and rows do not depend on m,n in any way that matters (see below), and a m=n=4096, k=2048 run proved in 6.8-8.5 s, same as small m,n.
- Peak RSS is the process peak through the proof (jemalloc; rep 0). Later reps in the same process raise it by up to ~1 GB as the circuit cache stays resident.
- Core-scaling for k=16384 at 1 thread ran at load 12.8, inflated. The k=4096 1-thread rerun (quiet) is the reliable anchor.

## k table (all 10 cores)
degree_bits from the prover's own code path (`degbits` crate, `compile().degree_bits()`; same rule as `pearl_program.rs:46-77`: rows = max(8 x blake instructions, matmul rows), padded to a power of two, minimum 2^13). Rate-expanded limit degree_bits+rate_bits0 <= 20 (`pearl_circuit.rs:159`) also holds: rate_bits0 = 1 for degree_bits >= 15, 2 below, so max is 13+2=15 ... 18+1=19.

| k | expected rows | degree_bits | prove cold s | prove warm s | verify s | peak RSS GB | ZK proof bytes |
|---|---|---|---|---|---|---|---|
| 2048 | 5,624 | 13 | 8.7 | 7.3-7.5 | 0.008 | 3.7 | 59,560 (+164 public) |
| 4096 | 11,064 | 14 | 11.1 | 9.7-9.8 | 0.007 | 4.3 | 59,560 |
| 8192 | 21,944 | 15 | 14.6 | 12.5-13.6 | 0.008 | 4.6 | 59,560 |
| 16384 | 43,704 | 16 | 19.9 | 17.9-18.6 | 0.010 | 6.5 | 59,560 |
| 32768 | 87,224 | 17 | 38.8 (1 rep, load ~7.8) | n/a | 0.018 | 6.7 | 59,560 |
| 65536 | 174,264 | 18 | 88.6 (1 rep, load ~7.2) | n/a | 0.068 | 9.4 | 59,560 |

Notes: proof size is constant (final recursion circuit). Cold-vs-warm gap is only ~1-3 s (circuit build is cheap relative to proving; the cache is in-process only, lost on restart; keep a long-lived prover and call `warmup_prove_v2`/first-proof at startup). degree_bits hits 19 only above k = 2^17 which C9 caps anyway (k <= 2^16). m,n sensitivity (k=16384): m=n=4096/8192/16384/2^20 -> rows 43,792/43,808/43,824/43,920, degree_bits 16 unchanged. Tile position (t_rows,t_cols) has no effect. k max for degree_bits <= 19 is therefore the C9 cap 65536 itself.

Latency is not linear in k: ~7 s fixed (recursion circuits) plus ~2-3 s per doubling of k over 2048..16384, then it jumps (38.8 s at 32768, 88.6 s at 65536) where the 2^17/2^18 STARK dominates.

## Core scaling (RAYON_NUM_THREADS; warm proof seconds; env var verified effective: plonky2/starky `parallel` feature uses rayon, and 1 thread is 3.2x slower than 10)
| threads | k=4096 | k=16384 (loaded) |
|---|---|---|
| 1 | 30.9-33.9 (quiet), 49.7 (loaded) | 90.9 |
| 2 | 19.1 | 45-46 |
| 4 | 16-20 | 26.5-29 |
| 6 | 11.6-12.6 | 21-23 |
| 10 (all) | 9.4-9.8 | 14.4-18 |

Fit for k=4096, T(n) = s + p/n with s = 7.3 s serial, p = 23.7 core-s parallel: predicts T2 = 19.2, T6 = 11.3, T10 = 9.7 (matches). Speedup at 10 cores is only ~3.2x; the serial/low-parallel part (witness/trace and recursion setup) is ~25% of one-core time.

Extrapolation (labeled estimate, not measured): M5 P-core-equivalent. Assume a server x86 core is 1.3-2x slower than an M5 P core (unmeasured assumption).
| host | k=2048 | k=4096 | k=16384 |
|---|---|---|---|
| M5 10 cores (measured) | 7.4 | 9.8 | 18 |
| 2-vCPU x86 VPS (est. 1.3-2x slower, 2 threads) | ~18-27 | ~25-38 | ~58-90 |
| 8-core x86 server (est.) | ~9-13 | ~12-19 | ~25-40 |
| 16-core x86 server (est.) | ~8-11 | ~10-16 | ~20-33 |
(2 vCPU may be 1 physical core with hyper-threads, which is worse. k=2048 1-thread was not run; its row is the k=4096 fit scaled by the measured all-core ratio 7.4/9.8.)
Memory is 3.7-6.5 GB at k<=16384 regardless of threads (1 thread: 3.5-4.6 GB), so a 2-vCPU VPS needs >= 8 GB RAM; the earlier simnet_e2e_m5.txt figure (25-31 s at k=2048) is 3-4x the quiet 7.4 s here; it was taken while the GPU miner and other jobs shared the CPU (not re-verified), so the real gateway latency under a running miner can be much higher. Rerun on the target host under load.

## Orphan model
Block spacing: `TargetTimePerBlock = 3m14s = 194 s` on all networks (`vendor/pearl/node/chaincfg/params.go:357,477,564,660,766`). Expected loss per find = 1 - e^(-L/194), L = find-to-submit latency (prove + verify + RPC; verify is ~10 ms).

| L (s) | loss % | where |
|---|---|---|
| 7.4 | 3.7 | M5, k=2048 |
| 9.8 | 4.9 | M5, k=4096 |
| 13 | 6.5 | M5, k=8192 |
| 18 | 8.9 | M5, k=16384 |
| 18-27 | 8.9-13 | est. 2-vCPU VPS, k=2048 |
| 31 | 14.8 | M5 1 thread, k=4096 |
| 38.8 | 18.1 | M5, k=32768 |
| 88.6 | 36.7 | M5, k=65536 |

This model is the first-order one (a competing block arriving within L). Overlap-aware loss also depends on propagation and chain tip changes; treat as upper-ish bound.

## Recommendation
- **k range for v1: 2048-8192** (<= ~6.5% on the M5, <= ~13 s). k=16384 is borderline (8.9% on the M5, worse on small hosts). Do not go above 16384; 32768 and 65536 are valid but cost 18-37% orphans.
- k=2048 minimizes orphan loss; choose higher k only if the GPU rate/difficulty trade (P6 objective ops/s x e^(-L/194)) gains more than the ~1.2%-per-doubling latency penalty; this bench did not measure GPU rate vs k.
- Keep the prover resident (warm cache) and off the gateway event loop; one proof uses all cores, so concurrent finds queue. At 194 s blocks and expected finds well below 1 per proof time this is fine for a single miner; a bounded queue (depth ~2) is enough.

## Pool/proving-server sizing
- Per-proof cost at k=2048: ~7.4 s wall on 10 M5 cores, ~30 core-s of CPU work at k=4096 (1 thread 31 s; k=2048 about 24 core-s est.). A 16-core x86 box proves ~1 share per 8-11 s, so ~330-450 shares/hour/box if saturated; a 2-vCPU VPS ~150-200 shares/hour at k=2048 with 18-27 s each, and loss ~10-13%.
- For a pool, share rate is bounded by difficulty, not CPU; only block-solving finds need a ZK proof. Size one 8-16 core, >= 16 GB (8 GB min) host per ~1 concurrent proof; add hosts for parallel proofs (memory 4-7 GB each at k<=16384, so do not run >2 concurrent per 16 GB).
