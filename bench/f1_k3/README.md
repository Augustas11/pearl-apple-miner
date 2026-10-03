# f1_k3: F1 feasibility gate — full K3-NA prototype (SPEC v0.2 §5.1, §8 P1, §9 F1)

Result and verdict: `docs/kb/f1-result.md`. Evidence: `bench/evidence/f1_k3_m5_{correct,perf}.txt`.

## Files
| File | What |
|---|---|
| `k3.metal` | Kernel, compiled at runtime (Metal 4.0) with `-D K3_VARIANT`: `2` = production K3-NA (matmul2d int8×int8→int32, `execution_simdgroups<4>`, 128×64 tile, RK=128, `multiply_accumulate`, `relaxed=false`; V6 fold into a per-lane 16-word transcript; keyed BLAKE3 single-block compress, flags CHUNK_START\|CHUNK_END\|ROOT\|KEYED_HASH, key = a_noise_seed; U256 LE compare vs block and share bounds; `atomic_uint` counters + bounds-checked slot writes; no C output). `0` = P1 baseline (int8bench matmul2d, same tile, C store disabled: cooperative-tensor destination + conditional sink). `1` = V6 fold only (no hash/compare/atomics). `3` = int8bench kernel unmodified (C stored; context) |
| `k3.swift` | Host. `run JOBDIR` (correctness, no lock) and `perf` (paired alternating rounds, GPU timestamps, lock + loadavg + power logging) |
| `oracle.py` | CPU oracle, numpy int64 + `blake3`: Pearl tiles (`threads_partition` / `offset_is_valid`), jackpot (`mine.rs:87-107`), hash (`proof_utils.rs:1502-1505`), bound (`extract_difficulty_bound`), and for the cross-check job key / raw roots / salted v3 seeds / noise. No zk-pow code imported |
| `harness.py` | Job-dir I/O and slot-by-slot comparison against the oracle |
| `crosscheck.py` | Oracle vs `pearl_mining` 0.3.1 `mine()` + `verify_plain_proof_for_cert_version(3, …)`, then the same job through the GPU kernel |
| `correct.py` | Tests (a)–(e) against the production kernel |

Slot layout (26 × u32): `t_rows, t_cols, transcript[16], hash[8]`. `t_rows`/`t_cols` are the global minima of the lane's
pattern (threadgroup origin + per-lane minimum from `get_multidimensional_index`), i.e. Pearl's valid offsets.
Classification is independent per array: block array gets every hash ≤ B, share array every hash ≤ S (a block find with
S ≥ B therefore appears in both). Counters keep counting past capacity; only `idx < cap` is written.
Host limits: M % 128 = 0, N % 64 = 0, K % 128 = 0 (v1 policy r = 128).

## Commands
```bash
cd <repo root>
swiftc -O "$PWD/bench/f1_k3/k3.swift" -o /tmp/k3      # absolute path: the binary finds k3.metal via #filePath

cd bench/f1_k3
# correctness (no GPU lock needed)
K3_BIN=/tmp/k3 K3_WORK=/tmp/k3_f1_jobs ../../.venv/bin/python crosscheck.py
K3_BIN=/tmp/k3 K3_WORK=/tmp/k3_f1_jobs ../../.venv/bin/python correct.py

# performance (takes /tmp/pmm-gpu-bench.lock per shape batch; waits up to 30 min for loadavg <= 8, then proceeds + flags)
/tmp/k3 perf 4096x4096x4096 8192x8192x4096 rounds=31
```
The evidence files were produced with exactly these commands (binary and job dir in the session scratchpad instead of `/tmp`).

## Perf method
- 4 variants per round (int8bench, base, fold, k3), order reversed on odd rounds, 1 warm-up each, 31 rounds per shape.
- Identical random operands in [-127,127] for all variants; GPU time = `gpuEndTime - gpuStartTime` per command buffer.
- P1 ratio per round = t_base / t_k3 (throughput ratio). Reported: median and a 90% percentile-bootstrap CI of the median
  (20,000 resamples, fixed seed); half-width = (hi − lo) / 2.
- Hash + compare + atomics overhead = median of t_k3 / t_fold − 1 over the same rounds.
- K3 runs with share bound 2^(258 − log2 tiles) − 1 (≈ 4 share finds per job), block bound 2^200 − 1, capacities 4 / 64.
- Each shape batch logs `vm.loadavg` and `pmset -g batt` before and after.
