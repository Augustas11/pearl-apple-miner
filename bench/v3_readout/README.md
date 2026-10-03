# v3_readout: R6 microbenchmark (accumulator readout every rank=128 K-steps)

Question: does reading the int32 `matmul2d` accumulators every 128 K-steps (Pearl v3 XOR-fold jackpot)
slow the M5 GPU Neural Accelerator path? Result and verdict: `docs/kb/r6-result.md`.

Single Swift host (`r6bench.swift`); Metal 4 kernels are generated as source and compiled at runtime
(`languageVersion = .version4_0`, MetalPerformancePrimitives `matmul2d`, int8 x int8 -> int32,
`execution_simdgroups<4>`, 128 threads per threadgroup).

## Variants
| Name | What it does |
|---|---|
| `V0_BMxBN` | Baseline, same as `bench/int8bench.swift`: one `run()` over the whole K into device C |
| `V1_BMxBN_rkRK` | K loop: `run()` per RK chunk, `multiply_accumulate` into an int32 cooperative tensor, no readout, `store` C at the end |
| `V2_BMxBN_rkRK` | V1 + every 128 K: XOR-fold all accumulator elements of the lane (plus lane^1 via `simd_shuffle_xor` for 64x32, whose 16-element lane set is below Pearl's 32 minimum), `rotl13` into a 16-word per-lane transcript slot `(chunk) % 16`. Writes only transcripts (16 words per lane), no C |
| `V3_BMxBN_rkRK` | Fallback: every 128 K store the cooperative tensor to threadgroup memory, barrier, each thread folds a 2 x W contiguous pattern, barrier |
| `V4_BMxBN_rkRK` | Software-pipelined fold: chunk c accumulates into a second cooperative tensor while acc_{c-1} is folded, then `cT += cN` (exact, but slow: two cooperative tensors) |
| `V5_BMxBN_rkRK` | Diagnostic only (not Pearl-valid): reads a single accumulator element every 128 K. Separates "any readout syncs the accelerator" from "per-element cost" |
| `V6_BMxBN_rkRK` | Same semantics as V2, but folds through a raw pointer (`&cT[0]` as `uint4*`, compile-time count, no per-element `is_valid_element` / accessor calls). Valid only because the probe showed every element valid and `&cT[i] == &cT[0] + i` on this OS/GPU, so a production kernel must re-probe and fail closed |
| `_mort` suffix | Morton threadgroup traversal instead of row-major |
| `_dump` suffix | (correctness only) also writes every accumulator at every 128-K boundary |

## Build and run
```bash
cd <repo root>
swiftc -O bench/v3_readout/r6bench.swift -o /tmp/r6bench

/tmp/r6bench probe     > bench/evidence/v3_readout_m5_probe.txt     # layout + PeriodicPattern validity
/tmp/r6bench correct   > bench/evidence/v3_readout_m5_correct.txt   # bit-exact vs CPU int64 oracle
/tmp/r6bench perf      > bench/evidence/v3_readout_m5_perf.txt      # 4096^3, 4096x4096x{2048,8192}, 8192x8192x{2048,8192}
/tmp/r6bench diag  > bench/evidence/v3_readout_m5_diag.txt        # V0/V1/V5/V2/V4/V6, random operands
/tmp/r6bench diag zero >> bench/evidence/v3_readout_m5_diag.txt   # same, all-zero operands
/tmp/r6bench final > bench/evidence/v3_readout_m5_final.txt       # decision variants, 15 rounds, 5 shapes
/tmp/r6bench sustain 180 8192x8192x8192 128 64 128 6 > bench/evidence/v3_readout_m5_sustain.txt
#                  secs  shape          BM  BN  RK V [mort]   (RK 0 + V 0 = baseline)
```
`perf`, `diag`, `final` and `sustain` take the shared GPU lock `mkdir /tmp/pmm-gpu-bench.lock` (retry every 15 s), log
`sysctl vm.loadavg` before/after, and `rmdir` the lock after each shape batch. `probe` and `correct` take no lock.

Timing: GPU timestamps (`gpuEndTime - gpuStartTime`) per command buffer, 1 warm-up per variant, 7 timed rounds (15 for `final`),
variant order reversed on odd rounds to cancel drift. TOPS = 2mnk / time.

Correctness (`correct`): operands uniform in [-127,127]; shapes 256x256x8192, 128x128x65536 and 256x128x2112
(trailing k % 128 = 64, which must not enter the transcript). Checked against a CPU int64 oracle: C for V0/V1,
all accumulators at every 128-K boundary for `_dump`, and every transcript word for V2/V3
(Pearl `mine.rs:87-107` semantics).

Limits: M, N must be multiples of the tile; K must be a multiple of RK (the kernel's tail loop steps by RK).
Morton needs power-of-two grid dims.
