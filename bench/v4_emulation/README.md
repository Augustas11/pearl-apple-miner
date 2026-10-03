# v4 (FP8 certificate) emulation on Apple GPUs

How fast can an Apple GPU produce Pearl certificate-v4 FP8 matmul results that are
bit-exact with Pearl's B200 device model? Results and verdict: `docs/kb/v4-emulation-result.md`.
Raw outputs: `bench/evidence/v4_emulation_m5_*.txt`.

## Layout

| Path | What |
|---|---|
| `oracle/` | Rust oracle. Compiles Pearl's own source files verbatim via `#[path]` from `vendor/pearl-fp8/zk-pow/src` (fp8-scheme 2569546): `utils.rs` (B200/H100 `matmul_fp8`), `quantization.rs`, `jackpot_policy.rs`, `dtype.rs`, `compute.rs`, `layout.rs`, `circuit/fp8/unpredictability.rs`. `exact_norms`/`open_prequant` come from the zk-pow crate itself. Two small stubs: `Device` enum (copied) and the noise-line sampler (copied; fixed BLAKE3 key instead of the header-derived subkey). |
| `oracle/src/families.rs` | Operand families. Policy families go through Pearl's miner path (int8 + BF16 block scales -> `exact_norms` -> rank-32 noise -> `noisy_quantize`, B200). Adversarial families write E4M3 codes directly. |
| `metal/kernels.metal` | All Metal kernels (compiled at runtime, Metal 4.0, `mathMode = .safe`). |
| `metal/v4emu.swift` | Harness: host prep, bit-exact verification vs the oracle, timing with the GPU lock and an int8 baseline. |
| `gen_vectors.sh`, `gen_vectors_xl.sh` | Deterministic test vectors (`vectors/`, git-ignored). |
| `run_verify.sh`, `run_bench.sh`, `run_bench_e.sh` | Produce the evidence files (`run_bench_e.sh`: kernel E after its rewrite). |
| `summarize.py` | Markdown tables from the bench / verify evidence. |
| `metal/microbench/` | `mma.swift` (peak simdgroup MMA / FMA rates), `gemm.swift` (plain fp32 simdgroup GEMM), `probe.swift` (cooperative-tensor element layout). |

## Operand families

| Family | How | Jackpot policy (Pearl `JackpotPolicy::evaluate`, B200, 4x64 tiles, k=4096) |
|---|---|---|
| `const` | X = +-64 random sign, unit scales (KB "constant magnitude +-X + delta=0.5 noise") | PASS |
| `uniform` | X uniform int8 in [-127,127], unit scales | PASS |
| `gauss` | X = round(24 N(0,1)), log-normal per-block BF16 scales | PASS |
| `adv_uniform` | uniform random E4M3 codes (NaN codes excluded) | not checked (adversarial) |
| `adv_edge` | per 32-group modes: +-0 heavy, +-448 with subnormals, far binades, cancelling pairs, constant, max binade, subnormal-only | not checked (adversarial) |

## Kernels

| Kernel | Hardware | Idea | Valid for |
|---|---|---|---|
| A | shader ALU | Direct B200 emulation: per 32-group anchor (pass 1 skipped when the carry provably dominates), truncated terms via exact fp32 product * 2^(25-E) -> int, int64 group sum, RZ to fp32. | any input |
| B | NA int8 x4 | Per (row, group) balanced int8 limbs of sig*2^(e-l1); 4 int8 `matmul2d` runs per group give the exact group dot; ALU proves "no term or carry truncated" from per-group metadata and applies RZ(C + dot); otherwise the SIMD-group recomputes that cell-group exactly (one product per lane). | any input |
| B3 | NA int8 x4 | B with a cheaper predicate and smaller code (more fallback on wide-binade data). | any input |
| C2 | NA int8 x4 | Grid-exact: if every element is a multiple of 0.5 and every group-boundary partial sum stays below 2^22, B200 never rounds, so C = exact integer dot. Limbs of 2a, int32 carry, exact per-group check, cooperative fallback. | any input (fast only on the 0.5 grid) |
| D | NA fp16 | Grid-exact on the fp16 NA with the carry kept in the fp32 accumulator; before each group a Cauchy-Schwarz bound proves every partial sum (any order) stays below 2^22. | any input (fast only on the 0.5 grid) |
| E | shader fp32 `simdgroup_matrix` | D's contract on the shader cores (the M1-M4 / M3 Ultra path); 2x2 8x8 blocks per SIMD-group, fragments loaded from device memory, guard every 32 K (`V4EMU_EW` groups). `E_NO_CHECK` = same kernel without the guard (reference fp32 GEMM). | any input (fast only on the 0.5 grid) |

Other variants in `kernels.metal` (B2, C with W>1, D2) were tried and are slower; they are kept
for the record and are bit-exact on the small sets.

## Commands

Needs `vendor/pearl-fp8`: run `scripts/fetch_vendor.sh --fp8` from the repo root first.

```bash
cd bench/v4_emulation
./gen_vectors.sh small       # 5 families at 256x256x4096 (+ Pearl reference outputs)
./gen_vectors.sh             # const/uniform/gauss 2048^2 x 4096, adversarial 1024^2 x 4096
./gen_vectors_xl.sh          # 4096^2 x 4096 (x5) + adversarial 2048^2 x 4096 (x2)
./run_verify.sh              # bit-exactness of every kernel on every set (no lock needed)
./run_bench.sh               # timing; takes /tmp/pmm-gpu-bench.lock per batch
./run_bench_e.sh             # kernel E + its no-check fp32 GEMM reference
python3 summarize.py         # tables

# single runs
oracle/target/release/v4-oracle analyze vectors/gauss_256 256 256 4096 4096   # where B200 is non-linear
cd metal && swiftc -O v4emu.swift -o v4emu
./v4emu verify ../vectors/const_2048 2048 2048 4096 a
V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1 ./v4emu bench ../vectors/const_2048 2048 2048 4096 5 d
```

Harness environment knobs: `V4EMU_TM`/`V4EMU_TN` (tile), `V4EMU_WG` (groups per window for C/D),
`V4EMU_DEFINES` (comma list of kernel `#define`s, e.g. `B_SKIP_EPI`, `E_NO_CHECK`), `V4EMU_COOL`
(seconds of idle before each timed batch; this M5 is a fanless MacBook Air), `V4EMU_ROUNDS`.

Every timed batch takes `mkdir /tmp/pmm-gpu-bench.lock` (retry every 15 s), times the int8
`matmul2d` baseline (the unmodified `bench/int8bench.swift` kernel, 4096^3, tile 128x64) three times
before and three times after the kernel, prints `sysctl vm.loadavg`, and removes the lock.
