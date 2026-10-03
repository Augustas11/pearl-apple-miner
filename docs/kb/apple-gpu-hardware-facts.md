# Apple GPU facts relevant to int8 GEMM

Each fact is tagged P = published by Apple, M = measured (by whom), U = unknown / UNVERIFIED.

## M1–M4 (Apple7–Apple9 families)
- **Matrix ops:** `simdgroup_matrix` 8×8 in half/float, plus bfloat from MSL 3.1 (P, MSL spec). **No int8 MMA type** (P: absent from the spec's type list).
- **Metal 4 TensorOps:**
  - Supported on M1 and later (P, https://developer.apple.com/metal/whats-new/).
  - These chips have no Neural Accelerators, so TensorOps run on the shader cores. Apple says they fall back to "optimized shader implementations" (P, tech talk 111432).
  - On M4 Max, `matmul2d` is 1.05–1.21× simdgroup_matrix and accumulates in ≥ fp32 (M, Rigel arXiv 2606.12765).
  - int8 `matmul2d` speed and exactness on M1–M4: **U**.
- **M3 Ultra:**
  - 80-core GPU, 819 GB/s (P, Apple 2025 Mac Studio specs; the M5 Ultra release calls M5 Ultra's 1.2 TB/s "50% higher").
  - MLX fp32 GEMM 16.6–17.2 TOPS, bit-exact (M, local `bench/exact_test.py`).
- **philipturner/metal-flash-attention** reaches 83% ALU on M1 Max and 71–94% forward on M3/M4 (M, project README).

## M5 family (Apple10 family; GPU Neural Accelerator in every GPU core)

| Chip | GPU cores | Memory bandwidth | Source |
|---|---|---|---|
| M5 | 10 (8 or 10 per Wikipedia) | 153 GB/s | P https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ |
| M5 Pro | up to 20 | up to 307 GB/s | P https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/ |
| M5 Max | up to 40 | up to 614 GB/s | P (same) |
| M5 Ultra | 64 standard, up to 80 | 1.2 TB/s; "up to 4.5x the peak GPU compute for AI compared to M3 Ultra" | P https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/ and https://support.apple.com/en-us/128107 (MacRumors says 4.3x; Apple's page is authoritative) |
| M6 (announced Aug 2026) | 12, Neural Accelerator per core, ~30% more peak AI than M5 | 170 GB/s | P (same page) |

- **Peak AI vs M4:** "over 4x the peak GPU compute for AI" (P). **Apple publishes no TOPS figure for the GPU Neural Accelerators (U).**
- **Supported dtypes:**
  - fp16 and int8→int32 on the Neural Accelerators; no bf16 (M, tzakharko, Oct 2025, https://tzakharko.github.io/apple-neural-accelerators-benchmark/).
  - **Conflict:** Apple's tech talk says bfloat arrived in 26.1 and 4/8-bit integer tensors in 26.4 (P, https://developer.apple.com/videos/play/tech-talks/111432/). tzakharko tested an earlier OS. Treat this as: int8 needs **macOS ≥ 26.4**, bf16 needs ≥ 26.1.
  - fp8, 2-bit, and MX formats arrive in OS 27 (P, WWDC26-330). Rigel says fp8 is emulated on M4.
- **int8 rate:**
  - ~1900–2048 ops/clk/core, i.e. 13.4 TOPS on a 5-core A19 at ~1460 MHz (M, tzakharko).
  - Derived M5 10-core estimate: 28–33 TOPS at 1.5–1.6 GHz (**derived, U**: the clock is estimated).
  - Measured locally: 19 TOPS for a naive `matmul2d` at 4096³, roughly 58–67% of that estimate (M, `bench/int8bench.swift`).
- **fp16 rate:** ~1024 FLOP/clk/core (M, tzakharko). kvad measured 20–27 TFLOP/s dense f16 on M5 Pro, with drift (M, kvad PR #58).
- **Shape requirements:**
  - M or N must be a multiple of 16; K must be dynamic or a multiple of 16 (P, header `MPPTensorOpsMatMul2dImpl.h:4259,4272`).
  - tzakharko reports ≥32×32 as optimal and some shape combinations miscompile (M).
- **This machine:** Apple M5, 10-core GPU, Metal 4, 32 GB, macOS 26.5, metal compiler 32023.883 (local `sysctl` / `system_profiler` / `xcrun metal --version`).
- **Thermals:** a slowdown at 8192³ on M5 was seen in the first benchmarks (MLX fp32 1.8 TOPS; Metal int8 11.1 TOPS vs 19.0 at 4096³). The cause is **U**: thermal throttling or cache behaviour, not yet separated.
