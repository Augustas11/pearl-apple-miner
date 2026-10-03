# How serious open-source projects write Metal compute kernels

## 1. Two paths to the matrix hardware

### simdgroup_matrix (M1–M5)
- 8×8 fragments with types half, float, or bfloat (bfloat since MSL 3.1). There is no int8 or int32 MMA type. (MSL spec, https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- Each lane holds 2 elements: 1 row, 2 adjacent columns.
- MLX computes the lane's position in `get_coord`: `qid=lane/4; fm=(qid&4)+((lane/2)%4); fn=(qid&2)*2+(lane%2)*2` (https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h, lines 49-55).

### Metal 4 TensorOps / MPP `matmul2d` + `cooperative_tensor`
- **Availability.** Requires macOS 26+ and a deployment target ≥ 26.2 (`MPPTensorOpsAvailability.h:10`).
- **Only route to the M5 Neural Accelerators.** This is the only way to reach the M5 GPU Neural Accelerators (llama.cpp PR #27461, https://github.com/ggml-org/llama.cpp/pull/27461).
- **int8 support.** The supported dtypes include **int8×int8→int32** (`MPPTensorOpsMatMul2d.h:32`). The implementation accepts an int32 destination (`MPPTensorOpsMatMul2dImpl.h:2530-2536`).
- **Shape constraints.**
  - At least one of M or N must be a multiple of 16, and K must be dynamic or a multiple of 16 (`MPPTensorOpsMatMul2dImpl.h:4259, 4272`).
  - This constraint arrived in a later OS and broke llama.cpp's probes: https://github.com/ggml-org/llama.cpp/pull/21048.
- **Default mode hazard.** `matmul2d_descriptor` defaults to `mode::multiply`, not accumulate (`MPPTensorOpsMatMul2d.h:349-367`). K-looping into a cooperative tensor needs `mode::multiply_accumulate`, which is what MLX uses (nax.h lines 401-408, https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/nax.h).
- **Cooperative tensor layout is opaque.** It is "implementation defined" (`MPPTensorOpsMatMul2d.h:224, 247`).
  - Access is through `get_capacity`, `get_mask(i)`, `cT[i]`, `get_multidimensional_index(i)` (`:255-291`), plus iterators / `map_iterator` (WWDC26 session 330, https://developer.apple.com/videos/play/wwdc2026/330/).
- **Built-in reductions.** `reduce_rows` and `reduce_columns` support sum, max, and min only. There is **no XOR** (`MPPTensorOpsMatMul2d.h:342-347, 588-609`).
- **M1–M4 behaviour.** TensorOps "falls back to optimized shader implementations" there (Apple tech talk, https://developer.apple.com/videos/play/tech-talks/111432/).

(Header paths are under `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/`.)

## 2. Project survey

| Project | Tiling / threadgroup memory | MMA path | Epilogue fusion | int8 | Build | Correctness testing | Perf / % peak |
|---|---|---|---|---|---|---|---|
| **llama.cpp ggml-metal** | Tensor kernel split from the legacy kernel. Tiles NRA×NRB default to 64×128 (legacy 64×32). B is read straight from device memory, and threadgroup memory holds only the dequantized A (PR #20962, https://github.com/ggml-org/llama.cpp/pull/20962) | `simdgroup_matrix` in the legacy kernel; Metal 4 `matmul2d` when `has_tensor` (initial PR #16634, https://github.com/ggml-org/llama.cpp/pull/16634) | Per-element device writes via `cT.get_multidimensional_index` / `cT[i]` (#20962) | Quant blocks dequantized to f16, not native int8 MMA | **Runtime source compile** by default (`GGML_METAL_EMBED_LIBRARY`); probe kernels decide `has_tensor`. A precompiled metallib lacked the tensor flag and returned garbage, fixed with a guard (#27461). Probes needed language version 4.0 (#27461) | `test-backend-ops` MUL_MAT (9/11 vs 11/11 in #27461) | +26% geomean from #20962 on M5 Max; 2–3× prefill once the tensor API was enabled (#27461) |
| **MLX Steel GEMM** | Threadgroup tiles BM×BN×BK with `simdgroup_matrix` 8×8 fragments (mma.h) | simdgroup_matrix; NAX path for M5 (PR #2772, https://github.com/ml-explore/mlx/pull/2772) | In-register transforms (`transforms.h`) | NAX supports int types via MPP; Steel is float-only | metallib built at package time. NAX needs `-mmacosx-version-min=26.2` / `MACOSX_DEPLOYMENT_TARGET=26.2` (PR #3622, https://github.com/ml-explore/mlx/pull/3622). Also "Fixing cooperative tensor build issues on Mac OS 27" (#4594) | Python test suite | Local: MLX fp32 ~8 TOPS on M5, ~17 TOPS on M3 Ultra (repo README) |
| **MLX NAX (M5)** | 16×16 fragments. **Each lane holds 2 rows (jump 8) × 4 contiguous cols** (`kElemRows=2, kElemCols=4, kElemRowsJump=8`; `get_coord: qid=lane>>2; fm=(qid&4)\|((lane>>1)&3); fn=((qid&2)\|(lane&1))*4`, nax.h ~lines 27-60) | `matmul2d<desc(16,32,16,…, relaxed=true, multiply_accumulate), execution_simdgroup>` with **cooperative tensors as both inputs and outputs**, copied element-wise from MLX's own fragment registers (nax.h 393-457) | Registers stay MLX-owned, so any epilogue works | n/a | As above | — | — |
| **mx.fast.metal_kernel** | User code | Any | User code | Any | JIT from a source string, cached per build. Supports `header`, `template`, `init_value`, `atomic_outputs` (https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) | — | The docs do not mention MPP / tensor headers (UNVERIFIED whether includes work) |
| **MLX C++ Primitive extension** | Full control | Any | Any | Any | CMake + metallib (PR #4004 "Metal extension build support", open) | — | — |
| **philipturner/metal-flash-attention** | Register blocking with deliberate spilling; a third blocking dimension along D (https://github.com/philipturner/metal-flash-attention) | simdgroup_matrix + custom | Fused attention | None | **All JIT-compiled at runtime from Swift-generated source** | Swift test suite (~10 s) | 83% ALU on M1 Max; 71–94% forward on M3/M4 |
| **PyTorch MPS custom ops** | User | Any | User | — | `torch.mps.compile_shader(src)` returns a callable library (https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py) | — | Each call is a separate dispatch from Python |
| **BaseRT (arXiv 2607.19438)** | Each threadgroup owns an output tile. Per K step, `matmul2d` accumulates into a register-resident cooperative tensor. Operands stream from device memory: "threadgroup-memory staging is not necessary for peak GEMM" (https://arxiv.org/html/2607.19438v1) | Metal 4 tensor | Fused MoE gate/up | 4/8-bit quant | `-std=metal4.0`, C++ runtime, SIMD fallback before M5 | Not described | Up to 6.4× llama.cpp prefill on M5 Pro; no %-of-peak given |
| **bisand/kvad PR #58** | K step 32. Unpacks a 64×32 Q8_0 slab to f16 in threadgroup memory; 64×64 cooperative-tensor tile; double-buffered register prefetch (https://github.com/bisand/kvad/pull/58) | `matmul2d` f16 | f32 out | Q8_0 dequantized to f16 (not int8 MMA) | Rust (objc2-metal); Apple10 family gate; `KVAD_GPU_MPP=0` off-switch | NaN-initialized outputs, `sum_all().is_finite()` check, deliberate-breakage test | M5 Pro dense f16 at 20–27 TFLOP/s. Notes large run-to-run drift, so it alternates rounds |
| **arcusis/Zerm PR #382** | — | Via llama.cpp / whisper.cpp | — | — | Gates the tensor path on `supportsFamily(.apple10)` (https://github.com/arcusis/Zerm/pull/382) | Identical transcripts | Not verified on M5 |
| **Rigel (arXiv 2606.12765)** | Microbenchmark of `matmul2d` on M4 Max (https://arxiv.org/abs/2606.12765) | Finds that on M4, `matmul2d` runs on shader cores at only 1.05–1.21× simdgroup_matrix, accumulates in ≥fp32, and emulates fp8 | — | — | Metal 4.1 needs Xcode 27 **and** macOS 27; 4.1 binaries refuse to load on 26.5 | Checksum-gated harness | Reconstructs the 8×8 fragment layout as "lane owns two vertically adjacent rows of one column". **This conflicts with MLX's assumed layout, and was measured on M4, not M5 int32.** |

## 3. Takeaways for the Pearl kernel
1. **Use `matmul2d` int8×int8→int32 on M5 (Apple10 family).**
   - The local naive kernel measured 19 TOPS at 4096³ and was bit-exact at K=8192 (`bench/int8bench.swift`).
   - It uses `tensor_inline` + `slice` and works on M5 / macOS 26.5. Rigel's claim that slicing `tensor_inline` doesn't offset reads (M4, Metal 4.1) did not reproduce here.
2. **Loop over K explicitly** in chunks that divide r, with `multiply_accumulate` and `relaxed_precision=false`.
   - What relaxed mode does to ints is UNVERIFIED. MLX uses `relaxed=true` for floats.
3. **No split-K.** Each threadgroup/simdgroup owns the full K range of its tile.
4. **Derive the per-lane coordinate set at startup.** Use a probe kernel that writes `get_multidimensional_index(i)` for every i, instead of hard-coding MLX's layout. The layout is officially opaque, and MLX and Rigel disagree.
5. **Ship a precompiled metallib plus a runtime-compiled fallback.** Build the metallib with `-std=metal4.0 -mmacosx-version-min=26.2`. The llama.cpp metallib/tensor-flag bug shows the two must be kept consistent.
6. **Gate on `MTLGPUFamilyApple10`** (as Zerm and kvad do).
7. **Keep buffers under 2 GB, or build tensors from computed pointers.** Large-offset slices are buggy: a slice ≥2 GB from the tensor base came out 4 GB off on M5 / macOS 26.6.2 (llama.cpp PR #28748, open).
8. **Measure performance carefully.**
   - Alternate A/B rounds (kvad) and report the median.
   - Count 2·m·n·k ops.
   - Check exactness against an int64 CPU oracle with full-range operands.
   - Profile in Metal System Trace, which shows Neural Accelerator utilization (Apple tech talk).
   - Traverse threadgroups in Morton order: Apple's demo cut a 4K² GEMM from 0.5 s to 0.33 s (tech talk).
