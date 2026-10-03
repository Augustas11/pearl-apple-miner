# Host side: driving the kernel and the CPU/GPU split

## 1. What Pearl's CUDA path runs where
- **GPU:**
  - Commitment hashing: tensor_hash Merkle kernels (`pearl-gemm/csrc/tensor_hash/`).
  - Noise generation, noising A/B, NoisyGEMM + jackpot + PoW check, and denoise. `pearl-gemm/src/pearl_gemm/pearl_gemm_interface.py:61-548` wraps these as torch ops.
- **CPU:**
  - Job key and seed derivation (tiny).
  - Proof construction: Merkle multiproofs, Rust `mine.rs:381-391`. This runs after the host polls the pinned `HostSignalHeader` (`host_signal_header.hpp:22-47`, `pearl_gemm_kernel.h:267-273`).
  - Async CUDA event processing (`settings.py:38`).

## 2. Why OpenJarvis is slow (4.2 GOPS measured locally)
- Source: `vendor/openjarvis/src/openjarvis/mining/_mps_miner_loop_main.py`.
- Python runs a triple loop over 128-wide output tiles and rank chunks (`:82-104, 124-133`).
- Each chunk is a separate `torch.matmul` on int32-upcast operands on MPS (`:86-90`).
- Hash tiles are copied back to the CPU for NumPy XOR and BLAKE3 (`:46-47`).
- The cost is thousands of tiny dispatches plus a sync and a copy per chunk. The GPU is not the limit.

## 3. Host language options

| Option | Dispatch overhead | Pros | Cons |
|---|---|---|---|
| Swift + Metal (the bench already uses it: `bench/int8bench.swift`) | µs | First-class Metal 4 API, `MTLCompileOptions.languageVersion = .version4_0` | Needs FFI to Rust for proofs |
| ObjC++ / metal-cpp (llama.cpp style) | µs | C ABI is easy to call from Rust or Python | Verbose |
| Rust + objc2-metal (kvad PR #58) | µs | Links `zk-pow` / `pearl_blake3` directly for commitments and proofs | Maturity of objc2's Metal 4 tensor bindings is UNVERIFIED |
| Python + pyobjc / `torch.mps.compile_shader` / `mx.fast.metal_kernel` | ~10–100 µs per call (UNVERIFIED) | Fast prototyping | Fine **only** with one dispatch per GEMM. Whether the MLX/PyTorch JIT supports the MPP headers is UNVERIFIED |

Recommendation:
- A native host (Swift or Rust) owns the hot loop.
- Python stays only for gateway glue (`miner_base` talks to `pearl-gateway`).
- **One compute dispatch covers the whole m×n grid. Never dispatch per tile.**

## 4. Per-job pipeline (double-buffered)
The CPU prepares job i+1 while the GPU runs job i. Use shared-storage `MTLBuffer`s (unified memory, zero copy) with 2–3 rotating slots.

1. **CPU:** generate A and B in [-64, 64].
2. **CPU:** compute the commitment hashes `hash_a` and `hash_b`, then the seeds.
   - The commitment is a keyed BLAKE3 over chunk-padded data, equal to the Merkle root (`mine.rs:556-570`).
   - Seed derivation: `mine.rs:442-479`; Salted vs Legacy via `seed.rs:20-25`.
3. **GPU kernel 1:** noise generation. One BLAKE3 per 32 bytes: (m+n)·r/32 + 2·k/8 compressions.
4. **GPU kernel 2:** noising via gather-subtract (`pearl_noise.rs:35-43`). Writes A' and B' as int8. Memory-bound, O((m+n)k).
5. **GPU kernel 3:** fused GEMM + jackpot + PoW. Writes only a found-slot buffer.
6. **CPU:** in the completion handler, read the found slots and call Rust to build the proof (`mine.rs:111-122`).

Put steps 3–5 in one command buffer. Use `addCompletedHandler` instead of `waitUntilCompleted`, so the CPU never blocks the GPU.

### Cost scaling
- The GEMM is O(mnk). Hashing and noising are O((m+n)k).
- At m=n=k=8192, the GEMM is 1.1e12 ops (~58 ms at 19 TOPS), while the commitment hashes 128 MiB.
- CPU BLAKE3 throughput on M5 is UNVERIFIED. At ≥4 GB/s multithreaded it would hide under the GEMM. **Measure first.**

### Possible shortcuts (both UNVERIFIED as consensus-acceptable)
- Keep A fixed across jobs and change only B, which caches `hash_a`.
- Change only a few chunks of B and update the Merkle root incrementally.

## 5. BLAKE3 options on Apple
- **CPU:** the `blake3` Rust crate (NEON), or `pearl_blake3`, which is what zk-pow uses (`mine.rs:13, 383-386`).
- **GPU:** port `pearl-gemm/csrc/blake3/blake3.cuh`, or reuse the zkMetal / stwo-zig Metal shaders (see fused-epilogue-patterns §5).
- Noise-generation BLAKE3 belongs on the GPU: it is embarrassingly parallel, keyed, and single-block. The commitment can start on the CPU.

## 6. Found-block signaling
- Metal has no pinned host-mapped memory as such, but shared buffers are visible to the CPU after the command buffer completes.
- Use a device `atomic_uint` counter and a fixed slot array of `(offset_r, offset_c, transcript[16])`.
- Before submitting, the CPU recomputes the jackpot for the winning tile with the Rust reference. This also gives a free correctness check on every find.

## 7. Command-buffer sizing
- Keep each command buffer well under a second. The macOS GPU watchdog limit for long compute kernels is UNVERIFIED.
- If each GEMM job is short, batch several per command buffer.
