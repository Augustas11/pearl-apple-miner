# Pearl-format models on MLX: inference first, then mining while serving (T-mlx-models)

Date 2026-10-02. Research only: no weights were downloaded and nothing ran on the M3 Ultra.
- Model facts come from the HF API, `config.json`, the safetensors index, and safetensors **headers** fetched via HTTP Range.
- Weight value ranges come from three 1 MiB range reads.
- `vendor/pearl/` = Pearl HEAD 7039e66f. `vendor/pearl-fp8/` = branch fp8-scheme HEAD 25695462.
- UNVERIFIED = inferred or estimated, not measured.

## 0. Answer first
- **Path A, inference only: feasible and cheap.**
  - int7 weights map **losslessly** into MLX affine 8-bit with group 128.
  - FP8 block weights can be dequantized with `mx.from_fp8` and requantized to affine 8-bit.
  - Effort: about 1–2 weeks for all four models; about 3–5 days for Llama-3.1-8B alone.
  - **For plain inference, Pearl weights add nothing.** mlx-community already publishes 4/8-bit conversions of all four base models, and the Pearl variants score slightly lower (Qwen MMLU-Pro 77.33 vs 77.88). Pearl weights only matter as the base for Path B.
- **Path B, mining while serving: hard and time-boxed.**
  - It is technically possible under cert v3.
  - It needs a new serving variant of K3: the weight as B, A per call, noising inside the kernel, exact denoise, and GPU BLAKE3. It also needs MLX/Swift plumbing.
  - Rough effort is 2.5–4 engineer-months, on top of K3-SG, which doesn't exist yet.
  - **Mainnet MoE certificates have been invalid since height 91630**, so Qwen's experts cannot be mined.
  - The FP8 cert-v4 branch replaces the whole int7 scheme. When v4 activates, the Path B investment is lost.
- **Mining yield while serving comes almost entirely from prefill.** Decode at m=1 mines about 0.7 TOPS-eq on the 8B model. Pure mining on M3 Ultra targets ~17 TOPS. (Both are UNVERIFIED estimates; see §3.3.)

## 1. The four models: exact format

### 1.1 Common to all four
- **Quantization config.**
  - `quant_method: "pearl"`, `format: "mixed-precision"`, compressed-tensors `version` 0.13–0.15.
  - `transform_config: {}` and `kv_cache_scheme: null`.
  - Source: each repo's `config.json`, e.g. https://huggingface.co/pearl-ai/Llama-3.1-8B-Instruct-pearl/raw/main/config.json
- **Two config groups.**
  - **int7 W7A7 ("mining").**
    - Weights: 7 bits, per-channel, symmetric, static. Stored as I8 `weight` plus BF16 `weight_scale [n,1]`.
    - Activations: 7 bits, per-token, dynamic, symmetric.
  - **FP8 ("non-mining").**
    - Weights: F8_E4M3 in 128×128 blocks, plus BF16 `weight_scale [ceil(n/128), ceil(k/128)]`.
    - Activations: FP8 per group of 128, dynamic.
  - Example from the safetensors headers: `model.layers.0.mlp.down_proj.weight F8_E4M3 [4096,14336]`, `weight_scale BF16 [32,112]`.
- **No Hadamard transform in these checkpoints.**
  - vLLM creates `hadamard_block_size` zero-initialised (`vendor/pearl/miner/vllm-miner/src/vllm_miner/vllm_scheme.py:131-136`), and 0 means no transform (`quantization_operators.py:7-8`).
  - No `hadamard*` tensor appears in any index or header.
  - The CuTe quantizer's default of 16 (`pearl-gemm/src/pearl_gemm/quantization/hadamard.py:525`) is not used on this path.
- **SmoothQuant.**
  - A BF16 `smooth_quant_scale [k]` is **multiplied into the activations** before the per-token amax (`hadamard.py:106-125`). So the weights were quantized pre-divided by it.
  - Where it appears:
    - o_proj in every layer of all four models (counts: 8B 32, 70B 80, Gemma 60, Qwen 48);
    - Qwen only: also q_proj (48) and one shared experts.0.gate_proj per layer.
  - vLLM loads the shared MoE scale for every expert's gate/up (`pearl_moe_method.py:38-70`).
  - For the fused q/k/v projection, vLLM appears to use q's scale for the whole fused input (inferred from `vllm_scheme.py:123-129`). UNVERIFIED; confirm with a logit-parity test.
- **Activation quantization (mining path).**
  - `scale = amax(x·s)/63` and `q = clamp(rint(x·s·(63/amax)), ±63)`, with an approximate reciprocal (`hadamard.py:443-456`). max_val is 63 (`quantization_operators.py:5, 72-79`).
  - The non-mining 8-bit path uses 127 (`:82-89`), but none of these checkpoints has an int8 group.
- **Observed weight range.**
  - Three samples (8B o_proj, 8B gate/up, Qwen o_proj) show |w| ≤ 62, with per-row maxima of 60–62.
  - That is inside the protocol's signal range [-64, 64].
  - The full-tensor range is UNVERIFIED.

### 1.2 Per model
Sizes come from HF API `safetensors.parameters` and file sizes (`https://huggingface.co/api/models/pearl-ai/<repo>?blobs=true`).

| Model (sha) | Arch | Params I8 / F8 / BF16 | Repo size | int7 (mineable) GEMMs | FP8 GEMMs | Ignored (BF16) | Card eval |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct-pearl (5dcb348a) | LlamaForCausalLM: 32 layers, h4096, ffn 14336, GQA 32/8 | 4.70G / 2.28G / 1.05G | 9.10 GB (2 shards) | o_proj (all), gate/up (all), q/k/v layers 16–31 | down (all), q/k/v layers 0–15 | lm_head, embed, norms | none |
| Llama-3.3-70B-Instruct-pearl (6cc401ca) | Llama: 80 layers, h8192, ffn 28672, GQA 64/8 | 46.31G / 22.15G / 2.11G | 72.7 GB (15 shards) | o (all), gate/up (all), q/k/v layers 40–79 | down (all), q/k/v layers 0–39 | lm_head | MMLU 0.8190 (PP=4) vs Meta 0.8198 |
| Gemma-4-31B-it-pearl (f1dfba68) | Gemma4ForConditionalGeneration: 60 text layers (50 sliding + 10 full), h5376, ffn 21504, `attention_k_eq_v` (full layers have no v_proj), tied embeddings, vision tower | 16.96G / 12.33G / 1.99G | 33.3 GB (1 file) | o_proj (all), gate/up (all) | q/k/v (all), down (all) | vision tower + projector (192 ignore entries) | GPQA 77.37 vs 77.27; MMLU 90.56 vs 90.93 |
| Qwen3-30B-A3B-Instruct-2507-pearl (751fd105) | Qwen3MoeForCausalLM: 48 layers, h2048, 128 experts, top-8, expert ffn 768 | 20.23G / 9.66G / 0.65G | 31.2 GB (7 shards) | attention q/k/v/o (all), experts gate/up | experts down | router `mlp.gate` (48), lm_head | MMLU-Pro 77.33 vs 77.88 |

Notes:
- **70B q/k/v split:** the group_0 regex `([0-3]?[0-9])` covers q/k/v in layers 0–39.
- **Gemma group_1:** its target is `Linear`, meaning everything not in group_0 and not ignored.
- **Licenses** (HF `cardData`): llama3.1, llama3.3, gemma, apache-2.0.
- **No community MLX or GGUF conversions exist.**
  - HF `filter=base_model:pearl-ai/<repo>` returns only `Avifenesh/pearl-eagle31-scratch-200k-seq4096`, which is an EAGLE draft head, not a conversion.
  - mlx-community's "Pearl-7B" is an unrelated 2024 Mistral merge.

### 1.3 What mining actually touches (vLLM miner)
- **Which layers mine.** Only W7A7 per-token symmetric dynamic layers (`vllm_config.py:67-84`).
  - FP8 layers fall through to vLLM's stock FP8 path (`:205-206`).
  - MoE gets `PearlMoEMethod` only when gate/up are int7 and down is FP8-block (`:142-166`).
- **Per-call gate.** A noisy GEMM runs only if `m ≥ 1024 and n ≥ 256 and k ≥ 1024` (`config.yaml:9-12`, `config.py:36-44`). Otherwise the layer does int7 quant + a plain GEMM (`vllm_kernels.py:202-224`).
  - **So decode never mines unless the batch has ≥ 1024 tokens.**
- **Operands.** A = int7 activations (m×k). B = the int7 weight (n×k) as stored (`vllm_kernels.py:204-215`). That stored layout is exactly the protocol's committed "Bᵀ row-major".
- **MoE.** Only GEMM1 (gate/up) mines, per expert (`pearl_moe_experts.py:188-300`). GEMM2 (down, FP8) does not (`:241-243`).
- **MoE is dead on mainnet.**
  - `DenseOnlyForkHeight: 91630` (`vendor/pearl/node/chaincfg/params.go:341, 367`).
  - `CheckCertificateRules` rejects MoE certificates after that height (`node/blockchain/validate.go:542-546`).
  - So **on mainnet only Qwen's attention GEMMs are usefully mineable**: about 0.91 G MAC/token, vs 1.21 G for the expert gate/up that can no longer be mined.
- **Mined MAC per token** (dense, mainnet-valid; equals the int7 parameter count, since each weight is used once per token):

  | Model | Mined MAC/token |
  |---|---|
  | 8B | 4.70 G |
  | 70B | 46.3 G |
  | Gemma | 16.96 G |
  | Qwen | ~0.91 G |

- **Protocol shapes with r=128.** k must be in [2048, 65536] with k%64 == 0. Every mined k passes:

  | Model | Mined k values |
  |---|---|
  | 8B | 4096 |
  | 70B | 8192 |
  | Gemma | 5376, 8192, 16384 |
  | Qwen | 2048, 4096 |

## 2. Path A: inference only on MLX

### 2.1 What MLX supports today
- **Affine quantization.**
  - `mx.quantize` affine supports bits 2/3/4/5/6/8 with group sizes 32/64/128. Scale and bias are stored in the input dtype; dequant is `w = s·q + β` with q unsigned.
  - Source: https://github.com/ml-explore/mlx/blob/main/python/src/ops.cpp, docstring ~L4808-4870; also local mlx 0.31.2.
  - **There is no 7-bit mode.**
- **Other quantized formats.**
  - mxfp4, mxfp8 (E4M3 with an e8m0 power-of-two scale, group 32) and nvfp4.
  - `QQLinear` (activation quantization) exists only for nvfp4 and mxfp8 (local `mlx_lm/utils.py:393-402`).
  - mlx v0.32.3 was released 2026-09-29.
- **FP8.**
  - `mx.from_fp8` and `mx.to_fp8` exist, but there is no FP8 matmul.
  - mlx-lm dequantizes FP8-block checkpoints in `sanitize` (e.g. `mlx_lm/models/deepseek_v3.py:381-395`).
  - mlx-lm `main` refuses compressed-tensors `float-quantized` with "dequantize to bf16 before converting" (`mlx_lm/utils.py`, `_compressed_tensors_quantization` ~L308-327).
  - `quant_method "pearl"` matches no branch (~L511-546). The model would load unquantized and strict loading would likely fail on the extra keys and dtypes (UNVERIFIED; not run).
- **Architectures.**
  - mlx-lm (local 0.31.3) has `llama.py`, `qwen3_moe.py` (stacks experts into `switch_mlp` in `sanitize`, L232-243) and `gemma4_text.py` (handles `attention_k_eq_v`, L40, L197-208).
  - mlx-swift-lm 3.32.3 has `Libraries/MLXLLM/Models/{Llama,Qwen3MoE,Gemma4Text}.swift`; `Gemma4Text.swift` handles `attention_k_eq_v` at L65/103/148.
- **Custom-kernel hooks.**
  - Python: `mx.fast.metal_kernel(...)` (https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
  - C++ primitives (https://ml-explore.github.io/mlx/build/html/dev/extensions.html).
  - Swift: `MLXFast.metalKernel` (used in `mlx-swift-lm/Libraries/MLXLMCommon/ParoQuant/PairwiseRotation.swift:185`).
- **Swift precedent for a custom loader: ParoQuant.**
  - It detects `quant_method == "paroquant"` (`ParoQuantLoader.swift:9-55`).
  - `RotateQuantizedLinear: QuantizedLinear` overrides `callAsFunction` to run a Metal kernel on x before `quantizedMM` (`RotateQuantizedLinear.swift:10-18, 72-90`).
  - Pearl's smooth scale is the same shape of problem.

### 2.2 Options
| Option | int7 layers | FP8 layers | Smooth scale | Runtime code | Fidelity vs Pearl weights |
|---|---|---|---|---|---|
| **A0** use mlx-community base models | — | — | — | none | Not Pearl; base-model quality (`Meta-Llama-3.1-8B-Instruct-8bit`, `Llama-3.3-70B-Instruct-{4,8}bit`, `Qwen3-30B-A3B-Instruct-2507-{4,8}bit`, `gemma-4-31b-it-{4,8}bit` exist) |
| **A1 "std"** standard MLX checkpoint | dequant `int7·s`, fold s into the weight columns (`W·diag(smooth)`), requantize affine 8-bit g64 (or 4-bit) | `from_fp8` × block scale → affine 8-bit g64 | folded into W | **none**: stock mlx-lm / mlx-swift-lm (per-layer quant map, `Load.swift:370-394`) | ≈ lossless at 8-bit; requant loss at 4-bit |
| **A2 "exact"** | **lossless**: q = int7+64 packed as affine 8-bit g128, scales = s_row, biases = −64·s_row (exact in bf16) | as A1, or bf16 | runtime `x·s`: folded into the preceding RMSNorm for qkv / gate_up / experts; an elementwise multiply for o_proj | small `PearlQuantizedLinear` (Python + Swift, ParoQuant pattern) | weights bit-identical to Pearl's int7 grid; activations stay bf16 (W7A16), likely ≥ W7A7 quality (UNVERIFIED) |
| A3 native FP8-block / int7 Metal kernels | custom qmv | custom E4M3 LUT qmv | runtime | heavy | Saves only ~3% bytes vs A2. Not worth it now |

Notes:
- **mxfp8 is a poor fit for Pearl's FP8 weights.** It needs power-of-two scales, so it would re-round E4M3 (~3% relative error; UNVERIFIED estimate).
- **A2 must pack the values directly.** `mx.quantize` picks its own per-group min/max and would not reproduce the int7 grid.

### 2.3 Size and speed (decode is memory-bandwidth-bound)
Method:
- bytes/token = Σ active linear weights + lm_head.
- Bits per weight: affine-8 g128 = 8.25, affine-8 g64 = 8.5, 4-bit g64 = 4.5.
- M3 Ultra bandwidth is 819 GB/s. Assume ≈70% efficiency, since Qwen3.6-27B-4bit measured 37 tok/s on the M3 Ultra.
- M5 has 153 GB/s, so scale the M3 Ultra numbers by ×0.19.
- **All rows are UNVERIFIED estimates.**

| Model | MLX A2/A1 8-bit resident | Decode bytes/token (8-bit / 4-bit) | M3 Ultra est. tok/s (8-bit / 4-bit) |
|---|---|---|---|
| 8B | ~8.4 GB | 7.8 / 4.2 GB | ~73 / ~136 |
| 70B | ~73.5 GB | 72.4 / 39.1 GB | ~8 / ~15 |
| Gemma 31B | ~33 GB (+~1 GB vision bf16) | 32.1 / 17.3 GB | ~18 / ~33 |
| Qwen 30B-A3B | ~32.4 GB | ~3.2 / ~1.8 GB active | bandwidth ceiling ~180 at 8-bit, but MLX MoE is overhead-bound (a 4-bit A3B measured 86 tok/s at 1 row on the M3 Ultra) |

Rule of thumb: Pearl at 8-bit moves about 2× the bytes per token of 4-bit models, so expect about ½ the decode tok/s.

### 2.4 Steps for the converter (`pearl2mlx.py`, modes A1 and A2)
1. Stream shards one at a time with `mx.load` / safetensors, on CPU.
2. Convert each tensor:
   - **I8 + scale → A2:** pack (q+64) into uint32, 4 values per word, low bits first. Scales = s, biases = −64·s, group 128. (For A1: dequantize, then requantize.)
   - **F8 + block scale:** `from_fp8` × `scale[i//128, j//128]` → affine 8-bit g64.
   - **Smooth scale:** fold into the RMSNorm for qkv / gate_up / experts; keep it as a param for o_proj.
3. Model-specific handling:
   - Qwen: stack the experts into `switch_mlp.{gate,up,down}_proj`.
   - Gemma: keep the `model.language_model.*` names and leave the vision tower in bf16.
4. Write `config.json` with `"quantization": {"group_size":…, "bits":8, "mode":"affine", <per-layer overrides>}` and drop `quantization_config`.
5. Unit-test on synthetic tensors (§4.1), then convert the 8B model on the M3 Ultra (§4.3).

Effort:

| Item | Estimate |
|---|---|
| A1 for 8B, incl. eval | 3–5 days |
| Qwen MoE stacking | +1–2 days |
| Gemma VLM naming / k_eq_v | +1–2 days |
| 70B | +1 day |
| A2 custom layer | +3–5 days |

## 3. Path B: mining while serving

### 3.1 What one mining Linear does per forward (vLLM reference, `gemm_operators.py:68-264`)
1. Quantize activations: `x_q, x_s = quant_7bit(x, smooth)`.
2. Fetch the job, build the config (k, r=128, patterns) and the adjusted target.
3. Compute `job_key`.
4. **Tensor-hash A and B on the GPU on every call** (`:124-138`).
5. Compute the salted v3 commitment seeds (`:141-150`).
6. Generate the noise (`:267-333`).
7. Run one `noisy_gemm`: noise → int GEMM with jackpot fold → denoise → scale → C (`:188-217`).
8. Run the async status/proof step (`:219-234`).

Properties of this design:
- Every eligible call is an independent mining instance.
- There is no cross-call batching.
- There is **no caching of the weight commitment**: B is re-hashed and re-noised on every call.
- MoE does this per expert (`pearl_moe_experts.py:262-300`).
- The model cards use vLLM `--enforce-eager`.

### 3.2 What must be bit-exact and what is free
- **Bit-exact (required by the protocol):**
  - the committed int8 A and B bytes;
  - seeds and noise bytes;
  - A' and B';
  - the int32 cumulative accumulators at each 128-K boundary;
  - the tile set, the XOR fold with rotl13 into 16 slots;
  - the keyed BLAKE3;
  - the U256 compare.
- **Free (implementation choices):**
  - the activation-quant arithmetic (CUDA uses `rcp_approx`; you commit to whatever int8 A you produce, as long as it stays in [-64,64]);
  - how the smooth scale is applied;
  - the denoise method and C precision;
  - tiles and the hash pattern (any valid PeriodicPattern);
  - scheduling.
- **The weights don't need to be Pearl's.** The protocol commits to arbitrary int matrices; Pearl checkpoints are just a quality-validated W7A7 recipe.
- **Retention:** A (m×k int8) must be kept per mined call until its scan finishes, because a find needs it for the Merkle proof.

### 3.3 Decode vs prefill
- Yield ∝ Σ 2·m·n·k, because difficulty is normalized by work.
- **Prefill** (m = chunk size; mlx-lm `prefill_step_size` defaults to 2048, `mlx_lm/generate.py:316`) is compute-bound.
  - 8B on M3 Ultra: ~1,000 prefill tok/s × 9.4 GOP ≈ 9–10 TOPS-eq, about 55–60% of pure-mining K3-SG (UNVERIFIED).
- **Decode** at m=1: 73 tok/s × 9.4 GOP ≈ 0.7 TOPS-eq on 8B. Batch 8 gives roughly 8× that.
- **Pad-to-fill idea (UNVERIFIED).**
  - Decode is bandwidth-bound, so the ALUs sit idle.
  - Padding A with extra committed rows could fill that idle compute while each weight byte is still read only once. That is pure mining riding on weight streaming.
  - It needs a small-m tile with a valid hash pattern (e.g. h=2) and a probe of the latency cost.
- **Comparison point:** pure mining on random matrices already reaches ~17 TOPS on an idle M3 Ultra.
- **So Path B's real value is:**
  - mining during busy prefill without stopping serving;
  - the useful-work / PIP narrative.

### 3.4 How K3 must change for real weights
K3 today (SPEC §5.1):
- random A and B;
- A fixed per template, B fresh per job (`pmkcore/src/lib.rs:219-299`);
- host-side M%128, N%64, K%128;
- **no C output**.

Serving inverts this:

| # | Change | Why / how |
|---|---|---|
| 1 | B = weight W (n×k, fixed). Commit W **once per (job_key, layer)**, and cache `raw_root_b` + `b_noise_seed` | `b_seed = blake3(job_key ‖ salted root_b)` depends only on job and layer (`lib.rs:250-259`), so vLLM's per-call re-hash is waste. The per-job cost is hashing every mined weight: 4.7 GB for 8B, 46 GB for 70B (throughput UNVERIFIED). Swap jobs only after the new hashes are ready |
| 2 | Commit A per call **on the GPU** (port `pearl-gemm/csrc/tensor_hash`), plus K1 seeds/noise | `a_seed = blake3(b_seed ‖ salted root_a)` gates the noise, so it sits on the critical path. CPU hashing would force a GPU↔CPU sync per layer per forward |
| 3 | Compute B' = W + E_BL·E_BR **inside the kernel** from cached E_BR (n×128 int8) and the per-l (i0,i1) | Avoids a second int8 weight copy (+4.7 / 46 GB). A' (m×k) can be materialized |
| 4 | Read the weight from MLX packed affine-8 (subtract 64 on load) for both the GEMM and the hash | No duplicate int8 copy. The hash kernel must emit the exact padded row-major int8 byte stream |
| 5 | **C output with exact integer denoise:** `AB = A'B' − (A'·E_BL)·E_BR − E_AL·(E_AR·B)` in int32, then × x_s × w_s → bf16 | Exact, unlike CUDA's fp16 ×2^12 denoise, so outputs with mining on and off are bit-identical. `E_AR·B` (r×n) changes per call; accumulate it while streaming B tiles. **Decode alternative:** compute A·B and A'·B' from the same B tile load (2× MACs, which is free when bandwidth-bound) |
| 6 | Small-m tiles with a valid PeriodicPattern for m ∈ {1..16}, plus large-m prefill tiles; pad m to the row period | The pattern must have h·w ≥ 32 with h and w even |
| 7 | M3 Ultra uses **K3-SG** (fp32 simdgroup_matrix, fresh accumulator per 128-chunk; exact since 128·127² < 2^24; SPEC §5.3), still research-first. M5 uses K3-NA (R6 V6 fold) | The M3 Ultra has no Neural Accelerators |
| 8 | pmkcore API: cached `commit_weight(job, layer, W)` and `commit_activations(call, A)`; jobs keyed by (job_key, layer, m) | Today `template_init` commits A and `commit_job` commits a caller-supplied B per job |
| 9 | Found slots + proof: keep A per call and a reference to the resident W; run the verifier gate before submit (R-A3) | Same as SPEC §4.1 |
| 10 | Enforce the dense-only rule: never emit MoE certificates | `validate.go:542-546` |

### 3.5 MLX / Swift integration
- **Python prototype:** a `PearlMiningLinear` chaining `mx.fast.metal_kernel` calls: act-quant → hash-A → K1 noise → fused noisy GEMM + denoise → C.
  - Found counters use `atomic_outputs` and are read after `mx.async_eval`, so token generation isn't blocked.
- **Swift (`mlx-swift-lm`):** same structure as ParoQuant:
  - a `PearlQuantizedLinear: QuantizedLinear` override;
  - `MLXFast.metalKernel` for the kernels;
  - pmkcore via its C ABI;
  - a gateway job feed over UDS.
- **Open questions (UNVERIFIED):**
  - Do custom kernels coexist with compiled decode?
  - Can Metal-4 `matmul2d` be used inside `metal_kernel` source? (Needed for M5.)
  - How much per-forward launch overhead do ~3 mined GEMMs × layers add?
- **Effort (UNVERIFIED, after K3-SG exists):**

  | Item | Estimate |
  |---|---|
  | Kernels | 4–8 weeks |
  | MLX Python prototype | 2–3 weeks |
  | Swift | 2–4 weeks |
  | Gateway / job plumbing | 1–2 weeks |
  | **Total** | **≈ 2.5–4 months** |

## 4. Testing plan (Mac Studio M3 Ultra, 256 GB, macOS 26.4.1)

All end-to-end model work runs on the M3 Ultra (SPEC §11, T-mlx-models), on an otherwise idle machine with other GPU workloads paused.

### 4.1 Off-box (any Mac or CPU; no dedicated window)
1. **Config and header audit.** Done in this doc. Pin the shas from §1.2.
2. **Converter unit tests on tiny synthetic tensors.**
   - A2 pack/unpack: dequant == int7·s **bit-exact** in fp32.
   - FP8 block dequant vs a numpy reference, including ragged edges (14336/128 = 112; Qwen down 768 → 6 blocks).
   - Smooth-fold equivalence: `norm(x)·(g·s) == norm(x)·g·s` within bf16 tolerance.
   - Qwen expert stacking order, Gemma name mapping, and k_eq_v layers.
3. **Tiny synthetic Pearl checkpoint** (2-layer Llama, h=256, in exact Pearl format):
   - convert it;
   - load it with `mlx_lm.load` (strict);
   - compare logits against `pearl_ref.py`, a numpy W7 dequant plus optional A7 per-token emulation.
   - Target: max |Δlogit| < 1e-2 in bf16.
4. **Path B kernels on synthetic data with real shapes** (n×k from §1.3; m ∈ {1, 2, 8, 16, 128, 2048}):
   - transcripts, finds and C bit-exact against the int64 oracle (`bench/f1_k3/oracle.py`, extended for C/denoise);
   - cross-check with `pearl_mining` `mine()` + `verify_plain_proof_for_cert_version(3)`.
   - M5 can test the K3-NA variant. The K3-SG design/probe needs a dedicated M3 Ultra window (SPEC §5.3).

### 4.2 Memory budget (check at the start of each window, read-only)
- **Preflight:**
  - `sysctl hw.memsize`
  - `sysctl iogpu.wired_limit_mb` (read it; never change it)
  - `vm_stat`
  - `df -h ~`
  - RSS of any other resident GPU workload
- **If another model server stays resident during a soft pause (UNVERIFIED),** budget for it: up to ~40 GB (A3B 4-bit measured ~38.5 GB RSS) plus ~15 GB of OS, leaving **~200 GB usable**.

| Load | Peak RAM | Fits with a ~40 GB model server resident? |
|---|---|---|
| Convert any model (shard streaming, CPU) | ≤ ~15 GB | yes |
| 8B Pearl-MLX + 8k KV | ~10 GB | yes |
| 8B bf16 reference (mlx-community bf16) | ~16 GB | yes (load sequentially) |
| Qwen / Gemma Pearl-MLX | ~33–35 GB | yes |
| 70B Pearl-MLX + 8k KV (~2.7 GB bf16) | ~77–80 GB | yes. Skip the 141 GB bf16 reference and compare against mlx-community 8-bit |

**Disk:** downloads ≈ 146 GB, conversions ≈ 147 GB, plus references. Needs **~320 GB free**.

### 4.3 Minimal first experiment: Llama-3.1-8B-Instruct-pearl

**Before every window:** run on an otherwise idle machine, pause other GPU workloads, and confirm none are running (for example `pgrep -fl llama-server` returns nothing).

**Downloads.** `hf download` (network and disk only) may run outside a window at low priority; otherwise it uses window W0.
- Pin `--revision 5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0`.
- Also fetch `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`, `-4bit` and `-bf16`.

**W1 (≤ 45 min), `w1`:**
1. Preflight (§4.2), logged.
2. Convert both ways on CPU, ~5–10 min each (UNVERIFIED):
   - `python pearl2mlx.py --mode exact --out ~/pearl2mlx-out/l8-exact`
   - `python pearl2mlx.py --mode std8 --out ~/pearl2mlx-out/l8-std8`
3. `mlx_lm.generate` smoke test on both conversions (chat template, 64 tokens).
4. `mlx_lm.perplexity --model <m> --sequence-length 1024 --num-samples 128` for exact, std8, mlx-community 8bit and bf16 (~3 min each, UNVERIFIED).
5. `mlx_lm.benchmark --model <m> --prompt-tokens 512 --generation-tokens 128 --num-trials 3` for exact, mlx-community 8bit and 4bit.

**W2 (≤ 45 min):**
- Logit parity vs the CPU `pearl_ref.py` on 32 prompts: top-1 agreement and mean KL, with and without A7 emulation.
- `mlx_lm.evaluate --tasks mmlu --limit 200` for exact vs bf16.

**Pass criteria:**
- exact-mode weights bit-identical to int7·s;
- perplexity ≤ +1% vs mlx-community 8-bit and ≤ +3% vs bf16;
- top-1 agreement ≥ 99% vs the reference;
- decode tok/s within 10% of mlx-community 8-bit.

**After every window:** resume the paused workloads and confirm they are healthy before the next window.

**Later windows:**
- W3–W5: Qwen, then Gemma, then 70B, one model per window (convert in one window and evaluate in the next if needed).
- Path B: K3-SG probe and design windows, then an 8B mining-on run on regtest following `bench/evidence/regtest_e2e_m5.txt`. Gates:
  - logits with mining on == mining off, bit-identical (from the exact denoise);
  - every find passes `verify_plain_proof_for_cert_version(3)`;
  - dumped (A, W, job) transcripts match the oracle off-box;
  - report decode/prefill tok/s with mining on vs off, and mined TOPS-eq (Σ2mnk ÷ wall) vs pure-mining K3-SG on the same machine.

## 5. Risks, blockers, verdict

| Risk | Severity | Notes |
|---|---|---|
| **v4 (FP8 cert) replaces the int7 scheme** | **Blocker for B's long-term value** | The fp8 branch mines BF16 weights encoded to "FP10 planes" at load, and decodes quantized checkpoints (FP8/NVFP4/MXFP4) back to BF16 lossily (`vendor/pearl-fp8/miner/vllm-miner/src/vllm_miner/vllm_pearl_config.py:1-25`, `upcast.py:1-19`). Operands become int8 values with 1×8 BF16 block scales (`miner-base/src/miner_base/prequant.py:1-9`). Pearl-ai int7 checkpoints become irrelevant for mining. Apple v4 arithmetic runs ~3–10× below v3. B lives only in the v3 window |
| MoE mining invalid on mainnet | High (Qwen) | `params.go:367`, `validate.go:542`. Only attention can be mined |
| FP8 layers on Apple | Low for A | No fp8 matmul, so dequant + requant (~+3% bytes vs native) |
| Quality | Low | Card evals are within ~0.5 pt of the originals. W7A16 on MLX is likely ≥ the W7A7 reference (UNVERIFIED) |
| Speed | Medium | 8-bit Pearl decodes at about ½ the tok/s of 4-bit. 70B ≈ 8 tok/s on M3 Ultra (est.) |
| Memory | Low | All four fit in 256 GB with another server resident |
| License | Medium (needs legal review) | Llama 3.1/3.3 community licenses, Gemma terms, Qwen Apache-2.0. Redistributing MLX conversions needs review; specific clauses UNVERIFIED |
| Path B kernel risk | High | K3-SG is unproven. GPU BLAKE3 on the critical path, launch overhead, and interplay with compiled decode are all UNVERIFIED |
| Economics of B | Medium | Prefill mines at ≈ 55–60% of the pure-mining rate; decode at ≈ 4% at m=1 (est.) |

**Verdict:**
- **Path A: feasible and low-risk.** About 1–2 weeks for all four models (A1), plus 3–5 days for the exact A2 mode. Do A2 for 8B first, since it is the foundation for B.
- **Path B: hard, but not technically blocked under v3.** About 2.5–4 engineer-months after K3-SG. Its value is time-boxed by v4 activation. Start it only if the v3 window is expected to last more than ~4 months, or for the PIP narrative.
