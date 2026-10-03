# pearl2mlx: Pearl checkpoints → MLX

Design in `docs/kb/mlx-pearl-models.md` §2.2 (A1/A2), §2.4, §4.

| File | What |
|---|---|
| `pearl2mlx.py` | Converter. `--src <HF snapshot> --out <dir> --mode exact\|std8\|std4 [--fp8 q8\|bf16] [--no-verify]` |
| `pearl_layers.py` | mlx-lm model file for exact mode (`PearlQuantizedLinear` = `QuantizedLinear` with `x·smooth`). Copied into exact outputs. |
| `pearl_ref.py` | numpy reference forward on the ORIGINAL Pearl checkpoint (`--layers N` prefix, `--a7` W7A7 emulation) |
| `studio/` | `lib.sh` (shared window helpers), `parity.py`, `summary.py`, `resolve.py` |
| `tests/` | pytest suite incl. a synthetic 2-layer Llama in the exact Pearl format (`tests/synth.py`) |

Only `model_type: llama` is supported (Llama-3.1-8B / 3.3-70B). Qwen MoE stacking and Gemma naming (kb §2.4 step 3) are not implemented yet.

## Local tests (any Mac, CPU; the tiny bf16 checks use the GPU for milliseconds)

```bash
cd tools/pearl2mlx
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx==0.31.2 mlx-lm==0.31.3 numpy safetensors pytest huggingface_hub
.venv/bin/python -m pytest tests -q        # 40 tests
```

## Output format

- **exact** (A2)
  - int7 layers → MLX affine 8-bit, group 128: `weight` = uint32 with 4 values `q = int7+64` per word, low byte first (checked against `mx.quantize`/`mx.dequantize` in the tests); `scales` = `s_row` repeated per group; `biases` = `−64·s_row` (bf16, exact).
  - The converter checks every int7 layer: `mx.dequantize` in fp32 == `int7·s`, bit for bit (`--no-verify` skips this).
  - FP8 layers → `mx.from_fp8` × block scale (fp32) → bf16 → affine 8-bit g64. `--fp8 bf16` keeps them as dense bf16 instead (+~2.2 GB on 8B, near-lossless).
  - SmoothQuant:
    - folded into the preceding RMSNorm when all of its consumers (q/k/v or gate/up) are int7 and share one identical scale;
    - otherwise kept as a runtime `<layer>.smooth` param, listed in `config["pearl_smooth"]`. This is every o_proj on Llama-8B.
    - FP8 layers with a scale fold it into the W columns.
  - embed, norms and lm_head stay bf16.
  - config gets `"quantization": {"group_size": 64, "bits": 8, "mode": "affine", "<int7 layer>": {"group_size": 128, ...}}` (also mirrored to `quantization_config`), plus `"model_file": "pearl_layers.py"` and `"pearl_smooth"`.
  - Stock `mlx_lm.load` and all `mlx_lm.*` CLIs honour `model_file`, so they work unchanged.
- **std8 / std4** (A1)
  - Every linear is dequantized; smooth is folded into the W columns; then affine 8-bit or 4-bit, g64.
  - Embedding and lm_head are quantized too, as `mlx_lm.convert` does. That keeps the results comparable with the mlx-community 8bit/4bit models.
  - Plain stock checkpoint: no `model_file`.
- **All modes**
  - `quantization_config` from Pearl is dropped. It is kept for provenance under `config["pearl2mlx"]["source_quantization_config"]`.
  - Tokenizer files and `chat_template.jinja` are copied.
  - One output shard per input shard. Stale `model*.safetensors` files in `--out` are deleted first.
- **Expected 8B sizes (estimates):** exact ≈ 9.4 GB, std8 ≈ 8.5 GB. Conversion peak RAM is about one shard (~5 GB in, ~5 GB out).

### Exactness facts found while building this (MLX 0.31.2)

- `mx.quantized_matmul` on the **GPU** with bf16 scales evaluates `s·q + b` in fp32 (qmv and qmm, m = 1…1024 tested).
  - So the kernel's effective weights are exactly `bf16(int7·s)`. `test_gpu_kernel_effective_weights_bit_exact_bf16` covers this; on the M3 Ultra, `parity.py` re-checks it on real layers.
- `mx.dequantize` in bf16 and the **CPU** bf16 `quantized_matmul` are **not** exact.
  - They round `s·q` to bf16 before adding `b`.
  - The CPU bf16 qmm also accumulates in bf16 (~7% relative error on a 1024-wide dot).
  - So use the GPU or fp32 for any check.
- `mx.from_fp8` decodes 0x7F/0xFF as ±480. OCP E4M3FN says those codes are NaN.
  - The converter rejects NaN codes. The tests compare `from_fp8` against an independent numpy decoder on all 254 finite codes (incl. subnormals and ±0) and on ragged block edges.

### Deviations from kb §2.4

1. **The `< 1e-2` bf16 target vs the pure Pearl reference.** It holds for exact `--fp8 bf16` (0.0085 / 0.0078), and for exact vs a reference that uses the same requantized FP8 weights (0.0073 / 0.0083).
   - With the default FP8→affine-8 requant, exact vs pure `pearl_ref` is 0.012–0.016 (fp32: 0.009).
   - That residual comes entirely from requantizing the FP8 layers. The int7 part has zero residual (fp32 Δ = 1e-6).
   - The synthetic model's logits are kept O(1). In [2,4), bf16 spacing (2⁻⁶) alone exceeds 1e-2.
2. **The fused-qkv smooth question (kb §1.1, UNVERIFIED) only matters for Qwen.** Each layer uses its own `smooth_quant_scale`.
   - When a norm's consumers disagree, the scales are not folded into the norm. They stay at runtime instead.
3. **Activation emulation.** `pearl_ref --a7` uses exact division `63/amax`, not `rcp_approx`. FP8 per-group activation quantization is not emulated.

### Swift mirror (not implemented)

In mlx-swift-lm, follow the ParoQuant pattern:
- When `config.json` has `pearl_smooth`, replace each listed Linear with a `PearlQuantizedLinear: QuantizedLinear` before `quantize(model:)`.
- That class holds `smooth: MLXArray [in]`, and its `callAsFunction` returns `quantizedMM(x * smooth, weight, scales, biases, transpose: true, groupSize:, bits:)`.
- Per-layer group sizes come from the standard `quantization` map that `Load.swift` already applies.

## Long-run windows

The window scripts (not published; `studio/lib.sh` has the shared helpers):
- use `~/pearl2mlx-work/venv/bin/python`;
- write everything to `~/pearl2mlx-work/logs/<w1|w2>-<UTC>.log`, plus per-step outputs in `<log>.d/`;
- set `HF_HUB_OFFLINE=1` so nothing downloads inside a window;
- never touch launchd, other services, locks or sysctl settings. `iogpu.wired_limit_mb` is only read.

### Prerequisites (outside any window)

```bash
# 1. copy the tool (from this repo)
rsync -a --exclude .venv --exclude tests --exclude __pycache__ tools/pearl2mlx/ <target-host>:~/pearl2mlx-work/pearl2mlx/
# 2. on the target Mac: HF cache (W0 or a low-priority download)
hf download pearl-ai/Llama-3.1-8B-Instruct-pearl --revision 5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0
hf download mlx-community/Meta-Llama-3.1-8B-Instruct-8bit --revision 142d42800404   # full sha if the CLI needs it
hf download mlx-community/Meta-Llama-3.1-8B-Instruct-bf16 --revision f8311090f9ee
hf download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
# 3. perplexity needs `datasets` + the tulu-3 dataset cached (mlx_lm.perplexity default data)
~/pearl2mlx-work/venv/bin/pip install datasets      # or: uv pip install --python ~/pearl2mlx-work/venv/bin/python datasets
~/pearl2mlx-work/venv/bin/python -c "from datasets import load_dataset; load_dataset('allenai/tulu-3-sft-mixture', split='train')"
# 4. optional, W2 MMLU: lm-eval + MMLU data (prefetch with a 1-question run on the small 4bit model)
~/pearl2mlx-work/venv/bin/pip install lm-eval
HF_HUB_OFFLINE=0 ~/pearl2mlx-work/venv/bin/mlx_lm.evaluate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --tasks mmlu --limit 1 --output-dir /tmp/mmlu-prefetch
```

`resolve.py` accepts the short shas above. It matches them against the cached revisions and never downloads. Each window's preflight prints `dep datasets: ok|MISSING` and `dep lm_eval: ok|MISSING`.

### Invocation

Run the steps below by hand or from your own window script that sources `studio/lib.sh`, on an otherwise idle machine with other GPU workloads paused. The launcher scripts used for the original runs are not published.

Env overrides: `PY`, `TMLX` (default `~/pearl2mlx-work`), `BUDGET_S` (2400), `HF_HUB_OFFLINE` (1), `MMLU_LIMIT` (20), `PMK_WATCH_PATTERN` (regex of other processes to list in preflight).

### W1 (≤ 40 min; step estimates UNVERIFIED)

| Step | Est / timeout | Req | Command |
|---|---|---|---|
| preflight | <1 min | – | `sysctl hw.memsize`, `sysctl iogpu.wired_limit_mb`, `vm_stat`, `df -h ~`, versions/deps; needs ≥ 25 GB free |
| convert-exact | 10 / 20 min | yes | `pearl2mlx.py --mode exact --out ~/pearl2mlx-work/l8-exact` |
| convert-std8 | 8 / 20 min | yes | `pearl2mlx.py --mode std8 --out ~/pearl2mlx-work/l8-std8` |
| gen-exact, gen-std8 | 1 / 5 min each | yes | `mlx_lm.generate --model <m> --prompt … --max-tokens 64` (chat template) |
| template-check | 20 s | no | token-hash of one chat with each tokenizer (ppl compares chat-formatted data) |
| ppl-exact, ppl-mc8 | 3 / 10 min each | yes | `mlx_lm.perplexity --model <m> --sequence-length 1024 --num-samples 128` |
| bench-exact, bench-mc8 | 1.5 / 5 min each | yes | `mlx_lm.benchmark --model <m> --prompt-tokens 512 --generation-tokens 128 --num-trials 3` |
| ppl-mcbf16, ppl-std8, bench-mc4 | 4, 3, 1 min | no | same commands |

Optional steps are skipped once `elapsed + estimate > BUDGET_S`. Every step also has a hard timeout.

### W2 (≤ 40 min)

| Step | Est / timeout | Req | Command |
|---|---|---|---|
| parity | 15 / 30 min | yes | `studio/parity.py --src <pearl snapshot> --mlx ~/pearl2mlx-work/l8-exact ~/pearl2mlx-work/l8-std8 --prompts 32 --a7 both` |
| mmlu-exact, mmlu-mcbf16 | 10 / 15 min each | no | `mlx_lm.evaluate --model <m> --tasks mmlu --limit $MMLU_LIMIT` (only if `lm_eval` imports) |

What `parity` does:
- runs `pearl_ref` (numpy fp32 on the CPU, all 32 prompts batched per layer) twice, plain W7A16 and `--a7`;
- reports top-1 agreement over all positions, mean/p99 KL(ref‖mlx) and max |Δlogit|, for each converted model;
- spot-checks 4 real int7 layers: the GPU kernel's effective weights must equal `bf16(int7·s)`.

MMLU notes:
- `MMLU_LIMIT=20` per subtask is about 1.1k questions per model.
- The kb's `--limit 200` is about 10k questions per model. That is UNVERIFIED but likely ≥ 1 h each, so it does not fit a window.
- **If `lm_eval` is missing**, MMLU is skipped (logged, not a failure). Quality evidence then rests on perplexity (W1) and parity top-1/KL (W2). To get MMLU, install `lm-eval` and prefetch outside the window (step 4 above), then rerun W2. Parity is cheap to repeat.

### Outputs and pass criteria (kb §4.3)

At the end of each window, `summary.py` prints a `SUMMARY {json}` line and `CRITERION PASS|FAIL` lines:

| Criterion | Source |
|---|---|
| exact-mode weights bit-identical to int7·s | `convert-exact` (`int7_verified_bit_exact` = 144 layers on 8B: o/gate/up × 32 + q/k/v × 16) and the W2 GPU kernel check |
| perplexity ≤ +1% vs mlx-community 8-bit, ≤ +3% vs bf16 | `ppl-exact` / `ppl-mc8` / `ppl-mcbf16` |
| top-1 agreement ≥ 99% vs the reference | `parity` (exact vs W7A16 ref; the A7 rows are informational) |
| decode tok/s within 10% of mlx-community 8-bit | `bench-exact` / `bench-mc8` `generation_tps` |

On the tok/s criterion: exact keeps lm_head in bf16, about +0.5 GB per token vs an 8-bit lm_head, so decode is likely ~5–9% slower than mlx-community 8bit (estimate). std8 is the A1 fallback if this criterion fails.

Exit codes:
- `0`: all steps OK and every criterion that has data passed.
- `1`: a step failed, timed out, or its model was not cached. A missing prerequisite (Pearl snapshot, disk, W1 outputs) also exits 1, immediately.
- `2`: steps OK but a criterion failed.

Steps that fail are logged verbatim in the window log.
