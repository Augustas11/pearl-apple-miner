"""Tiny synthetic Llama written in the exact Pearl checkpoint format.

Mirrors pearl-ai/Llama-3.1-8B-Instruct-pearl@5dcb348a: tensor names, dtypes
(I8 weight + BF16 weight_scale[n,1]; F8_E4M3 weight + BF16 weight_scale
[ceil(n/128), ceil(k/128)]; BF16 smooth_quant_scale[k] on o_proj; BF16
embed / norms / lm_head), 2 shards + index, and a quantization_config whose
regex groups put q/k/v of layer 0 and every down_proj in FP8 and o_proj,
gate/up and q/k/v of layer 1 in int7.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from pearl_ref import E4M3_LUT, f32_to_bf16_bits

H, FFN, VOCAB, NH, NKV, HD, NL = 256, 512, 1000, 4, 2, 64, 2

_FINITE = np.array([c for c in range(256) if not np.isnan(E4M3_LUT[c])], dtype=np.uint8)
_ORDER = np.argsort(E4M3_LUT[_FINITE], kind="stable")
_SORTED_CODES = _FINITE[_ORDER]
_SORTED_VALS = E4M3_LUT[_SORTED_CODES]


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """Nearest finite E4M3 code (|x| <= 448 assumed)."""
    x = np.asarray(x, dtype=np.float32)
    i = np.clip(np.searchsorted(_SORTED_VALS, x), 1, len(_SORTED_VALS) - 1)
    lo, hi = _SORTED_VALS[i - 1], _SORTED_VALS[i]
    pick = np.where(np.abs(x - lo) <= np.abs(hi - x), i - 1, i)
    return _SORTED_CODES[pick]


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def int7_quant(w: np.ndarray):
    s = np.abs(w).max(axis=1, keepdims=True) / 63.0
    s_bits = f32_to_bf16_bits(s)
    s_q = (s_bits.astype(np.uint32) << 16).view(np.float32)
    q = np.clip(np.rint(w / s_q), -63, 63).astype(np.int8)
    return q, s_bits


def fp8_block_quant(w: np.ndarray):
    n, k = w.shape
    bn, bk = -(-n // 128), -(-k // 128)
    s = np.zeros((bn, bk), dtype=np.float32)
    for i in range(bn):
        for j in range(bk):
            s[i, j] = np.abs(w[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]).max() / 448.0
    s_bits = f32_to_bf16_bits(s)
    s_q = (s_bits.astype(np.uint32) << 16).view(np.float32)
    full = np.repeat(np.repeat(s_q, 128, 0), 128, 1)[:n, :k]
    return e4m3_encode(np.clip(w / full, -448, 448)), s_bits


def config() -> dict:
    return {
        "architectures": ["LlamaForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": 1, "dtype": "bfloat16", "eos_token_id": [2], "head_dim": HD,
        "hidden_act": "silu", "hidden_size": H, "initializer_range": 0.02,
        "intermediate_size": FFN, "max_position_embeddings": 131072, "mlp_bias": False,
        "model_type": "llama", "num_attention_heads": NH, "num_hidden_layers": NL,
        "num_key_value_heads": NKV, "pretraining_tp": 1,
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                          "group_size": 128, "num_bits": 8, "observer": None,
                                          "observer_kwargs": {}, "strategy": "group",
                                          "symmetric": True, "type": "float"},
                    "output_activations": None,
                    "targets": ["re:.*\\.down_proj$", "re:model\\.layers\\.(0)\\.self_attn\\.[qkv]_proj$"],
                    "weights": {"actorder": None, "block_structure": [128, 128], "dynamic": False,
                                "group_size": None, "num_bits": 8, "observer": "minmax",
                                "observer_kwargs": {}, "strategy": "block", "symmetric": True,
                                "type": "float"},
                },
                "group_1": {
                    "format": "int-quantized",
                    "input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                          "group_size": None, "num_bits": 7, "observer": None,
                                          "observer_kwargs": {}, "strategy": "token",
                                          "symmetric": True, "type": "int"},
                    "output_activations": None,
                    "targets": ["re:.*self_attn\\.o_proj$", "re:.*\\.gate_proj$", "re:.*\\.up_proj$",
                                "re:model\\.layers\\.(1)\\.self_attn\\.[qkv]_proj$"],
                    "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                                "group_size": None, "num_bits": 7, "observer": "minmax",
                                "observer_kwargs": {}, "strategy": "channel", "symmetric": True,
                                "type": "int"},
                },
            },
            "format": "mixed-precision", "global_compression_ratio": None, "ignore": ["lm_head"],
            "kv_cache_scheme": None, "quant_method": "pearl", "quantization_status": "compressed",
            "sparsity_config": {}, "transform_config": {}, "version": "0.13.0",
        },
        "rms_norm_eps": 1e-05,
        "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                         "original_max_position_embeddings": 8192, "rope_type": "llama3"},
        "rope_theta": 500000.0, "tie_word_embeddings": False, "transformers_version": "4.57.6",
        "use_cache": True, "vocab_size": VOCAB,
    }


def write_tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"t{i}": i for i in range(3, VOCAB)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "model_max_length": 4096}))
    (d / "special_tokens_map.json").write_text(json.dumps(
        {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}))
    (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 1, "eos_token_id": 2}))


def make_pearl_checkpoint(d: Path, seed: int = 0, extra_smooth: bool = False) -> Path:
    """extra_smooth adds smooth scales that exercise every fold path:
    layer-1 gate/up share one scale (-> RMSNorm fold), layer-1 q_proj alone
    (ambiguous -> runtime smooth), layer-0 down_proj FP8 (-> W column fold)."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    bfb = lambda a: ("BF16", f32_to_bf16_bits(a.astype(np.float32)))  # noqa: E731
    shards: list[dict] = [{}, {}]
    shards[0]["model.embed_tokens.weight"] = bfb(rng.standard_normal((VOCAB, H)))
    dims = {"self_attn.q_proj": (NH * HD, H), "self_attn.k_proj": (NKV * HD, H),
            "self_attn.v_proj": (NKV * HD, H), "self_attn.o_proj": (H, NH * HD),
            "mlp.gate_proj": (FFN, H), "mlp.up_proj": (FFN, H), "mlp.down_proj": (H, FFN)}
    for li in range(NL):
        sh = shards[li]
        p = f"model.layers.{li}"
        sh[f"{p}.input_layernorm.weight"] = bfb(1 + 0.1 * rng.standard_normal(H))
        sh[f"{p}.post_attention_layernorm.weight"] = bfb(1 + 0.1 * rng.standard_normal(H))
        for name, (n, k) in dims.items():
            w = rng.standard_normal((n, k)).astype(np.float32) / np.sqrt(k)
            fp8 = name == "mlp.down_proj" or (li == 0 and name.startswith("self_attn.") and name != "self_attn.o_proj")
            if fp8:
                q, s = fp8_block_quant(w)
                sh[f"{p}.{name}.weight"] = ("F8_E4M3", q)
            else:
                q, s = int7_quant(w)
                sh[f"{p}.{name}.weight"] = ("I8", q)
            sh[f"{p}.{name}.weight_scale"] = ("BF16", s)
        sh[f"{p}.self_attn.o_proj.smooth_quant_scale"] = bfb(np.exp(0.3 * rng.standard_normal(NH * HD)))
        if extra_smooth and li == 1:
            gu = bfb(np.exp(0.3 * rng.standard_normal(H)))
            sh[f"{p}.mlp.gate_proj.smooth_quant_scale"] = gu
            sh[f"{p}.mlp.up_proj.smooth_quant_scale"] = gu
            sh[f"{p}.self_attn.q_proj.smooth_quant_scale"] = bfb(np.exp(0.3 * rng.standard_normal(H)))
        if extra_smooth and li == 0:
            shards[0][f"{p}.mlp.down_proj.smooth_quant_scale"] = bfb(np.exp(0.3 * rng.standard_normal(FFN)))
    shards[1]["model.norm.weight"] = bfb(1 + 0.1 * rng.standard_normal(H))
    # logits kept O(1): bf16 spacing is 2^-8 below 1 but 2^-6 in [2,4), which
    # alone would swamp a 1e-2 bf16 parity bound.
    shards[1]["lm_head.weight"] = bfb(rng.standard_normal((VOCAB, H)) / (4 * np.sqrt(H)))
    wm = {}
    for i, sh in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-00002.safetensors"
        write_safetensors(d / fn, sh)
        wm.update({k: fn for k in sh})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps(config(), indent=2))
    write_tokenizer(d)
    return d
