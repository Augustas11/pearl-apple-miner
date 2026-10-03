"""(d) end-to-end: synthetic Pearl checkpoint -> pearl2mlx (3 modes) ->
stock mlx_lm.load (strict) -> logits vs pearl_ref (no A7).

bf16 runs use the GPU (tiny model, negligible load): MLX's CPU bf16
quantized_matmul accumulates in bf16 and is not representative. fp32 runs use
the CPU.
"""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm import load

from pearl2mlx import convert
from pearl_ref import PearlCheckpoint, forward
from synth import VOCAB, make_pearl_checkpoint

PROMPTS = [[1, 17, 503, 999, 42, 7, 256, 3, 88, 640, 12, 5], [1, 900, 901, 902]]


@pytest.fixture(scope="module", params=[False, True], ids=["pearl-format", "extra-smooth"])
def src(request, tmp_path_factory):
    return make_pearl_checkpoint(tmp_path_factory.mktemp("pearl") / "ck", seed=11,
                                 extra_smooth=request.param)


@pytest.fixture(scope="module")
def ref_logits(src):
    return forward(PearlCheckpoint(src), PROMPTS)


def _logits(model, stream):
    with mx.stream(stream):
        return [np.array(model(mx.array([t])).astype(mx.float32))[0] for t in PROMPTS]


def _report(tag, got, ref):
    d = float(max(np.abs(g - r).max() for g, r in zip(got, ref)))
    top1 = float(np.mean(np.concatenate([g.argmax(-1) == r.argmax(-1) for g, r in zip(got, ref)])))
    print(f"\n[{tag}] max|dlogit|={d:.6f} top1={top1:.3f} "
          f"(max|logit|={max(np.abs(r).max() for r in ref):.2f})")
    for g in got:
        assert g.shape[1] == VOCAB and np.isfinite(g).all()
    return d, top1


def _convert_load(src, out, mode, **kw):
    summary = convert(str(src), str(out), mode, **kw)
    cfg = json.loads((out / "config.json").read_text())
    model, tok = load(str(out))  # stock mlx_lm.load, strict weight loading
    assert tok.encode("t5 t6", add_special_tokens=False) == [5, 6]
    return summary, cfg, model


def _requant_aware_ref(src, model, monkeypatch):
    """pearl_ref with each FP8 layer's weight replaced by what the MLX model
    actually holds for it (int7 layers keep Pearl's own int7*s)."""
    ck = PearlCheckpoint(src)
    mods = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    over = {}
    for p, m in mods.items():
        if not ck.has(p + ".weight_scale") or ck.linear_kind(p) != "fp8":
            continue
        w = np.array(mx.dequantize(m.weight, m.scales.astype(mx.float32), m.biases.astype(mx.float32),
                                   group_size=m.group_size, bits=m.bits))
        s = ck.smooth(p)
        over[p] = w / s if s is not None else w   # undo column fold; ref re-applies smooth
    orig = PearlCheckpoint.dequant
    monkeypatch.setattr(PearlCheckpoint, "dequant", lambda self, p: over[p] if p in over else orig(self, p))
    out = forward(ck, PROMPTS)
    monkeypatch.undo()
    return out


def test_exact(src, ref_logits, tmp_path, monkeypatch):
    summary, cfg, model = _convert_load(src, tmp_path / "exact", "exact")
    assert summary["int7_verified_bit_exact"] == 9  # L0: o,gate,up; L1: q,k,v,o,gate,up
    assert cfg["model_file"] == "pearl_layers.py" and (tmp_path / "exact" / "pearl_layers.py").exists()
    q = cfg["quantization"]
    assert q["model.layers.1.self_attn.q_proj"] == {"group_size": 128, "bits": 8, "mode": "affine"}
    assert "model.layers.0.self_attn.q_proj" not in q and q["group_size"] == 64
    assert cfg["quantization_config"] == q and "quant_method" not in q
    assert "model.layers.0.self_attn.o_proj" in cfg["pearl_smooth"]

    # 1) A2 is lossless on the int7 part: vs a reference that uses the same
    #    requantized FP8 weights, the only residual is arithmetic rounding.
    #    (A folded RMSNorm weight bf16(g*s) adds one bf16 rounding.)
    ref_rq = _requant_aware_ref(src, model, monkeypatch)
    got16 = _logits(model, mx.gpu)
    d16, _ = _report("exact bf16 vs requant-aware ref", got16, ref_rq)
    assert d16 < 1e-2
    # 2) vs the true Pearl reference: residual adds the FP8 -> affine-8 requant.
    d, top1 = _report("exact bf16 vs pearl_ref", got16, ref_logits)
    assert d < 5e-2 and top1 >= 0.9
    model.set_dtype(mx.float32)
    d32, _ = _report("exact fp32 vs requant-aware ref", _logits(model, mx.cpu), ref_rq)
    assert d32 < (1e-4 if summary["norm_fold"] == 0 else 5e-3)


def test_exact_fp8_bf16(src, ref_logits, tmp_path):
    _, cfg, model = _convert_load(src, tmp_path / "exact-bf16", "exact", fp8="bf16")
    assert cfg["pearl2mlx"]["fp8"] == "bf16"
    d16, _ = _report("exact --fp8 bf16, bf16 vs pearl_ref", _logits(model, mx.gpu), ref_logits)
    assert d16 < 1e-2
    model.set_dtype(mx.float32)
    d32, top1 = _report("exact --fp8 bf16, fp32 vs pearl_ref", _logits(model, mx.cpu), ref_logits)
    assert d32 < 1e-2 and top1 == 1.0


@pytest.mark.parametrize("mode,bound", [("std8", 0.25), ("std4", None)])
def test_std(src, ref_logits, tmp_path, mode, bound):
    _, cfg, model = _convert_load(src, tmp_path / mode, mode)
    assert "model_file" not in cfg and "pearl_smooth" not in cfg
    assert cfg["quantization"] == {"group_size": 64, "bits": 8 if mode == "std8" else 4, "mode": "affine"}
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    d, top1 = _report(f"{mode} bf16 vs pearl_ref", _logits(model, mx.gpu), ref_logits)
    if bound is not None:
        assert d < bound and top1 >= 0.9
