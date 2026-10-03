"""(c) SmoothQuant folding equivalences + fold planning."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pearl2mlx import EXACT_GROUP, pack_int7_exact, plan
from pearl_layers import PearlQuantizedLinear
from pearl_ref import PearlCheckpoint, bf16_bits_to_f32, f32_to_bf16_bits
from synth import make_pearl_checkpoint


def _bf16_round(x):
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


def test_norm_fold_equivalence():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 256)).astype(np.float32)
    g = _bf16_round(1 + 0.1 * rng.standard_normal(256).astype(np.float32))
    s = _bf16_round(np.exp(0.3 * rng.standard_normal(256)).astype(np.float32))
    xn = mx.array(x)
    folded = nn.RMSNorm(256, eps=1e-5)
    folded.weight = mx.array(_bf16_round(g * s))          # what the converter stores
    plain = nn.RMSNorm(256, eps=1e-5)
    plain.weight = mx.array(g)
    a = np.array(folded(xn))
    b = np.array(plain(xn)) * s
    np.testing.assert_allclose(a, b, rtol=2 ** -8, atol=1e-6)


def test_column_fold_equivalence():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 256)).astype(np.float32)
    w = rng.standard_normal((64, 256)).astype(np.float32)
    s = np.exp(0.3 * rng.standard_normal(256)).astype(np.float32)
    np.testing.assert_allclose((x * s) @ w.T, x @ (w * s[None, :]).T, rtol=1e-5, atol=1e-4)


def test_pearl_quantized_linear_applies_smooth():
    rng = np.random.default_rng(2)
    q = rng.integers(-63, 64, size=(32, 256)).astype(np.int8)
    s_bits = f32_to_bf16_bits(np.full((32, 1), 0.01, np.float32))
    wq, sc, bi = pack_int7_exact(q, s_bits)
    layer = PearlQuantizedLinear(256, 32, bias=False, group_size=EXACT_GROUP, bits=8)
    sm = mx.array(np.exp(0.3 * rng.standard_normal(256)).astype(np.float32))
    layer.update({"weight": wq, "scales": sc.astype(mx.float32), "biases": bi.astype(mx.float32), "smooth": sm})
    x = mx.array(rng.standard_normal((3, 256)).astype(np.float32))
    want = mx.quantized_matmul(x * sm, wq, sc.astype(mx.float32), bi.astype(mx.float32),
                               transpose=True, group_size=EXACT_GROUP, bits=8)
    assert mx.array_equal(layer(x), want).item()


def test_plan_fold_targets(tmp_path):
    ck = PearlCheckpoint(make_pearl_checkpoint(tmp_path / "ck", extra_smooth=True))
    _, norm_fold, runtime, col = plan(ck, "exact")
    assert norm_fold == {"model.layers.1.post_attention_layernorm.weight":
                         "model.layers.1.mlp.gate_proj.smooth_quant_scale"}
    assert runtime == {"model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj",
                       "model.layers.1.self_attn.q_proj"}
    assert col == {"model.layers.0.mlp.down_proj"}
    _, norm_fold, runtime, col = plan(ck, "std8")
    assert not norm_fold and not runtime and len(col) == 6
