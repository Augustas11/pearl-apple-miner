"""(a) exact-mode int7 packing: MLX 8-bit layout + bit-exact dequant."""
import mlx.core as mx
import numpy as np
import pytest

from pearl2mlx import EXACT_GROUP, pack_int7_exact
from pearl_ref import bf16_bits_to_f32, f32_to_bf16_bits


def test_mlx_8bit_layout_is_low_byte_first():
    rng = np.random.default_rng(0)
    q = rng.integers(0, 256, size=(8, 256), dtype=np.uint8)
    packed = mx.array(q.view("<u4"))
    ones = mx.ones((8, 256 // 128))
    deq = mx.dequantize(packed, ones, mx.zeros_like(ones), group_size=128, bits=8)
    assert np.array_equal(np.array(deq), q.astype(np.float32))


def test_mx_quantize_output_uses_same_layout():
    rng = np.random.default_rng(1)
    w = mx.array(rng.standard_normal((16, 256)).astype(np.float32))
    wq, s, b = mx.quantize(w, group_size=128, bits=8)
    q = np.array(wq).view(np.uint8).reshape(16, 256).astype(np.float32)
    s_full = np.repeat(np.array(s), 128, axis=1)
    b_full = np.repeat(np.array(b), 128, axis=1)
    ours = s_full * q + b_full
    theirs = np.array(mx.dequantize(wq, s, b, group_size=128, bits=8))
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)


def _case(n, k, seed):
    rng = np.random.default_rng(seed)
    q = rng.integers(-63, 64, size=(n, k)).astype(np.int8)
    q[0, :4] = [-64, 64, 63, -63]
    q[1, :] = 0
    mags = 2.0 ** rng.uniform(-20, 4, size=(n, 1))
    s_bits = f32_to_bf16_bits(mags.astype(np.float32))
    return q, s_bits


@pytest.mark.parametrize("n,k", [(4, 128), (64, 512), (24, 1024)])
def test_exact_dequant_bit_exact_fp32(n, k):
    q, s_bits = _case(n, k, n + k)
    wq, scales, biases = pack_int7_exact(q, s_bits)
    assert wq.dtype == mx.uint32 and wq.shape == (n, k // 4)
    assert scales.dtype == mx.bfloat16 and scales.shape == (n, k // EXACT_GROUP)
    ref32 = q.astype(np.float32) * bf16_bits_to_f32(s_bits)
    deq32 = mx.dequantize(wq, scales.astype(mx.float32), biases.astype(mx.float32),
                          group_size=EXACT_GROUP, bits=8)
    assert np.array_equal(np.array(deq32), ref32)


@pytest.mark.parametrize("m", [1, 4, 32, 33, 256])
def test_gpu_kernel_effective_weights_bit_exact_bf16(m):
    """quantized_matmul with bf16 scales on the GPU (qmv for small m, qmm for
    large m) evaluates s*q+b in fp32, so one-hot inputs read back exactly
    bf16(int7*s). (mx.dequantize in bf16 and the CPU bf16 qmm round s*q to
    bf16 first and are NOT exact; tests and converters use fp32 for those.)
    Tiny GPU workload: [m,256] x [64,256]."""
    q, s_bits = _case(64, 256, m)
    wq, scales, biases = pack_int7_exact(q, s_bits)
    want = mx.array(q.astype(np.float32) * bf16_bits_to_f32(s_bits)).astype(mx.bfloat16)
    cols = np.arange(m) % 256
    x = np.zeros((m, 256), np.float32)
    x[np.arange(m), cols] = 1
    y = mx.quantized_matmul(mx.array(x).astype(mx.bfloat16), wq, scales, biases, transpose=True,
                            group_size=EXACT_GROUP, bits=8, stream=mx.gpu)
    assert mx.array_equal(y, want.T[mx.array(cols)]).item()


def test_exact_quantized_matmul_matches_dense():
    q, s_bits = _case(32, 256, 7)
    wq, scales, biases = pack_int7_exact(q, s_bits)
    x = mx.array(np.random.default_rng(3).standard_normal((5, 256)).astype(np.float32))
    y = mx.quantized_matmul(x, wq, scales.astype(mx.float32), biases.astype(mx.float32),
                            transpose=True, group_size=EXACT_GROUP, bits=8)
    w = q.astype(np.float32) * bf16_bits_to_f32(s_bits)
    np.testing.assert_allclose(np.array(y), np.array(x) @ w.T, rtol=1e-5, atol=1e-5 * np.abs(w).max())


def test_out_of_range_rejected():
    q, s_bits = _case(2, 128, 0)
    q[0, 0] = -65
    with pytest.raises(ValueError):
        pack_int7_exact(q, s_bits)
