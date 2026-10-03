"""(b) FP8 E4M3 block dequant vs an independent numpy decoder."""
import mlx.core as mx
import numpy as np
import pytest

from pearl2mlx import fp8_dequant
from pearl_ref import E4M3_LUT, bf16_bits_to_f32, f32_to_bf16_bits


def test_lut_known_values():
    assert E4M3_LUT[0x00] == 0 and E4M3_LUT[0x01] == 2.0 ** -9      # min subnormal
    assert E4M3_LUT[0x07] == 7 * 2.0 ** -9                            # max subnormal
    assert E4M3_LUT[0x08] == 2.0 ** -6                                # min normal
    assert E4M3_LUT[0x7E] == 448 and E4M3_LUT[0xFE] == -448
    assert E4M3_LUT[0x38] == 1.0
    assert np.isnan(E4M3_LUT[0x7F]) and np.isnan(E4M3_LUT[0xFF])


def test_mx_from_fp8_matches_lut_on_all_finite_codes():
    codes = np.arange(256, dtype=np.uint8)
    got = np.array(mx.from_fp8(mx.array(codes), dtype=mx.float32))
    finite = ~np.isnan(E4M3_LUT)
    assert np.array_equal(got[finite], E4M3_LUT[finite])
    assert np.signbit(got[0x80]) and got[0x80] == 0


def _independent_block_dequant(codes, s_bits):
    n, k = codes.shape
    s = bf16_bits_to_f32(s_bits)
    out = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            out[i, j] = E4M3_LUT[codes[i, j]] * s[i // 128, j // 128]
    return out


@pytest.mark.parametrize("n,k", [(128, 128), (300, 200), (130, 768), (257, 769)])
def test_block_dequant_ragged(n, k):
    rng = np.random.default_rng(n * k)
    finite = np.array([c for c in range(256) if not np.isnan(E4M3_LUT[c])], dtype=np.uint8)
    codes = rng.choice(finite, size=(n, k))
    codes[0, :16] = np.arange(16)          # +0 and subnormals
    codes[-1, -8:] = 0x80 + np.arange(8)   # -0 and negative subnormals
    s_bits = f32_to_bf16_bits(2.0 ** rng.uniform(-12, 2, size=(-(-n // 128), -(-k // 128))).astype(np.float32))
    got = np.array(fp8_dequant(codes, s_bits))
    assert got.shape == (n, k)
    assert np.array_equal(got, _independent_block_dequant(codes, s_bits))


@pytest.mark.parametrize("nan_code", [0x7F, 0xFF])
def test_nan_codes_rejected(nan_code):
    codes = np.zeros((128, 128), dtype=np.uint8)
    codes[5, 7] = nan_code
    with pytest.raises(ValueError, match="NaN"):
        fp8_dequant(codes, f32_to_bf16_bits(np.ones((1, 1), np.float32)))


def test_scale_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="block scale shape"):
        fp8_dequant(np.zeros((300, 200), np.uint8), f32_to_bf16_bits(np.ones((2, 2), np.float32)))
