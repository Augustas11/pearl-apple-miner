"""pearl_ref: A7 emulation, N-layer prefix, CLI."""
import json

import numpy as np

import pearl_ref
from pearl_ref import PearlCheckpoint, forward
from synth import VOCAB, make_pearl_checkpoint


def test_a7_matches_explicit_int_gemm(tmp_path):
    ck = PearlCheckpoint(make_pearl_checkpoint(tmp_path / "ck", seed=3))
    p = "model.layers.0.self_attn.o_proj"
    x = np.random.default_rng(0).standard_normal((5, 256)).astype(np.float32)
    x[2] = 0  # all-zero token -> zero output, no NaN
    y = pearl_ref._linear(ck, p, x, a7=True)
    s = ck.smooth(p)
    xs = x * s
    amax = np.abs(xs).max(-1, keepdims=True)
    q = np.clip(np.rint(xs * 63 / np.where(amax == 0, 1, amax)), -63, 63).astype(np.int64)
    acc = q @ ck.raw(p + ".weight").astype(np.int64).T
    want = acc * (amax / 63) * ck.f32(p + ".weight_scale").reshape(1, -1)
    np.testing.assert_allclose(y, want, rtol=1e-6, atol=1e-7)
    assert np.all(y[2] == 0)
    # fp8 layers are untouched by --a7
    q0 = "model.layers.0.self_attn.q_proj"
    assert np.array_equal(pearl_ref._linear(ck, q0, x, True), pearl_ref._linear(ck, q0, x, False))


def test_prefix_and_a7_forward(tmp_path):
    ck = PearlCheckpoint(make_pearl_checkpoint(tmp_path / "ck", seed=4))
    toks = [[1, 5, 9, 200], [1, 7]]
    full = forward(ck, toks)
    one = forward(ck, toks, n_layers=1)
    a7 = forward(ck, toks, a7=True)
    assert [f.shape for f in full] == [(4, VOCAB), (2, VOCAB)] == [o.shape for o in one]
    assert not np.allclose(full[0], one[0])
    d = max(np.abs(a - f).max() for a, f in zip(a7, full))
    assert 0 < d < 0.1
    # batching prompts together == running them alone
    assert np.allclose(forward(ck, [toks[1]])[0], full[1], atol=1e-5)


def test_cli(tmp_path, capsys):
    src = make_pearl_checkpoint(tmp_path / "ck", seed=5)
    out = tmp_path / "l.npy"
    assert pearl_ref.main(["--src", str(src), "--tokens", "1,4,8", "--layers", "1", "--out", str(out)]) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert res["tokens"] == 3 and len(res["top5"]) == 5
    assert np.load(out).shape == (3, VOCAB)
