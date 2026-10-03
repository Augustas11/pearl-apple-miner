"""studio/parity.py on the synthetic checkpoint (tiny GPU workload) + shell syntax."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pearl2mlx import convert
from synth import make_pearl_checkpoint

STUDIO = Path(__file__).resolve().parent.parent / "studio"
sys.path.insert(0, str(STUDIO))


def test_parity_script(tmp_path):
    import mlx.core as mx
    import parity
    dev = mx.default_device()
    mx.set_default_device(mx.gpu)  # as on the Studio; CPU bf16 qmm is not exact
    try:
        _run_parity(parity, tmp_path)
    finally:
        mx.set_default_device(dev)


def _run_parity(parity, tmp_path):
    src = make_pearl_checkpoint(tmp_path / "ck", seed=21)
    convert(str(src), str(tmp_path / "exact"), "exact")
    convert(str(src), str(tmp_path / "std8"), "std8")
    out = tmp_path / "parity.json"
    rc = parity.main(["--src", str(src), "--mlx", str(tmp_path / "exact"), str(tmp_path / "std8"),
                      "--prompts", "6", "--min-top1", "0.9", "--json-out", str(out)])
    res = json.loads(out.read_text())
    assert rc == 0, res
    assert len(res["pairs"]) == 4 and res["kernel_check"] and all(res["kernel_check"].values())
    first = res["pairs"][0]
    assert first["ref"] == "w7a16" and first["top1"] >= 0.9 and first["mean_kl"] < 1e-3
    rc = parity.main(["--src", str(src), "--mlx", str(tmp_path / "exact"), "--prompts", "2",
                      "--layers", "1", "--a7", "on"])
    assert rc == 0


@pytest.mark.parametrize("script", ["lib.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(STUDIO / script)], check=True)
