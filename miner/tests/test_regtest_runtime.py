from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/pmk_regtest_e2e.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("pmk_regtest_e2e_runtime", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_run_root_is_external_temp_child() -> None:
    harness = load_harness()
    run = harness.make_run_root(None)
    try:
        assert run.name.startswith("pmk-regtest-")
        assert run.is_dir()
        assert not run.resolve().is_relative_to(ROOT.resolve())
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_run_root_rejects_repository_parent(tmp_path: Path) -> None:
    harness = load_harness()
    with pytest.raises(ValueError, match="outside the repository"):
        harness.make_run_root(ROOT / "miner")

    link = tmp_path / "repo-link"
    link.symlink_to(ROOT / "miner", target_is_directory=True)
    with pytest.raises(ValueError, match="outside the repository"):
        harness.make_run_root(link)


def test_explicit_parent_contents_preserved(tmp_path: Path) -> None:
    harness = load_harness()
    parent = tmp_path / "runs"
    parent.mkdir()
    sentinel = parent / "existing.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    run = harness.make_run_root(parent)
    try:
        assert run.parent == parent.resolve()
        assert run != parent
        assert sentinel.read_text(encoding="utf-8") == "keep me"
    finally:
        shutil.rmtree(run, ignore_errors=True)

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_cleanup_removes_only_owned_child(tmp_path: Path) -> None:
    harness = load_harness()
    parent = tmp_path / "runs"
    parent.mkdir()
    sentinel = parent / "existing.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    run = harness.make_run_root(parent)
    harness.write_text(run / "secret.txt", "rpc_password=supersecret")

    harness.cleanup_run_root(run, keep_run=False, secrets_to_hide=["supersecret"])

    assert not run.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_keep_run_preserves_sanitized_owned_child(tmp_path: Path) -> None:
    harness = load_harness()
    run = harness.make_run_root(tmp_path)
    secret = run / "logs" / "gateway.log"
    harness.write_text(secret, "rpc_password=supersecret wallet=rprl1leak")

    harness.cleanup_run_root(run, keep_run=True, secrets_to_hide=["supersecret", "rprl1leak"])

    assert run.exists()
    text = secret.read_text(encoding="utf-8")
    assert "supersecret" not in text
    assert "rprl1leak" not in text
    assert "<redacted>" in text
    shutil.rmtree(run, ignore_errors=True)


def test_harness_rpc_exception_cannot_disclose_credentials(monkeypatch):
    import traceback
    harness = load_harness()
    def fail(*args, **kwargs):
        raise RuntimeError('reg-user secret-password')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', fail)
    rpc = harness.Rpc('http://127.0.0.1:1', 'reg-user', 'secret-password')
    with pytest.raises(RuntimeError) as error:
        rpc.call('getblockcount')
    text = ''.join(traceback.format_exception(error.value))
    assert 'secret-password' not in text and 'reg-user' not in text
