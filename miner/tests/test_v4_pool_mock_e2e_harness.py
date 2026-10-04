# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
import base64
import pmk_miner.pool as pool_mod

import importlib.util
from pathlib import Path

from pmk_miner.pool import validate_pool_notify

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pmk_v4_pool_mock_e2e.py"
spec = importlib.util.spec_from_file_location("pmk_v4_pool_mock_e2e", SCRIPT)
harness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(harness)


def test_fixture_notify_plan_uses_pool_valid_bits_and_v4_ancestor_extension():
    header, ancestor, m, n, k = harness.load_fixture(harness.DEFAULT_FIXTURE)
    assert len(header) == 76
    assert len(ancestor) == 108
    assert (m, n, k) == (32, 64, 1024)
    assert int.from_bytes(header[72:76], "little") == harness.POOL_BITS

    params = harness.notify_params("v4-current", header, cert_version=4, ancestor=ancestor)
    assert params["job_id"] == "v4-current"
    assert params["cert_version"] == 4
    assert params["target"] == f"{harness.POOL_TARGET:064x}"
    assert base64.b64decode(params["ancestor_headers"][0], validate=True) == ancestor


def test_dry_plan_records_gpu_gate_and_loopback_checks():
    args = harness.build_parser().parse_args([])
    plan = harness.dry_plan(args)
    assert plan["schema"] == "pmk-v4-pool-mock-e2e-plan-v1"
    assert plan["real_pool"] is False
    assert plan["loopback_only"] is True
    assert plan["requires_real_v4_g3_admission_file"] is True
    assert plan["run_gpu_required_env"] == "PMK_GPU_LOCK_HELD=1"
    assert plan["shape"] == {"m": 2048, "n": 2048, "k": 4096, "slots": 2}
    assert plan["pool_difficulty"] == 1
    checks = " ".join(plan["checks"])
    assert "same Pipeline" in checks
    assert "real locally verified v3 plain_proof" in checks
    assert "late real v3 share returns stale" in checks
    assert "pmkcore_v4_verify_plain_proof" in checks
    assert plan["target_accepted_shares"] == 3


def test_cpu_check_uses_intersection_shape_not_tiny_fixture():
    args = harness.build_parser().parse_args([])
    result = harness.cpu_check(args)
    assert result["shape"] == {"m": 2048, "n": 2048, "k": 4096, "slots": 2}
    assert result["fixture_shape"] == {"m": 32, "n": 64, "k": 1024}
    assert result["pool_difficulty"] == 1


def test_argument_policy_keeps_v3_v4_intersection_shape():
    parser = harness.build_parser()
    harness.validate_args(parser.parse_args(["--m", "2048", "--n", "2048", "--k", "4096"]))
    try:
        harness.validate_args(parser.parse_args(["--m", "32", "--n", "64", "--k", "1024"]))
    except SystemExit as exc:
        assert "--m" in str(exc) or "--k" in str(exc)
    else:
        raise AssertionError("tiny v4-only shape must not be valid for no-restart harness")


def test_notify_params_are_valid_with_explicit_local_mock_floor(monkeypatch):
    header, ancestor, *_ = harness.load_fixture(harness.DEFAULT_FIXTURE)
    params = harness.notify_params("v4-current", header, cert_version=4, ancestor=ancestor)
    monkeypatch.setattr(pool_mod, "validate_v4_g3_admission_file", lambda: {"library_sha256": "a" * 64})
    job = validate_pool_notify(params, session_id=1, difficulty_floor=harness.POOL_DIFFICULTY)
    assert job.target == harness.POOL_TARGET
    assert job.cert_version == 4
