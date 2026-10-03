from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/pmk_regtest_e2e.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("pmk_regtest_e2e", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coinbase_payout_check_accepts_rawtx_shape() -> None:
    harness = load_harness()
    block = {
        "rawtx": [
            {
                "vout": [
                    {"scriptPubKey": {"hex": "51202d6c74a5d8af133f1c65c0a9f99c433498dc807dce022e559c60a3a70e0223ea"}}
                ]
            }
        ]
    }

    outputs = harness.coinbase_outputs(block)
    assert outputs[0]["scriptPubKey"]["hex"].startswith("5120")


def test_log_leak_detection_fails_on_raw_credentials(tmp_path: Path) -> None:
    harness = load_harness()
    log = tmp_path / "gateway.log"
    log.write_text("rpc_password=supersecret wallet=rprl1leak\n")

    with pytest.raises(RuntimeError, match="raw credential or payout material leaked"):
        harness.assert_no_log_leaks([log], ["supersecret", "rprl1leak"])


def test_runtime_scrub_redacts_existing_artifacts(tmp_path: Path) -> None:
    harness = load_harness()
    log = tmp_path / "logs" / "gateway.log"
    log.parent.mkdir()
    log.write_text("rpc_password=supersecret wallet=rprl1leak\n")

    harness.scrub_runtime_secrets(tmp_path, ["supersecret", "rprl1leak"])

    text = log.read_text()
    assert "supersecret" not in text
    assert "rprl1leak" not in text
    assert "<redacted>" in text


def test_gateway_patch_preflight_sentinel_passes(tmp_path: Path) -> None:
    harness = load_harness()
    harness.preflight_gateway_patch(tmp_path)


def test_miner_summary_requires_payout_and_stopped_counts(tmp_path: Path) -> None:
    harness = load_harness()
    log = tmp_path / "miner.log"
    log.write_text(
        "\n".join(
            [
                '{"event":"payout_verified","accepted":1}',
                '{"event":"python_overhead","job_id":7,"python_overhead_pct":0.5,"wall_seconds":1.2}',
                '{"event":"payout_verified","accepted":3}',
                '{"event":"stopped","accepted":3,"completed_ops":123}',
                "",
            ]
        )
    )

    summary = harness.parse_miner_summary(log)

    assert summary["payout_verified"] == 3
    assert summary["stopped_accepted"] == 3
    assert summary["python_overhead_events"] == [
        {"job_id": 7, "python_overhead_pct": 0.5, "wall_seconds": 1.2}
    ]


def test_regtest_e2e_accepts_blocks_and_rejects_corrupted_certificates() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "miner"))
    result = subprocess.run(
        [
            str(ROOT / "scripts/pmk_regtest_e2e.sh"),
            env.get("PMK_REGTEST_TARGET_BLOCKS", "3"),
            env.get("PMK_REGTEST_TIMEOUT_SECONDS", "1500"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(env.get("PMK_REGTEST_PYTEST_TIMEOUT", "1800")),
        check=False,
    )
    assert result.returncode == 0, result.stdout[-8000:]
    assert "RESULT accepted_blocks=" in result.stdout
    assert "coinbase_ok=True" in result.stdout
    assert "corrupted_certificate_rejection=True" in result.stdout
    assert "miner_graceful_stop=True" in result.stdout
