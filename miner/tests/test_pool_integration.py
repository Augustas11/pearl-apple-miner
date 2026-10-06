from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/pmk_pool_mock_e2e.py"
PYTHON = ROOT / ".venv/bin/python"
MAINNET_BITS = 0x177FD82E
REAL_DIFFICULTY_VECTORS = (2**21, 2_000_000, 50_000, 10_000)


def load_harness():
    spec = importlib.util.spec_from_file_location("pmk_pool_mock_e2e", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('already_exited', [False, True])
def test_mock_cleanup_handles_denied_process_group(monkeypatch, already_exited):
    harness = load_harness()
    process = subprocess.Popen([sys.executable, '-c',
        'pass' if already_exited else 'import time; time.sleep(60)'], start_new_session=True)
    try:
        if already_exited:
            process.wait(timeout=10)
        def denied(*_args):
            raise PermissionError('process group is no longer signalable')
        monkeypatch.setattr(harness.os, 'killpg', denied)
        harness.terminate(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_mock_pool_rejects_wrong_job_ids() -> None:
    harness = load_harness()

    async def scenario() -> bool:
        pool = harness.MockPool(difficulty=1000, rotate_seconds=15, block_difficulty=1_000_000_000)
        await pool.start()
        try:
            return await harness.direct_wrong_job_probe(pool)
        finally:
            await pool.stop()

    import asyncio

    assert asyncio.run(scenario())


def test_target_bits_for_mock_pool_difficulty() -> None:
    harness = load_harness()
    pool = harness.MockPool(difficulty=1000, rotate_seconds=15, block_difficulty=1_000_000_000)
    assert pool.target == harness.DIFF1_TARGET // 1000
    assert harness.bits_to_target(pool.share_nbits) <= pool.target


def test_mock_pool_can_emit_mainnet_bits_header() -> None:
    harness = load_harness()
    header = harness.make_header(1, block_difficulty=1_000_000_000, block_nbits=MAINNET_BITS)
    assert int.from_bytes(header[72:76], "little") == MAINNET_BITS


def test_pool_cli_parses_explicit_block_nbits(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = load_harness()
    monkeypatch.setattr(sys, "argv", ["pmk_pool_mock_e2e.py", "--block-nbits", "0x177fd82e"])
    assert harness.parse_args().block_nbits == MAINNET_BITS


def test_dispatch_config_sets_pool_runtime_max_jobs(tmp_path: Path) -> None:
    harness = load_harness()
    config = tmp_path / "pmk_pool.toml"
    harness.make_config(
        config,
        state_dir=tmp_path / "state",
        summary_file=tmp_path / "summary.json",
        max_accepted=0,
        max_jobs=2,
        max_seconds=30,
        difficulty_floor=10_000,
        m=8192,
        n=8192,
        k=4096,
        slots=2,
    )
    text = config.read_text(encoding="utf-8")
    assert "max_jobs = 2" in text
    assert "max_accepted = 0" in text


def gpu_pool_e2e_available() -> bool:
    return (
        PYTHON.exists()
        and (ROOT / "pmkcore/target/release/libpmkcore.dylib").exists()
        and (ROOT / "libpmk/.build/release/libpmk.dylib").exists()
    )


def run_harness(tmp_path: Path, name: str, *extra: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "miner"))
    env.setdefault("PMK_POOL_T1_RUN_ROOT", str(tmp_path / "runs"))
    evidence_dir = env.get("PMK_POOL_T1_EVIDENCE_DIR")
    result = subprocess.run(
        [str(PYTHON), str(HARNESS), *extra],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout if timeout is not None else int(env.get("PMK_POOL_T1_PYTEST_TIMEOUT", "1800")),
        check=False,
    )
    evidence = (Path(evidence_dir) if evidence_dir else tmp_path) / name
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(result.stdout, encoding="utf-8")
    return result


def test_pool_mock_e2e_accepts_50_real_k3sg_shares(tmp_path: Path, v3_g3_admission) -> None:
    if not PYTHON.exists():
        pytest.skip("repo .venv is required for the real pool-mode miner e2e")
    if not (ROOT / "pmkcore/target/release/libpmkcore.dylib").exists():
        pytest.skip("release pmkcore native library is required for pool e2e")
    if not (ROOT / "libpmk/.build/release/libpmk.dylib").exists():
        pytest.skip("release libpmk native library is required for pool e2e")

    result = run_harness(tmp_path, "b6_t1.txt")
    assert result.returncode == 0, result.stdout[-12000:]
    assert "RESULT accepted=" in result.stdout
    assert "corrupted_proof_gate_caught=True" in result.stdout
    assert "wrong_job_id_rejected=True" in result.stdout
    assert "mismatched_existing_job_rejected=True" in result.stdout
    assert "first_reject_stopped=True" in result.stdout
    assert "malicious_cert_refused=True" in result.stdout
    assert "malicious_target_refused=True" in result.stdout


@pytest.mark.skipif(not gpu_pool_e2e_available(), reason="release native libraries are required for GPU pool e2e")
def test_pool_mock_accepts_shortened_share_with_mainnet_bits(tmp_path: Path, v3_g3_admission) -> None:
    result = run_harness(
        tmp_path,
        "b6live_mainnet_bits_acceptance.txt",
        "--positive-only",
        "--difficulty", "1000",
        "--difficulty-floor", "1000",
        "--block-nbits", hex(MAINNET_BITS),
        "--target-shares", "1",
        "--max-seconds", os.environ.get("PMK_POOL_T1_SHORT_MAX_SECONDS", "300"),
    )
    assert result.returncode == 0, result.stdout[-12000:]
    assert "RESULT accepted=" in result.stdout
    assert "corrupted_proof_gate_caught=True" in result.stdout
    assert "mismatched_existing_job_rejected=True" in result.stdout


@pytest.mark.skipif(not gpu_pool_e2e_available(), reason="release native libraries are required for GPU pool e2e")
def test_pool_mock_real_difficulty_dispatch_vectors_complete_without_device_errors(
    tmp_path: Path, v3_g3_admission
) -> None:
    completed = {}
    for difficulty in REAL_DIFFICULTY_VECTORS:
        result = run_harness(
            tmp_path,
            f"b6live_dispatch_d{difficulty}.txt",
            "--dispatch-only",
            "--difficulty", str(difficulty),
            "--difficulty-floor", str(min(10_000, difficulty)),
            "--block-nbits", hex(MAINNET_BITS),
            "--target-completed-jobs", os.environ.get("PMK_POOL_T1_DISPATCH_JOBS", "2"),
            "--max-seconds", os.environ.get("PMK_POOL_T1_DISPATCH_MAX_SECONDS", "300"),
            "--m", os.environ.get("PMK_POOL_T1_DISPATCH_M", "8192"),
            "--n", os.environ.get("PMK_POOL_T1_DISPATCH_N", "8192"),
            "--k", os.environ.get("PMK_POOL_T1_DISPATCH_K", "4096"),
            "--slots", os.environ.get("PMK_POOL_T1_DISPATCH_SLOTS", "2"),
        )
        assert result.returncode == 0, result.stdout[-12000:]
        assert "fatal_events=0" in result.stdout
        assert "device_or_verifier_alerts=0" in result.stdout
        summary_path = tmp_path / "runs"
        completed[str(difficulty)] = "RESULT dispatch_completed_jobs=" in result.stdout
        assert not any("FatalDeviceError" in line for line in result.stdout.splitlines() if '"event":"fatal"' in line)
        assert summary_path.exists()
    assert completed == {str(difficulty): True for difficulty in REAL_DIFFICULTY_VECTORS}
