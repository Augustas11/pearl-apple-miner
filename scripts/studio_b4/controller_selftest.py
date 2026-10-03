#!/usr/bin/env python3
"""No-GPU checks for the B4 window controller."""

from __future__ import annotations

import json
import os
import argparse
import asyncio
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import ast
import builtins
import shutil
import signal
import time
from unittest.mock import patch

import b4_window
import pool_window
import perf


def check_manifest_corruption() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / "payload.txt"
        data.write_text("ok")
        manifest = root / "MANIFEST.sha256"
        manifest.write_text("0" * 64 + "  payload.txt\n")
        old_root, old_manifest = b4_window.ROOT, b4_window.MANIFEST
        b4_window.ROOT = root
        b4_window.MANIFEST = manifest
        try:
            try:
                b4_window.verify_manifest()
            except b4_window.StepFailure:
                return
            raise AssertionError("corrupt manifest passed")
        finally:
            b4_window.ROOT = old_root
            b4_window.MANIFEST = old_manifest


def check_p6_digest_matching() -> None:
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "p6.jsonl"
        rows = [
            {"event": "gpu_find", "proof_digest": "a", "header_hash": "h1", "gpu_find_time": 10.0},
            {"event": "proof_handoff", "proof_digest": "a", "header_hash": "h1", "handoff_time": 12.0},
            {"event": "submit_verdict", "proof_digest": "a", "header_hash": "h1", "verdict_time": 14.0, "verdict": "rejected"},
            {"event": "submit_verdict", "proof_digest": "z", "header_hash": "h2", "verdict_time": 15.0},
            {"event": "submit_verdict", "proof_digest": "a", "header_hash": "h1", "verdict_time": 20.0, "verdict": None},
            {"event": "submit_verdict", "proof_digest": "a", "header_hash": "h1", "verdict_time": 21.0, "verdict": "accepted"},
        ]
        log.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        matched = b4_window.p6_correlated_latencies(log)
        assert len(matched) == 1, matched
        assert matched[0]["proof_digest"] == "a"
        assert matched[0]["gpu_find_to_accept_seconds"] == 11.0


def check_timeout_kills_group() -> None:
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "ignore_term.py"
        marker = Path(td) / "marker"
        script.write_text(
            textwrap.dedent(
                f"""
                import os, signal, time
                signal.signal(signal.SIGTERM, lambda *_: None)
                open({str(marker)!r}, 'w').write(str(os.getpid()))
                while True:
                    time.sleep(1)
                """
            )
        )
        try:
            b4_window.run_cmd([sys.executable, str(script)], timeout=1, check=False)
        except b4_window.StepFailure:
            pass
        else:
            raise AssertionError("timeout command unexpectedly succeeded")
        pid = int(marker.read_text())
        probe = subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert probe.returncode != 0, f"child still alive pid={pid}"


def check_no_undefined_controller_refs() -> None:
    path = Path(__file__).with_name("b4_window.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    defined: set[str] = set(dir(builtins))
    defined.update({"__file__", "__name__", "__annotations__"})
    imports: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add((alias.asname or alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    missing = sorted(used - defined - imports)
    # Comprehension targets and local assignment flow are intentionally not
    # modeled fully; keep this allowlist small and concrete.
    missing = [name for name in missing if name not in {"p", "cmd", "row", "block", "proc", "candidate", "path", "raw", "rel", "secret"}]
    assert "peak_rss_kb" not in used
    assert not missing, missing


def check_pip_hash_lock_rejects_tampered_wheel() -> None:
    python = shutil.which("python3") or sys.executable
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wheel = root / "demo_pkg-1.0-py3-none-any.whl"
        wheel.write_bytes(b"tampered")
        lock = root / "requirements.lock"
        lock.write_text(
            "demo-pkg==1.0 "
            "--hash=sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        )
        out = root / "out"
        result = subprocess.run(
            [
                python,
                "-m",
                "pip",
                "download",
                "--no-index",
                "--find-links",
                str(root),
                "--require-hashes",
                "--no-deps",
                "-r",
                str(lock),
                "-d",
                str(out),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    assert result.returncode != 0, result.stdout
    assert "DO NOT MATCH THE HASHES" in result.stdout, result.stdout


def check_perf_failure_preserves_sustained_windows() -> None:
    stderr = "\n".join(
        [
            '{"completed_jobs":433,"elapsed_seconds":30.0,"event":"sustained_window","shares":0,"tops":7.9}',
            '{"completed_jobs":817,"elapsed_seconds":60.0,"event":"sustained_window","shares":1,"tops":7.0}',
        ]
    )
    result = b4_window._perf_failure_result(
        ["sustained"],
        timeout=840,
        returncode=-9,
        stdout="",
        stderr=stderr,
        error_type="ProcessFailure",
        error="perf.py failed rc=-9",
    )
    assert result["pass"] is False
    assert result["signal"] == "SIGKILL"
    assert result["jobs"] == 817
    assert len(result["windows"]) == 2
    assert result["last_window"]["tops"] == 7.0


def check_perf_stage_and_timeline_summary() -> None:
    class FakeRecord:
        def __init__(self, job_id, stages):
            self.job_id = job_id
            self.stage_seconds = stages
            self.cancelled = False

    records = [
        FakeRecord(1, {"build": 0.002, "k1": 0.003, "k3": 0.010}),
        FakeRecord(2, {"build": 0.004, "k1": 0.005, "k3": 0.012}),
    ]
    stages = perf._stage_summary(records)
    assert stages["build"]["count"] == 2
    assert stages["build"]["ms_median"] == 3.0
    logs = [
        {"event": "gpu_dispatch", "job_id": 1, "inflight": 1},
        {"event": "gpu_dispatch", "job_id": 2, "inflight": 2},
        {"event": "completed", "job_id": 1, "gpu_start_time": 10.0, "gpu_end_time": 20.0},
        {"event": "completed", "job_id": 2, "gpu_start_time": 15.0, "gpu_end_time": 25.0},
        {"event": "completed", "job_id": 3, "gpu_start_time": 30.0, "gpu_end_time": 35.0},
    ]
    timeline = perf._gpu_timeline(logs)
    assert timeline["max_inflight"] == 2
    assert timeline["gpu_union_seconds"] == 20.0
    assert timeline["gpu_idle_gap_seconds"] == 5.0
    assert timeline["gpu_overlap_seconds"] == 5.0


def check_run_perf_records_stderr_only_failure() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script_dir = root / "scripts" / "studio_b4"
        script_dir.mkdir(parents=True)
        fake = script_dir / "perf.py"
        fake.write_text(
            textwrap.dedent(
                """
                import sys
                print('{"event":"sustained_window","completed_jobs":3,"tops":1.25,"shares":0}', file=sys.stderr, flush=True)
                raise SystemExit(1)
                """
            )
        )
        old_root, old_run_root = b4_window.ROOT, b4_window.RUN_ROOT
        try:
            b4_window.ROOT = root
            b4_window.RUN_ROOT = root / ".b4_run"
            original_python_path = b4_window.python_path
            original_bundle_env = b4_window.bundle_env
            b4_window.python_path = lambda: Path(sys.executable)
            b4_window.bundle_env = lambda extra=None: os.environ.copy()
            result = b4_window.run_perf(["sustained"], timeout=5)
        finally:
            b4_window.ROOT = old_root
            b4_window.RUN_ROOT = old_run_root
            b4_window.python_path = original_python_path
            b4_window.bundle_env = original_bundle_env
        assert result["pass"] is False
        assert result["error_type"] == "MissingJSON"
        assert result["jobs"] == 3
        assert result["windows"][0]["tops"] == 1.25


def check_p2_brackets_k3_baselines() -> None:
    calls = []

    def fake_k3(args):
        calls.append("k3")
        rate = 100.0 if len(calls) == 1 else 120.0
        return {"ops_per_second": rate, "tops": rate / 1e12, "jobs": args.jobs}

    async def fake_pipeline(args, native):
        calls.append("pipeline")
        return {"ops_per_second": 90.0, "jobs": args.jobs}

    class FakeNative:
        def close(self):
            calls.append("close")

    old_k3, old_pipeline, old_native = perf.run_k3_alone_external, perf.run_pipeline_once, perf.new_native
    try:
        perf.run_k3_alone_external = fake_k3
        perf.run_pipeline_once = fake_pipeline
        perf.new_native = lambda: FakeNative()
        args = argparse.Namespace(
            m=128,
            n=128,
            k=4096,
            jobs=4,
            quick=True,
            min_ratio=0.85,
            share_calibration_tops=1.0,
        )
        result = asyncio.run(perf.run_p2(args))
    finally:
        perf.run_k3_alone_external = old_k3
        perf.run_pipeline_once = old_pipeline
        perf.new_native = old_native
    assert result["paired_order"] == ["k3_before", "pipeline", "k3_after"]
    assert result["ratio_before"] == 0.9
    assert result["ratio_after"] == 0.75
    assert result["ratio_conservative"] == 0.75
    assert result["ratio"] == result["ratio_conservative"]
    assert result["pass"] is False


def check_p6_step_is_structured() -> None:
    g5 = {
        "p6_correlated": [{"proof_digest": "a"}],
        "gpu_find_accept_latency_seconds": [1.5],
        "proof_handoff_accept_latency_seconds": [0.5],
        "p6_peak_gateway_tree_rss_kb": 123,
        "p6_log": "/tmp/p6.jsonl",
    }
    ok = b4_window.p6_step_from_g5(g5, 1)
    assert ok["pass"] is True
    assert ok["matched"] == 1
    assert ok["p6_peak_gateway_tree_rss_kb"] == 123
    fail = b4_window.p6_step_from_g5(g5, 2)
    assert fail["pass"] is False
    assert "need 2" in fail["error"]


def check_pipeline_profile_serial_is_pipeline_only() -> None:
    parser = perf.parser()
    pipeline_args = parser.parse_args(["pipeline", "--profile-serial", "--slots", "2", "--dry-run"])
    p2_args = parser.parse_args(["p2", "--slots", "2", "--dry-run"])
    shape = argparse.Namespace(slots=2)
    assert perf._pipeline_worker_count(pipeline_args, shape) == 1
    assert perf._pipeline_worker_count(p2_args, shape) == 2
    try:
        parser.parse_args(["p2", "--profile-serial"])
    except SystemExit:
        pass
    else:
        raise AssertionError("p2 unexpectedly accepted --profile-serial")


def check_sustained_window_diagnostics() -> None:
    def expected(ops, nbits):
        assert nbits == 0x1
        return ops / 100.0

    def poisson(value):
        return value - 1.0, value + 1.0

    window = perf.sustained_window_summary(
        elapsed_seconds=30.0,
        interval_seconds=10.0,
        delta_ops=200,
        completed_jobs=2,
        shares_total=7,
        shares_delta=2,
        expected_fn=expected,
        poisson_fn=poisson,
        share_nbits=0x1,
    )
    assert window["ops"] == 200
    assert window["shares"] == 7
    assert window["shares_delta"] == 2
    assert window["expected_shares"] == 2.0
    assert window["poisson_lower"] == 1.0
    assert window["poisson_upper"] == 3.0
    assert window["poisson_pass"] is True
    assert isinstance(window["peak_rss_bytes"], int)
    assert window["peak_rss_bytes"] > 0


def check_sustained_profile_retention() -> None:
    profile = perf.BoundedProfile(2)
    for job_id in range(20):
        record = argparse.Namespace(job_id=job_id + 1, cancelled=False, stage_seconds={})
        for event in ('gpu_dispatch', 'completed', 'python_overhead'):
            profile.add_log({'event': event, 'job_id': record.job_id})
        profile.add_record(record)
        row = profile.take_row_for_record(record, mode='sustained')
        assert row['completed']['job_id'] == record.job_id
        assert not profile.completed_by_job
        assert not profile.overhead_by_job
        assert not profile.dispatch_by_job
    assert len(profile.records) == 2 and len(profile.logs) <= 8


def check_sustained_short_run_keeps_tail() -> None:
    from dataclasses import dataclass
    @dataclass
    class Shape:
        slots: int = 2
        ops: int = 100
    class Pipeline:
        def __init__(self, native, shape, capture):
            self.capture = capture
            self.sequence = 0
        def set_template(self, source):
            pass
        def cancel(self):
            pass
        async def run(self, *args):
            await asyncio.sleep(.001)
            self.sequence += 1
            self.capture('completed', job_id=self.sequence, shares=0, blocks=0, wall_seconds=.001)
            return argparse.Namespace(job_id=self.sequence, cancelled=False, stage_seconds={})
    original = {key: getattr(perf, key) for key in
                ('_load_miner_modules', 'validate_shape', 'new_native', '_append_profile', 'synthetic_job')}
    mods = {
        'Pipeline': Pipeline,
        'bits_to_target': lambda nbits: nbits,
        'choose_share_nbits': lambda ops_per_second, target_shares_per_minute: 1,
        'expected_shares': lambda ops, nbits: 0.0,
        'poisson_interval': lambda expected: (-1.0, 1.0),
    }
    try:
        perf._load_miner_modules = lambda: mods
        perf.validate_shape = lambda args: (Shape(), 0)
        perf.new_native = lambda: argparse.Namespace(close=lambda: None)
        perf._append_profile = lambda rows: None
        perf.synthetic_job = lambda bits: object()
        args = perf.parser().parse_args(['sustained', '--seconds', '.02', '--report-interval', '30',
                                         '--ops-per-second', '1000000000000'])
        result = asyncio.run(perf.run_sustained(args))
    finally:
        for key, value in original.items():
            setattr(perf, key, value)
    assert result['jobs'] > 0
    assert len(result['windows']) == 1
    assert result['windows'][0]['partial'] is True
    assert result['windows'][0]['ops'] == result['ops']
    assert result['windows'][0]['interval_seconds'] > 0


def check_pool_window_command_and_summary() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wallet = root / "wallet.txt"
        allowlist = root / "allowlist.txt"
        source = root / "base.toml"
        summary_path = root / "summary.json"
        wallet.write_text("prl1redacted\n")
        allowlist.write_text("prl1redacted\n")
        source.write_text("m = 128\nn = 256\nk = 4096\nslots = 1\n")
        args = pool_window.parser().parse_args(
            [
                "--pool-url",
                "stratum+tcp://127.0.0.1:1234",
                "--wallet-file",
                str(wallet),
                "--wallet-allowlist",
                str(allowlist),
                "--worker",
                "studio-test",
                "--config",
                str(source),
                "--max-accepted",
                "1",
                "--max-submitted",
                "2",
                "--max-seconds",
                "3",
                "--summary",
                str(summary_path),
                "--dry-run",
                "--no-require-inherited-gpu-lock",
            ]
        )
        old_root, old_run_root = pool_window.ROOT, pool_window.RUN_ROOT
        try:
            pool_window.ROOT = root
            pool_window.RUN_ROOT = root / ".pool_run"
            original_create_offline_venv = pool_window.create_offline_venv
            pool_window.create_offline_venv = lambda: {"python": sys.executable, "status": "test"}
            try:
                assert pool_window.main([
                    "--pool-url",
                    args.pool_url,
                    "--wallet-file",
                    str(args.wallet_file),
                    "--wallet-allowlist",
                    str(args.wallet_allowlist),
                    "--worker",
                    args.worker,
                    "--config",
                    str(args.config),
                    "--max-accepted",
                    "1",
                    "--max-submitted",
                    "2",
                    "--max-seconds",
                    "3",
                    "--summary",
                    str(args.summary),
                    "--dry-run",
                    "--no-require-inherited-gpu-lock",
                ]) == 0
            finally:
                pool_window.create_offline_venv = original_create_offline_venv
            run_dirs = list((root / ".pool_run").iterdir())
            assert len(run_dirs) == 1
            generated = run_dirs[0] / "pmk-pool.toml"
            text = generated.read_text()
            assert "Only root shape keys are copied" in text
            assert "m = 128" in text
            assert "max_accepted = 1" in text
            assert "max_submitted = 2" in text
            command = pool_window.miner_cmd(args, generated)
            assert command[1].endswith("pool_miner_entry.py")
            assert "--wallet-allowlist" in command
            assert "--lab-owner-token" not in command
            assert "--pool-stats-url" not in command
            assert "prl1redacted" not in " ".join(command)
            summary = json.loads(summary_path.read_text())
            assert summary["result"] == "DRY_RUN"
            assert summary["criteria"]["pool"] == "PASS"
            assert summary["steps"]["pool"]["submitted"] == 2
            assert summary["steps"]["pool"]["accepted_ratio"] == 1.0
        finally:
            pool_window.ROOT = old_root
            pool_window.RUN_ROOT = old_run_root


def check_pool_window_repairs_stale_pool_helper_venv() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        venv_python = root / ".venv-b4/bin/python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("# fake python\n")
        pip = root / ".venv-b4/bin/pip"
        pip.write_text("# fake pip\n")
        wheels = root / "wheels"
        wheels.mkdir()
        runtime_lock = root / "miner/requirements.lock"
        runtime_lock.parent.mkdir()
        runtime_lock.write_text("runtime\n")
        local_lock = root / "local-wheels.lock"
        local_lock.write_text("local\n")
        old_root = pool_window.ROOT
        calls: list[list[str]] = []

        def fake_run_cmd(args: list[str], *, timeout: int, check: bool = True):
            calls.append(args)
            if args[:2] == [str(venv_python), "-c"] and "nbits_to_difficulty" in args[2] and len([c for c in calls if c[:2] == [str(venv_python), "-c"] and "nbits_to_difficulty" in c[2]]) == 1:
                return subprocess.CompletedProcess(args, 1, stdout="missing pool helper(s): extract_difficulty_bound\n")
            return subprocess.CompletedProcess(args, 0, stdout="")

        try:
            pool_window.ROOT = root
            with patch.object(pool_window, "run_cmd", fake_run_cmd):
                result = pool_window.create_offline_venv()
            assert result == {"python": str(venv_python), "status": "repaired"}
            assert calls[0][:2] == [str(venv_python), "-c"]
            assert str(local_lock) in calls[0]
            assert calls[1][:2] == [str(venv_python), "-c"]
            assert "nbits_to_difficulty" in calls[1][2]
            assert calls[2][:2] == [str(pip), "install"]
            assert str(runtime_lock) in calls[2]
            assert calls[3][:2] == [str(pip), "install"]
            assert str(local_lock) in calls[3]
            assert calls[4] == [str(pip), "check"]
            assert calls[5][:2] == [str(venv_python), "-c"]
            assert str(local_lock) in calls[5]
            assert calls[6][:2] == [str(venv_python), "-c"]
            assert "nbits_to_difficulty" in calls[6][2]
        finally:
            pool_window.ROOT = old_root


def check_pool_window_repairs_stale_local_wheel_hash_venv() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        venv_python = root / ".venv-b4/bin/python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("# fake python\n")
        pip = root / ".venv-b4/bin/pip"
        pip.write_text("# fake pip\n")
        wheels = root / "wheels"
        wheels.mkdir()
        runtime_lock = root / "miner/requirements.lock"
        runtime_lock.parent.mkdir()
        runtime_lock.write_text("runtime\n")
        local_lock = root / "local-wheels.lock"
        local_lock.write_text("py-pearl-mining==0.3.1 --hash=sha256:e5018d3118d5756a1bed6427c0185e6429ab73156462593ddc54d3a4846bbac6\n")
        old_root = pool_window.ROOT
        calls: list[list[str]] = []

        def fake_run_cmd(args: list[str], *, timeout: int, check: bool = True):
            calls.append(args)
            if args[:2] == [str(venv_python), "-c"] and "import base64" in args[2] and len([c for c in calls if c[:2] == [str(venv_python), "-c"] and "import base64" in c[2]]) == 1:
                return subprocess.CompletedProcess(args, 1, stdout="py-pearl-mining: installed file hash mismatch pearl_mining/pearl_mining.abi3.so\n")
            return subprocess.CompletedProcess(args, 0, stdout="")

        try:
            pool_window.ROOT = root
            with patch.object(pool_window, "run_cmd", fake_run_cmd):
                result = pool_window.create_offline_venv()
            assert result == {"python": str(venv_python), "status": "repaired"}
            assert calls[0][:2] == [str(venv_python), "-c"]
            assert "import base64" in calls[0][2]
            assert not any("nbits_to_difficulty" in c[2] for c in calls[:1] if c[:2] == [str(venv_python), "-c"])
            assert calls[1][:2] == [str(pip), "install"]
            assert str(runtime_lock) in calls[1]
            assert calls[2][:2] == [str(pip), "install"]
            assert str(local_lock) in calls[2]
            assert calls[3] == [str(pip), "check"]
            assert calls[4][:2] == [str(venv_python), "-c"]
            assert "import base64" in calls[4][2]
            assert calls[5][:2] == [str(venv_python), "-c"]
            assert "nbits_to_difficulty" in calls[5][2]
        finally:
            pool_window.ROOT = old_root


def check_pool_window_rejects_empty_or_malformed_local_wheel_lock() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wheels = root / "wheels"
        wheels.mkdir()
        local_lock = root / "local-wheels.lock"

        for lock_text in (
            "# empty\n",
            "py-pearl-mining==0.3.1\n",
            "not a requirement\n",
        ):
            local_lock.write_text(lock_text)
            ok = pool_window.verify_local_wheel_records(Path(sys.executable), wheels, local_lock, check=False)
            assert ok is False


def check_pool_window_accepts_tls_alias() -> None:
    assert pool_window.sanitize_pool_url("stratum+tls://127.0.0.1:8048") == "stratum+tls://127.0.0.1:8048"


def check_pool_window_requires_inherited_lock() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wallet = root / "wallet.txt"
        allowlist = root / "allowlist.txt"
        wallet.write_text("prl1redacted\n")
        allowlist.write_text("prl1redacted\n")
        args = argparse.Namespace(
            wallet_file=wallet,
            wallet_allowlist=allowlist,
            require_inherited_gpu_lock=True,
            dry_run=True,
            max_accepted=1,
            max_submitted=None,
            max_seconds=10,
            shutdown_grace_seconds=1,
        )
        with patch.dict(os.environ, {}, clear=True):
            try:
                pool_window.validate_paths(args)
            except pool_window.PoolWindowError as exc:
                assert "PMK_GPU_LOCK_HELD" in str(exc)
            else:
                raise AssertionError("pool window accepted a missing inherited GPU lock")


def check_pool_window_real_run_rejects_disabled_inherited_lock() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wallet = root / "wallet.txt"
        allowlist = root / "allowlist.txt"
        wallet.write_text("prl1redacted\n")
        allowlist.write_text("prl1redacted\n")
        args = argparse.Namespace(
            wallet_file=wallet,
            wallet_allowlist=allowlist,
            require_inherited_gpu_lock=False,
            dry_run=False,
            max_accepted=1,
            max_submitted=None,
            max_seconds=10,
            shutdown_grace_seconds=1,
        )
        with patch.dict(os.environ, {"B4_LAB_OWNER_TOKEN": "owner"}, clear=True):
            try:
                pool_window.validate_paths(args)
            except pool_window.PoolWindowError as exc:
                assert "inherited GPU lock" in str(exc)
            else:
                raise AssertionError("pool window accepted a real run with inherited lock enforcement disabled")


def check_pool_window_real_run_requires_lab_token() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wallet = root / "wallet.txt"
        allowlist = root / "allowlist.txt"
        gpu_lock = root / "gpu-lock"
        wallet.write_text("prl1redacted\n")
        allowlist.write_text("prl1redacted\n")
        gpu_lock.mkdir()
        args = argparse.Namespace(
            wallet_file=wallet,
            wallet_allowlist=allowlist,
            require_inherited_gpu_lock=True,
            dry_run=False,
            max_accepted=1,
            max_submitted=None,
            max_seconds=10,
            shutdown_grace_seconds=1,
        )
        old_gpu_lock = pool_window.GPU_LOCK_DIR
        try:
            pool_window.GPU_LOCK_DIR = gpu_lock
            with patch.dict(os.environ, {"PMK_GPU_LOCK_HELD": "1"}, clear=True):
                try:
                    pool_window.validate_paths(args)
                except pool_window.PoolWindowError as exc:
                    assert "lab owner token" in str(exc)
                else:
                    raise AssertionError("pool window accepted a real run without env token")
        finally:
            pool_window.GPU_LOCK_DIR = old_gpu_lock


def check_pool_window_rejects_secret_pool_urls_without_summary_leak() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wallet = root / "wallet.txt"
        allowlist = root / "allowlist.txt"
        summary_path = root / "summary.json"
        wallet.write_text("prl1redacted\n")
        allowlist.write_text("prl1redacted\n")
        old_root, old_run_root = pool_window.ROOT, pool_window.RUN_ROOT
        try:
            pool_window.ROOT = root
            pool_window.RUN_ROOT = root / ".pool_run"
            for bad in (
                "stratum+tcp://user:secret@127.0.0.1:1234",
                "stratum+tcp://127.0.0.1:1234?wallet=secret",
            ):
                rc = pool_window.main([
                    "--pool-url",
                    bad,
                    "--wallet-file",
                    str(wallet),
                    "--wallet-allowlist",
                    str(allowlist),
                    "--worker",
                    "studio-test",
                    "--summary",
                    str(summary_path),
                    "--dry-run",
                    "--no-require-inherited-gpu-lock",
                ])
                assert rc == 1
                text = summary_path.read_text()
                assert "secret" not in text
                assert "wallet=secret" not in text
                summary = json.loads(text)
                assert summary["pool_endpoint"] is None
        finally:
            pool_window.ROOT = old_root
            pool_window.RUN_ROOT = old_run_root


def check_pool_window_summary_criteria() -> None:
    args = argparse.Namespace(max_accepted=20, max_submitted=20)
    good = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 20, "stale": 0, "rejected": 0, "gate_failures": 0, "poisson_ok": True, "pass": True},
        args,
        returncode=0,
    )
    assert good["pass"] is True
    stale = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 19, "stale": 1, "rejected": 0, "gate_failures": 0, "poisson_ok": True, "pass": True},
        args,
        returncode=0,
    )
    assert stale["submitted"] == 20
    assert stale["rejected"] == 0
    assert stale["pass"] is False
    assert any("stale ratio" in reason for reason in stale["fail_reasons"])
    incomplete = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 3, "stale": 0, "rejected": 0, "gate_failures": 0, "poisson_ok": True, "pass": True},
        args,
        returncode=0,
    )
    assert incomplete["pass"] is False
    explicit_reject = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 19, "stale": 0, "rejected": 1, "invalid": 1, "gate_failures": 0, "poisson_ok": True, "pass": True},
        args,
        returncode=0,
    )
    assert explicit_reject["rejected"] == 1
    submitted_event = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 19, "stale": 1, "submitted": 20, "failed": 1, "rejected": 0, "invalid": 1, "gate_failures": 0, "poisson_ok": True, "pass": True},
        args,
        returncode=0,
    )
    assert submitted_event["submitted"] == 20
    assert submitted_event["failed"] == 1
    assert submitted_event["rejected"] == 0
    failed_event = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 20, "stale": 0, "rejected": 0, "gate_failures": 0, "poisson_ok": True, "pass": False},
        args,
        returncode=0,
    )
    assert failed_event["pass"] is False

    t2 = argparse.Namespace(max_accepted=1, max_submitted=1)
    accepted_first = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 1, "stale": 0, "rejected": 0, "submitted": 1, "gate_failures": 0, "poisson_ok": True, "pass": True},
        t2,
        returncode=0,
    )
    assert accepted_first["status"] == "PASS"
    rejected_first = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 0, "stale": 0, "rejected": 1, "submitted": 1, "gate_failures": 0, "poisson_ok": True, "pass": False},
        t2,
        returncode=1,
    )
    assert rejected_first["status"] == "FAIL"
    assert rejected_first["exit_code"] == 2
    timeout_no_verdict = pool_window.classify_summary(
        {"event": "pool_summary", "accepted": 0, "stale": 0, "rejected": 0, "submitted": 1, "timeout": 1, "gate_failures": 0, "poisson_ok": True, "pass": True},
        t2,
        returncode=0,
    )
    assert timeout_no_verdict["status"] == "INCONCLUSIVE"
    assert timeout_no_verdict["exit_code"] == 3


def check_pool_window_signal_reaps_miner_group() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script_dir = root / "scripts" / "studio_b4"
        script_dir.mkdir(parents=True)
        marker = root / "miner-pid.txt"
        (script_dir / "pool_miner_entry.py").write_text(
            textwrap.dedent(
                f"""
                import os
                import signal
                import time

                signal.signal(signal.SIGTERM, lambda *_: None)
                {str(marker)!r} and open({str(marker)!r}, "w").write(str(os.getpid()))
                while True:
                    time.sleep(1)
                """
            )
        )
        controller = root / "controller.py"
        controller.write_text(
            textwrap.dedent(
                f"""
                import argparse
                import sys
                from pathlib import Path

                sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
                import pool_window

                root = Path({str(root)!r})
                pool_window.ROOT = root
                pool_window.RUN_ROOT = root / ".pool_run"
                args = argparse.Namespace(
                    pool_url="stratum+tcp://127.0.0.1:1",
                    wallet_file=root / "wallet.txt",
                    wallet_allowlist=root / "allowlist.txt",
                    worker="studio-test",
                    max_accepted=1,
                    max_submitted=1,
                    max_seconds=3600,
                    shutdown_grace_seconds=1,
                )
                raise SystemExit(0 if pool_window.run_pool_miner(args, root / "cfg.toml", root / "miner.log") else 1)
                """
            )
        )
        proc = subprocess.Popen([sys.executable, str(controller)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 5
            while not marker.exists() and time.time() < deadline:
                time.sleep(0.05)
            assert marker.exists(), (proc.poll(), proc.stdout.read() if proc.stdout else "")
            miner_pid = int(marker.read_text())
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=8)
            deadline = time.time() + 5
            while time.time() < deadline:
                probe = subprocess.run(["ps", "-p", str(miner_pid)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                if probe.returncode != 0:
                    break
                time.sleep(0.05)
            assert subprocess.run(["ps", "-p", str(miner_pid)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).returncode != 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def main() -> int:
    check_manifest_corruption()
    check_p6_digest_matching()
    check_timeout_kills_group()
    check_no_undefined_controller_refs()
    check_pip_hash_lock_rejects_tampered_wheel()
    check_perf_failure_preserves_sustained_windows()
    check_perf_stage_and_timeline_summary()
    check_run_perf_records_stderr_only_failure()
    check_p2_brackets_k3_baselines()
    check_p6_step_is_structured()
    check_pipeline_profile_serial_is_pipeline_only()
    check_sustained_window_diagnostics()
    check_sustained_profile_retention()
    check_sustained_short_run_keeps_tail()
    check_pool_window_command_and_summary()
    check_pool_window_repairs_stale_pool_helper_venv()
    check_pool_window_repairs_stale_local_wheel_hash_venv()
    check_pool_window_rejects_empty_or_malformed_local_wheel_lock()
    check_pool_window_accepts_tls_alias()
    check_pool_window_requires_inherited_lock()
    check_pool_window_real_run_rejects_disabled_inherited_lock()
    check_pool_window_real_run_requires_lab_token()
    check_pool_window_rejects_secret_pool_urls_without_summary_leak()
    check_pool_window_summary_criteria()
    check_pool_window_signal_reaps_miner_group()
    print("controller selftest ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
