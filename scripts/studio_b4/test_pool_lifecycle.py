from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "pmk_pool_mock_e2e.py"
POOL_WINDOW = ROOT / "scripts" / "studio_b4" / "pool_window.py"
B4_WINDOW = ROOT / "scripts" / "studio_b4" / "b4_window.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
GPU_LOCK_DIR = Path("/tmp/pmm-gpu-bench.lock")


def load_harness() -> Any:
    spec = importlib.util.spec_from_file_location("pmk_pool_mock_e2e", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_real_miner() -> None:
    if not PYTHON.exists():
        pytest.skip("repo .venv is required for the real pool-mode controller test")
    if not (ROOT / "pmkcore" / "target" / "release" / "libpmkcore.dylib").exists():
        pytest.skip("release pmkcore native library is required for the real pool-mode controller test")
    if not (ROOT / "libpmk" / ".build" / "release" / "libpmk.dylib").exists():
        pytest.skip("release libpmk native library is required for the real pool-mode controller test")


def process_exists(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def wait_until_gone(pid: int, *, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.1)
    return not process_exists(pid)


def process_group_members(pgid: int) -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,args="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    members: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and int(fields[1]) == pgid:
            members.append((int(fields[0]), fields[2]))
    return members


def wait_until_group_gone(pgid: int, *, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_members(pgid):
            return True
        time.sleep(0.1)
    return not process_group_members(pgid)


def wait_for_group_payload(pgid: int, needle: str, *, timeout: float = 20.0) -> tuple[int, str] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pid, command in process_group_members(pgid):
            if "process_guard.py" not in command and needle in command:
                return pid, command
        time.sleep(0.1)
    return None


def write_controller_inputs(root: Path, wallet: str) -> tuple[Path, Path, Path, Path]:
    wallet_file = root / "wallet.txt"
    allowlist = root / "wallet-allowlist.txt"
    config = root / "shape.toml"
    summary = root / "pool-summary.json"
    wallet_file.write_text(wallet + "\n", encoding="utf-8")
    allowlist.write_text(wallet + "\n", encoding="utf-8")
    config.write_text("m = 128\nn = 128\nk = 4096\nslots = 2\n", encoding="utf-8")
    return wallet_file, allowlist, config, summary


def controller_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PMK_GPU_LOCK_HELD"] = "1"
    for key in tuple(env):
        if key.startswith(("B4_LAB_", "PMK_LAB_", "PMK_B4_LAB_")):
            env.pop(key)
    env["PYTHONPATH"] = f"{ROOT / 'miner'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    return env


@contextmanager
def exclusive_gpu_lock(timeout: float = 180.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            GPU_LOCK_DIR.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out waiting for exclusive GPU lock {GPU_LOCK_DIR}")
            time.sleep(0.25)
    try:
        yield
    finally:
        GPU_LOCK_DIR.rmdir()


def controller_command(pool_url: str, wallet_file: Path, allowlist: Path, config: Path, summary: Path, *, max_seconds: int) -> list[str]:
    # Exercise the real child-lifecycle API locally. The live-lab main wrapper
    # requires a provider pause/resume owner; no provider is involved here.
    runner = summary.parent / "pool-controller.py"
    runner.write_text(
        "import sys, json\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(POOL_WINDOW.parent)!r})\n"
        "import pool_window as window\n"
        "args = window.parser().parse_args()\n"
        "config = args.summary.parent / 'miner.toml'\n"
        "window.write_pool_config(config, source=args.config, state_dir=args.summary.parent / 'state', "
        "max_accepted=args.max_accepted, max_submitted=args.max_submitted, max_seconds=args.max_seconds)\n"
        "result = window.run_pool_miner(args, config, args.summary.parent / 'miner.jsonl')\n"
        "summary = {'result': result['status'], 'criteria': {'pool': result['status']}, 'steps': {'pool': result}}\n"
        "args.summary.write_text(json.dumps(summary))\n"
        "raise SystemExit(result['exit_code'])\n"
    )
    return [
        str(PYTHON),
        str(runner),
        "--pool-url",
        pool_url,
        "--wallet-file",
        str(wallet_file),
        "--wallet-allowlist",
        str(allowlist),
        "--worker",
        "pytest-lifecycle",
        "--config",
        str(config),
        "--max-accepted",
        "1",
        "--max-seconds",
        str(max_seconds),
        "--shutdown-grace-seconds",
        "1",
        "--summary",
        str(summary),
    ]


def read_until_miner_start(process: subprocess.Popen[str], *, timeout: float = 20.0) -> tuple[int, str]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = ""
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            for key, _ in selector.select(timeout=0.5):
                line = key.fileobj.readline()
                if not line:
                    continue
                output += line
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "miner_start":
                    return int(event["pid"]), output
            if process.poll() is not None:
                break
    finally:
        selector.close()
    raise AssertionError(f"controller did not start miner; rc={process.poll()} output:\n{output}")


def run_controller_timeout(pool_url: str, tmp_path: Path, wallet: str) -> tuple[int, dict[str, Any], str]:
    wallet_file, allowlist, config, summary = write_controller_inputs(tmp_path, wallet)
    env = controller_env()
    with exclusive_gpu_lock():
        result = subprocess.run(
            controller_command(pool_url, wallet_file, allowlist, config, summary, max_seconds=1),
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
    assert summary.exists(), result.stdout
    starts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    groups = [row['pid'] for row in starts if row.get('event') == 'miner_start']
    assert len(groups) == 1, result.stdout
    assert wait_until_group_gone(groups[0]), process_group_members(groups[0])
    return result.returncode, json.loads(summary.read_text(encoding="utf-8")), result.stdout


@pytest.mark.parametrize("controller_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGKILL])
def test_pool_window_signal_reaps_real_miner(tmp_path: Path, controller_signal: signal.Signals) -> None:
    require_real_miner()
    harness = load_harness()

    async def scenario() -> None:
        pool = harness.MockPool(difficulty=10**18, rotate_seconds=60, block_difficulty=10**18)
        await pool.start()
        wallet_file, allowlist, config, summary = write_controller_inputs(tmp_path, harness.DEFAULT_WALLET)
        env = controller_env()
        process: subprocess.Popen[str] | None = None
        miner_pid: int | None = None
        with exclusive_gpu_lock():
            try:
                process = subprocess.Popen(
                    controller_command(pool.url(), wallet_file, allowlist, config, summary, max_seconds=60),
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                miner_pid, output = await asyncio.to_thread(read_until_miner_start, process)
                assert process_exists(miner_pid), output
                payload = await asyncio.to_thread(wait_for_group_payload, miner_pid, "pool_miner_entry.py")
                deadline = time.monotonic() + 20
                while pool.stats.authorized < 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                assert payload is not None and payload[0] != miner_pid, (payload, process_group_members(miner_pid), output)
                assert pool.stats.authorized >= 1, f"real miner never authorized with loopback pool: {output}"
                os.kill(process.pid, controller_signal)
                stdout, _ = process.communicate(timeout=20)
                expected = -signal.SIGKILL if controller_signal == signal.SIGKILL else 1
                assert process.returncode == expected, output + stdout
                assert wait_until_group_gone(miner_pid), f"miner group still alive pgid={miner_pid}\n{output}{stdout}"
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                if miner_pid is not None and process_group_members(miner_pid):
                    os.killpg(miner_pid, signal.SIGKILL)
        await pool.stop()

    asyncio.run(scenario())


def test_b4_controller_sigkill_reaps_real_pearld_gateway_and_miner(tmp_path: Path) -> None:
    require_real_miner()
    controller = tmp_path / "b4-lifecycle-controller.py"
    controller.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(B4_WINDOW.parent)!r})",
                "import b4_window",
                f"b4_window.python_path = lambda: Path({str(PYTHON)!r})",
                f"b4_window.RUN_ROOT = Path({str(tmp_path / 'b4-run')!r})",
                "b4_window.SUMMARY = b4_window.RUN_ROOT / 'summary.json'",
                "b4_window.regtest_g5(shape=b4_window.QUICK, target_blocks=100, timeout_seconds=300, name='lifecycle_sigkill')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = controller_env()
    process: subprocess.Popen[str] | None = None
    groups: dict[str, int] = {}
    output = ""
    with exclusive_gpu_lock():
        try:
            process = subprocess.Popen(
                [str(PYTHON), str(controller)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 90
            try:
                while len(groups) < 3 and time.monotonic() < deadline:
                    for key, _ in selector.select(timeout=0.5):
                        line = key.fileobj.readline()
                        output += line
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("event") == "proc_start":
                            groups[str(event["name"])] = int(event["pid"])
                    if process.poll() is not None:
                        break
            finally:
                selector.close()
            assert set(groups) == {"pearld", "gateway", "miner"}, output
            payloads = {
                "pearld": wait_for_group_payload(groups["pearld"], "pearld"),
                "gateway": wait_for_group_payload(groups["gateway"], "gateway"),
                "miner": wait_for_group_payload(groups["miner"], "b4_miner_entry.py"),
            }
            assert all(payloads.values()), (payloads, {name: process_group_members(pgid) for name, pgid in groups.items()}, output)
            assert all(payload and payload[0] != groups[name] for name, payload in payloads.items()), payloads
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            assert process.returncode == -signal.SIGKILL
            for name, pgid in groups.items():
                assert wait_until_group_gone(pgid), f"{name} group still alive pgid={pgid}: {process_group_members(pgid)}"
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            for pgid in groups.values():
                if process_group_members(pgid):
                    os.killpg(pgid, signal.SIGKILL)


def test_pool_window_timeout_without_verdict_is_inconclusive(tmp_path: Path) -> None:
    require_real_miner()
    harness = load_harness()

    async def scenario() -> None:
        pool = harness.MockPool(difficulty=10**18, rotate_seconds=60, block_difficulty=10**18)
        await pool.start()
        try:
            returncode, summary, output = await asyncio.to_thread(
                run_controller_timeout, pool.url(), tmp_path, harness.DEFAULT_WALLET
            )
            assert returncode == 3, output
            assert summary["result"] == "INCONCLUSIVE"
            assert summary["criteria"]["pool"] == "INCONCLUSIVE"
            assert summary["steps"]["pool"]["status"] == "INCONCLUSIVE"
            assert summary["steps"]["pool"]["accepted"] == 0
        finally:
            await pool.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["startup_exception", "payload_sigkill"])
def test_b4_startup_exception_reaps_real_pearld(tmp_path: Path, monkeypatch, failure: str) -> None:
    """Fault after a real spawn, before the caller can append its Proc."""
    spec = importlib.util.spec_from_file_location("b4_startup_failure", B4_WINDOW)
    assert spec and spec.loader
    window = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, window)
    spec.loader.exec_module(window)
    groups = []

    def fail_after_spawn(event, **fields):
        if event != "proc_start":
            return
        pgid = fields["pid"]
        groups.append(pgid)
        payload = wait_for_group_payload(pgid, "pearld")
        assert payload is not None
        if failure == "startup_exception":
            raise RuntimeError("injected failure after real service spawn")
        os.kill(payload[0], signal.SIGKILL)

    monkeypatch.setattr(window, "jlog", fail_after_spawn)
    command = [str(window.find_pearld()), "--regtest", "--nodnsseed", "--nolisten", "--norpc",
               "--connect=127.0.0.1:1", f"--datadir={tmp_path / 'data'}", f"--logdir={tmp_path / 'logs'}"]
    procs = []
    try:
        if failure == "startup_exception":
            with pytest.raises(RuntimeError, match="injected failure"):
                procs.append(window.start_proc("pearld", command, tmp_path / "pearld.log", env=window.bundle_env()))
            assert not procs, "fault must occur before caller registers Proc"
        else:
            procs.append(window.start_proc("pearld", command, tmp_path / "pearld.log", env=window.bundle_env()))
            assert procs[0].process.wait(timeout=5) == -signal.SIGKILL
        window.stop_all(procs)
        assert len(groups) == 1
        assert wait_until_group_gone(groups[0]), process_group_members(groups[0])
    finally:
        window.stop_all(procs)
        for pgid in groups:
            if process_group_members(pgid):
                os.killpg(pgid, signal.SIGKILL)
