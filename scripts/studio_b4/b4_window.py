#!/usr/bin/env python3
"""B4 long-run window controller.

The controller runs entirely from the copied bundle on the target Mac. It only
starts loopback regtest services that it owns, writes logs under ~/pmk-b4-work/logs,
and cleans up its child process groups on every exit path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("B4_WINDOW_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUN_ROOT = ROOT / ".b4_run"
SUMMARY = ROOT / "b4_summary.json"
MANIFEST = ROOT / "MANIFEST.sha256"
G3_ADMISSION_FILE = RUN_ROOT / "g3-admission.json"
LAB_RESUME_REPORT = RUN_ROOT / "lab-resume-report.json"
GPU_LOCK_DIR = Path("/tmp/pmm-gpu-bench.lock")
MINING_ADDR = "rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"
EXPECTED_SCRIPT = "51202d6c74a5d8af133f1c65c0a9f99c433498dc807dce022e559c60a3a70e0223ea"
PRODUCTION = {"m": 8192, "n": 8192, "k": 4096, "slots": 2}
QUICK = {"m": 128, "n": 128, "k": 4096, "slots": 2}
WINDOW_BUDGET_SECONDS = min(2400, int(os.environ.get("B4_WINDOW_BUDGET_SECONDS", "2400")))
# B7 LabSession allows up to 600s for confirmed provider recovery.
CLEANUP_RESERVE_SECONDS = 615
EXIT_STEP_FAILURE = 1
EXIT_CRITERION_FAILURE = 2
_OWNED_LAUNCHES: dict[int, tuple[subprocess.Popen[str], int]] = {}


def controller_signals() -> list[signal.Signals]:
    values = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        values.append(signal.SIGHUP)
    return values


@contextmanager
def cleanup_signal_shield():
    """Keep repeated interrupts non-raising until recovery is recorded.

    Block only handler transitions: recovery hooks must not inherit a blocked
    TERM mask (their timeout still needs to work). Python no-op handlers reset
    on exec, unlike SIG_IGN, so hooks retain normal signal behavior.
    """
    handled = [*controller_signals(), signal.SIGALRM]
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
    previous = {signum: signal.getsignal(signum) for signum in handled}
    try:
        for signum in handled:
            signal.signal(signum, lambda _signum, _frame: None)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, handled)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


@dataclass(slots=True)
class Proc:
    name: str
    process: subprocess.Popen[str]
    log_path: Path
    sentinel_write_fd: int


@dataclass(slots=True)
class LabWindow:
    token: str | None
    resume_hook: Path | None
    report: Path | None
    lock: Path
    session: Any | None = None


class CriterionFailure(RuntimeError):
    pass


class StepFailure(RuntimeError):
    pass


class Rpc:
    def __init__(self, url: str, user: str, password: str) -> None:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.url = url
        self.authorization = f"Basic {token}"
        self.request_id = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, method: str, params: list[Any] | None = None, *, timeout: float = 15) -> Any:
        self.request_id += 1
        payload = json.dumps(
            {"jsonrpc": "1.0", "id": self.request_id, "method": method, "params": params or []}
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"content-type": "text/plain", "authorization": self.authorization},
            method="POST",
        )
        with self.opener.open(req, timeout=timeout) as response:
            body = json.loads(response.read().decode())
        if body.get("error") is not None:
            raise RuntimeError(f"{method} failed: {body['error']}")
        return body["result"]


def jlog(event: str, **fields: Any) -> None:
    print(json.dumps({"time": time.time(), "event": event, **fields}, separators=(",", ":")), flush=True)


def guarded_popen(args: list[str], **kwargs: Any) -> tuple[subprocess.Popen[str], int]:
    sentinel_read_fd, sentinel_write_fd = os.pipe()
    guard = Path(__file__).resolve().with_name("process_guard.py")
    command = [str(Path(sys.executable)), str(guard), str(sentinel_read_fd), "--", *args]
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, controller_signals())
    try:
        try:
            process = subprocess.Popen(command, pass_fds=(sentinel_read_fd,), **kwargs)
        except BaseException:
            os.close(sentinel_read_fd)
            os.close(sentinel_write_fd)
            raise
        os.close(sentinel_read_fd)
        _OWNED_LAUNCHES[process.pid] = (process, sentinel_write_fd)
        return process, sentinel_write_fd
    except BaseException:
        raise
    finally:
        # If a controller signal arrived during Popen, it is delivered only
        # after the sentinel and process are present in the owned registry.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def release_owned_process(process: subprocess.Popen[str], fd: int) -> None:
    _OWNED_LAUNCHES.pop(process.pid, None)
    try:
        os.close(fd)
    except OSError:
        pass


def run_cmd(
    args: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    jlog("cmd_start", argv=args, cwd=str(cwd), timeout=timeout)
    process, sentinel_write_fd = guarded_popen(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process_group(process)
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            kill_process_group(process)
            stdout, _ = process.communicate()
        kill_process_group(process)
        release_owned_process(process, sentinel_write_fd)
        print(stdout, end="")
        raise StepFailure(f"command timed out after {timeout}s: {' '.join(args)}") from exc
    except BaseException:
        terminate_process_group(process)
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            kill_process_group(process)
            process.communicate()
        kill_process_group(process)
        release_owned_process(process, sentinel_write_fd)
        raise
    kill_process_group(process)
    release_owned_process(process, sentinel_write_fd)
    result = subprocess.CompletedProcess(args, process.returncode, stdout, None)
    print(result.stdout, end="")
    jlog("cmd_end", argv=args[:2], returncode=result.returncode)
    if check and result.returncode != 0:
        raise StepFailure(f"command failed rc={result.returncode}: {' '.join(args)}")
    return result


def remaining_timeout(start: float, requested: int) -> int:
    remaining = WINDOW_BUDGET_SECONDS - CLEANUP_RESERVE_SECONDS - int(time.monotonic() - start)
    if remaining <= 0:
        raise StepFailure(f"B4 window exceeded {WINDOW_BUDGET_SECONDS}s budget")
    return max(1, min(requested, remaining))


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def start_proc(name: str, args: list[str], log_path: Path, *, env: dict[str, str]) -> Proc:
    secure_dir(log_path.parent)
    log_path.touch(mode=0o600, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8")
    try:
        process, sentinel_write_fd = guarded_popen(
            args,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        fh.close()
    jlog("proc_start", name=name, pid=process.pid, log=str(log_path))
    return Proc(name, process, log_path, sentinel_write_fd)


def stop_all(procs: list[Proc]) -> None:
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, controller_signals())
    try:
        owned = list(_OWNED_LAUNCHES.values())
        for process, _fd in reversed(owned):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 8
        for process, _fd in reversed(owned):
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.1)
        for process, _fd in reversed(owned):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        for process, fd in owned:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                kill_process_group(process)
                process.wait(timeout=2)
            finally:
                release_owned_process(process, fd)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def assert_alive(procs: list[Proc], *, allow_success: set[str] | None = None) -> None:
    allow_success = allow_success or set()
    for proc in procs:
        rc = proc.process.poll()
        if rc is None or (proc.name in allow_success and rc == 0):
            continue
        tail = ""
        if proc.log_path.exists():
            tail = "\n".join(proc.log_path.read_text(errors="replace").splitlines()[-80:])
        raise StepFailure(f"{proc.name} exited rc={rc}; tail:\n{tail}")


def wait_for_rpc(rpc: Rpc, deadline: float) -> int:
    last: Exception | None = None
    while time.time() < deadline:
        try:
            return int(rpc.call("getblockcount", timeout=5))
        except (OSError, urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            last = exc
            time.sleep(0.5)
    raise StepFailure(f"pearld RPC did not become ready: {last}")


def load_harness_module() -> Any:
    path = ROOT / "scripts" / "pmk_regtest_e2e.py"
    spec = importlib.util.spec_from_file_location("pmk_regtest_e2e_bundle", path)
    if spec is None or spec.loader is None:
        raise StepFailure(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def python_path() -> Path:
    return ROOT / ".venv-b4" / "bin" / "python"


def load_runtime_module() -> Any:
    path = ROOT / "miner" / "pmk_miner" / "runtime.py"
    spec = importlib.util.spec_from_file_location("pmk_miner_runtime_bundle", path)
    if spec is None or spec.loader is None:
        raise StepFailure(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bundle_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    dyld = [str(ROOT / "libpmk" / ".build" / "release"), str(ROOT / "pmkcore" / "target" / "release")]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "miner"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join([*dyld, env.get("DYLD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    env["PMK_RESOURCE_BUNDLE"] = str(ROOT / "libpmk" / ".build" / "release" / "libpmk_PMK.bundle")
    env["PMK_B4_ROOT"] = str(ROOT)
    env["PMKCORE_DYLIB"] = str(ROOT / "pmkcore" / "target" / "release" / "libpmkcore.dylib")
    env["LIBPMK_DYLIB"] = str(ROOT / "libpmk" / ".build" / "release" / "libpmk.dylib")
    env["PMK_G3_ADMISSION_FILE"] = str(G3_ADMISSION_FILE)
    for key in list(env):
        if key.lower().endswith("_proxy"):
            env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost,::1"
    env["no_proxy"] = env["NO_PROXY"]
    if extra:
        env.update(extra)
    return env


def prepare_lab_window() -> LabWindow:
    lock = Path.home() / ".lab-window.lock"
    token = os.environ.get("B4_LAB_OWNER_TOKEN") or os.environ.get("PMK_LAB_OWNER_TOKEN")
    hook_raw = os.environ.get("B4_LAB_RESUME_HOOK") or os.environ.get("PMK_LAB_RESUME_HOOK")
    hook = Path(hook_raw).expanduser() if hook_raw else None
    report_raw = os.environ.get("B4_LAB_RESUME_REPORT") or os.environ.get("PMK_LAB_RESUME_REPORT")
    report = Path(report_raw).expanduser() if report_raw else (LAB_RESUME_REPORT if token or hook else None)
    runtime = load_runtime_module()
    session = runtime.LabSession(token=token, resume_hook=hook, report=report, lock=lock)
    window = LabWindow(token=token, resume_hook=hook, report=report, lock=lock, session=session)
    if lock.exists() or token or hook:
        session.check()
        os.environ["PMK_B4_CONTROLLER_LAB_SESSION"] = str(os.getpid())
        if token:
            os.environ["PMK_B4_LAB_OWNER_TOKEN"] = token
        if hook:
            os.environ["PMK_B4_LAB_RESUME_HOOK"] = str(hook)
        if report:
            os.environ["PMK_B4_LAB_RESUME_REPORT"] = str(report)
    return window


def finish_lab_window(window: LabWindow | None) -> dict[str, Any] | None:
    if window is None or window.session is None or not getattr(window.session, "owned", False):
        return None
    outcome = window.session.finish()
    if isinstance(outcome, dict):
        return outcome
    return None


def cleanup_lab_window(window: LabWindow | None, summary: dict[str, Any]) -> None:
    with cleanup_signal_shield():
        # Cancel the active deadline only after its handler is shielded.
        signal.alarm(0)
        try:
            stop_all([])
        finally:
            try:
                outcome = finish_lab_window(window)
            except Exception:
                summary["criteria"]["lab_resume"] = "FAIL"
                raise
            if outcome:
                summary["steps"]["lab_resume"] = outcome
                summary["criteria"]["lab_resume"] = "PASS"
                finish_summary(summary)


@contextmanager
def gpu_window_lock():
    runtime = load_runtime_module()
    previous = os.environ.get("PMK_GPU_LOCK_HELD")
    with runtime.gpu_lock(jlog, path=GPU_LOCK_DIR):
        os.environ["PMK_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("PMK_GPU_LOCK_HELD", None)
            else:
                os.environ["PMK_GPU_LOCK_HELD"] = previous


def find_pearld() -> Path:
    for candidate in (
        ROOT / "bin" / "pearld",
        ROOT / "vendor" / "pearl" / "bin" / "pearld",
        ROOT / "vendor" / "pearl-fp8" / "bin" / "pearld",
    ):
        if candidate.exists():
            return candidate
    raise StepFailure("missing pearld in bundle")


def gateway_source() -> Path:
    for candidate in (
        ROOT / "vendor" / "pearl" / "miner" / "pearl-gateway",
        ROOT / "vendor" / "pearl-fp8" / "miner" / "pearl-gateway",
        ROOT / "vendor" / "pearl-patched" / "pearl-gateway",
    ):
        if candidate.exists():
            return candidate
    raise StepFailure("missing pearl-gateway source in bundle")


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise StepFailure("missing MANIFEST.sha256")
    checked = 0
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 2:
            raise StepFailure(f"bad manifest line: {raw!r}")
        expected, rel = parts[0], parts[-1].lstrip("*")
        if rel == "MANIFEST.sha256":
            continue
        path = ROOT / rel
        if not path.is_file():
            raise StepFailure(f"manifest file missing: {rel}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise StepFailure(f"manifest mismatch: {rel}")
        checked += 1
    if not checked:
        raise StepFailure("empty manifest")
    return {"checked_files": checked}


def create_offline_venv() -> dict[str, Any]:
    py = Path(os.environ.get("B4_STUDIO_PYTHON") or os.environ.get("PMK_PYTHON") or sys.executable)
    if not py.exists():
        py = Path(sys.executable)
    venv = ROOT / ".venv-b4"
    wheels = ROOT / "wheels"
    if not wheels.is_dir():
        raise StepFailure("missing offline wheels/ directory")
    if not (venv / "bin" / "python").exists():
        run_cmd([str(py), "-m", "venv", str(venv)], timeout=180)
    version = run_cmd(
        [str(venv / "bin" / "python"), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
        timeout=30,
    ).stdout.strip()
    if version != "3.12":
        raise StepFailure(f"offline venv must use Python 3.12, got {version}")
    pip = venv / "bin" / "pip"
    wheel_count = len(list(wheels.glob("*.whl")))
    runtime_lock = ROOT / "miner" / "requirements.lock"
    local_lock = ROOT / "local-wheels.lock"
    if not runtime_lock.is_file():
        raise StepFailure("missing miner/requirements.lock")
    if not local_lock.is_file():
        raise StepFailure("missing local-wheels.lock")
    if wheel_count == 0:
        raise StepFailure("offline wheels/ directory contains no wheels")
    run_cmd(
        [
            str(pip),
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--require-hashes",
            "--force-reinstall",
            "-r",
            str(runtime_lock),
        ],
        timeout=600,
    )
    run_cmd(
        [
            str(pip),
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--require-hashes",
            "--no-deps",
            "--force-reinstall",
            "-r",
            str(local_lock),
        ],
        timeout=300,
    )
    run_cmd([str(pip), "check"], timeout=60)
    run_cmd([str(python_path()), "-c", "import pytest, pearl_mining, pmk_miner; print('offline venv ok')"], timeout=60)
    return {"python": str(python_path()), "wheel_count": wheel_count, "runtime_lock": str(runtime_lock), "local_lock": str(local_lock)}


def preflight() -> dict[str, Any]:
    details: dict[str, Any] = {
        "manifest": verify_manifest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for cmd in (
        ["sysctl", "hw.model"],
        ["sysctl", "machdep.cpu.brand_string"],
        ["sysctl", "hw.memsize"],
        ["sysctl", "kern.osproductversion"],
        ["vm_stat"],
        ["df", "-h", str(Path.home())],
    ):
        result = run_cmd(cmd, timeout=30, check=False)
        details[" ".join(cmd)] = {"rc": result.returncode, "out": result.stdout[-4000:]}
    processes = subprocess.check_output(["ps", "-axo", "pid=,rss=,comm="], text=True)
    # Optional: PMK_PREFLIGHT_PS_PATTERN (regex) lists other heavy processes (pid, RSS) for the record.
    ps_pattern = os.environ.get("PMK_PREFLIGHT_PS_PATTERN")
    provider_rows = [line for line in processes.splitlines() if ps_pattern and re.search(ps_pattern, line, re.I)]
    print("watched process RSS (KiB):\n" + "\n".join(provider_rows), flush=True)
    details["provider_rss"] = provider_rows
    details["venv"] = create_offline_venv()
    return details


def g3_admission() -> dict[str, Any]:
    G3_ADMISSION_FILE.unlink(missing_ok=True)
    helper = ROOT / "bin" / "g3-admit"
    if not helper.is_file():
        raise StepFailure("missing bin/g3-admit full G3 admission helper")
    result = run_cmd([str(helper)], env=bundle_env(), timeout=600, check=False)
    if result.returncode != 0:
        print("DO NOT MINE")
        raise CriterionFailure("full G3 admission helper failed; DO NOT MINE")
    admission: dict[str, Any] | None = None
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("devices"), list):
            admission = value
    if admission is None:
        raise StepFailure("full G3 helper produced no admission record")
    secure_dir(G3_ADMISSION_FILE.parent)
    write_text(G3_ADMISSION_FILE, json.dumps(admission, indent=2, sort_keys=True) + "\n")
    return {"admission_file": str(G3_ADMISSION_FILE), "devices": admission["devices"]}


def probe() -> dict[str, Any]:
    code = (
        "from pmk_miner.native import Native\n"
        "n=Native(); print('probe_key', n.probe_key); n.close()\n"
    )
    result = run_cmd([str(python_path()), "-c", code], env=bundle_env(), timeout=180, check=False)
    if result.returncode != 0:
        print("DO NOT MINE")
        raise CriterionFailure("libpmk K3-SG probe failed; DO NOT MINE")
    match = re.search(r"probe_key\s+(\S+)", result.stdout)
    key = match.group(1) if match else None
    records = json.loads(G3_ADMISSION_FILE.read_text())["devices"]
    if not key or not any(record.get("cache_key") == key for record in records):
        raise CriterionFailure("production dylib probe differs from G3 admission; DO NOT MINE")
    return {"probe_key": key}


def production_smoke() -> dict[str, Any]:
    result = run_cmd(
        [str(python_path()), str(ROOT / "scripts" / "pmk_production_smoke.py"), "--slots", "2"],
        env=bundle_env({"PMK_GPU_LOCK_HELD": "1"}),
        timeout=600,
        check=False,
    )
    summary: dict[str, Any] | None = None
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") == "production_shape_smoke":
            summary = value
    if result.returncode != 0 or summary is None or not summary.get("passed"):
        raise CriterionFailure("production-shape Native + Pipeline smoke failed")
    if summary.get("verified_submissions", 0) < 1:
        raise CriterionFailure("production-shape smoke produced no verified proof")
    poll = summary.get("poll_result", {})
    if poll.get("status") != 0 or poll.get("overflow"):
        raise CriterionFailure(f"production-shape smoke returned invalid GPU result: {poll}")
    return summary


def pytest_fast() -> dict[str, Any]:
    args = [
        str(python_path()),
        "-m",
        "pytest",
        "-q",
        "miner/tests",
        "-k",
        "not regtest_e2e_accepts_blocks_and_rejects_corrupted_certificates and not pool_mock_e2e_accepts_50_real_k3sg_shares",
    ]
    result = run_cmd(args, env=bundle_env(), timeout=300, check=False)
    if result.returncode != 0:
        raise StepFailure("fast miner pytest subset failed")
    return {"returncode": result.returncode}


def make_config(path: Path, *, env_file: Path, gateway_log: Path, state_dir: Path, target: int, shape: dict[str, int], rpc_url: str, rpc_user: str, rpc_pass: str) -> None:
    write_text(
        path,
        f"""m = {shape['m']}
n = {shape['n']}
k = {shape['k']}
slots = {shape['slots']}

[gateway]
env_file = "{env_file}"
log_file = "{gateway_log}"

[node_rpc]
rpc_url = "{rpc_url}"
rpc_user = "{rpc_user}"
rpc_password = "{rpc_pass}"
mining_address = "{MINING_ADDR}"

[payout]
hrp = "rprl"
script = "{EXPECTED_SCRIPT}"

[run]
state_dir = "{state_dir}"
max_accepted = {target}
stop_after_cert_rejection = true
template_poll_seconds = 0.15
""",
    )


def gateway_cmd(run: Path) -> list[str]:
    if (ROOT / "patched-gateway" / "src").is_dir():
        return [str(python_path()), str(ROOT / "scripts" / "studio_b4" / "b4_gateway_tap.py")]
    return [
        str(python_path()),
        "-m",
        "pmk_miner.gateway_launcher",
        "--source",
        str(gateway_source()),
        "--copy-parent",
        str(run / "gateway-copy"),
        "--tap-script",
        str(ROOT / "scripts" / "studio_b4" / "b4_gateway_tap.py"),
        "--",
        "start",
        "--debug",
    ]


def miner_cmd(config: Path, gw_port: int) -> list[str]:
    cmd = [
        str(python_path()),
        str(ROOT / "scripts" / "studio_b4" / "b4_miner_entry.py"),
        "--mode",
        "solo",
        "--gateway",
        f"127.0.0.1:{gw_port}",
        "--config",
        str(config),
    ]
    token = os.environ.get("PMK_B4_LAB_OWNER_TOKEN")
    hook = os.environ.get("PMK_B4_LAB_RESUME_HOOK")
    report = os.environ.get("PMK_B4_LAB_RESUME_REPORT")
    if token and hook and report:
        cmd.extend(["--resume-hook", hook, "--resume-report", report])
    return cmd


def parse_json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def p6_correlated_latencies(p6_log: Path) -> list[dict[str, Any]]:
    if not p6_log.exists():
        return []
    finds: dict[str, dict[str, Any]] = {}
    handoffs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for line in p6_log.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = event.get("event")
        digest = event.get("proof_digest")
        if not isinstance(digest, str):
            continue
        if name == "gpu_find":
            finds[digest] = event
        elif name == "proof_handoff":
            handoffs[digest] = event
        elif name == "submit_verdict":
            if event.get("verdict") != "accepted":
                continue
            find = finds.get(digest)
            handoff = handoffs.get(digest)
            if not find or not handoff:
                continue
            verdict_time = float(event.get("verdict_time", event.get("time", 0.0)))
            gpu_time = float(find["gpu_find_time"])
            rows.append(
                {
                    "header_hash": event.get("header_hash") or find.get("header_hash"),
                    "proof_digest": digest,
                    "header_fields": event.get("header_fields"),
                    "gpu_find_time": gpu_time,
                    "proof_handoff_time": handoff.get("handoff_time"),
                    "accept_time": verdict_time,
                    "gpu_find_to_accept_seconds": max(0.0, verdict_time - gpu_time),
                    "proof_handoff_to_accept_seconds": max(0.0, verdict_time - float(handoff.get("handoff_time", verdict_time))),
                    "verdict": event.get("verdict"),
                }
            )
    return rows


def process_tree_rss_kb(root_pid: int) -> tuple[int, list[int]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid, ppid, value = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = value
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return sum(rss.get(pid, 0) for pid in seen), sorted(seen)


def regtest_g5(*, shape: dict[str, int], target_blocks: int, timeout_seconds: int, name: str) -> dict[str, Any]:
    harness = load_harness_module()
    run = RUN_ROOT / name
    if run.exists():
        shutil.rmtree(run)
    secure_dir(run)
    secure_dir(run / "logs")
    rpc_port, p2p_port, gw_port, dead_port = free_port(), free_port(), free_port(), free_port()
    rpc_user = f"b4-{secrets.token_hex(4)}"
    rpc_pass = secrets.token_hex(16)
    rpc_url = f"http://127.0.0.1:{rpc_port}"
    env_file = run / "gateway.env"
    config_file = run / "pmk.toml"
    pearld_log = run / "logs" / "pearld.out"
    pearld_conf = run / "pearld.conf"
    gateway_log = run / "logs" / "gateway.log"
    state_dir = run / "state"
    tap_log = run / "logs" / "tap.log"
    p6_log = run / "logs" / "p6.jsonl"
    miner_log = run / "logs" / "miner.log"
    env = bundle_env()
    gateway_env = dict(env)
    gateway_env.update(
        {
            "PEARLD_RPC_URL": rpc_url,
            "PEARLD_RPC_USER": rpc_user,
            "PEARLD_RPC_PASSWORD": rpc_pass,
            "PEARLD_MINING_ADDRESS": MINING_ADDR,
            "MINER_RPC_TRANSPORT": "tcp",
            "MINER_RPC_HOST": "127.0.0.1",
            "MINER_RPC_PORT": str(gw_port),
            "RTAP_LOG": str(tap_log),
            "RTAP_CORRUPT_FIRST": "1",
            "PMK_B4_P6_LOG": str(p6_log),
        }
    )
    if (ROOT / "patched-gateway" / "src").is_dir():
        gateway_env["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT / "scripts" / "studio_b4"),
                str(ROOT / "patched-gateway" / "src"),
                str(ROOT / "miner"),
                gateway_env.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)
    env["PMK_B4_P6_LOG"] = str(p6_log)
    write_text(env_file, f"PEARLD_RPC_URL={rpc_url}\nPEARLD_RPC_USER={rpc_user}\nPEARLD_RPC_PASSWORD={rpc_pass}\nPEARLD_MINING_ADDRESS={MINING_ADDR}\n")
    write_text(pearld_conf, "")
    secure_dir(state_dir)
    make_config(config_file, env_file=env_file, gateway_log=gateway_log, state_dir=state_dir, target=target_blocks, shape=shape, rpc_url=rpc_url, rpc_user=rpc_user, rpc_pass=rpc_pass)
    procs: list[Proc] = []
    summary: dict[str, Any] = {"shape": shape, "target_blocks": target_blocks, "run": str(run)}
    try:
        pearld_args = [
            str(find_pearld()),
            "--regtest",
            f"--configfile={pearld_conf}",
            f"--datadir={run / 'pearld'}",
            f"--logdir={run / 'logs/pearld'}",
            f"--rpcuser={rpc_user}",
            f"--rpcpass={rpc_pass}",
            f"--rpclisten=127.0.0.1:{rpc_port}",
            f"--listen=127.0.0.1:{p2p_port}",
            "--nodnsseed",
            f"--connect=127.0.0.1:{dead_port}",
            "--notls",
            "--addrindex",
            "--txindex",
            f"--miningaddr={MINING_ADDR}",
            "--debuglevel=info",
        ]
        procs.append(start_proc("pearld", pearld_args, pearld_log, env=env))
        rpc = Rpc(rpc_url, rpc_user, rpc_pass)
        start_height = wait_for_rpc(rpc, time.time() + 60)
        template = rpc.call("getblocktemplate", [{"rules": ["segwit"]}])
        if int(template.get("requiredcertversion", -1)) != 3:
            raise CriterionFailure(f"expected requiredcertversion=3, got {template.get('requiredcertversion')}")
        procs.append(start_proc("gateway", gateway_cmd(run), gateway_log, env=gateway_env))
        time.sleep(2)
        assert_alive(procs)
        miner_proc = start_proc("miner", miner_cmd(config_file, gw_port), miner_log, env=env)
        procs.append(miner_proc)
        goal = start_height + target_blocks
        deadline = time.time() + timeout_seconds
        peak_gateway_tree_rss = 0
        peak_gateway_tree_pids: list[int] = []
        final_height = start_height
        while time.time() < deadline:
            for _ in range(20):
                gateway_proc = next((p for p in procs if p.name == "gateway"), None)
                if gateway_proc is not None:
                    rss_kb, tree_pids = process_tree_rss_kb(gateway_proc.process.pid)
                    if rss_kb > peak_gateway_tree_rss:
                        peak_gateway_tree_rss = rss_kb
                        peak_gateway_tree_pids = tree_pids
                time.sleep(0.25)
                if time.time() >= deadline:
                    break
            final_height = int(rpc.call("getblockcount", timeout=8))
            jlog(
                "g5_progress",
                height=final_height,
                goal=goal,
                peak_gateway_tree_rss_kb=peak_gateway_tree_rss,
                gateway_tree_pids=peak_gateway_tree_pids,
            )
            if final_height >= goal:
                break
            assert_alive(procs, allow_success={"miner"})
        if final_height < goal:
            raise CriterionFailure(f"G5 timed out at height {final_height}; goal {goal}")
        miner_proc.process.wait(timeout=45)
        if miner_proc.process.returncode != 0:
            raise StepFailure(f"miner exited with {miner_proc.process.returncode}")
        miner_summary = harness.parse_miner_summary(miner_log)
        if miner_summary["payout_verified"] < target_blocks or (miner_summary["stopped_accepted"] or 0) < target_blocks:
            raise CriterionFailure("miner did not confirm the required payout count before clean exit")
        final_height = int(rpc.call("getblockcount", timeout=8))
        paid = [h for h in range(start_height + 1, final_height + 1) if harness.coinbase_pays_script(rpc, h, EXPECTED_SCRIPT)]
        blocks = harness.summarize_blocks(rpc, start_height, final_height)
        tap_text = tap_log.read_text(errors="replace") if tap_log.exists() else ""
        corrupt_ok = "NEGATIVE corrupt_proof verdict=rejected" in tap_text and "NEGATIVE corrupt_public_data verdict=rejected" in tap_text
        miner_events = parse_json_events(miner_log)
        completed = [e for e in miner_events if e.get("event") == "completed"]
        overhead = [e for e in miner_events if e.get("event") == "python_overhead"]
        p6_rows = p6_correlated_latencies(p6_log)
        summary.update(
            {
                "start_height": start_height,
                "final_height": final_height,
                "accepted_blocks": final_height - start_height,
                "blocks": blocks,
                "miner_graceful_stop": True,
                "miner_payout_verified": miner_summary["payout_verified"],
                "coinbase_ok": len(paid) == final_height - start_height,
                "coinbase_checked_heights": paid,
                "corrupted_certificate_rejection": corrupt_ok,
                "p6_correlated": p6_rows,
                "gpu_find_accept_latency_seconds": [row["gpu_find_to_accept_seconds"] for row in p6_rows],
                "proof_handoff_accept_latency_seconds": [row["proof_handoff_to_accept_seconds"] for row in p6_rows],
                "p6_peak_gateway_tree_rss_kb": peak_gateway_tree_rss,
                "p6_peak_gateway_tree_pids": peak_gateway_tree_pids,
                "completed_jobs": len(completed),
                "overhead_events": overhead[-10:],
                "miner_log": str(miner_log),
                "gateway_log": str(gateway_log),
                "tap_log": str(tap_log),
                "p6_log": str(p6_log),
            }
        )
        if len(p6_rows) < target_blocks:
            raise CriterionFailure(f"P6 correlation missing: matched {len(p6_rows)} < {target_blocks}")
        if len(blocks) < len(p6_rows):
            raise CriterionFailure(f"P6 accepted rows exceed chain blocks: p6={len(p6_rows)} chain={len(blocks)}")
        # Acceptance order need not equal block order. Match every header field.
        chain_details = {b["hash"]: rpc.call("getblock", [b["hash"], 1]) for b in blocks}
        matched_hashes = set()
        for row in p6_rows:
            fields = row.get("header_fields")
            if not fields:
                raise CriterionFailure("P6 accepted submission has no header identity")
            matches = [b for b in blocks if all(chain_details[b["hash"]].get(k) == v for k, v in fields.items())]
            if len(matches) != 1 or matches[0]["hash"] in matched_hashes:
                raise CriterionFailure("P6 submission does not match a unique accepted chain block")
            row["chain_confirmed"] = True
            row["accepted_block"] = matches[0]
            matched_hashes.add(matches[0]["hash"])
        if summary["accepted_blocks"] < target_blocks or not summary["coinbase_ok"] or not corrupt_ok:
            raise CriterionFailure(f"G5 criterion failed: {summary}")
        return summary
    finally:
        stop_all(procs)
        write_text(run / "summary.json", json.dumps(summary, indent=2) + "\n")


def p2(shape: dict[str, int], jobs: int) -> dict[str, Any]:
    summary = run_perf(
        [
            "p2",
            "--m",
            str(shape["m"]),
            "--n",
            str(shape["n"]),
            "--k",
            str(shape["k"]),
            "--slots",
            str(shape["slots"]),
            "--jobs",
            str(jobs),
            "--min-ratio",
            "0.85",
        ],
        timeout=2100,
    )
    return summary


def sustained(shape: dict[str, int], seconds: int) -> dict[str, Any]:
    summary = run_perf(
        [
            "sustained",
            "--m",
            str(shape["m"]),
            "--n",
            str(shape["n"]),
            "--k",
            str(shape["k"]),
            "--slots",
            str(shape["slots"]),
            "--seconds",
            str(seconds),
            "--report-interval",
            "30",
            "--shares-per-minute",
            "1.0",
        ],
        timeout=seconds + 240,
    )
    return summary


def _json_dicts(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _perf_signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


def _perf_failure_result(
    args: list[str],
    *,
    timeout: int,
    returncode: int | None,
    stdout: str,
    stderr: str,
    error_type: str,
    error: str,
    parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = _json_dicts(stdout) + _json_dicts(stderr)
    windows = [row for row in events if row.get("event") == "sustained_window"]
    result: dict[str, Any] = {
        "criterion": args[0] if args else "perf",
        "pass": False,
        "error_type": error_type,
        "error": error,
        "returncode": returncode,
        "signal": _perf_signal_name(returncode),
        "timeout_seconds": timeout,
        "partial_events": events[-20:],
    }
    if parsed:
        result.update({k: v for k, v in parsed.items() if k not in {"pass", "error", "error_type"}})
        result["pass"] = False
        result["error_type"] = parsed.get("error_type", error_type)
        result["error"] = parsed.get("error", error)
    if windows:
        result["windows"] = windows
        result["last_window"] = windows[-1]
        result["jobs"] = windows[-1].get("completed_jobs")
        result["tops"] = windows[-1].get("tops")
        result["shares"] = windows[-1].get("shares")
    return result


def _perf_final_json(stdout: str, stderr: str) -> dict[str, Any] | None:
    candidates = _json_dicts(stdout) + _json_dicts(stderr)
    for value in reversed(candidates):
        if any(key in value for key in ("criterion", "mode", "pass", "error_type", "error")):
            return value
    return None


def run_perf(args: list[str], *, timeout: int) -> dict[str, Any]:
    cmd = [str(python_path()), str(ROOT / "scripts" / "studio_b4" / "perf.py"), *args]
    jlog("cmd_start", argv=cmd, cwd=str(ROOT), timeout=timeout)
    env = bundle_env()
    profile = RUN_ROOT / f"perf-{args[0] if args else 'run'}-profile.jsonl"
    env.setdefault("PMK_PROFILE_JSONL", str(profile))
    process, sentinel_write_fd = guarded_popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            kill_process_group(process)
            stdout, stderr = process.communicate()
        kill_process_group(process)
        release_owned_process(process, sentinel_write_fd)
        print(stdout, end="")
        print(stderr, end="")
        jlog("cmd_end", argv=cmd[:2], returncode=process.returncode, timed_out=True)
        return _perf_failure_result(
            args,
            timeout=timeout,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            error_type=type(exc).__name__,
            error=f"perf.py timed out after {timeout}s",
            parsed=_perf_final_json(stdout, stderr),
        )
    except BaseException:
        terminate_process_group(process)
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            kill_process_group(process)
            process.communicate()
        kill_process_group(process)
        release_owned_process(process, sentinel_write_fd)
        raise
    kill_process_group(process)
    release_owned_process(process, sentinel_write_fd)
    print(stdout, end="")
    print(stderr, end="")
    jlog("cmd_end", argv=cmd[:2], returncode=process.returncode)
    parsed = _perf_final_json(stdout, stderr)
    if parsed is None:
        return _perf_failure_result(
            args,
            timeout=timeout,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            error_type="MissingJSON",
            error="perf.py produced no final JSON result",
        )
    if process.returncode not in (0, 2):
        return _perf_failure_result(
            args,
            timeout=timeout,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            error_type="ProcessFailure",
            error=f"perf.py failed rc={process.returncode}",
            parsed=parsed,
        )
    if profile.exists():
        parsed.setdefault("profile_jsonl", str(profile))
    return parsed


def finish_summary(summary: dict[str, Any]) -> None:
    write_text(SUMMARY, json.dumps(summary, indent=2, sort_keys=True) + "\n", mode=0o644)
    jlog("summary_written", path=str(SUMMARY))


def not_run_step(criterion: str, reason: str = "not_reached") -> dict[str, Any]:
    return {"criterion": criterion, "pass": False, "status": "NOT_RUN", "reason": reason}


def p6_step_from_g5(g5: dict[str, Any], required: int, *, error: str | None = None) -> dict[str, Any]:
    rows = g5.get("p6_correlated", []) if isinstance(g5, dict) else []
    matched = len(rows) if isinstance(rows, list) else 0
    result: dict[str, Any] = {
        "criterion": "P6",
        "pass": matched >= required and error is None,
        "required": required,
        "matched": matched,
        "gpu_find_accept_latency_seconds": g5.get("gpu_find_accept_latency_seconds", []) if isinstance(g5, dict) else [],
        "proof_handoff_accept_latency_seconds": g5.get("proof_handoff_accept_latency_seconds", []) if isinstance(g5, dict) else [],
        "p6_peak_gateway_tree_rss_kb": g5.get("p6_peak_gateway_tree_rss_kb") if isinstance(g5, dict) else None,
        "p6_peak_gateway_tree_pids": g5.get("p6_peak_gateway_tree_pids", []) if isinstance(g5, dict) else [],
        "p6_log": g5.get("p6_log") if isinstance(g5, dict) else None,
    }
    if error:
        result["error"] = error
    elif matched < required:
        result["error"] = f"P6 matched {matched} accepted proof(s), need {required}"
    return result


def read_g5_partial_summary(name: str) -> dict[str, Any]:
    path = RUN_ROOT / name / "summary.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="local validation: production-shape smoke plus 128x128x4096 regtest")
    parser.add_argument("--skip-long", action="store_true", help="skip P2 and sustained run after G5")
    parser.add_argument("--p2-jobs", type=int, default=200)
    parser.add_argument("--sustain-seconds", type=int, default=600)
    parser.add_argument("--resume-hook", type=Path, default=Path(os.environ["B4_LAB_RESUME_HOOK"]) if os.environ.get("B4_LAB_RESUME_HOOK") else (Path(os.environ["PMK_LAB_RESUME_HOOK"]) if os.environ.get("PMK_LAB_RESUME_HOOK") else None))
    parser.add_argument("--resume-report", type=Path, default=Path(os.environ["B4_LAB_RESUME_REPORT"]) if os.environ.get("B4_LAB_RESUME_REPORT") else (Path(os.environ["PMK_LAB_RESUME_REPORT"]) if os.environ.get("PMK_LAB_RESUME_REPORT") else None))
    args = parser.parse_args()
    if args.resume_hook:
        os.environ["B4_LAB_RESUME_HOOK"] = str(args.resume_hook)
    if args.resume_report:
        os.environ["B4_LAB_RESUME_REPORT"] = str(args.resume_report)
    if not args.quick:
        if args.p2_jobs < 200:
            raise SystemExit("--p2-jobs must be >= 200 for production B4")
        if args.sustain_seconds < 600:
            raise SystemExit("--sustain-seconds must be >= 600 for production B4")
    shape = QUICK if args.quick else PRODUCTION
    summary: dict[str, Any] = {
        "root": str(ROOT),
        "quick": args.quick,
        "steps": {
            "p6": not_run_step("P6"),
            "p2": not_run_step("P2"),
            "sustained": not_run_step("b4_sustained_smoke"),
        },
        "criteria": {k: "NOT_RUN" for k in ("preflight", "lab", "g3_admission", "probe", "production_smoke", "pytest_fast", "g5", "p6", "p2", "sustained")},
    }
    active_step = "preflight"
    start = time.monotonic()
    lab_window: LabWindow | None = None
    def on_alarm(_signum, _frame):
        raise StepFailure(f"B4 window exceeded active budget; reserving {CLEANUP_RESERVE_SECONDS}s for cleanup")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.signal(signal.SIGTERM, on_alarm)
    signal.signal(signal.SIGINT, on_alarm)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_alarm)
    signal.alarm(max(1, WINDOW_BUDGET_SECONDS - CLEANUP_RESERVE_SECONDS))
    try:
        active_step = "lab"
        jlog("step_start", step="lab", elapsed_seconds=time.monotonic() - start)
        lab_window = prepare_lab_window()
        summary["steps"]["lab"] = {
            "lock": str(lab_window.lock),
            "owned": bool(lab_window.session and getattr(lab_window.session, "owned", False)),
            "resume_hook": str(lab_window.resume_hook) if lab_window.resume_hook else None,
            "resume_report": str(lab_window.report) if lab_window.report else None,
        }
        summary["criteria"]["lab"] = "PASS"
        finish_summary(summary)
        try:
            active_step = "preflight"
            jlog("step_start", step="preflight", elapsed_seconds=time.monotonic() - start)
            summary["steps"]["preflight"] = preflight()
            summary["criteria"]["preflight"] = "PASS"
            finish_summary(summary)
            with gpu_window_lock():
                steps: list[tuple[str, Any]] = [
                    ("g3_admission", g3_admission),
                    ("probe", probe),
                    ("production_smoke", production_smoke),
                    ("pytest_fast", pytest_fast),
                    ("g5", lambda: regtest_g5(shape=shape, target_blocks=1 if args.quick else 3, timeout_seconds=600 if args.quick else 1500, name="g5_quick" if args.quick else "g5")),
                ]
                for name, fn in steps:
                    active_step = name
                    jlog("step_start", step=name, elapsed_seconds=time.monotonic() - start)
                    summary["steps"][name] = fn()
                    summary["criteria"][name] = "PASS"
                    if name == "g5":
                        required = 1 if args.quick else 3
                        summary["steps"]["p6"] = p6_step_from_g5(summary["steps"]["g5"], required)
                        summary["criteria"]["p6"] = "PASS" if summary["steps"]["p6"]["pass"] else "FAIL"
                        if not summary["steps"]["p6"]["pass"]:
                            raise CriterionFailure(str(summary["steps"]["p6"]["error"]))
                    finish_summary(summary)
                if not args.quick and not args.skip_long:
                    active_step = "p2"
                    jlog("step_start", step="p2", elapsed_seconds=time.monotonic() - start)
                    summary["steps"]["p2"] = p2(shape, args.p2_jobs)
                    summary["criteria"]["p2"] = "PASS" if summary["steps"]["p2"]["pass"] else "FAIL"
                    finish_summary(summary)
                    active_step = "sustained"
                    jlog("step_start", step="sustained", elapsed_seconds=time.monotonic() - start)
                    summary["steps"]["sustained"] = sustained(shape, args.sustain_seconds)
                    summary["criteria"]["sustained"] = "PASS" if summary["steps"]["sustained"]["pass"] else "FAIL"
                else:
                    reason = "quick" if args.quick else "skip_long"
                    summary["criteria"]["p2"] = f"SKIPPED:{reason}"
                    summary["criteria"]["sustained"] = f"SKIPPED:{reason}"
                    summary["steps"]["p2"] = not_run_step("P2", reason)
                    summary["steps"]["sustained"] = not_run_step("b4_sustained_smoke", reason)
        finally:
            cleanup_lab_window(lab_window, summary)
        summary["elapsed_seconds"] = time.monotonic() - start
        failed = any(value == "FAIL" for value in summary["criteria"].values())
        summary["result"] = "CRITERION_FAIL" if failed else ("PASS" if not args.quick and not args.skip_long else "PASS_SELECTED")
        finish_summary(summary)
        jlog("criteria", **summary["criteria"])
        return EXIT_CRITERION_FAILURE if failed else 0
    except CriterionFailure as exc:
        if active_step == "g5":
            required = 1 if args.quick else 3
            partial = summary["steps"].get("g5") if isinstance(summary["steps"].get("g5"), dict) else read_g5_partial_summary("g5_quick" if args.quick else "g5")
            summary["steps"]["p6"] = p6_step_from_g5(partial, required, error=str(exc))
            summary["criteria"]["p6"] = "PASS" if summary["steps"]["p6"]["pass"] else "FAIL"
        summary["criteria"][active_step] = "FAIL"
        summary["elapsed_seconds"] = time.monotonic() - start
        summary["result"] = "CRITERION_FAIL"
        summary["error"] = str(exc)
        finish_summary(summary)
        jlog("criterion_failure", error=str(exc))
        return EXIT_CRITERION_FAILURE
    except Exception as exc:
        if active_step == "g5":
            required = 1 if args.quick else 3
            partial = summary["steps"].get("g5") if isinstance(summary["steps"].get("g5"), dict) else read_g5_partial_summary("g5_quick" if args.quick else "g5")
            summary["steps"]["p6"] = p6_step_from_g5(partial, required, error=f"{type(exc).__name__}: {exc}")
            summary["criteria"]["p6"] = "PASS" if summary["steps"]["p6"]["pass"] else "FAIL"
        summary["criteria"][active_step] = "FAIL"
        summary["elapsed_seconds"] = time.monotonic() - start
        summary["result"] = "STEP_FAIL"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        finish_summary(summary)
        jlog("step_failure", error_type=type(exc).__name__, error=str(exc))
        return EXIT_STEP_FAILURE
    finally:
        with cleanup_signal_shield():
            signal.alarm(0)
            # Also covers a signal/exception after Popen but before a local Proc
            # was appended to its phase list.
            stop_all([])


if __name__ == "__main__":
    raise SystemExit(main())
