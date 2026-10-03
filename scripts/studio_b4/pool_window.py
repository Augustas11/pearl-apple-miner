#!/usr/bin/env python3
"""Pool-mode long-run window controller.

The controller is intentionally narrow: an optional outer wrapper handles
provider state and GPU lock ownership, while pmk owns pool protocol behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import sys
import textwrap
import time
import tomllib
import urllib.parse
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("POOL_WINDOW_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUN_ROOT = ROOT / ".pool_run"
SUMMARY = ROOT / "pool_summary.json"
GPU_LOCK_DIR = Path("/tmp/pmm-gpu-bench.lock")
DEFAULT_SHAPE = {"m": 8192, "n": 8192, "k": 4096, "slots": 2}
SUMMARY_FIELDS = (
    "accepted",
    "stale",
    "submitted",
    "failed",
    "rejected",
    "duplicate",
    "low_difficulty",
    "invalid",
    "transport",
    "timeout",
    "gate_failures",
    "completed_ops",
    "completed_macs",
    "observed",
    "expected",
    "lower",
    "upper",
    "poisson_ok",
    "elapsed_seconds",
    "ops_per_second",
    "pool_hashrate",
    "block_candidates",
)
REJECT_CLASS_FIELDS = ("duplicate", "low_difficulty", "invalid")


class PoolWindowError(RuntimeError):
    pass


class PoolWindowInterrupted(PoolWindowError):
    pass


def jlog(event: str, **fields: Any) -> None:
    print(json.dumps({"time": time.time(), "event": event, **fields}, separators=(",", ":")), flush=True)


def secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def python_path() -> Path:
    return ROOT / ".venv-b4" / "bin" / "python"


def run_cmd(args: list[str], *, timeout: int, check: bool = True) -> subprocess.CompletedProcess[str]:
    jlog("cmd_start", argv=args[:3], timeout=timeout)
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    print(result.stdout, end="")
    jlog("cmd_end", returncode=result.returncode)
    if check and result.returncode != 0:
        raise PoolWindowError(f"command failed rc={result.returncode}: {' '.join(args)}")
    return result


def verify_manifest() -> dict[str, Any]:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        return {"checked_files": 0, "status": "missing"}
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            raise PoolWindowError(f"manifest file missing: {rel}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise PoolWindowError(f"manifest mismatch: {rel}")
        checked += 1
    return {"checked_files": checked, "status": "ok"}


def create_offline_venv() -> dict[str, Any]:
    venv_python = python_path()
    wheels = ROOT / "wheels"
    runtime_lock = ROOT / "miner" / "requirements.lock"
    local_lock = ROOT / "local-wheels.lock"
    if not wheels.is_dir() or not runtime_lock.is_file() or not local_lock.is_file():
        if venv_python.exists():
            verify_pool_bound_helpers(venv_python)
            return {"python": str(venv_python), "status": "existing"}
        if (ROOT / "miner").is_dir():
            return {"python": sys.executable, "status": "source-tree"}
        raise PoolWindowError("missing offline bundle wheels or lock files")
    if venv_python.exists():
        if verify_local_wheel_records(venv_python, wheels, local_lock, check=False) and verify_pool_bound_helpers(venv_python, check=False):
            return {"python": str(venv_python), "status": "existing"}
        install_offline_requirements(wheels, runtime_lock, local_lock)
        verify_local_wheel_records(venv_python, wheels, local_lock)
        verify_pool_bound_helpers(venv_python)
        return {"python": str(venv_python), "status": "repaired"}
    py = Path(os.environ.get("B4_STUDIO_PYTHON") or os.environ.get("POOL_STUDIO_PYTHON") or os.environ.get("PMK_PYTHON") or sys.executable)
    if not py.exists():
        py = Path(sys.executable)
    run_cmd([str(py), "-m", "venv", str(ROOT / ".venv-b4")], timeout=180)
    install_offline_requirements(wheels, runtime_lock, local_lock)
    verify_local_wheel_records(venv_python, wheels, local_lock)
    verify_pool_bound_helpers(venv_python)
    return {"python": str(venv_python), "status": "created"}


def install_offline_requirements(wheels: Path, runtime_lock: Path, local_lock: Path) -> None:
    pip = ROOT / ".venv-b4" / "bin" / "pip"
    run_cmd([str(pip), "install", "--no-index", "--find-links", str(wheels), "--require-hashes", "--force-reinstall", "-r", str(runtime_lock)], timeout=600)
    run_cmd([str(pip), "install", "--no-index", "--find-links", str(wheels), "--require-hashes", "--no-deps", "--force-reinstall", "-r", str(local_lock)], timeout=300)
    run_cmd([str(pip), "check"], timeout=60)


def verify_local_wheel_records(venv_python: Path, wheels: Path, local_lock: Path, *, check: bool = True) -> bool:
    code = r"""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import re
import sys
import zipfile
from pathlib import Path

wheels = Path(sys.argv[1])
local_lock = Path(sys.argv[2])


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


errors = []
requirements = {}
for raw in local_lock.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        continue
    if "==" not in raw:
        errors.append(f"malformed local lock line: {raw[:80]}")
        continue
    left, _, rest = raw.partition("==")
    match = re.search(r"--hash=sha256:([0-9a-fA-F]{64})", rest)
    if not match:
        errors.append(f"missing sha256 hash for local requirement: {left}")
        continue
    requirements[normalize(left)] = match.group(1).lower()

if not requirements:
    errors.append("local lock has no hashed requirements")

wheel_by_name = {}
for wheel in wheels.glob("*.whl"):
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    stem = wheel.name.split("-", 2)
    if len(stem) >= 2:
        wheel_by_name[(normalize(stem[0]), digest)] = wheel

for name, expected_digest in requirements.items():
    wheel = wheel_by_name.get((name, expected_digest))
    if wheel is None:
        errors.append(f"{name}: locked wheel sha256 missing from wheels/")
        continue
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        errors.append(f"{name}: package is not installed")
        continue
    with zipfile.ZipFile(wheel) as archive:
        record_name = next((entry for entry in archive.namelist() if entry.endswith(".dist-info/RECORD")), None)
        if record_name is None:
            errors.append(f"{name}: wheel lacks RECORD")
            continue
        rows = csv.reader(archive.read(record_name).decode("utf-8").splitlines())
        for rel, digest_field, _size in rows:
            if not digest_field.startswith("sha256="):
                continue
            installed = Path(dist.locate_file(rel))
            if not installed.is_file():
                errors.append(f"{name}: missing installed file {rel}")
                continue
            expected = digest_field.removeprefix("sha256=")
            actual = base64.urlsafe_b64encode(hashlib.sha256(installed.read_bytes()).digest()).rstrip(b"=").decode("ascii")
            if actual != expected:
                errors.append(f"{name}: installed file hash mismatch {rel}")

if errors:
    raise SystemExit("; ".join(errors[:5]))
"""
    result = run_cmd([str(venv_python), "-c", textwrap.dedent(code), str(wheels), str(local_lock)], timeout=60, check=False)
    ok = result.returncode == 0
    if check and not ok:
        raise PoolWindowError("offline venv does not match shipped local wheel hashes")
    return ok


def verify_pool_bound_helpers(venv_python: Path, *, check: bool = True) -> bool:
    code = (
        "import pearl_mining as pm\n"
        "missing=[name for name in ('nbits_to_difficulty','extract_difficulty_bound') if not callable(getattr(pm,name,None))]\n"
        "if missing:\n"
        "    raise SystemExit('missing pool helper(s): '+','.join(missing))\n"
        "def compact_to_target(nbits):\n"
        "    size=(nbits >> 24) & 0xff\n"
        "    mantissa=nbits & 0x007fffff\n"
        "    if nbits & 0x00800000:\n"
        "        raise ValueError('negative compact target')\n"
        "    return mantissa >> (8 * (3 - size)) if size <= 3 else mantissa << (8 * (size - 3))\n"
        "def target_bound(target,cfg):\n"
        "    bound=target*32*cfg.common_dim\n"
        "    penalized=pm.penalized_target_bound(target,cfg)\n"
        "    if int(penalized) != bound:\n"
        "        raise SystemExit('pool helper semantic check penalized bound mismatch')\n"
        "    return bound\n"
        "cfg=pm.MiningConfiguration(common_dim=4096,rank=128,mma_type=pm.MMAType.Int7xInt7ToInt32,"
        "rows_pattern=pm.PeriodicPattern.from_list([0,8,64,72]),"
        "cols_pattern=pm.PeriodicPattern.from_list([0,1,8,9,32,33,40,41]),moe=None)\n"
        "vectors=[0x177fd82e,0x1A07FFF8,0x1A086373,0x1B014F8A,0x1B068DB2]\n"
        "for bits in vectors:\n"
        "    target=int(pm.nbits_to_difficulty(bits))\n"
        "    expected_target=compact_to_target(bits)\n"
        "    if target != expected_target:\n"
        "        raise SystemExit(f'pool helper semantic check target mismatch bits=0x{bits:08x}')\n"
        "    bound=int(pm.extract_difficulty_bound(bits,cfg))\n"
        "    expected_bound=target_bound(target,cfg)\n"
        "    if bound != expected_bound or not (0 < bound < 2**256):\n"
        "        raise SystemExit(f'pool helper semantic check bound mismatch bits=0x{bits:08x}')\n"
    )
    result = run_cmd([str(venv_python), "-c", code], timeout=30, check=False)
    ok = result.returncode == 0
    if check and not ok:
        raise PoolWindowError("offline venv lacks py-pearl-mining pool difficulty-bound helpers")
    return ok


def bundle_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "miner"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(ROOT / "libpmk" / ".build" / "release"),
            str(ROOT / "pmkcore" / "target" / "release"),
            env.get("DYLD_LIBRARY_PATH", ""),
        ]
    ).rstrip(os.pathsep)
    env["PMK_RESOURCE_BUNDLE"] = str(ROOT / "libpmk" / ".build" / "release" / "libpmk_PMK.bundle")
    env["PMKCORE_DYLIB"] = str(ROOT / "pmkcore" / "target" / "release" / "libpmkcore.dylib")
    env["LIBPMK_DYLIB"] = str(ROOT / "libpmk" / ".build" / "release" / "libpmk.dylib")
    for key in list(env):
        if key.lower().endswith("_proxy"):
            env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost,::1"
    env["no_proxy"] = env["NO_PROXY"]
    if extra:
        env.update(extra)
    return env


def load_shape(config: Path | None) -> dict[str, int]:
    shape = dict(DEFAULT_SHAPE)
    if config and config.exists():
        data = tomllib.loads(config.read_text(encoding="utf-8"))
        for key in shape:
            if key in data:
                shape[key] = int(data[key])
    return shape


def write_pool_config(path: Path, *, source: Path | None, state_dir: Path, max_accepted: int, max_submitted: int | None, max_seconds: int) -> None:
    shape = load_shape(source)
    lines = [
        "# Generated by pool_window.py. Only root shape keys are copied from --config;",
        "# pool transport/security policy stays in pmk_miner and CLI inputs.",
        *[f"{key} = {shape[key]}" for key in ("m", "n", "k", "slots")],
    ]
    lines.extend(
        [
            "",
            "[run]",
            f'state_dir = "{state_dir}"',
            f'routine_log_file = "{state_dir / "routine.jsonl"}"',
            f"max_accepted = {max_accepted}",
            f"max_seconds = {max_seconds}",
        ]
    )
    if max_submitted is not None:
        lines.append(f"max_submitted = {max_submitted}")
    write_text(path, "\n".join(lines) + "\n")


def validate_paths(args: argparse.Namespace) -> None:
    for name in ("wallet_file", "wallet_allowlist"):
        path = getattr(args, name)
        if not path.is_file():
            raise PoolWindowError(f"{name.replace('_', '-')} is missing: {path}")
    if args.max_accepted <= 0:
        raise PoolWindowError("--max-accepted must be positive")
    if args.max_submitted is not None and args.max_submitted <= 0:
        raise PoolWindowError("--max-submitted must be positive when supplied")
    if args.max_seconds <= 0:
        raise PoolWindowError("--max-seconds must be positive")
    if args.shutdown_grace_seconds <= 0:
        raise PoolWindowError("--shutdown-grace-seconds must be positive")
    if not args.dry_run and not args.require_inherited_gpu_lock:
        raise PoolWindowError("real pool runs must require the inherited GPU lock from the outer wrapper")
    if args.require_inherited_gpu_lock:
        if os.environ.get("PMK_GPU_LOCK_HELD") != "1":
            raise PoolWindowError("pool window requires inherited PMK_GPU_LOCK_HELD=1 from the outer wrapper")
        if not GPU_LOCK_DIR.is_dir():
            raise PoolWindowError(f"inherited GPU lock is absent: {GPU_LOCK_DIR}")
    if not args.dry_run and not os.environ.get("B4_LAB_OWNER_TOKEN") and not os.environ.get("PMK_LAB_OWNER_TOKEN"):
        raise PoolWindowError("pool window requires lab owner token in the environment for real runs")


def sanitize_pool_url(raw: str) -> str:
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"stratum+tcp", "stratum+ssl", "stratum+tls", "tls"}:
        raise PoolWindowError("pool URL scheme must be stratum+tcp, stratum+ssl, stratum+tls or tls")
    if parsed.username or parsed.password:
        raise PoolWindowError("pool URL must not contain userinfo")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise PoolWindowError("pool URL must contain only scheme, host and port")
    if not parsed.hostname:
        raise PoolWindowError("pool URL host is required")
    try:
        port = parsed.port
    except ValueError:
        raise PoolWindowError("pool URL port is invalid") from None
    if not port:
        raise PoolWindowError("pool URL port is required and must be nonzero")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}:{port}"


def miner_cmd(args: argparse.Namespace, config: Path) -> list[str]:
    cmd = [
        str(python_path() if python_path().exists() else Path(sys.executable)),
        str(ROOT / "scripts" / "studio_b4" / "pool_miner_entry.py"),
        "--mode",
        "pool",
        "--pool-url",
        args.pool_url,
        "--wallet-file",
        str(args.wallet_file),
        "--wallet-allowlist",
        str(args.wallet_allowlist),
        "--worker",
        args.worker,
        "--config",
        str(config),
    ]
    return cmd


def guarded_command(args: list[str], sentinel_fd: int) -> list[str]:
    return [
        str(python_path() if python_path().exists() else Path(sys.executable)),
        str(Path(__file__).resolve().with_name("process_guard.py")),
        str(sentinel_fd),
        "--",
        *args,
    ]


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


def append_stdout(text: str, log_file: Any) -> None:
    if not text:
        return
    log_file.write(text)
    log_file.flush()
    print(text, end="")


def update_latest_from_text(text: str, latest: dict[str, Any] | None) -> dict[str, Any] | None:
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "pool_summary":
            latest = event
    return latest


def drain_stdout(
    stdout_fd: int,
    pending: bytes,
    latest: dict[str, Any] | None,
    log_file: Any,
) -> tuple[bytes, dict[str, Any] | None, bool]:
    eof = False
    while True:
        try:
            chunk = os.read(stdout_fd, 65536)
        except BlockingIOError:
            break
        if not chunk:
            eof = True
            break
        pending += chunk
        while b"\n" in pending:
            raw_line, pending = pending.split(b"\n", 1)
            line = raw_line.decode("utf-8", errors="replace") + "\n"
            append_stdout(line, log_file)
            latest = update_latest_from_text(line, latest)
    return pending, latest, eof


def flush_pending_stdout(
    pending: bytes,
    latest: dict[str, Any] | None,
    log_file: Any,
) -> tuple[bytes, dict[str, Any] | None]:
    if not pending:
        return pending, latest
    text = pending.decode("utf-8", errors="replace")
    append_stdout(text, log_file)
    latest = update_latest_from_text(text, latest)
    return b"", latest


def finish_process_group(
    process: subprocess.Popen[bytes],
    stdout_fd: int,
    pending: bytes,
    latest: dict[str, Any] | None,
    log_file: Any,
    *,
    stdout_timeout: float,
) -> tuple[bytes, dict[str, Any] | None]:
    deadline = time.monotonic() + stdout_timeout
    selector = selectors.DefaultSelector()
    try:
        selector.register(stdout_fd, selectors.EVENT_READ)
        while process.poll() is None and time.monotonic() < deadline:
            for _key, _ in selector.select(timeout=0.1):
                pending, latest, _eof = drain_stdout(stdout_fd, pending, latest, log_file)
        if process.poll() is None:
            kill_process_group(process)
        kill_deadline = time.monotonic() + 2.0
        while process.poll() is None and time.monotonic() < kill_deadline:
            for _key, _ in selector.select(timeout=0.1):
                pending, latest, _eof = drain_stdout(stdout_fd, pending, latest, log_file)
        if process.poll() is None:
            raise PoolWindowError("pool miner process group did not exit after SIGKILL")
        process.wait(timeout=1)
        kill_process_group(process)
        drain_deadline = time.monotonic() + stdout_timeout
        while time.monotonic() < drain_deadline:
            pending, latest, eof = drain_stdout(stdout_fd, pending, latest, log_file)
            if eof:
                break
            time.sleep(0.01)
        pending, latest = flush_pending_stdout(pending, latest, log_file)
        return pending, latest
    finally:
        selector.close()


def stop_process_group(
    process: subprocess.Popen[bytes],
    stdout_fd: int,
    pending: bytes,
    latest: dict[str, Any] | None,
    log_file: Any,
    *,
    stdout_timeout: float,
) -> tuple[bytes, dict[str, Any] | None]:
    if process.poll() is None:
        terminate_process_group(process)
    return finish_process_group(process, stdout_fd, pending, latest, log_file, stdout_timeout=stdout_timeout)


def install_miner_signal_handlers(process_ref: dict[str, subprocess.Popen[bytes] | None]) -> dict[int, Any]:
    handled = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        handled.append(signal.SIGHUP)
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        # A second interrupt must not cut short the first one's termination
        # request or the finally block that drains and reaps the owned group.
        ignore_signal_handlers(previous)
        process = process_ref.get("process")
        jlog("miner_controller_signal", signal=signum, pid=process.pid if process is not None else None)
        if process is not None:
            terminate_process_group(process)
        raise PoolWindowInterrupted(f"controller interrupted by signal {signum}")

    for signum in handled:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def controller_signals() -> list[signal.Signals]:
    handled = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        handled.append(signal.SIGHUP)
    return handled


def ignore_signal_handlers(previous: dict[int, Any]) -> None:
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, previous)
    try:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def classify_summary(event: dict[str, Any] | None, args: argparse.Namespace, *, returncode: int | None, error: str | None = None, timed_out: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {field: (event.get(field) if event else None) for field in SUMMARY_FIELDS}
    accepted = int(summary.get("accepted") or 0)
    stale = int(summary.get("stale") or 0)
    explicit_rejected = int(summary.get("rejected") or 0) if event and "rejected" in event else None
    by_class = sum(int(summary.get(field) or 0) for field in REJECT_CLASS_FIELDS)
    rejected = explicit_rejected if explicit_rejected is not None else by_class
    submitted = int(summary.get("submitted") or 0) if event and "submitted" in event else accepted + stale + rejected
    target = args.max_submitted or args.max_accepted
    accepted_ratio = (accepted / submitted) if submitted else 0.0
    stale_ratio = (stale / submitted) if submitted else 0.0
    throughput = {"ops_per_second": summary.get("ops_per_second"), "pool_hashrate": summary.get("pool_hashrate"), "p2_ratio": None, "p2_ratio_gate": False}
    reasons: list[str] = []
    if event is None:
        reasons.append("missing pool_summary event")
    if returncode != 0:
        reasons.append(f"miner returncode {returncode}")
    if event and event.get("pass") is False:
        reasons.append("miner reported failed pool_summary")
    if submitted < target:
        reasons.append(f"submitted {submitted}, need {target}")
    if accepted < args.max_accepted:
        reasons.append(f"accepted {accepted}, need {args.max_accepted}")
    if rejected or by_class:
        reasons.append("pool rejection: config incompatibility")
    if int(summary.get("transport") or 0) or int(summary.get("timeout") or 0):
        reasons.append("pool verdict unavailable: transport/timeout")
    if summary.get("failed"):
        reasons.append("miner reported failure")
    if args.max_accepted == 1 and stale:
        reasons.append("first share was stale, not accepted")
    if args.max_accepted > 1 and accepted_ratio < 0.95:
        reasons.append(f"accepted ratio {accepted_ratio:.3f} below 0.95")
    if args.max_accepted > 1 and stale_ratio >= 0.02:
        reasons.append(f"stale ratio {stale_ratio:.3f} is not below 0.02")
    if event and event.get("poisson_ok") is False:
        reasons.append("poisson check failed")
    if int(summary.get("gate_failures") or 0) != 0:
        reasons.append("gate failures were reported")
    summary.update(
        {
            "accepted": accepted,
            "stale": stale,
            "rejected": rejected,
            "submitted": submitted,
            "required_submitted": target,
            "accepted_ratio": accepted_ratio,
            "stale_ratio": stale_ratio,
            "throughput": throughput,
            "raw_event": event,
            "returncode": returncode,
            "pass": not reasons,
        }
    )
    if error:
        summary["error"] = error
        reasons.append(error)
        summary["pass"] = False
    # A bounded T2 window with no pool verdict cannot establish compatibility.
    # Preserve actual failures (rejections, verifier/device faults, interruption).
    no_verdict = accepted == 0 and stale == 0 and rejected == 0 and by_class == 0
    inconclusive = (
        args.max_accepted == 1 and no_verdict
        and (returncode == 0 or timed_out)
        and (not error or timed_out)
        and not summary.get("failed")
        and not summary.get("gate_failures")
        and not (event and (event.get("pass") is False or event.get("poisson_ok") is False))
        and (event is not None or timed_out)
    )
    summary["status"] = "INCONCLUSIVE" if inconclusive else ("FAIL" if reasons else "PASS")
    summary["pass"] = summary["status"] == "PASS"
    summary["exit_code"] = {"PASS": 0, "FAIL": 2, "INCONCLUSIVE": 3}[summary["status"]]
    if summary["status"] == "INCONCLUSIVE":
        summary["inconclusive_reasons"] = reasons
    elif reasons:
        summary["fail_reasons"] = reasons
    return summary


def run_pool_miner(args: argparse.Namespace, config: Path, log_path: Path) -> dict[str, Any]:
    cmd = miner_cmd(args, config)
    env = bundle_env({"PMK_POOL_WINDOW_LOG": str(log_path.with_suffix(".entry.jsonl"))})
    latest: dict[str, Any] | None = None
    pending = b""
    secure_dir(log_path.parent)
    process_ref: dict[str, subprocess.Popen[bytes] | None] = {"process": None}
    signal_handlers = install_miner_signal_handlers(process_ref)
    process: subprocess.Popen[bytes] | None = None
    sentinel_write_fd: int | None = None
    stdout_fd: int | None = None
    selector = selectors.DefaultSelector()
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            sentinel_read_fd, sentinel_write_fd = os.pipe()
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, controller_signals())
            try:
                process = subprocess.Popen(
                    guarded_command(cmd, sentinel_read_fd),
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(sentinel_read_fd,),
                )
                process_ref["process"] = process
            finally:
                os.close(sentinel_read_fd)
                # Deliver a pending controller signal only after the new group
                # is reachable through process_ref and protected by its pipe.
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if process.stdout is None:
                raise PoolWindowError("miner stdout pipe was not created")
            stdout_fd = process.stdout.fileno()
            os.set_blocking(stdout_fd, False)
            selector.register(stdout_fd, selectors.EVENT_READ)
            deadline = time.monotonic() + args.max_seconds + args.shutdown_grace_seconds
            jlog("miner_start", argv=[cmd[0], "pool_miner_entry.py", "--mode", "pool"], log=str(log_path), pid=process.pid)
            while True:
                for _key, _ in selector.select(timeout=1.0):
                    pending, latest, _eof = drain_stdout(stdout_fd, pending, latest, log_file)
                if process.poll() is not None:
                    kill_process_group(process)
                    pending, latest = finish_process_group(
                        process,
                        stdout_fd,
                        pending,
                        latest,
                        log_file,
                        stdout_timeout=2.0,
                    )
                    break
                if time.monotonic() >= deadline:
                    ignore_signal_handlers(signal_handlers)
                    selector.unregister(stdout_fd)
                    pending, latest = stop_process_group(
                        process,
                        stdout_fd,
                        pending,
                        latest,
                        log_file,
                        stdout_timeout=max(1.0, float(args.shutdown_grace_seconds)),
                    )
                    return classify_summary(
                        latest,
                        args,
                        returncode=process.returncode,
                        error="pool miner exceeded window deadline",
                        timed_out=True,
                    )
        except BaseException:
            ignore_signal_handlers(signal_handlers)
            if process is not None and stdout_fd is not None:
                try:
                    selector.unregister(stdout_fd)
                except (KeyError, ValueError):
                    pass
                pending, latest = stop_process_group(process, stdout_fd, pending, latest, log_file, stdout_timeout=2.0)
            raise
        finally:
            ignore_signal_handlers(signal_handlers)
            selector.close()
            if process is not None:
                if process.poll() is None and stdout_fd is not None:
                    stop_process_group(process, stdout_fd, pending, latest, log_file, stdout_timeout=2.0)
                else:
                    kill_process_group(process)
                    if process.poll() is None:
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            kill_process_group(process)
                            process.wait(timeout=2)
            if sentinel_write_fd is not None:
                os.close(sentinel_write_fd)
            restore_signal_handlers(signal_handlers)
    return classify_summary(latest, args, returncode=process.returncode)


def write_summary(summary: dict[str, Any], path: Path) -> None:
    write_text(path, json.dumps(summary, indent=2, sort_keys=True) + "\n", mode=0o644)
    jlog("summary_written", path=str(path), result=summary.get("result"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-url", required=True)
    p.add_argument("--wallet-file", type=Path, required=True)
    p.add_argument("--wallet-allowlist", type=Path, required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--max-accepted", type=int, default=20)
    p.add_argument("--max-submitted", type=int)
    p.add_argument("--max-seconds", type=int, default=8 * 3600)
    p.add_argument("--shutdown-grace-seconds", type=int, default=60)
    p.add_argument("--summary", type=Path, default=SUMMARY)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-inherited-gpu-lock", dest="require_inherited_gpu_lock", action="store_true", default=True)
    p.add_argument("--no-require-inherited-gpu-lock", dest="require_inherited_gpu_lock", action="store_false")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    start = time.monotonic()
    secure_dir(RUN_ROOT)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    active_root = RUN_ROOT / run_id
    secure_dir(active_root)
    state_dir = active_root / "state"
    config = active_root / "pmk-pool.toml"
    miner_log = active_root / "miner-pool.jsonl"
    summary: dict[str, Any] = {
        "result": "RUNNING",
        "root": str(ROOT),
        "pool_endpoint": None,
        "worker": args.worker,
        "run_id": run_id,
        "steps": {},
        "criteria": {},
        "machine": {"platform": platform.platform(), "machine": platform.machine()},
    }
    try:
        args.pool_url = sanitize_pool_url(args.pool_url)
        summary["pool_endpoint"] = args.pool_url
        validate_paths(args)
        summary["steps"]["preflight"] = {"manifest": verify_manifest(), "venv": create_offline_venv()}
        summary["criteria"]["preflight"] = "PASS"
        write_pool_config(
            config,
            source=args.config,
            state_dir=state_dir,
            max_accepted=args.max_accepted,
            max_submitted=args.max_submitted,
            max_seconds=args.max_seconds,
        )
        summary["steps"]["config"] = {"path": str(config), "max_accepted": args.max_accepted, "max_submitted": args.max_submitted, "max_seconds": args.max_seconds}
        if args.dry_run:
            target = args.max_submitted or args.max_accepted
            pool = classify_summary({"event": "pool_summary", "accepted": target, "stale": 0, "rejected": 0, "gate_failures": 0, "poisson_ok": True, "pass": True}, args, returncode=0)
            summary["result"] = "DRY_RUN"
        else:
            pool = run_pool_miner(args, config, miner_log)
            summary["result"] = pool["status"]
        summary["steps"]["pool"] = pool
        summary["criteria"]["pool"] = pool["status"]
        summary["elapsed_seconds"] = time.monotonic() - start
        write_summary(summary, args.summary)
        return 0 if summary["result"] == "DRY_RUN" else pool["exit_code"]
    except Exception as exc:
        summary["result"] = "STEP_FAIL"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["elapsed_seconds"] = time.monotonic() - start
        write_summary(summary, args.summary)
        jlog("step_failure", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
