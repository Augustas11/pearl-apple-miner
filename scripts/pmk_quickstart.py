#!/usr/bin/env python3
"""Shared quick-start state and G3 admission operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
G3_HELPER = ROOT / "dist" / "quickstart" / "bin" / "g3-admit"
G3_ROOT = ROOT / "dist" / "quickstart"
VALID_HOURS = 6.0


def pmk_home() -> Path:
    configured = os.environ.get("PMK_HOME")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".pmk"


def admission_path() -> Path:
    return pmk_home() / "g3-admission.json"


def _runtime_imports() -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(ROOT / "miner"))
    from pmk_miner.native import Native
    from pmk_miner.runtime import atomic_json, gpu_lock

    return Native, atomic_json, gpu_lock


def _environment(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "miner"), env.get("PYTHONPATH", "")) if part
    )
    env["PMK_B4_ROOT"] = str(G3_ROOT)
    env["PMK_RESOURCE_BUNDLE"] = str(
        ROOT / "libpmk" / ".build" / "release" / "libpmk_PMK.bundle"
    )
    env["PMK_G3_ADMISSION_FILE"] = str(path)
    return env


def _fresh_record(path: Path, now: float) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        devices = value["devices"]
        if not isinstance(devices, list) or not devices:
            return False
        return any(_record_fresh(record, now) for record in devices)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return False


def _record_fresh(record: Any, now: float) -> bool:
    if not isinstance(record, dict) or record.get("g3_passed") is not True:
        return False
    try:
        last_probe = float(record["last_probe_unix"])
        valid_hours = min(float(record.get("valid_hours", VALID_HOURS)), VALID_HOURS)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        valid_hours > 0
        and last_probe > 0
        and last_probe <= now
        and now - last_probe <= valid_hours * 3600
    )


def _production_probe(Native: Any, path: Path) -> None:
    previous = os.environ.get("PMK_G3_ADMISSION_FILE")
    os.environ["PMK_G3_ADMISSION_FILE"] = str(path)
    native = None
    try:
        native = Native()
        admission = json.loads(path.read_text(encoding="utf-8"))
        records = admission.get("devices", [])
        if not any(
            isinstance(record, dict)
            and record.get("cache_key") == native.probe_key
            and _record_fresh(record, time.time())
            for record in records
        ):
            raise RuntimeError("production probe differs from the G3 admission")
    finally:
        if native is not None:
            native.close()
        if previous is None:
            os.environ.pop("PMK_G3_ADMISSION_FILE", None)
        else:
            os.environ["PMK_G3_ADMISSION_FILE"] = previous


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_full_g3(
    path: Path,
    atomic_json: Any,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    if not G3_HELPER.is_file():
        raise RuntimeError("missing G3 helper; rerun scripts/install.sh")
    deadline = time.monotonic() + 1200
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [str(G3_HELPER)],
            cwd=ROOT,
            env=_environment(path),
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    raise InterruptedError("G3 admission cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("G3 admission timed out after 1200 seconds")
                try:
                    process.wait(timeout=min(0.5, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            _stop_process(process)
            raise
        process.wait()
        output.seek(0)
        stdout = output.read()
    print(stdout, end="")
    if process.returncode != 0:
        raise RuntimeError(f"G3 helper exited with status {process.returncode}")
    admission = None
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("devices"), list):
            admission = candidate
    if admission is None:
        raise RuntimeError("G3 helper produced no admission record")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    atomic_json(path, admission)


def ensure_admission(
    *,
    force: bool = False,
    inherited: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Ensure a current, exact-device G3 admission under the shared GPU lock."""
    Native, atomic_json, gpu_lock = _runtime_imports()
    path = admission_path()

    def lock_log(event: str, **_fields: Any) -> None:
        if event == "gpu_lock_wait":
            print("Waiting for another GPU task to finish...", flush=True)

    try:
        with gpu_lock(lock_log, inherited=inherited):
            if not force and _fresh_record(path, time.time()):
                try:
                    _production_probe(Native, path)
                    print(f"G3 admission is current: {path}")
                    return path
                except Exception:
                    # A copied record or an OS/kernel change must be recertified.
                    pass
            # A new full check supersedes the previous pass, even if interrupted.
            path.unlink(missing_ok=True)
            _run_full_g3(path, atomic_json, cancelled)
            _production_probe(Native, path)
            print(f"G3 admission passed: {path}")
    except InterruptedError:
        raise
    except Exception:
        print("this Mac failed the correctness check; do not mine", file=sys.stderr)
        path.unlink(missing_ok=True)
        raise
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pearl miner quick-start helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    admission = subparsers.add_parser("admission", help="ensure the six-hour G3 admission")
    admission.add_argument("--force", action="store_true", help="rerun the full G3 suite")
    args = parser.parse_args(argv)
    try:
        if args.command == "admission":
            ensure_admission(force=args.force, inherited=False)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            print(f"G3 admission error: {exc}", file=sys.stderr)
        else:
            print("G3 admission interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
