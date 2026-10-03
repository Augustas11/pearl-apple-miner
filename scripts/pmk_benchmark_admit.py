#!/usr/bin/env python3
"""Run the full device G3 suite before an offline miner benchmark."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


def _load_quickstart():
    path = ROOT / "scripts" / "pmk_quickstart.py"
    if not path.is_file() or path.resolve() == Path(__file__).resolve():
        return None
    spec = importlib.util.spec_from_file_location("pmk_quickstart", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load G3 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _private_admission(cancelled: Callable[[], bool]) -> Path:
    """Run the bundled full G3 executable in a private development checkout."""
    bundle = ROOT / "dist" / "studio_b4"
    helper = bundle / "bin" / "g3-admit"
    if not helper.is_file():
        raise RuntimeError("missing full G3 admission helper; rerun the package build")
    path = Path(os.environ.get("PMK_HOME", Path.home() / ".pmk")) / "g3-admission.json"
    env = os.environ.copy()
    env.update(
        PMK_B4_ROOT=str(bundle),
        PMK_RESOURCE_BUNDLE=str(bundle / "libpmk/.build/release/libpmk_PMK.bundle"),
        PMK_G3_ADMISSION_FILE=str(path),
    )
    path.unlink(missing_ok=True)
    deadline = time.monotonic() + 1200
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [str(helper)], cwd=ROOT, env=env, text=True,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError("G3 admission cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("G3 admission timed out after 1200 seconds")
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            _stop_process(process)
            raise
        output.seek(0)
        stdout = output.read()
    if process.returncode != 0:
        raise RuntimeError(f"full G3 helper exited with status {process.returncode}: {stdout[-1000:]}")
    admission = None
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("devices"), list):
            admission = candidate
    if admission is None:
        raise RuntimeError("full G3 helper produced no admission record")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(admission, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)

    previous = os.environ.get("PMK_G3_ADMISSION_FILE")
    os.environ["PMK_G3_ADMISSION_FILE"] = str(path)
    native = None
    try:
        sys.path.insert(0, str(ROOT / "miner"))
        from pmk_miner.native import Native

        native = Native()
        if not any(
            isinstance(record, dict) and record.get("cache_key") == native.probe_key
            for record in admission["devices"]
        ):
            raise RuntimeError("production probe differs from the full G3 admission")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if native is not None:
            native.close()
        if previous is None:
            os.environ.pop("PMK_G3_ADMISSION_FILE", None)
        else:
            os.environ["PMK_G3_ADMISSION_FILE"] = previous
    return path


def ensure_benchmark_admission(
    cancelled: Callable[[], bool] = lambda: False, *, inherited: bool = False,
) -> Path:
    """Force a full G3 run; cached probe-only admission is insufficient."""
    quickstart = _load_quickstart()
    if quickstart is not None:
        path = quickstart.ensure_admission(force=True, inherited=inherited, cancelled=cancelled)
    elif inherited:
        path = _private_admission(cancelled)
    else:
        sys.path.insert(0, str(ROOT / "miner"))
        from pmk_miner.runtime import gpu_lock

        with gpu_lock(lambda *_args, **_kwargs: None):
            path = _private_admission(cancelled)
    os.environ["PMK_G3_ADMISSION_FILE"] = str(path)
    return Path(path)


def main() -> int:
    try:
        print(ensure_benchmark_admission())
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"G3 admission error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
