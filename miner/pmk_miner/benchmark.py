"""Offline production-pipeline benchmark for paste-ready device reports."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import time
from typing import Any

from . import __version__
from .kernel import apply_kernel_environment, resolve_v3_kernel, scheme_for_kernel
from .monitor import DIFF1_TARGET, POW_DENOMINATOR, bits_to_target
from .native import Native
from .pipeline import Pipeline, Shape
from .transport import GatewayJob


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_SHAPE = Shape(8192, 8192, 4096, 2)
BENCHMARK_BITS = 0x01010000
BENCHMARK_TARGET = 1
DIFF1_MACS = POW_DENOMINATOR / (2 * DIFF1_TARGET)


def synthetic_job() -> GatewayJob:
    header = (
        (1).to_bytes(4, "little") + bytes(32) + bytes(32)
        + (1).to_bytes(4, "little") + BENCHMARK_BITS.to_bytes(4, "little")
    )
    if bits_to_target(BENCHMARK_BITS) != BENCHMARK_TARGET:
        raise AssertionError("benchmark compact target is not exact")
    return GatewayJob(header, BENCHMARK_TARGET, 3)


def _admission_module():
    path = ROOT / "scripts" / "pmk_benchmark_admit.py"
    if not path.is_file():
        raise RuntimeError("missing benchmark G3 admission helper")
    spec = importlib.util.spec_from_file_location("pmk_benchmark_admit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load benchmark G3 admission helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _ensure_admission(stop: asyncio.Event) -> Path:
    module = _admission_module()
    task = asyncio.create_task(asyncio.to_thread(
        module.ensure_benchmark_admission, stop.is_set, inherited=True,
    ))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            stop.set()
    if cancelled:
        raise asyncio.CancelledError
    return result


def _reuse_admission_path() -> Path:
    raw = os.environ.get("PMK_G3_ADMISSION_FILE")
    if not raw:
        raise RuntimeError("benchmark admission reuse requires PMK_G3_ADMISSION_FILE")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError("benchmark admission reuse file is missing")
    return path


def _sysctl(name: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip() or default
    except (OSError, subprocess.CalledProcessError):
        return default


def _host_fields() -> dict[str, Any]:
    chip = _sysctl("machdep.cpu.brand_string", _sysctl("hw.model"))
    cores = "unknown"
    try:
        displays = json.loads(subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=30,
        ))
        items = displays.get("SPDisplaysDataType", [])
        if items:
            chip = str(items[0].get("sppci_chipset_model", chip))
            cores = str(items[0].get("sppci_cores", cores))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            TypeError, ValueError, json.JSONDecodeError):
        pass
    return {
        "chip": chip,
        "gpu_cores": cores,
        "macos": platform.mac_ver()[0] or platform.platform(),
        "pmk_version": __version__,
    }


def _paste_block(summary: dict[str, Any]) -> str:
    shape = summary["shape"]
    expected = summary["expected_share_seconds"]
    expected_text = (f"{expected/3600:.1f} h" if expected >= 3600 else f"{expected/60:.1f} min") if expected is not None else "n/a"
    return "\n".join((
        "```text",
        f"PMK {summary['pmk_version']} | macOS {summary['macos']}",
        f"Chip: {summary['chip']} | cores: {summary['gpu_cores']} | power: {summary['power_source']}",
        f"Kernel: K3-{summary['kernel'].upper()} | Shape: {shape['m']}x{shape['n']}x{shape['k']} ({shape['slots']} slots)",
        f"Duration: {summary['elapsed_seconds']:.2f}s | jobs: {summary['jobs']}",
        f"Throughput: {summary['tops']:.3f} TOPS | {summary['macs_per_second']:,.0f} MAC/s (pool H/s) | {summary['jobs_per_second']:.3f} jobs/s",
        f"GPU busy: {summary['gpu_busy_pct']:.1f}% | difficulty: {summary['difficulty']:.0f} | expected share: {expected_text}",
        "```",
    ))


def _install_signal_handlers(stop: asyncio.Event):
    loop = asyncio.get_running_loop()
    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)
            loop.add_signal_handler(sig, stop.set)
            installed.append((sig, previous))
        except (NotImplementedError, RuntimeError):
            pass
    return loop, installed


async def run_benchmark(args, log) -> dict[str, Any]:
    """Run the real offline mining pipeline for the requested wall duration."""
    duration = int(args.benchmark if args.benchmark is not None else 60)
    difficulty = float(getattr(args, "difficulty", 2**21))
    if not 10 <= duration <= 600:
        raise ValueError("benchmark must be in [10, 600]")
    if not 1 <= difficulty <= 2**64:
        raise ValueError("difficulty must be in [1, 2^64]")
    kernel, device_class = resolve_v3_kernel(getattr(args, "kernel", "auto"))
    apply_kernel_environment(kernel)
    scheme = scheme_for_kernel(kernel)

    stop = asyncio.Event()
    loop, installed = _install_signal_handlers(stop)
    native = None
    pipeline = None
    records = []
    desktop = getattr(args, "desktop", None)
    clock = getattr(args, "benchmark_clock", time.monotonic)
    try:
        reuse_admission = bool(getattr(args, "benchmark_reuse_admission", False))
        admission = _reuse_admission_path() if reuse_admission else await _ensure_admission(stop)
        if stop.is_set():
            raise InterruptedError("benchmark interrupted during G3 admission")
        log("benchmark_admission", path=str(admission), full_g3=not reuse_admission,
            reused=reuse_admission, kernel=kernel)
        source = synthetic_job()
        native = _native_for_kernel(kernel)
        kernel_metadata = (native.validate_v3_kernel(scheme)
                           if hasattr(native, "validate_v3_kernel") else None)
        pipeline = _pipeline_for_scheme(
            native, PRODUCTION_SHAPE, log, scheme,
            max_gpu_seconds=None, desktop=desktop,
        )
        pipeline.set_template(source)
        power_source = desktop.power_source() if desktop is not None else "unknown"
        busy_before = float(desktop.busy_seconds) if desktop is not None else 0.0
        started = clock()
        deadline = started + duration
        log("benchmark_started", duration_seconds=duration, difficulty=difficulty,
            shape={"m": 8192, "n": 8192, "k": 4096, "slots": 2},
            power_source=power_source, requested_kernel=getattr(args, "kernel", "auto"),
            effective_kernel=kernel, device_class=device_class,
            kernel_metadata=kernel_metadata)

        async def submit(*_ignored) -> None:
            raise AssertionError("tiny-target offline benchmark unexpectedly found a candidate")

        async def run_slot(index: int) -> None:
            while not stop.is_set() and clock() < deadline:
                record = await pipeline.run(
                    index, BENCHMARK_TARGET, BENCHMARK_BITS, submit,
                    lambda candidate: (candidate is source and not stop.is_set()
                                       and clock() < deadline),
                )
                if record.completed_ops:
                    records.append(record)

        lane_results = await asyncio.gather(
            *(run_slot(index) for index in range(PRODUCTION_SHAPE.slots)),
            return_exceptions=True,
        )
        for result in lane_results:
            if isinstance(result, BaseException):
                raise result
        elapsed = max(clock() - started, 1e-9)
        completed_ops = sum(record.completed_ops for record in records)
        jobs = len(records)
        ops_per_second = completed_ops / elapsed
        macs_per_second = ops_per_second / 2
        busy_seconds = max(0.0, (float(desktop.busy_seconds) - busy_before) if desktop else sum(r.gpu_seconds for r in records))
        ordered_gpu_seconds = [record.gpu_seconds for record in sorted(records, key=lambda record: getattr(record, "job_id", 0))
                               if record.gpu_seconds > 0]
        drift_window = max(1, len(ordered_gpu_seconds) // 10) if ordered_gpu_seconds else 0
        early_gpu_seconds = (statistics.median(ordered_gpu_seconds[:drift_window])
                             if drift_window else None)
        late_gpu_seconds = (statistics.median(ordered_gpu_seconds[-drift_window:])
                            if drift_window else None)
        max_gpu_seconds = max(ordered_gpu_seconds) if ordered_gpu_seconds else None
        over_budget_jobs = sum(value > 0.4 for value in ordered_gpu_seconds)
        gpu_job_time_drift_pct = (
            100.0 * (late_gpu_seconds / early_gpu_seconds - 1.0)
            if early_gpu_seconds and late_gpu_seconds else None
        )
        summary = {
            "event": "benchmark_summary",
            "offline": True,
            "shape": {"m": 8192, "n": 8192, "k": 4096, "slots": 2},
            "kernel": kernel,
            "device_class": device_class,
            "kernel_metadata": kernel_metadata,
            "requested_seconds": duration,
            "elapsed_seconds": elapsed,
            "jobs": jobs,
            "jobs_per_second": jobs / elapsed,
            "completed_ops": completed_ops,
            "ops_per_second": ops_per_second,
            "tops": ops_per_second / 1e12,
            "macs_per_second": macs_per_second,
            "pool_hashrate": macs_per_second,
            "gpu_busy_seconds": busy_seconds,
            "gpu_busy_pct": min(100.0, 100 * busy_seconds / elapsed),
            "gpu_job_seconds_early_median": early_gpu_seconds,
            "gpu_job_seconds_late_median": late_gpu_seconds,
            "gpu_job_time_drift_pct": gpu_job_time_drift_pct,
            "gpu_job_seconds_max": max_gpu_seconds,
            "command_budget_seconds": 0.4,
            "command_budget_overruns": over_budget_jobs,
            "command_budget_passed": bool(ordered_gpu_seconds) and over_budget_jobs == 0,
            "difficulty": difficulty,
            "diff1_macs": DIFF1_MACS,
            "expected_share_seconds": (difficulty * DIFF1_MACS / macs_per_second) if macs_per_second else None,
            "power_source": power_source,
            **_host_fields(),
        }
        summary["paste_block"] = _paste_block(summary)
        log("benchmark_summary", **{key: value for key, value in summary.items() if key != "event"})
        print(summary["paste_block"], flush=True)
        return summary
    finally:
        stop.set()
        if pipeline is not None:
            pipeline.cancel()
        if native is not None:
            native.close()
        for sig, previous in installed:
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous)


def _native_for_kernel(kernel):
    try:
        return Native(kernel=kernel)
    except TypeError:
        return Native()


def _pipeline_for_scheme(native, shape, log, scheme, **kwargs):
    try:
        return Pipeline(native, shape, log, scheme=scheme, **kwargs)
    except TypeError as exc:
        if "scheme" not in str(exc):
            raise
        return Pipeline(native, shape, log, **kwargs)
