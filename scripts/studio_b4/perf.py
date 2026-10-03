#!/usr/bin/env python3
"""B4 performance gates for the bundle.

The runner is local-only. It creates synthetic cert-v3 jobs, exercises the real
pmk_miner Pipeline for K3-SG, and compares completed work with the companion
Swift K3-only benchmark compiled from the same production k3sg.metal source.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import pathlib
import platform
import inspect
import resource
import statistics
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterable
from typing import Any


HEADER_BITS = 0x177FD82E
CERT_VERSION = 3


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    candidates = [
        pathlib.Path(os.environ.get("PMK_B4_ROOT", "")).expanduser()
        if os.environ.get("PMK_B4_ROOT")
        else None,
        pathlib.Path.cwd(),
        here.parents[2] if len(here.parents) > 2 else None,
        here.parents[1] if len(here.parents) > 1 else None,
    ]
    for candidate in candidates:
        if candidate and (candidate / "miner" / "pmk_miner").is_dir():
            return candidate
    return pathlib.Path.cwd()


ROOT = _repo_root()
if str(ROOT / "miner") not in sys.path:
    sys.path.insert(0, str(ROOT / "miner"))


def _load_miner_modules():
    import pearl_mining as pm
    from pmk_miner.monitor import (
        bits_to_target,
        choose_share_nbits,
        expected_shares,
        poisson_interval,
    )
    from pmk_miner.native import Native
    from pmk_miner.pipeline import Pipeline, Shape
    from pmk_miner.transport import GatewayJob

    return {
        "pm": pm,
        "bits_to_target": bits_to_target,
        "choose_share_nbits": choose_share_nbits,
        "expected_shares": expected_shares,
        "poisson_interval": poisson_interval,
        "Native": Native,
        "Pipeline": Pipeline,
        "Shape": Shape,
        "GatewayJob": GatewayJob,
    }


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def emit_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default), flush=True)


def note(message: str) -> None:
    print(f"[b4-perf] {message}", file=sys.stderr, flush=True)


def note_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default), file=sys.stderr, flush=True)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_rate(ops: int, seconds: float) -> float:
    return 0.0 if seconds <= 0 else ops / seconds


def peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _profile_path() -> pathlib.Path | None:
    raw = os.environ.get("PMK_PROFILE_JSONL")
    if not raw:
        return None
    return pathlib.Path(raw).expanduser()


def _append_profile(rows: list[dict[str, Any]]) -> None:
    path = _profile_path()
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default) + "\n")


def _stage_summary(records: list[Any]) -> dict[str, dict[str, float | int | None]]:
    return _stage_summary_from_maps([getattr(record, "stage_seconds", {}) for record in records])


def _stage_summary_from_maps(stage_maps: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    values: dict[str, list[float]] = {}
    for stage_map in stage_maps:
        for name, seconds in stage_map.items():
            try:
                value = float(seconds)
            except (TypeError, ValueError):
                continue
            values.setdefault(str(name), []).append(value)
    return {
        name: {
            "count": len(samples),
            "ms_median": (_median(samples) or 0.0) * 1e3,
            "ms_p99": (_percentile(samples, 0.99) or 0.0) * 1e3,
            "seconds_total": sum(samples),
        }
        for name, samples in sorted(values.items())
    }


def _gpu_timeline(logs: list[dict[str, Any]]) -> dict[str, Any]:
    dispatches = [row for row in logs if row.get("event") == "gpu_dispatch"]
    intervals: list[tuple[float, float]] = []
    for row in logs:
        if row.get("event") != "completed":
            continue
        try:
            start = float(row.get("gpu_start_time"))
            end = float(row.get("gpu_end_time"))
        except (TypeError, ValueError):
            continue
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    union_seconds = 0.0
    gap_seconds = 0.0
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            if merged:
                gap_seconds += max(0.0, start - merged[-1][1])
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    union_seconds = sum(end - start for start, end in merged)
    total_span = (merged[-1][1] - merged[0][0]) if merged else 0.0
    return {
        "gpu_intervals": len(intervals),
        "gpu_union_seconds": union_seconds,
        "gpu_span_seconds": total_span,
        "gpu_idle_gap_seconds": gap_seconds,
        "gpu_idle_gap_pct": (100.0 * gap_seconds / total_span) if total_span > 0 else None,
        "gpu_overlap_seconds": max(0.0, sum(end - start for start, end in intervals) - union_seconds),
        "max_inflight": max((int(row.get("inflight", 0)) for row in dispatches), default=0),
        "dispatch_events": len(dispatches),
    }


def _profile_rows(records: list[Any], logs: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    completed = {int(row["job_id"]): row for row in logs if row.get("event") == "completed" and "job_id" in row}
    overhead = {int(row["job_id"]): row for row in logs if row.get("event") == "python_overhead" and "job_id" in row}
    dispatch = {int(row["job_id"]): row for row in logs if row.get("event") == "gpu_dispatch" and "job_id" in row}
    rows: list[dict[str, Any]] = []
    for record in records:
        job_id = int(getattr(record, "job_id", 0))
        row = {
            "mode": mode,
            "job_id": job_id,
            "cancelled": bool(getattr(record, "cancelled", False)),
            "stage_seconds": dict(getattr(record, "stage_seconds", {})),
        }
        if job_id in completed:
            row["completed"] = completed[job_id]
        if job_id in overhead:
            row["python_overhead"] = overhead[job_id]
        if job_id in dispatch:
            row["gpu_dispatch"] = dispatch[job_id]
        rows.append(row)
    return rows


def _profile_row_from_maps(
    record: Any,
    *,
    mode: str,
    completed: dict[int, dict[str, Any]],
    overhead: dict[int, dict[str, Any]],
    dispatch: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    job_id = int(getattr(record, "job_id", 0))
    row = {
        "mode": mode,
        "job_id": job_id,
        "cancelled": bool(getattr(record, "cancelled", False)),
        "stage_seconds": dict(getattr(record, "stage_seconds", {})),
    }
    if job_id in completed:
        row["completed"] = completed[job_id]
    if job_id in overhead:
        row["python_overhead"] = overhead[job_id]
    if job_id in dispatch:
        row["gpu_dispatch"] = dispatch[job_id]
    return row


class BoundedProfile:
    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self.records_seen = 0
        self.logs_seen = 0
        self.records_truncated = 0
        self.logs_truncated = 0
        self.records: deque[Any] = deque(maxlen=self.limit)
        self.logs: deque[dict[str, Any]] = deque(maxlen=self.limit * 4 if self.limit else 0)
        self.completed_by_job: dict[int, dict[str, Any]] = {}
        self.overhead_by_job: dict[int, dict[str, Any]] = {}
        self.dispatch_by_job: dict[int, dict[str, Any]] = {}

    def add_log(self, row: dict[str, Any]) -> None:
        self.logs_seen += 1
        try:
            job_id = int(row["job_id"])
        except (KeyError, TypeError, ValueError):
            job_id = 0
        if job_id:
            event = row.get("event")
            if event == "completed":
                self.completed_by_job[job_id] = row
            elif event == "python_overhead":
                self.overhead_by_job[job_id] = row
            elif event == "gpu_dispatch":
                self.dispatch_by_job[job_id] = row
        before = len(self.logs)
        self.logs.append(row)
        if len(self.logs) == before and self.limit:
            self.logs_truncated += 1
        elif not self.limit:
            self.logs_truncated += 1

    def add_record(self, record: Any) -> None:
        self.records_seen += 1
        before = len(self.records)
        self.records.append(record)
        if len(self.records) == before and self.limit:
            self.records_truncated += 1
        elif not self.limit:
            self.records_truncated += 1

    def metadata(self) -> dict[str, int]:
        return {
            "retained_record_limit": self.limit,
            "records_seen": self.records_seen,
            "records_retained": len(self.records),
            "records_truncated": self.records_truncated,
            "logs_seen": self.logs_seen,
            "logs_retained": len(self.logs),
            "logs_truncated": self.logs_truncated,
        }

    def take_row_for_record(self, record: Any, *, mode: str) -> dict[str, Any]:
        row = _profile_row_from_maps(
            record,
            mode=mode,
            completed=self.completed_by_job,
            overhead=self.overhead_by_job,
            dispatch=self.dispatch_by_job,
        )
        for index in (self.completed_by_job, self.overhead_by_job, self.dispatch_by_job):
            index.pop(record.job_id, None)
        return row


def _pipeline_worker_count(args: argparse.Namespace, shape: Any) -> int:
    if getattr(args, "command", "") == "pipeline" and getattr(args, "profile_serial", False):
        return 1
    return int(shape.slots)


def sustained_window_summary(
    *,
    elapsed_seconds: float,
    interval_seconds: float,
    delta_ops: int,
    completed_jobs: int,
    shares_total: int,
    shares_delta: int,
    expected_fn: Any,
    poisson_fn: Any,
    share_nbits: int,
) -> dict[str, Any]:
    expected = expected_fn(delta_ops, share_nbits)
    lower, upper = poisson_fn(expected)
    return {
        "event": "sustained_window",
        "elapsed_seconds": elapsed_seconds,
        "interval_seconds": interval_seconds,
        "ops": delta_ops,
        "tops": _safe_rate(delta_ops, interval_seconds) / 1e12,
        "completed_jobs": completed_jobs,
        "shares": shares_total,
        "shares_delta": shares_delta,
        "expected_shares": expected,
        "poisson_lower": lower,
        "poisson_upper": upper,
        "poisson_pass": lower <= shares_delta <= upper,
        "peak_rss_bytes": peak_rss_bytes(),
    }


def _find_existing(paths: Iterable[pathlib.Path], label: str) -> pathlib.Path:
    for path in paths:
        if path.is_file():
            return path
    joined = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"could not find {label}; checked {joined}")


def dylib_paths() -> tuple[pathlib.Path, pathlib.Path]:
    core_env = os.environ.get("PMKCORE_DYLIB")
    metal_env = os.environ.get("LIBPMK_DYLIB")
    core_candidates = []
    metal_candidates = []
    if core_env:
        core_candidates.append(pathlib.Path(core_env))
    if metal_env:
        metal_candidates.append(pathlib.Path(metal_env))
    core_candidates.extend(
        [
            ROOT / "lib" / "libpmkcore.dylib",
            ROOT / "bin" / "libpmkcore.dylib",
            ROOT / "pmkcore" / "target" / "release" / "libpmkcore.dylib",
        ]
    )
    metal_candidates.extend(
        [
            ROOT / "lib" / "libpmk.dylib",
            ROOT / "bin" / "libpmk.dylib",
            ROOT / "libpmk" / ".build" / "release" / "libpmk.dylib",
            ROOT / "libpmk" / ".build-b2" / "arm64-apple-macosx" / "release" / "libpmk.dylib",
        ]
    )
    return _find_existing(core_candidates, "libpmkcore.dylib"), _find_existing(
        metal_candidates, "libpmk.dylib"
    )


def machine_summary() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hw_model": run(["sysctl", "-n", "hw.model"]),
        "cpu_brand": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memsize": run(["sysctl", "-n", "hw.memsize"]),
        "macos": run(["sw_vers", "-productVersion"]),
        "python": sys.version.split()[0],
    }


def synthetic_job(bits: int = HEADER_BITS):
    mods = _load_miner_modules()
    pm = mods["pm"]
    GatewayJob = mods["GatewayJob"]
    bits_to_target = mods["bits_to_target"]
    header = pm.IncompleteBlockHeader(
        version=1,
        prev_block=bytes(32),
        merkle_root=bytes(32),
        timestamp=1,
        nbits=bits,
    )
    return GatewayJob(bytes(header.to_bytes()), bits_to_target(bits), CERT_VERSION)


def new_native():
    mods = _load_miner_modules()
    Native = mods["Native"]
    core, metal = dylib_paths()
    return Native(core=core, metal=metal)


def validate_shape(args: argparse.Namespace):
    Shape = _load_miner_modules()["Shape"]
    shape = Shape(args.m, args.n, args.k, args.slots)
    memory_budget = shape.validate()
    return shape, memory_budget


def new_pipeline(Pipeline: Any, native: Any, shape: Any, capture: Any, args: argparse.Namespace) -> Any:
    if getattr(args, "allow_slow_gpu", False) and "max_gpu_seconds" in inspect.signature(Pipeline).parameters:
        return Pipeline(native, shape, capture, max_gpu_seconds=None)
    return Pipeline(native, shape, capture)


def k3_alone_binary(args: argparse.Namespace) -> pathlib.Path | None:
    raw = args.k3_alone_bin or os.environ.get("PMK_K3_ALONE_BIN")
    candidates = []
    if raw:
        candidates.append(pathlib.Path(raw).expanduser())
    candidates.extend(
        [
            ROOT / "bin" / "k3_alone",
            ROOT / "bin" / "k3-alone",
            ROOT / "k3_alone",
            ROOT / "k3-alone",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_k3_alone_external(args: argparse.Namespace) -> dict[str, Any] | None:
    binary = k3_alone_binary(args)
    if binary is None:
        raise FileNotFoundError(
            "K3-alone benchmark binary not found; build scripts/studio_b4/k3_alone.swift to bin/k3-alone"
        )
    env = os.environ.copy()
    env.setdefault("PMK_B4_ROOT", str(ROOT))
    cmd = [
        str(binary),
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--jobs",
        str(args.jobs),
        "--inflight",
        str(args.slots),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=args.job_timeout * max(1, args.jobs) + 30)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} failed with {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{binary} did not emit JSON: {proc.stdout!r}") from exc
    result["binary"] = str(binary)
    result["scope"] = "k3_only"
    return result


async def run_pipeline_once(args: argparse.Namespace, native: Any | None = None) -> dict[str, Any]:
    mods = _load_miner_modules()
    Pipeline = mods["Pipeline"]
    bits_to_target = mods["bits_to_target"]
    choose_share_nbits = mods["choose_share_nbits"]
    expected_shares = mods["expected_shares"]
    shape, memory_budget = validate_shape(args)
    owned_native = native is None
    native = native or new_native()
    source = synthetic_job(args.bits)
    logs: list[dict[str, Any]] = []

    def capture(event: str, **fields: Any) -> None:
        logs.append({"event": event, **fields})

    pipeline = new_pipeline(Pipeline, native, shape, capture, args)
    records = []
    submissions = 0
    worker_count = _pipeline_worker_count(args, shape)
    share_nbits = choose_share_nbits(args.share_calibration_tops * 1e12, target_shares_per_minute=args.shares_per_minute)
    share_target = bits_to_target(share_nbits)

    async def submit(_job: Any, _proof: str) -> None:
        nonlocal submissions
        submissions += 1

    async def worker(slot_index: int, counter: list[int], lock: asyncio.Lock) -> None:
        while True:
            async with lock:
                if counter[0] >= args.jobs:
                    return
                counter[0] += 1
            record = await pipeline.run(slot_index, share_target, share_nbits, submit, lambda job: job == source)
            if not record.cancelled:
                records.append(record)

    start = time.monotonic()
    try:
        await asyncio.to_thread(pipeline.set_template, source)
        counter = [0]
        lock = asyncio.Lock()
        tasks = [asyncio.create_task(worker(i, counter, lock)) for i in range(worker_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
    finally:
        pipeline.cancel()
        if owned_native:
            native.close()
    wall_seconds = time.monotonic() - start
    completed = [row for row in logs if row["event"] == "completed"]
    overhead = [row for row in logs if row["event"] == "python_overhead"]
    stages = _stage_summary(records)
    build_seconds = [float(row.stage_seconds.get("build", 0.0)) for row in records]
    gpu_seconds = [float(row.get("gpu_seconds", 0.0)) for row in completed]
    wall_job_seconds = [float(row.get("wall_seconds", 0.0)) for row in completed]
    python_seconds = [float(row.get("python_seconds", 0.0)) for row in overhead]
    shares = sum(int(row.get("shares", 0)) for row in completed)
    blocks = sum(int(row.get("blocks", 0)) for row in completed)
    ops = shape.ops * len(records)
    python_overhead_pct = 100.0 * sum(python_seconds) / sum(wall_job_seconds) if wall_job_seconds else None
    profile_rows = _profile_rows(records, logs, mode="pipeline")
    _append_profile(profile_rows)
    return {
        "mode": "pipeline",
        "jobs_requested": args.jobs,
        "jobs": len(records),
        "shape": dataclasses.asdict(shape),
        "diagnostic_concurrency": {
            "requested_slots": int(shape.slots),
            "worker_count": worker_count,
            "profile_serial": bool(getattr(args, "profile_serial", False) and getattr(args, "command", "") == "pipeline"),
            "scope": "pipeline_only",
        },
        "memory_budget_bytes": memory_budget,
        "peak_rss_bytes": peak_rss_bytes(),
        "wall_seconds": wall_seconds,
        "ops": ops,
        "ops_per_second": _safe_rate(ops, wall_seconds),
        "tops": _safe_rate(ops, wall_seconds) / 1e12,
        "share_nbits": f"{share_nbits:08x}",
        "shares": shares,
        "blocks": blocks,
        "expected_shares": expected_shares(ops, share_nbits),
        "proof_submissions": submissions,
        "gpu_ms_median": (_median(gpu_seconds) or 0.0) * 1e3,
        "gpu_ms_p99": (_percentile(gpu_seconds, 0.99) or 0.0) * 1e3,
        "job_wall_ms_median": (_median(wall_job_seconds) or 0.0) * 1e3,
        "job_wall_ms_p99": (_percentile(wall_job_seconds, 0.99) or 0.0) * 1e3,
        "build_ms_median": (_median(build_seconds) or 0.0) * 1e3,
        "build_ms_p99": (_percentile(build_seconds, 0.99) or 0.0) * 1e3,
        "stage_ms": stages,
        "gpu_timeline": _gpu_timeline(logs),
        "profile_jsonl": str(_profile_path()) if _profile_path() else None,
        "allow_slow_gpu": bool(getattr(args, "allow_slow_gpu", False)),
        "studio_eligible": not bool(getattr(args, "allow_slow_gpu", False)),
        "python_overhead_pct": python_overhead_pct,
        "probe_key": native.probe_key,
    }


async def run_p2(args: argparse.Namespace) -> dict[str, Any]:
    production_shape = (args.m, args.n, args.k) == (8192, 8192, 4096)
    if production_shape and not args.quick and args.jobs < 200:
        raise ValueError("production P2 requires --jobs >= 200; use --quick only for explicit local smoke runs")
    note(f"running K3-alone baseline before pipeline: jobs={args.jobs} shape={args.m}x{args.n}x{args.k}")
    k3_before = run_k3_alone_external(args)
    native = new_native()
    try:
        args.share_calibration_tops = max(k3_before["tops"], 0.001)
        note(f"running pmk_miner pipeline: jobs={args.jobs} shape={args.m}x{args.n}x{args.k}")
        pipeline = await run_pipeline_once(args, native)
    finally:
        native.close()
    note(f"running K3-alone baseline after pipeline: jobs={args.jobs} shape={args.m}x{args.n}x{args.k}")
    k3_after = run_k3_alone_external(args)
    before_rate = float(k3_before.get("ops_per_second") or 0.0)
    after_rate = float(k3_after.get("ops_per_second") or 0.0)
    conservative_rate = max(before_rate, after_rate)
    geometric_rate = math.sqrt(before_rate * after_rate) if before_rate > 0 and after_rate > 0 else 0.0
    ratio_before = pipeline["ops_per_second"] / before_rate if before_rate else 0.0
    ratio_after = pipeline["ops_per_second"] / after_rate if after_rate else 0.0
    ratio_conservative = pipeline["ops_per_second"] / conservative_rate if conservative_rate else 0.0
    ratio_geometric = pipeline["ops_per_second"] / geometric_rate if geometric_rate else 0.0
    conservative_k3 = k3_before if before_rate >= after_rate else k3_after
    return {
        "criterion": "P2",
        "pass": ratio_conservative >= args.min_ratio and pipeline["jobs"] >= args.jobs and (args.quick or not production_shape or pipeline["jobs"] >= 200),
        "target_ratio": args.min_ratio,
        "ratio": ratio_conservative,
        "ratio_policy": "pipeline_ops_per_second / max(k3_before_ops_per_second, k3_after_ops_per_second)",
        "ratio_before": ratio_before,
        "ratio_after": ratio_after,
        "ratio_geometric": ratio_geometric,
        "ratio_conservative": ratio_conservative,
        "paired_same_run": True,
        "paired_order": ["k3_before", "pipeline", "k3_after"],
        "quick": args.quick,
        "k3_alone": conservative_k3,
        "k3_before": k3_before,
        "k3_after": k3_after,
        "pipeline": pipeline,
        "machine": machine_summary(),
    }


async def run_sustained(args: argparse.Namespace) -> dict[str, Any]:
    mods = _load_miner_modules()
    Pipeline = mods["Pipeline"]
    bits_to_target = mods["bits_to_target"]
    choose_share_nbits = mods["choose_share_nbits"]
    expected_shares = mods["expected_shares"]
    poisson_interval = mods["poisson_interval"]
    shape, memory_budget = validate_shape(args)

    if args.ops_per_second <= 0:
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.jobs = max(2, min(args.warmup_jobs, args.jobs))
        warmup_args.shares_per_minute = 0.01
        warmup = await run_pipeline_once(warmup_args)
        args.ops_per_second = max(warmup["ops_per_second"], 1.0)

    native = new_native()
    source = synthetic_job(args.bits)
    profile = BoundedProfile(getattr(args, "retained_profile_jobs", 4096))
    counters = {"shares": 0, "blocks": 0}

    def capture(event: str, **fields: Any) -> None:
        profile.add_log({"event": event, **fields})
        if event == "completed":
            counters["shares"] += int(fields.get("shares", 0))
            counters["blocks"] += int(fields.get("blocks", 0))

    pipeline = new_pipeline(Pipeline, native, shape, capture, args)
    start = 0.0
    last_tick = 0.0
    last_ops = 0
    last_shares = 0
    windows: list[dict[str, Any]] = []
    completed_ops = 0
    completed_jobs = 0
    submissions = 0

    share_nbits = choose_share_nbits(args.ops_per_second, target_shares_per_minute=args.shares_per_minute)
    share_target = bits_to_target(share_nbits)

    async def submit(_job: Any, _proof: str) -> None:
        nonlocal submissions
        submissions += 1

    async def worker(slot_index: int) -> None:
        nonlocal completed_ops, completed_jobs, last_tick, last_ops, last_shares
        while time.monotonic() - start < args.seconds:
            record = await pipeline.run(slot_index, share_target, share_nbits, submit, lambda job: job == source)
            if record.cancelled:
                continue
            completed_jobs += 1
            profile.add_record(record)
            _append_profile([profile.take_row_for_record(record, mode="sustained")])
            completed_ops += shape.ops
            now = time.monotonic()
            if now - last_tick >= args.report_interval:
                delta_ops = completed_ops - last_ops
                shares_delta = counters["shares"] - last_shares
                window = sustained_window_summary(
                    elapsed_seconds=now - start,
                    interval_seconds=now - last_tick,
                    delta_ops=delta_ops,
                    completed_jobs=completed_jobs,
                    shares_total=counters["shares"],
                    shares_delta=shares_delta,
                    expected_fn=expected_shares,
                    poisson_fn=poisson_interval,
                    share_nbits=share_nbits,
                )
                windows.append(window)
                note_json(window)
                last_tick = now
                last_ops = completed_ops
                last_shares = counters["shares"]

    try:
        await asyncio.to_thread(pipeline.set_template, source)
        start = time.monotonic()
        last_tick = start
        tasks = [asyncio.create_task(worker(i)) for i in range(shape.slots)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
    finally:
        pipeline.cancel()
        native.close()
    elapsed = time.monotonic() - start
    tail_ops = completed_ops - last_ops
    tail_interval = start + elapsed - last_tick if last_tick else 0.0
    if tail_ops > 0 and tail_interval > 0:
        tail = sustained_window_summary(
            elapsed_seconds=elapsed,
            interval_seconds=tail_interval,
            delta_ops=tail_ops,
            completed_jobs=completed_jobs,
            shares_total=counters["shares"],
            shares_delta=counters["shares"] - last_shares,
            expected_fn=expected_shares,
            poisson_fn=poisson_interval,
            share_nbits=share_nbits,
        )
        tail["partial"] = True
        windows.append(tail)
        note_json(tail)
    retained_logs = list(profile.logs)
    retained_records = list(profile.records)
    completed = [row for row in retained_logs if row["event"] == "completed"]
    overhead = [row for row in retained_logs if row["event"] == "python_overhead"]
    wall_job_seconds = [float(row.get("wall_seconds", 0.0)) for row in completed]
    python_seconds = [float(row.get("python_seconds", 0.0)) for row in overhead]
    python_overhead_pct = 100.0 * sum(python_seconds) / sum(wall_job_seconds) if wall_job_seconds else None
    expected = expected_shares(completed_ops, share_nbits)
    lower, upper = poisson_interval(expected)
    shares = counters["shares"]
    blocks = counters["blocks"]
    return {
        "criterion": "b4_sustained_smoke",
        "satisfies_spec_p4": False,
        "pass": lower <= shares <= upper and elapsed >= args.seconds,
        "seconds_target": args.seconds,
        "elapsed_seconds": elapsed,
        "shape": dataclasses.asdict(shape),
        "memory_budget_bytes": memory_budget,
        "peak_rss_bytes": peak_rss_bytes(),
        "jobs": completed_jobs,
        "ops": completed_ops,
        "tops": _safe_rate(completed_ops, elapsed) / 1e12,
        "share_nbits": f"{share_nbits:08x}",
        "shares": shares,
        "blocks": blocks,
        "proof_submissions": submissions,
        "expected_shares": expected,
        "poisson_lower": lower,
        "poisson_upper": upper,
        "windows": windows,
        "stage_ms": _stage_summary(retained_records),
        "gpu_timeline": _gpu_timeline(retained_logs),
        "retention": profile.metadata(),
        "profile_jsonl": str(_profile_path()) if _profile_path() else None,
        "allow_slow_gpu": bool(getattr(args, "allow_slow_gpu", False)),
        "studio_eligible": not bool(getattr(args, "allow_slow_gpu", False)),
        "python_overhead_pct": python_overhead_pct,
        "machine": machine_summary(),
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    shape = argparse.Namespace(slots=args.slots)
    worker_count = _pipeline_worker_count(args, shape)
    return {
        "mode": args.command,
        "dry_run": True,
        "root": ROOT,
        "machine": machine_summary(),
        "shape": {"m": args.m, "n": args.n, "k": args.k, "slots": args.slots},
        "allow_slow_gpu": bool(getattr(args, "allow_slow_gpu", False)),
        "studio_eligible": not bool(getattr(args, "allow_slow_gpu", False)),
        "retained_profile_jobs": int(getattr(args, "retained_profile_jobs", 0)),
        "diagnostic_concurrency": {
            "requested_slots": int(args.slots),
            "worker_count": worker_count,
            "profile_serial": bool(getattr(args, "profile_serial", False) and args.command == "pipeline"),
            "scope": "pipeline_only",
        },
        "dylibs": {
            "libpmkcore": str(dylib_paths()[0]) if not args.skip_dylib_check else None,
            "libpmk": str(dylib_paths()[1]) if not args.skip_dylib_check else None,
        },
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--m", type=int, default=8192)
        sp.add_argument("--n", type=int, default=8192)
        sp.add_argument("--k", type=int, default=4096)
        sp.add_argument("--slots", type=int, default=2)
        sp.add_argument("--bits", type=lambda x: int(x, 0), default=HEADER_BITS)
        sp.add_argument("--jobs", type=int, default=200)
        sp.add_argument("--share-calibration-tops", type=float, default=16.0)
        sp.add_argument("--shares-per-minute", type=float, default=1.0)
        sp.add_argument("--job-timeout", type=float, default=30.0)
        sp.add_argument("--k3-alone-bin", default="")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--skip-dylib-check", action="store_true")
        sp.add_argument("--allow-slow-gpu", action="store_true", help="diagnostic-only: disable the 400 ms command-buffer guard when supported")
        sp.add_argument("--retained-profile-jobs", type=int, default=4096, help="max retained records/log samples for long-run summaries; PMK_PROFILE_JSONL still streams per-job rows")

    sp = sub.add_parser("k3-alone", help="direct libpmk K3-only wall-rate benchmark")
    add_common(sp)
    sp = sub.add_parser("pipeline", help="pmk_miner Pipeline benchmark")
    add_common(sp)
    sp.add_argument("--profile-serial", action="store_true", help="diagnostic-only: use one worker with the requested slot/buffer shape")
    sp = sub.add_parser("p2", help="B4 P2 ratio: pipeline throughput / K3-alone throughput")
    add_common(sp)
    sp.add_argument("--min-ratio", type=float, default=0.85)
    sp.add_argument("--quick", action="store_true")
    sp = sub.add_parser("sustained", help="B4 10-minute sustained K3-SG share monitor")
    add_common(sp)
    sp.add_argument("--seconds", type=float, default=600.0)
    sp.add_argument("--report-interval", type=float, default=30.0)
    sp.add_argument("--ops-per-second", type=float, default=0.0)
    sp.add_argument("--warmup-jobs", type=int, default=8)
    return p


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        return dry_run(args)
    if args.command == "k3-alone":
        return run_k3_alone_external(args)
    if args.command == "pipeline":
        return await run_pipeline_once(args)
    if args.command == "p2":
        return await run_p2(args)
    if args.command == "sustained":
        return await run_sustained(args)
    raise ValueError(args.command)


def main() -> int:
    args = parser().parse_args()
    try:
        result = asyncio.run(async_main(args))
        emit_json(result)
        return 0 if result.get("pass", True) else 2
    except BaseException as exc:
        emit_json(
            {
                "pass": False,
                "criterion": getattr(args, "command", "perf"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "machine": machine_summary(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
