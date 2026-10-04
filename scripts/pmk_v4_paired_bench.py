#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Paired V4 kernel-E/K3-only vs V3 K3-SG benchmark harness.

This script is intentionally inert by default. Use ``--dry-run`` to inspect the
plan, or pass ``--run-gpu`` with ``PMK_GPU_LOCK_HELD=1`` already exported by the
coordinator. It does not acquire the GPU lock itself.

Measurement design:
- shape defaults to the production comparison point, 8192 x 8192 x 4096;
- V3 uses the current libpmk v3 ``pmk_run_job`` path with valid constant int8
  operands and deterministic seeds;
- V4 first builds a fresh pmkcore_v4 Grid-B200 job and runs production
  ``pmk_v4_run_job`` once to obtain quantized E4M3 code buffers from the current
  V4 implementation;
- measured V4 samples use ``pmk_v4_run_codes_diagnostic`` so the timed value is
  the V2/V4 K3 kernel path only; reported throughput claims are therefore
  labelled K3-only;
- measured V3 samples include the current v3 libpmk K3-SG command-buffer stages,
  including jackpot/lottery compare and slot atomics;
- jobs are interleaved after warmup to reduce heat/order bias.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from pmk_v4_paths import bundle_root

ROOT = bundle_root(__file__)
U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32
U64 = C.c_uint64
PTR = C.c_void_p
P = C.POINTER
PMK_PENDING = 1
PMK_SUCCESS = 0
DEFAULT_PROBE = ROOT / "libpmk/resources/v4_probe/manifest.json"
DEFAULT_OUTPUT = ROOT / "bench/evidence/b9_v5_paired_bench.txt"


class Slot(C.Structure):
    _fields_ = [("t_rows", U32), ("t_cols", U32), ("transcript", U32 * 16), ("hash", U32 * 8)]


class V3Desc(C.Structure):
    _fields_ = [(x, U32) for x in ("abi_version", "m", "n", "k")] + [
        ("a", PTR), ("bt", PTR), ("a_bytes", U64), ("bt_bytes", U64)
    ] + [(x, U32 * 8) for x in ("a_seed", "b_seed", "block_bound", "share_bound")] + [
        ("block_capacity", U32), ("share_capacity", U32), ("cert_version", U32), ("rank", U32), ("job_id", U64)
    ]


class V3Result(C.Structure):
    _fields_ = [("abi_version", U32), ("status", C.c_int32), ("job_id", U64)] + [
        (x, U32) for x in ("block_count", "share_count", "block_stored", "share_stored", "overflow", "recovered")
    ] + [("blocks", P(Slot)), ("shares", P(Slot)), ("gpu_start_time", C.c_double), ("gpu_end_time", C.c_double)]


class Pmk4GpuJobDesc(C.Structure):
    _fields_ = [(x, U32) for x in ("m", "n", "k", "rank", "tile_rows", "tile_cols", "row_period", "col_period")] + [
        ("a_values", P(C.c_int8)), ("a_scales", P(U16)), ("bt_values", P(C.c_int8)), ("bt_scales", P(U16)),
        ("a_noised", P(U8)), ("bt_noised", P(U8)),
        ("a_noise_e", P(U8)), ("a_noise_f", P(U8)), ("bt_noise_e", P(U8)), ("bt_noise_f", P(U8)),
        ("a_alpha", P(U16)), ("a_beta", P(U16)), ("a_l2", P(U16)),
        ("bt_alpha", P(U16)), ("bt_beta", P(U16)), ("bt_l2", P(U16)),
    ] + [(x, U8 * 32) for x in ("key_a", "key_b", "hash_a", "hash_b", "noise_seed_a", "noise_seed_b", "jackpot_key")]


class Pmk4TileResult(C.Structure):
    _fields_ = [
        ("t_rows", U32), ("t_cols", U32), ("message", U8 * 64), ("hash", U8 * 32),
        ("policy_pass", U32), ("is_share", U32), ("is_block", U32),
    ]


class V4OperandDesc(C.Structure):
    _fields_ = [
        ("clean_values", P(C.c_int8)), ("clean_value_count", U64),
        ("noise_e_codes", P(U8)), ("noise_e_count", U64),
        ("noise_f_codes", P(U8)), ("noise_f_count", U64),
        ("alpha_bf16", P(U16)), ("beta_bf16", P(U16)), ("scale_count", U64),
    ]


class V4JobDesc(C.Structure):
    _fields_ = [(x, U32) for x in ("abi_version", "m", "n", "k", "r")] + [
        ("a", V4OperandDesc), ("bt", V4OperandDesc),
        ("jackpot_key", U32 * 8), ("block_bound", U32 * 8), ("share_bound", U32 * 8),
        ("block_capacity", U32), ("share_capacity", U32), ("job_id", U64),
    ]


class V4CodesDesc(C.Structure):
    _fields_ = [(x, U32) for x in ("abi_version", "m", "n", "k")] + [
        ("a_codes", P(U8)), ("bt_codes", P(U8)), ("a_code_count", U64), ("bt_code_count", U64),
        ("jackpot_key", U32 * 8), ("block_bound", U32 * 8), ("share_bound", U32 * 8),
        ("block_capacity", U32), ("share_capacity", U32), ("job_id", U64),
    ]


class V4QuantizedCodes(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32), ("a_codes", P(U8)), ("bt_codes", P(U8)), ("a_code_count", U64), ("bt_code_count", U64)]


class V4Stats(C.Structure):
    _fields_ = [("abi_version", U32), ("flags", U32)] + [
        (x, U64) for x in (
            "fallback_groups", "total_groups", "quantized_a", "quantized_b",
            "quant_saturated_a", "quant_saturated_b", "quant_nan_a", "quant_nan_b",
        )
    ] + [("layout_failures", U32), ("fallback_alert", U32)]


class V4Result(C.Structure):
    _fields_ = [("abi_version", U32), ("status", C.c_int32), ("job_id", U64)] + [
        (x, U32) for x in ("block_count", "share_count", "block_stored", "share_stored", "overflow", "recovered")
    ] + [
        ("blocks", P(Slot)), ("shares", P(Slot)), ("stats", V4Stats),
        ("c_bits", P(U32)), ("c_count", U64), ("gpu_start_time", C.c_double), ("gpu_end_time", C.c_double),
    ]


def words(value: int | bytes) -> U32 * 8:
    data = value if isinstance(value, bytes) else int(value).to_bytes(32, "little")
    return (U32 * 8).from_buffer_copy(data)


def u8_buffer(data: bytes) -> C.Array[U8]:
    return (U8 * len(data)).from_buffer_copy(data)


def seconds(result: Any) -> float:
    value = float(result.gpu_end_time - result.gpu_start_time)
    if value <= 0:
        raise RuntimeError(f"non-positive GPU timing interval: {value}")
    return value


def summary(samples: list[dict[str, Any]], m: int, n: int, k: int) -> dict[str, Any]:
    times = sorted(float(row["gpu_seconds"]) for row in samples)
    if not times:
        raise ValueError("no samples")
    tops = sorted(2.0 * m * n * k / t / 1e12 for t in times)
    return {
        "count": len(times),
        "seconds_median": statistics.median(times),
        "seconds_p25": percentile(times, 0.25),
        "seconds_p75": percentile(times, 0.75),
        "tops_eq_median": statistics.median(tops),
        "tops_eq_p25": percentile(tops, 0.25),
        "tops_eq_p75": percentile(tops, 0.75),
    }


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


class LibPMK:
    def __init__(self, path: Path, *, diagnostic_v4_init: bool = False):
        self.lib = C.CDLL(str(path))
        self.cb_type = C.CFUNCTYPE(None, PTR, PTR)
        self.callbacks: list[Any] = []
        declarations = {
            "pmk_init": [P(PTR), PTR, U64],
            "pmk_buffer_alloc": [PTR, U64, P(PTR)],
            "pmk_buffer_release": [PTR, PTR],
            "pmk_run_job": [PTR, P(V3Desc), self.cb_type, PTR, P(PTR)],
            "pmk_poll": [PTR, P(V3Result)],
            "pmk_job_wait_callback": [PTR],
            "pmk_job_release": [PTR],
            "pmk_context_error": [PTR, PTR, U64],
            "pmk_job_error": [PTR, PTR, U64],
            "pmk_v4_init_diagnostic" if diagnostic_v4_init else "pmk_v4_init": [P(PTR), PTR, U64],
            "pmk_v4_run_job": [PTR, P(V4JobDesc), self.cb_type, PTR, P(PTR)],
            "pmk_v4_run_codes_diagnostic": [PTR, P(V4CodesDesc), self.cb_type, PTR, P(PTR)],
            "pmk_v4_job_quantized_codes": [PTR, P(V4QuantizedCodes)],
            "pmk_v4_poll": [PTR, P(V4Result)],
            "pmk_v4_job_wait_callback": [PTR],
            "pmk_v4_job_release": [PTR],
            "pmk_v4_admission_metadata": [PTR, PTR, U64],
            "pmk_v4_context_error": [PTR, PTR, U64],
            "pmk_v4_job_error": [PTR, PTR, U64],
        }
        for name, args in declarations.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = C.c_int32
        self.lib.pmk_destroy.argtypes = [PTR]
        self.lib.pmk_destroy.restype = None
        self.lib.pmk_v4_destroy.argtypes = [PTR]
        self.lib.pmk_v4_destroy.restype = None
        self.ctx = PTR()
        self.v4_ctx = PTR()
        error = C.create_string_buffer(4096)
        self.check(self.lib.pmk_init(C.byref(self.ctx), error, len(error)), error, "pmk_init")
        init_name = "pmk_v4_init_diagnostic" if diagnostic_v4_init else "pmk_v4_init"
        self.check(getattr(self.lib, init_name)(C.byref(self.v4_ctx), error, len(error)), error, init_name)
        self.buffers: list[PTR] = []

    def close(self):
        for ptr in list(self.buffers):
            self.release_buffer(ptr)
        if self.v4_ctx:
            self.lib.pmk_v4_destroy(self.v4_ctx)
            self.v4_ctx = PTR()
        if self.ctx:
            self.lib.pmk_destroy(self.ctx)
            self.ctx = PTR()

    def check(self, rc: int, error=None, name: str = "native") -> None:
        if rc == PMK_SUCCESS:
            return
        detail = ""
        if error is not None and getattr(error, "value", b""):
            detail = error.value.decode(errors="replace")
        raise RuntimeError(f"{name} failed rc={rc}: {detail}")

    def alloc_const_i8(self, count: int, value: int) -> PTR:
        ptr = PTR()
        self.check(self.lib.pmk_buffer_alloc(self.ctx, count, C.byref(ptr)), name="pmk_buffer_alloc")
        C.memset(ptr, value & 0xFF, count)
        self.buffers.append(ptr)
        return ptr

    def release_buffer(self, ptr: PTR) -> None:
        self.check(self.lib.pmk_buffer_release(self.ctx, ptr), name="pmk_buffer_release")
        self.buffers.remove(ptr)

    def callback(self):
        def mark(_job, _user):
            return None
        cb = self.cb_type(mark)
        self.callbacks.append(cb)
        return cb

    def wait_v3(self, job: PTR, timeout: float) -> V3Result:
        result = V3Result()
        deadline = time.monotonic() + timeout
        while True:
            rc = self.lib.pmk_poll(job, C.byref(result))
            if rc != PMK_PENDING:
                self.check(rc, name="pmk_poll")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("v3 job timed out")
            time.sleep(0.002)
        if result.status != PMK_SUCCESS:
            raise RuntimeError(f"v3 job status={result.status}")
        self.check(self.lib.pmk_job_wait_callback(job), name="pmk_job_wait_callback")
        return result

    def wait_v4(self, job: PTR, timeout: float) -> V4Result:
        result = V4Result()
        deadline = time.monotonic() + timeout
        while True:
            rc = self.lib.pmk_v4_poll(job, C.byref(result))
            if rc != PMK_PENDING:
                self.check(rc, name="pmk_v4_poll")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("v4 job timed out")
            time.sleep(0.002)
        if result.status != PMK_SUCCESS:
            raise RuntimeError(f"v4 job status={result.status}")
        self.check(self.lib.pmk_v4_job_wait_callback(job), name="pmk_v4_job_wait_callback")
        return result

    def release_v3_job(self, job: PTR) -> None:
        self.check(self.lib.pmk_job_release(job), name="pmk_job_release")

    def release_v4_job(self, job: PTR) -> None:
        self.check(self.lib.pmk_v4_job_release(job), name="pmk_v4_job_release")


class CoreV4:
    def __init__(self, path: Path):
        self.lib = C.CDLL(str(path))
        declarations = {
            "pmkcore_v4_init": [U32],
            "pmkcore_v4_job_create_grid_b200": [PTR, PTR, PTR, U64, U32, U32, U32, P(PTR)],
            "pmkcore_v4_gpu_descriptor": [PTR, P(Pmk4GpuJobDesc)],
            "pmkcore_v4_prepare_oracle_noised": [PTR],
            "pmkcore_v4_tile_cpu_oracle": [PTR, U32, U32, PTR, PTR, P(Pmk4TileResult)],
        }
        for name, args in declarations.items():
            if hasattr(self.lib, name):
                fn = getattr(self.lib, name)
                fn.argtypes = args
                fn.restype = C.c_int32
        self.lib.pmkcore_v4_job_free.argtypes = [PTR]
        self.lib.pmkcore_v4_job_free.restype = None
        self.lib.pmkcore_v4_strerror.argtypes = [C.c_int32]
        self.lib.pmkcore_v4_strerror.restype = C.c_char_p
        self.check(self.lib.pmkcore_v4_init(0), "pmkcore_v4_init")

    def check(self, rc: int, name: str) -> None:
        if rc == 0:
            return
        msg = self.lib.pmkcore_v4_strerror(rc)
        detail = msg.decode(errors="replace") if msg else ""
        raise RuntimeError(f"{name} failed rc={rc}: {detail}")

    def create_job(self, header: bytes, ancestor: bytes, m: int, n: int, k: int) -> PTR:
        out = PTR()
        self.check(self.lib.pmkcore_v4_job_create_grid_b200(u8_buffer(header), u8_buffer(ancestor), None, 0, m, n, k, C.byref(out)), "pmkcore_v4_job_create_grid_b200")
        return out

    def descriptor(self, job: PTR) -> Pmk4GpuJobDesc:
        desc = Pmk4GpuJobDesc()
        self.check(self.lib.pmkcore_v4_gpu_descriptor(job, C.byref(desc)), "pmkcore_v4_gpu_descriptor")
        return desc

    def spot_check_tile(self, job: PTR, t_rows: int, t_cols: int, share_bound: int, block_bound: int) -> dict[str, Any]:
        out = Pmk4TileResult()
        share = u8_buffer(int(share_bound).to_bytes(32, "little"))
        block = u8_buffer(int(block_bound).to_bytes(32, "little"))
        self.check(
            self.lib.pmkcore_v4_tile_cpu_oracle(job, t_rows, t_cols, share, block, C.byref(out)),
            "pmkcore_v4_tile_cpu_oracle",
        )
        return {
            "event": "v4_cpu_oracle_spot_check",
            "t_rows": int(out.t_rows),
            "t_cols": int(out.t_cols),
            "policy_pass": bool(out.policy_pass),
            "is_share": bool(out.is_share),
            "is_block": bool(out.is_block),
            "message_prefix_hex": bytes(out.message[:8]).hex(),
            "hash_prefix_hex": bytes(out.hash[:8]).hex(),
        }

    def free_job(self, job: PTR) -> None:
        self.lib.pmkcore_v4_job_free(job)


def v4_operand(ptr: Any, rows: int, k: int, r: int, noise_e, noise_f, alpha, beta) -> V4OperandDesc:
    return V4OperandDesc(ptr, rows * k, noise_e, rows * r, noise_f, k * r, alpha, beta, rows)


def make_v4_job_desc(core_desc: Pmk4GpuJobDesc, block_bound: int, share_bound: int, job_id: int) -> V4JobDesc:
    return V4JobDesc(
        abi_version=1, m=core_desc.m, n=core_desc.n, k=core_desc.k, r=core_desc.rank,
        a=v4_operand(core_desc.a_values, core_desc.m, core_desc.k, core_desc.rank, core_desc.a_noise_e, core_desc.a_noise_f, core_desc.a_alpha, core_desc.a_beta),
        bt=v4_operand(core_desc.bt_values, core_desc.n, core_desc.k, core_desc.rank, core_desc.bt_noise_e, core_desc.bt_noise_f, core_desc.bt_alpha, core_desc.bt_beta),
        jackpot_key=words(bytes(core_desc.jackpot_key)),
        block_bound=words(block_bound), share_bound=words(share_bound),
        block_capacity=64, share_capacity=64, job_id=job_id,
    )


def extract_v4_codes(lib: LibPMK, v4_desc: V4JobDesc, timeout: float) -> tuple[bytes, bytes, dict[str, Any]]:
    job = PTR()
    lib.check(lib.lib.pmk_v4_run_job(lib.v4_ctx, C.byref(v4_desc), lib.callback(), None, C.byref(job)), name="pmk_v4_run_job")
    try:
        result = lib.wait_v4(job, timeout)
        q = V4QuantizedCodes()
        lib.check(lib.lib.pmk_v4_job_quantized_codes(job, C.byref(q)), name="pmk_v4_job_quantized_codes")
        if q.abi_version != 1 or q.a_code_count != v4_desc.m * v4_desc.k or q.bt_code_count != v4_desc.n * v4_desc.k:
            raise RuntimeError("unexpected V4 quantized code descriptor")
        a_codes = C.string_at(q.a_codes, q.a_code_count)
        bt_codes = C.string_at(q.bt_codes, q.bt_code_count)
        stats = stats_dict(result.stats)
        stats["production_e_gpu_seconds_for_codegen"] = seconds(result)
        return a_codes, bt_codes, stats
    finally:
        lib.release_v4_job(job)


def run_v3_sample(lib: LibPMK, a_ptr: PTR, bt_ptr: PTR, args: argparse.Namespace, job_id: int) -> dict[str, Any]:
    desc = V3Desc()
    desc.abi_version = 1
    desc.m, desc.n, desc.k = args.m, args.n, args.k
    desc.a, desc.bt = a_ptr, bt_ptr
    desc.a_bytes, desc.bt_bytes = args.m * args.k, args.n * args.k
    desc.a_seed = words(bytes.fromhex("03" * 32))
    desc.b_seed = words(bytes.fromhex("04" * 32))
    desc.block_bound = words(0)
    desc.share_bound = words(0)
    desc.block_capacity = 64
    desc.share_capacity = 64
    desc.cert_version = 3
    desc.rank = 128
    desc.job_id = job_id
    job = PTR()
    lib.check(lib.lib.pmk_run_job(lib.ctx, C.byref(desc), lib.callback(), None, C.byref(job)), name="pmk_run_job")
    try:
        result = lib.wait_v3(job, args.job_timeout)
        return {"family": "v3_k3_sg", "job_id": job_id, "gpu_seconds": seconds(result), "lottery_included": True, "fallback_groups": None, "total_groups": None}
    finally:
        lib.release_v3_job(job)


def run_v4_codes_sample(lib: LibPMK, a_codes: bytes, bt_codes: bytes, args: argparse.Namespace, job_id: int) -> dict[str, Any]:
    a_buf = (U8 * len(a_codes)).from_buffer_copy(a_codes)
    bt_buf = (U8 * len(bt_codes)).from_buffer_copy(bt_codes)
    desc = V4CodesDesc()
    desc.abi_version = 1
    desc.m, desc.n, desc.k = args.m, args.n, args.k
    desc.a_codes = C.cast(a_buf, P(U8))
    desc.bt_codes = C.cast(bt_buf, P(U8))
    desc.a_code_count = len(a_codes)
    desc.bt_code_count = len(bt_codes)
    desc.jackpot_key = words(bytes.fromhex("05" * 32))
    desc.block_bound = words(0)
    desc.share_bound = words(0)
    desc.block_capacity = 64
    desc.share_capacity = 64
    desc.job_id = job_id
    job = PTR()
    lib.check(lib.lib.pmk_v4_run_codes_diagnostic(lib.v4_ctx, C.byref(desc), lib.callback(), None, C.byref(job)), name="pmk_v4_run_codes_diagnostic")
    try:
        result = lib.wait_v4(job, args.job_timeout)
        stats = stats_dict(result.stats)
        return {"family": "v4_codes_k3_only", "job_id": job_id, "gpu_seconds": seconds(result), "lottery_included": False, **stats}
    finally:
        lib.release_v4_job(job)


def stats_dict(stats: V4Stats) -> dict[str, int | float]:
    row = {name: int(getattr(stats, name)) for name, _ in V4Stats._fields_}
    row["fallback_per_job"] = row["fallback_groups"]
    row["fallback_percent"] = 100.0 * row["fallback_groups"] / max(1, row["total_groups"])
    return row


def interleaved_order(warmup: int, measured: int) -> Iterable[tuple[str, bool]]:
    for _ in range(warmup):
        yield "v3", True
        yield "v4", True
    for _ in range(measured):
        yield "v3", False
        yield "v4", False


def run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PMK_GPU_LOCK_HELD") != "1":
        raise SystemExit("Refusing GPU benchmark without PMK_GPU_LOCK_HELD=1 from coordinator/root")
    if args.m != 8192 or args.n != 8192 or args.k != 4096:
        raise SystemExit("This paired benchmark is scoped to 8192^2 x 4096")
    minimum = (1, 2) if args.quick else (3, 10)
    if args.warmup < minimum[0] or args.measured < minimum[1]:
        raise SystemExit(f"Need at least {minimum[0]} warmup and {minimum[1]} measured iterations")
    # Use a header/ancestor pair generated and authenticated by Pearl, rather
    # than inventing unrelated bytes that fail the PR #355 ancestry check.
    fixture = json.loads(DEFAULT_PROBE.read_text())
    header = bytes.fromhex(args.header_hex or fixture["proposed_header"])
    ancestor = bytes.fromhex(args.ancestor_header_hex or fixture["ancestor_header"])
    if len(header) != 76 or len(ancestor) != 108:
        raise SystemExit("header must be 76 bytes and ancestor header must be 108 bytes")
    rows: list[dict[str, Any]] = []
    lib = LibPMK(args.libpmk, diagnostic_v4_init=args.diagnostic_v4_init)
    core = CoreV4(args.pmkcore_v4)
    core_job = PTR()
    try:
        metadata_buffer = C.create_string_buffer(16384)
        lib.check(lib.lib.pmk_v4_admission_metadata(lib.v4_ctx, metadata_buffer, len(metadata_buffer)), name="pmk_v4_admission_metadata")
        metadata = json.loads(metadata_buffer.value)
        if metadata["library_sha256"] != hashlib.sha256(args.libpmk.read_bytes()).hexdigest():
            raise RuntimeError("benchmark library identity differs from the loaded image")
        rows.append({"event": "environment", "native": metadata,
                     "core_sha256": hashlib.sha256(args.pmkcore_v4.read_bytes()).hexdigest(),
                     "measured_unix": time.time()})
        a_ptr = lib.alloc_const_i8(args.m * args.k, args.v3_const)
        bt_ptr = lib.alloc_const_i8(args.n * args.k, args.v3_const)
        core_job = core.create_job(header, ancestor, args.m, args.n, args.k)
        core_desc = core.descriptor(core_job)
        v4_desc = make_v4_job_desc(core_desc, block_bound=0, share_bound=0, job_id=1)
        a_codes, bt_codes, codegen_stats = extract_v4_codes(lib, v4_desc, args.job_timeout)
        rows.append({"event": "v4_code_generation", **codegen_stats})
        if args.v4_oracle_spot_check:
            rows.append(core.spot_check_tile(core_job, 0, 0, share_bound=0, block_bound=0))
        job_id = 10
        measured_rows: list[dict[str, Any]] = []
        for family, warm in interleaved_order(args.warmup, args.measured):
            job_id += 1
            if family == "v3":
                row = run_v3_sample(lib, a_ptr, bt_ptr, args, job_id)
            else:
                row = run_v4_codes_sample(lib, a_codes, bt_codes, args, job_id)
            row["warmup"] = warm
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
            if not warm:
                measured_rows.append(row)
        v3 = [row for row in measured_rows if row["family"] == "v3_k3_sg"]
        v4 = [row for row in measured_rows if row["family"] == "v4_codes_k3_only"]
        result = {
            "schema": "pmk-v4-paired-bench-v1",
            "shape": {"m": args.m, "n": args.n, "k": args.k},
            "warmup_per_family": args.warmup,
            "measured_per_family": args.measured,
            "interleaved": True,
            "native": metadata,
            "v3_label": "v3 K3-SG via pmk_run_job; jackpot compare/slot atomics included; no CPU proof",
            "v4_label": "v4 codes diagnostic K3-only using codes extracted from fresh production kernel-E job",
            "v4_oracle_spot_check": bool(args.v4_oracle_spot_check),
            "v3": summary(v3, args.m, args.n, args.k),
            "v4": summary(v4, args.m, args.n, args.k),
            "v4_fallback_per_job": [int(row["fallback_per_job"]) for row in v4],
            "v4_fallback_percent_median": statistics.median(float(row["fallback_percent"]) for row in v4),
        }
        result["v4_to_v3_tops_ratio"] = result["v4"]["tops_eq_median"] / result["v3"]["tops_eq_median"]
        rows.append({"event": "summary", **result})
        output = args.output or DEFAULT_OUTPUT
        write_jsonl(output, rows)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    finally:
        if core_job:
            core.free_job(core_job)
        lib.close()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    plan = {
        "schema": "pmk-v4-paired-bench-plan-v1",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "warmup_per_family": args.warmup,
        "measured_per_family": args.measured,
        "order": list(interleaved_order(args.warmup, args.measured)),
        "requires_gpu_lock_env": "PMK_GPU_LOCK_HELD=1",
        "libpmk": str(args.libpmk),
        "pmkcore_v4": str(args.pmkcore_v4),
        "v3_label": "v3 K3-SG via pmk_run_job; jackpot compare/slot atomics included; no CPU proof",
        "v4_label": "v4 codes diagnostic K3-only using codes extracted from fresh production kernel-E job",
        "v4_oracle_spot_check": bool(args.v4_oracle_spot_check),
        "recommended_result_output": str(DEFAULT_OUTPUT),
        "current_output_arg": str(args.output) if args.output else None,
        "run_command_after_gpu_lock": [
            "PMK_GPU_LOCK_HELD=1",
            str(Path(__file__).resolve()),
            "--run-gpu",
            "--output",
            str(DEFAULT_OUTPUT),
        ],
        "summary_fields": [
            "v3.seconds_median", "v3.seconds_p25", "v3.seconds_p75", "v3.tops_eq_median",
            "v4.seconds_median", "v4.seconds_p25", "v4.seconds_p75", "v4.tops_eq_median",
            "v4_to_v3_tops_ratio", "v4_fallback_per_job", "v4_fallback_percent_median",
        ],
    }
    if args.output:
        write_jsonl(args.output, [{"event": "plan", **plan}])
    print(json.dumps(plan, sort_keys=True))
    return plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measured", type=int, default=10)
    p.add_argument("--job-timeout", type=float, default=120.0)
    p.add_argument("--libpmk", type=Path, default=ROOT / "libpmk/.build/release/libpmk.dylib")
    p.add_argument("--pmkcore-v4", type=Path, default=ROOT / "pmkcore/v4/target/release/libpmkcore_v4.dylib")
    p.add_argument("--output", type=Path, default=None, help="JSONL output path; GPU runs default to bench/evidence/b9_v5_paired_bench.txt")
    p.add_argument("--header-hex", default="")
    p.add_argument("--ancestor-header-hex", default="")
    p.add_argument("--v3-const", type=int, default=64, help="constant int8 operand value for v3 inputs")
    p.add_argument("--diagnostic-v4-init", action="store_true", help="diagnostic only; production runs should use pmk_v4_init")
    p.add_argument("--no-v4-oracle-spot-check", dest="v4_oracle_spot_check", action="store_false", help="skip the bounded pmkcore CPU oracle tile check outside timed samples")
    p.set_defaults(v4_oracle_spot_check=True)
    p.add_argument("--run-gpu", action="store_true", help="actually run GPU jobs; requires PMK_GPU_LOCK_HELD=1")
    p.add_argument("--quick", action="store_true", help="keep the production shape but allow 1 warmup and 2 measured samples per family")
    p.add_argument("--dry-run", action="store_true", help="print and optionally write the planned run without loading dylibs")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.m != 8192 or args.n != 8192 or args.k != 4096:
        raise SystemExit("B9 paired benchmark is scoped to --m 8192 --n 8192 --k 4096")
    minimum = (1, 2) if args.quick else (3, 10)
    if args.warmup < minimum[0] or args.measured < minimum[1]:
        raise SystemExit(f"Use at least --warmup {minimum[0]} and --measured {minimum[1]}")
    if not -64 <= args.v3_const <= 64:
        raise SystemExit("--v3-const must be in the valid v3 signal range [-64,64]")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.dry_run or not args.run_gpu:
        dry_run(args)
        return 0
    run_gpu(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
