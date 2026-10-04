#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Run K3-V4 against checksummed Pearl CPU references, then admit this device.

The manifest records the pinned source, the V1 oracle executable and unique
input/reference hashes. Admission is written only after the complete matrix
and adversarial cases pass. Small manifests remain useful diagnostic runs.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

from pmk_v4_paths import bundle_root, verify_pearl_pin

ROOT = bundle_root(__file__)
PIN = "f696760b259500ecb608469ea3953aeabbe78948"
U32, U64, PTR = C.c_uint32, C.c_uint64, C.c_void_p


class Slot(C.Structure):
    _fields_ = [("row", U32), ("col", U32), ("message", U32 * 16), ("hash", U32 * 8)]


class Codes(C.Structure):
    _fields_ = [(x, U32) for x in ("abi_version", "m", "n", "k")] + [
        ("a_codes", PTR), ("bt_codes", PTR),
        ("a_code_count", U64), ("bt_code_count", U64),
    ] + [(x, U32 * 8) for x in ("jackpot_key", "block_bound", "share_bound")] + [
        ("block_capacity", U32), ("share_capacity", U32), ("job_id", U64),
    ]


class Stats(C.Structure):
    _fields_ = [("abi_version", U32), ("flags", U32)] + [
        (x, U64) for x in (
            "fallback_groups", "total_groups", "quantized_a", "quantized_b",
            "quant_saturated_a", "quant_saturated_b", "quant_nan_a", "quant_nan_b",
        )
    ] + [("layout_failures", U32), ("fallback_alert", U32)]


class Result(C.Structure):
    _fields_ = [("abi_version", U32), ("status", C.c_int32), ("job_id", U64)] + [
        (x, U32) for x in (
            "block_count", "share_count", "block_stored", "share_stored", "overflow", "recovered",
        )
    ] + [
        ("blocks", C.POINTER(Slot)), ("shares", C.POINTER(Slot)), ("stats", Stats),
        ("c_bits", PTR), ("c_count", U64),
        ("gpu_start_time", C.c_double), ("gpu_end_time", C.c_double),
    ]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def integration_passed(path: Path | None, metadata: dict, library: Path) -> bool:
    if path is None:
        return False
    record = json.loads(path.read_text())
    if not (
        record.get("schema") == "pmk-v4-integration-v1"
        and record.get("upstream_pin") == PIN
        and record.get("passed") is True
        and record.get("mismatches") == 0
        and record.get("cache_key") == metadata.get("cache_key")
        and record.get("library_sha256") == metadata.get("library_sha256")
    ):
        raise ValueError("fused integration evidence has a wrong schema, source, kernel, or result")
    for field in ("quantized_cells_checked", "fold_tiles_checked", "proofs_accepted", "corrupt_proofs_rejected"):
        count = record.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"fused integration evidence is missing {field}")
    for field, expected in (("library", library), ("core_library", None)):
        binary = Path(record[field])
        if not binary.is_absolute():
            binary = ROOT / binary
        if expected is not None and binary.resolve() != expected.resolve():
            raise ValueError("integration and G3 libpmk paths differ")
        hash_field = "core_sha256" if field == "core_library" else "library_sha256"
        if digest(binary.read_bytes()) != record.get(hash_field):
            raise ValueError(f"{field} changed after fused integration checks")
    cases = {(int(c["m"]), int(c["n"]), int(c["k"])) for c in record["cases"]}
    if not ({1024, 4096, 16384} <= {k for _, _, k in cases}
            and any(m < n for m, n, _ in cases)
            and any(m > n for m, n, _ in cases)):
        raise ValueError("fused integration does not cover supported k and both rectangular orientations")
    return True


@contextlib.contextmanager
def gpu_lock():
    path = Path("/tmp/pmm-gpu-bench.lock")
    owned = os.environ.get("PMK_GPU_LOCK_HELD") != "1"
    if owned:
        while True:
            try:
                path.mkdir()
                break
            except FileExistsError:
                print("GPU lock occupied; retrying in 15 seconds.", flush=True)
                time.sleep(15)
    try:
        yield
    finally:
        if owned:
            path.rmdir()


class Metal:
    def __init__(self, path: Path):
        self.lib = C.CDLL(str(path))
        definitions = {
            "pmk_v4_init_diagnostic": [C.POINTER(PTR), PTR, U64],
            "pmk_v4_admission_metadata": [PTR, PTR, U64],
            "pmk_v4_run_codes_diagnostic": [PTR, C.POINTER(Codes), PTR, PTR, C.POINTER(PTR)],
            "pmk_v4_poll": [PTR, C.POINTER(Result)],
            "pmk_v4_job_release": [PTR],
            "pmk_v4_job_wait_callback": [PTR],
            "pmk_v4_write_admission_record": [PTR, C.c_char_p, U64, PTR, U64],
        }
        for name, args in definitions.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int32
        self.lib.pmk_v4_destroy.argtypes = [PTR]
        self.lib.pmk_v4_destroy.restype = None
        self.context = PTR()
        error = C.create_string_buffer(4096)
        self.check(self.lib.pmk_v4_init_diagnostic(C.byref(self.context), error, len(error)), error)

    @staticmethod
    def check(code: int, error=None):
        if code != 0:
            detail = error.value.decode(errors="replace") if error is not None else ""
            raise RuntimeError(f"libpmk v4 error {code}: {detail}")

    def metadata(self):
        out = C.create_string_buffer(16384)
        self.check(self.lib.pmk_v4_admission_metadata(self.context, out, len(out)))
        return json.loads(out.value)

    def close(self):
        if self.context:
            self.lib.pmk_v4_destroy(self.context)
            self.context = PTR()

    def verify(self, case: dict, number: int):
        m, n, k = (int(case[x]) for x in ("m", "n", "k"))
        directory = Path(case["directory"])
        if not directory.is_absolute():
            directory = ROOT / directory
        payloads = []
        for name, size in (("a", m * k), ("b", n * k), ("c_b200", m * n * 4)):
            data = (directory / f"{name}.bin").read_bytes()
            if len(data) != size or digest(data) != case[f"{name}_sha256"]:
                raise ValueError(f"reference provenance/size mismatch: {directory}/{name}.bin")
            payloads.append(data)
        a, b, expected = payloads
        a_buffer, b_buffer = C.create_string_buffer(a), C.create_string_buffer(b)
        desc = Codes()
        desc.abi_version, desc.m, desc.n, desc.k = 1, m, n, k
        desc.a_codes, desc.bt_codes = C.cast(a_buffer, PTR), C.cast(b_buffer, PTR)
        desc.a_code_count, desc.bt_code_count = len(a), len(b)
        desc.block_capacity, desc.share_capacity = 4, 64
        desc.job_id = number
        job = PTR()
        self.check(self.lib.pmk_v4_run_codes_diagnostic(self.context, C.byref(desc), None, None, C.byref(job)))
        try:
            result = Result()
            deadline = time.monotonic() + 1800
            while True:
                status = self.lib.pmk_v4_poll(job, C.byref(result))
                if status != 1:
                    self.check(status)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("K3-V4 diagnostic job exceeded 30 minutes")
                time.sleep(0.01)
            self.check(result.status)
            if result.c_count != m * n or not result.c_bits:
                raise ValueError("diagnostic result does not contain the complete C matrix")
            actual = C.string_at(result.c_bits, m * n * 4)
            mismatches = 0
            if actual != expected:
                got, want = array.array("I"), array.array("I")
                got.frombytes(actual)
                want.frombytes(expected)
                mismatches = sum(x != y for x, y in zip(got, want))
            stats = {name: int(getattr(result.stats, name)) for name, _ in Stats._fields_}
            record = {
                "case": directory.name, "family": case["family"], "m": m, "n": n, "k": k,
                "cells": m * n, "mismatches": mismatches, **stats,
                "gpu_seconds": result.gpu_end_time - result.gpu_start_time,
                "fallback_percent": 100 * stats["fallback_groups"] / max(1, stats["total_groups"]),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if mismatches or stats["layout_failures"] or result.overflow:
                raise ValueError("G3-v4 failed: mismatch, layout error, or overflow")
            if stats["total_groups"] != m * n * (k // 32):
                raise ValueError("fallback denominator does not cover every cell/group")
            if case["family"].startswith("adv_") and stats["fallback_groups"] == 0:
                raise ValueError("adversarial fixture did not exercise the exact fallback")
            return record
        finally:
            self.lib.pmk_v4_job_wait_callback(job)
            self.lib.pmk_v4_job_release(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--library", type=Path, default=ROOT / "libpmk/.build/release/libpmk.dylib")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--integration", type=Path, help="checksummed fused quantization/fold/proof evidence")
    parser.add_argument("--admission-requires-integration", action="store_true", help="also require --integration before writing admission")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_pearl_pin(ROOT, PIN)
    pin = PIN
    if manifest["upstream_pin"] != PIN:
        raise ValueError("G3 source pin differs from B9")
    oracle = Path(manifest["oracle_executable"])
    if not oracle.is_absolute():
        oracle = ROOT / oracle
    if digest(oracle.read_bytes()) != manifest["oracle_sha256"]:
        raise ValueError("V1 oracle executable changed since reference generation")
    identities = set()
    for case in manifest["cases"]:
        identity = (case["a_sha256"], case["b_sha256"], case["m"], case["n"], case["k"])
        if identity in identities:
            raise ValueError("duplicate inputs cannot inflate the G3 cell count")
        identities.add(identity)
    with gpu_lock():
        metal = Metal(args.library)
        try:
            metadata = metal.metadata()
            library_sha = digest(args.library.read_bytes())
            if metadata.get("library_sha256") != library_sha:
                raise ValueError("loaded native library identity differs from the G3 binary")
            print(json.dumps({"device": metadata, "upstream_pin": pin, "manifest_sha256": digest(args.manifest.read_bytes())}), flush=True)
            results = [metal.verify(case, i + 1) for i, case in enumerate(manifest["cases"])]
            const = [r for r in results if r["family"] == "const"]
            shapes = {(r["m"], r["n"], r["k"]) for r in const}
            cells = sum(r["cells"] for r in const)
            full_matrix = (
                cells >= 100_000_000
                and {(256, 256, 4096), (2048, 2048, 4096), (4096, 4096, 4096)} <= shapes
                and {1024, 16384} <= {r["k"] for r in const}
                and {"adv_uniform", "adv_edge"} <= {r["family"] for r in results}
            )
            # Fused quantization and fold/hash/proof checks are separately required
            # and tied to this exact host binary and compiled Metal key.
            integration_ok = integration_passed(args.integration, metadata, args.library) if args.integration else None
            full = full_matrix and (integration_ok is not False)
            print(json.dumps({"exact_const_cells": cells, "exact_total_cells": sum(r["cells"] for r in results), "g3_v4_complete": full, "g3_v4_matrix_complete": full_matrix, "integration_complete": integration_ok}), flush=True)
            if args.admission:
                if not full_matrix or (args.admission_requires_integration and not integration_ok):
                    raise ValueError("incomplete G3 matrix or fused quantization/fold integration evidence; admission refused")
                if digest(args.library.read_bytes()) != library_sha:
                    raise ValueError("native library changed during G3; admission refused")
                error = C.create_string_buffer(4096)
                metal.check(metal.lib.pmk_v4_write_admission_record(metal.context, os.fsencode(args.admission), cells, error, len(error)), error)
                admission = json.loads(args.admission.read_text())
                records = admission.get("devices", [admission])
                if not any(isinstance(record, dict)
                           and record.get("library_sha256") == library_sha
                           and record.get("cache_key") == metadata["cache_key"]
                           for record in records):
                    args.admission.unlink()
                    raise ValueError("native admission omitted the tested binary identity")
                print(f"G3-v4 admission written: {args.admission}", flush=True)
        finally:
            metal.close()


if __name__ == "__main__":
    main()
