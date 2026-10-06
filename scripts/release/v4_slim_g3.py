#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run v4 G3 admission from a slim manifest without shipping reference C blobs."""
from __future__ import annotations

import argparse
import array
import contextlib
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

PIN = "f696760b259500ecb608469ea3953aeabbe78948"
U32, U64, PTR = C.c_uint32, C.c_uint64, C.c_void_p


class Codes(C.Structure):
    _fields_ = [(x, U32) for x in ("abi_version", "m", "n", "k")] + [
        ("a_codes", PTR),
        ("bt_codes", PTR),
        ("a_code_count", U64),
        ("bt_code_count", U64),
    ] + [(x, U32 * 8) for x in ("jackpot_key", "block_bound", "share_bound")] + [
        ("block_capacity", U32),
        ("share_capacity", U32),
        ("job_id", U64),
    ]


class Slot(C.Structure):
    _fields_ = [("row", U32), ("col", U32), ("message", U32 * 16), ("hash", U32 * 8)]


class Stats(C.Structure):
    _fields_ = [("abi_version", U32), ("flags", U32)] + [
        (
            x,
            U64,
        )
        for x in (
            "fallback_groups",
            "total_groups",
            "quantized_a",
            "quantized_b",
            "quant_saturated_a",
            "quant_saturated_b",
            "quant_nan_a",
            "quant_nan_b",
        )
    ] + [("layout_failures", U32), ("fallback_alert", U32)]


class Result(C.Structure):
    _fields_ = [("abi_version", U32), ("status", C.c_int32), ("job_id", U64)] + [
        (
            x,
            U32,
        )
        for x in (
            "block_count",
            "share_count",
            "block_stored",
            "share_stored",
            "overflow",
            "recovered",
        )
    ] + [
        ("blocks", C.POINTER(Slot)),
        ("shares", C.POINTER(Slot)),
        ("stats", Stats),
        ("c_bits", PTR),
        ("c_count", U64),
        ("gpu_start_time", C.c_double),
        ("gpu_end_time", C.c_double),
    ]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
                print("Waiting for another GPU task to finish...", flush=True)
                time.sleep(15)
    try:
        yield
    finally:
        if owned:
            path.rmdir()


def adversarial(directory: Path, family: str, m: int, n: int, k: int, seed: int) -> None:
    for side, rows in (("a", m), ("b", n)):
        rng = random.Random(f"B9/{family}/{seed}/{side}")
        data = bytearray(rows * k)
        for base in range(0, len(data), 32):
            mode = rng.randrange(5) if family == "adv_edge" else 0
            for offset in range(32):
                if mode == 1:
                    choices = (0x00, 0x80, 0x00, 0x80, 0x01, 0x81)
                elif mode == 2:
                    choices = (0x7E, 0xFE, 0x01, 0x81, 0x08, 0x88)
                elif mode == 3:
                    choices = (0x01, 0x02, 0x07, 0x81, 0x82, 0x87)
                elif mode == 4:
                    choices = (0x78, 0xF8, 0x7E, 0xFE)
                else:
                    choices = None
                if choices:
                    code = rng.choice(choices)
                else:
                    code = rng.randrange(256)
                    while code & 0x7F == 0x7F:
                        code = rng.randrange(256)
                data[base + offset] = code
        (directory / f"{side}.bin").write_bytes(data)


def materialize_inputs(case: dict, oracle: Path, directory: Path) -> tuple[bytes, bytes]:
    m, n, k, seed = (int(case[x]) for x in ("m", "n", "k", "seed"))
    family = str(case["family"])
    directory.mkdir(parents=True, exist_ok=True)
    if family == "const":
        seed_hex = hashlib.sha256(f"B9/{seed}".encode()).hexdigest()
        subprocess.run(
            [str(oracle), "gen", str(m), str(n), str(k), seed_hex, str(directory)],
            check=True,
        )
    elif family in {"adv_uniform", "adv_edge"}:
        adversarial(directory, family, m, n, k, seed)
    else:
        raise ValueError(f"unsupported v4 G3 family: {family}")
    a = (directory / "a.bin").read_bytes()
    b = (directory / "b.bin").read_bytes()
    if len(a) != m * k or digest(a) != case["a_sha256"]:
        raise ValueError(f"A input provenance mismatch: {directory.name}")
    if len(b) != n * k or digest(b) != case["b_sha256"]:
        raise ValueError(f"B input provenance mismatch: {directory.name}")
    return a, b


class Metal:
    def __init__(self, library: Path):
        self.lib = C.CDLL(str(library))
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
            fn.argtypes = args
            fn.restype = C.c_int32
        self.lib.pmk_v4_destroy.argtypes = [PTR]
        self.lib.pmk_v4_destroy.restype = None
        self.context = PTR()
        error = C.create_string_buffer(4096)
        self.check(self.lib.pmk_v4_init_diagnostic(C.byref(self.context), error, len(error)), error)

    @staticmethod
    def check(code: int, error=None) -> None:
        if code != 0:
            detail = error.value.decode(errors="replace") if error is not None else ""
            raise RuntimeError(f"libpmk v4 error {code}: {detail}")

    def metadata(self) -> dict:
        out = C.create_string_buffer(16384)
        self.check(self.lib.pmk_v4_admission_metadata(self.context, out, len(out)))
        return json.loads(out.value)

    def verify(self, case: dict, number: int, oracle: Path, work: Path) -> dict:
        m, n, k = (int(case[x]) for x in ("m", "n", "k"))
        a, b = materialize_inputs(case, oracle, work / str(case["name"]))
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
                raise ValueError("diagnostic result does not contain complete C")
            actual = C.string_at(result.c_bits, m * n * 4)
            actual_sha = digest(actual)
            expected_sha = str(case["c_b200_sha256"])
            if actual_sha != expected_sha:
                got, want = array.array("I"), None
                got.frombytes(actual)
                raise ValueError(
                    f"G3-v4 failed for {case['name']}: C sha256 {actual_sha} != {expected_sha}; "
                    f"first_word={got[0] if got else 'none'}"
                )
            stats = {name: int(getattr(result.stats, name)) for name, _ in Stats._fields_}
            record = {
                "case": case["name"],
                "family": case["family"],
                "m": m,
                "n": n,
                "k": k,
                "cells": m * n,
                "c_b200_sha256": actual_sha,
                **stats,
                "gpu_seconds": result.gpu_end_time - result.gpu_start_time,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if stats["layout_failures"] or result.overflow:
                raise ValueError("G3-v4 failed: layout error or overflow")
            if stats["total_groups"] != m * n * (k // 32):
                raise ValueError("fallback denominator does not cover every cell/group")
            if str(case["family"]).startswith("adv_") and stats["fallback_groups"] == 0:
                raise ValueError("adversarial fixture did not exercise exact fallback")
            return record
        finally:
            self.lib.pmk_v4_job_wait_callback(job)
            self.lib.pmk_v4_job_release(job)

    def write_admission(self, path: Path, cells: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        error = C.create_string_buffer(4096)
        self.check(
            self.lib.pmk_v4_write_admission_record(
                self.context, os.fsencode(path), cells, error, len(error)
            ),
            error,
        )

    def close(self) -> None:
        if self.context:
            self.lib.pmk_v4_destroy(self.context)
            self.context = PTR()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("upstream_pin") != PIN:
        raise ValueError("G3 source pin differs from B9")
    if digest_path(args.oracle) != manifest.get("oracle_sha256"):
        raise ValueError("oracle executable changed since v4 G3 manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty v4 G3 manifest")
    with gpu_lock(), tempfile.TemporaryDirectory(prefix="pmk-v4-g3-") as tmp:
        metal = Metal(args.library)
        try:
            metadata = metal.metadata()
            library_sha = digest_path(args.library)
            if metadata.get("library_sha256") != library_sha:
                raise ValueError("loaded native library identity differs from G3 binary")
            print(json.dumps({"device": metadata, "upstream_pin": PIN, "manifest_sha256": digest_path(args.manifest)}), flush=True)
            results = [metal.verify(case, index + 1, args.oracle, Path(tmp)) for index, case in enumerate(cases)]
            const = [record for record in results if record["family"] == "const"]
            shapes = {(record["m"], record["n"], record["k"]) for record in const}
            cells = sum(record["cells"] for record in const)
            full_matrix = (
                cells >= 100_000_000
                and {(256, 256, 4096), (2048, 2048, 4096), (4096, 4096, 4096)} <= shapes
                and {1024, 16384} <= {record["k"] for record in const}
                and {"adv_uniform", "adv_edge"} <= {record["family"] for record in results}
            )
            if not full_matrix:
                raise ValueError("incomplete slim G3-v4 matrix; admission refused")
            if args.admission:
                metal.write_admission(args.admission, cells)
                admission = json.loads(args.admission.read_text(encoding="utf-8"))
                if not any(record.get("library_sha256") == library_sha for record in admission.get("devices", [admission])):
                    args.admission.unlink(missing_ok=True)
                    raise ValueError("native admission omitted tested binary identity")
                print(f"G3-v4 admission written: {args.admission}", flush=True)
        finally:
            metal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
