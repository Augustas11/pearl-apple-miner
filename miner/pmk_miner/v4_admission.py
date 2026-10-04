# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""V4 G3 admission-file guard shared by miner parsers and pipeline."""
from __future__ import annotations

import json
from contextlib import contextmanager
import re
import math
import os
import time
from pathlib import Path
from typing import Any

V4_G3_ADMISSION_ENV = "PMK_V4_G3_ADMISSION_FILE"
V4_G3_SCHEMA = "pmk-v4-admission-v1"
V4_VENDOR_PEARL_FP8 = "f696760b259500ecb608469ea3953aeabbe78948"
V4_EXACT_CELLS_GATE = 100_000_000
V4_ADMISSION_VALID_SECONDS = 7 * 24 * 3600
_LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class V4AdmissionError(ValueError):
    """A local admission failure, distinct from a malformed gateway job."""


@contextmanager
def configured_v4_admission(config: dict[str, Any], config_path: Path):
    """Use an explicit environment override or the miner's [v4] admission_file.

    Export the same path for Python validation and native Metal initialization.
    Never discover or synthesize an admission record implicitly.
    """
    previous = os.environ.get(V4_G3_ADMISSION_ENV)
    if previous is not None:
        yield
        return
    section = config.get("v4", {})
    if not isinstance(section, dict):
        raise V4AdmissionError("v4 configuration must be a table")
    raw_path = section.get("admission_file")
    if raw_path is None:
        yield
        return
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise V4AdmissionError("v4.admission_file must be a nonempty path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    os.environ[V4_G3_ADMISSION_ENV] = str(path.resolve())
    try:
        yield
    finally:
        os.environ.pop(V4_G3_ADMISSION_ENV, None)


def _finite_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V4AdmissionError(f"v4 G3 admission malformed {field}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise V4AdmissionError(f"v4 G3 admission malformed {field}")
    return number


def _nonempty_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise V4AdmissionError(f"v4 G3 admission missing {field}")
    return value


def _validate_record(record: Any, now: float) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise V4AdmissionError("v4 G3 admission malformed record")
    if record.get("schema") != V4_G3_SCHEMA:
        raise V4AdmissionError("v4 G3 admission schema mismatch")
    if record.get("v4_g3_passed") is not True:
        raise V4AdmissionError("v4 G3 admission has no passing G3 record")
    for field in ("gpu_name", "device_class", "os_build", "cache_key"):
        _nonempty_string(record, field)
    library_sha256 = _nonempty_string(record, "library_sha256")
    if not _LOWER_SHA256_RE.fullmatch(library_sha256):
        raise V4AdmissionError("v4 G3 admission malformed library_sha256")
    if record.get("kernel") != "E":
        raise V4AdmissionError("v4 G3 admission kernel mismatch")
    if record.get("metal_language") != "3.1":
        raise V4AdmissionError("v4 G3 admission Metal language mismatch")
    if record.get("vendor_pearl_fp8") != V4_VENDOR_PEARL_FP8:
        raise V4AdmissionError("v4 G3 admission vendor pin mismatch")
    exact = record.get("exact_cells")
    if isinstance(exact, bool) or not isinstance(exact, int) or exact < V4_EXACT_CELLS_GATE:
        raise V4AdmissionError("v4 G3 admission exact_cells below G3 gate")
    last_probe = _finite_positive_number(record.get("last_probe_unix"), "last_probe_unix")
    valid_until = _finite_positive_number(record.get("valid_until_unix"), "valid_until_unix")
    if last_probe > now or valid_until < now or valid_until - last_probe > V4_ADMISSION_VALID_SECONDS + 60:
        raise V4AdmissionError("v4 G3 admission stale or mismatched")
    return record


def validate_v4_g3_admission_file(path: str | os.PathLike[str] | None = None, *, now: float | None = None) -> dict[str, Any]:
    raw_path = path if path is not None else os.environ.get(V4_G3_ADMISSION_ENV)
    if not raw_path:
        raise V4AdmissionError("missing v4 G3 admission file")
    try:
        root = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4AdmissionError("malformed v4 G3 admission file") from exc
    now_value = time.time() if now is None else float(now)
    records: list[Any]
    if isinstance(root, dict) and isinstance(root.get("devices"), list):
        records = list(root["devices"])
    else:
        records = [root]
    errors: list[str] = []
    for record in records:
        try:
            return _validate_record(record, now_value)
        except ValueError as exc:
            errors.append(str(exc))
    detail = errors[0] if errors else "v4 G3 admission has no records"
    raise V4AdmissionError(detail)
