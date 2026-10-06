"""Kernel selection for cert-v3 mining."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from .scheme import V3_NA_SCHEME, V3_SG_SCHEME, Scheme


VALID_KERNELS = ("auto", "sg", "na")
KERNEL_ENV = "PMK_KERNEL"


def normalize_kernel(value: str | None) -> str:
    kernel = (value or "auto").lower()
    if kernel not in VALID_KERNELS:
        raise ValueError("kernel must be one of auto, sg, na")
    return kernel


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, timeout=5).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def detect_device_class() -> str:
    override = os.environ.get("PMK_DEVICE_CLASS")
    if override:
        return override
    chip = ""
    try:
        data = json.loads(subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=30,
        ))
        items = data.get("SPDisplaysDataType", [])
        if items:
            chip = str(items[0].get("sppci_chipset_model", ""))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            TypeError, ValueError, json.JSONDecodeError):
        pass
    text = f"{chip} {_sysctl('machdep.cpu.brand_string')} {_sysctl('hw.model')}".lower()
    if "m5" in text or "mac17," in text:
        return "Apple10"
    if "m4" in text or "m3" in text:
        return "Apple9"
    if "m2" in text:
        return "Apple8"
    if "m1" in text:
        return "Apple7"
    return "unknown"


def device_class_generation(device_class: str) -> int | None:
    if not device_class.startswith("Apple"):
        return None
    try:
        return int(device_class[5:])
    except ValueError:
        return None


def resolve_v3_kernel(requested: str | None, device_class: str | None = None) -> tuple[str, str]:
    kernel = normalize_kernel(requested)
    detected = device_class or detect_device_class()
    generation = device_class_generation(detected)
    if kernel == "na" and generation is not None and generation < 10:
        raise ValueError(f"K3-NA requires Apple10+, got {detected}")
    if kernel == "auto":
        kernel = "na" if generation is not None and generation >= 10 else "sg"
    return kernel, detected


def scheme_for_kernel(kernel: str) -> Scheme:
    normalized = normalize_kernel(kernel)
    if normalized == "auto":
        normalized, _device_class = resolve_v3_kernel(normalized)
    return V3_NA_SCHEME if normalized == "na" else V3_SG_SCHEME


def apply_kernel_environment(kernel: str) -> None:
    normalized = normalize_kernel(kernel)
    if normalized == "auto":
        raise ValueError("cannot apply unresolved auto kernel")
    os.environ[KERNEL_ENV] = normalized


def metadata_kernel(metadata: dict[str, Any]) -> str | None:
    raw = str(metadata.get("kernel", metadata.get("v3_kernel", ""))).lower()
    if raw in ("na", "k3-na", "k3_na", "k3na"):
        return "na"
    if raw in ("sg", "k3-sg", "k3_sg", "k3sg"):
        return "sg"
    return None
