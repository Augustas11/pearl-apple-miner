#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Relocatable path and pinned-source checks shared by v4 tooling."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def bundle_root(script_file: str) -> Path:
    """Resolve the repo-shaped runtime root without embedding a user home."""
    override = os.environ.get("PMK_V4_ROOT")
    return Path(override).expanduser().resolve() if override else Path(script_file).resolve().parents[1]


def verify_pearl_pin(root: Path, expected: str) -> Path:
    """Verify the live git vendor or a manifest-protected offline export.

    A source stamp identifies an exported tree; a live checkout is additionally
    required to be at the exact commit with no tracked modifications.
    """
    vendor = Path(os.environ.get("PMK_PEARL_V4_VENDOR", root / "vendor" / "pearl-fp8")).resolve()
    if (vendor / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"wrong Pearl source pin: {actual}")
        for flags in ((), ("--cached",)):
            subprocess.run(
                ["git", "-C", str(vendor), "diff", *flags, "--exit-code", "--quiet"],
                check=True,
            )
        return vendor

    stamp = vendor / ".pmk-v4-pin"
    try:
        actual = stamp.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"offline Pearl source is missing pin stamp: {stamp}") from exc
    if actual != expected:
        raise ValueError(f"offline Pearl source pin mismatch: {actual}")
    return vendor
