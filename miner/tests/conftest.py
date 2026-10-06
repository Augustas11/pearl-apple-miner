"""Shared real-device fixtures that never consume operator admission state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from pmk_miner.runtime import atomic_json, gpu_lock


ROOT = Path(__file__).resolve().parents[2]


def _packaged_g3_helper() -> tuple[Path, Path, Path]:
    candidates = list((ROOT / "dist" / "release-work").glob(
        "pmk-macos-arm64-*/dist/quickstart/bin/g3-admit"
    ))
    candidates.append(ROOT / "dist" / "quickstart" / "bin" / "g3-admit")
    helpers = [path for path in candidates if path.is_file()]
    assert helpers, "release G3 helper is required for real-GPU tests"
    helper = max(helpers, key=lambda path: path.stat().st_mtime_ns)
    quickstart = helper.parent.parent
    package_root = quickstart.parent.parent
    if package_root == ROOT / "dist":
        package_root = ROOT
    bundle = package_root / "libpmk/.build/release/libpmk_PMK.bundle"
    assert bundle.is_dir(), f"release resource bundle is missing: {bundle}"
    return helper, quickstart, bundle


@pytest.fixture(scope="session")
def v3_g3_admission(tmp_path_factory):
    """Create an exact packaged-runtime v3 admission for this pytest session."""
    helper, quickstart, bundle = _packaged_g3_helper()
    path = tmp_path_factory.mktemp("v3-g3-admission") / "g3-admission.json"
    env = os.environ.copy()
    env.update(
        PMK_B4_ROOT=str(quickstart),
        PMK_RESOURCE_BUNDLE=str(bundle),
        PMK_G3_ADMISSION_FILE=str(path),
    )
    with gpu_lock(lambda *_args, **_kwargs: None):
        result = subprocess.run(
            [str(helper)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=1200, check=False,
        )
    assert result.returncode == 0, result.stdout[-4000:]
    admission = None
    for line in result.stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("devices"), list):
            admission = candidate
    assert admission is not None, "packaged G3 helper produced no admission record"
    kernels = {record.get("kernel") for record in admission["devices"]}
    assert {"sg", "na"}.issubset(kernels), f"packaged G3 omitted kernels: {kernels}"
    atomic_json(path, admission)
    previous = os.environ.get("PMK_G3_ADMISSION_FILE")
    os.environ["PMK_G3_ADMISSION_FILE"] = str(path)
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop("PMK_G3_ADMISSION_FILE", None)
        else:
            os.environ["PMK_G3_ADMISSION_FILE"] = previous
