"""Runtime launcher for pmk-miner's patched pearl-gateway copy.

The vendor gateway is treated as read-only. This module copies it to a runtime
directory, applies the B3 patch there, and starts Python with that copied source
first on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "vendor" / "pearl" / "miner" / "pearl-gateway"
DEFAULT_PATCH = ROOT / "miner" / "gateway_patches" / "0001-b3-safe-async-proving.patch"


@dataclass(frozen=True)
class PatchedGatewayCopy:
    source_dir: Path
    root_dir: Path
    src_dir: Path
    patch_file: Path

    def cleanup(self) -> None:
        runtime_dir = self.root_dir.parent
        if runtime_dir.name.startswith("pmk-gateway-"):
            shutil.rmtree(runtime_dir, ignore_errors=True)


def _run_patch_apply(copy_root: Path, patch_file: Path) -> None:
    normalized_patch = _normalize_unified_diff(patch_file)
    try:
        check = subprocess.run(
            ["patch", "-p1", "-C", "-t", "-i", str(normalized_patch)],
            cwd=copy_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check.returncode != 0:
            raise RuntimeError(f"gateway patch check failed: {check.stderr.strip()}")
        applied = subprocess.run(
            ["patch", "-p1", "-N", "-t", "-i", str(normalized_patch)],
            cwd=copy_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if applied.returncode != 0:
            raise RuntimeError(f"gateway patch failed: {applied.stderr.strip()}")
    finally:
        normalized_patch.unlink(missing_ok=True)


def _normalize_unified_diff(patch_file: Path) -> Path:
    """Return a git-apply-safe copy of a diff with explicit empty context lines."""
    lines = patch_file.read_text().splitlines()
    normalized: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@ "):
            in_hunk = True
        if in_hunk and line == "":
            normalized.append(" ")
        else:
            normalized.append(line)
    handle = tempfile.NamedTemporaryFile(
        "w", prefix="pmk-gateway-patch-", suffix=".patch", delete=False
    )
    with handle:
        handle.write("\n".join(normalized))
        handle.write("\n")
    return Path(handle.name)


def patch_gateway_copy(
    source_dir: Path | str = DEFAULT_SOURCE,
    patch_file: Path | str = DEFAULT_PATCH,
    copy_parent: Path | str | None = None,
) -> PatchedGatewayCopy:
    """Copy vendor pearl-gateway and apply the B3 runtime patch to the copy."""
    source = Path(source_dir).resolve()
    patch = Path(patch_file).resolve()
    if not source.exists():
        raise FileNotFoundError(f"gateway source not found: {source}")
    if not patch.exists():
        raise FileNotFoundError(f"gateway patch not found: {patch}")

    if copy_parent is None:
        root = Path(tempfile.mkdtemp(prefix="pmk-gateway-"))
    else:
        parent = Path(copy_parent).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="pmk-gateway-", dir=parent))

    destination = root / "pearl-gateway"
    ignore = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache")
    try:
        shutil.copytree(source, destination, ignore=ignore)
        _run_patch_apply(destination, patch)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return PatchedGatewayCopy(
        source_dir=source,
        root_dir=destination,
        src_dir=destination / "src",
        patch_file=patch,
    )


def build_gateway_command(
    patched: PatchedGatewayCopy,
    gateway_args: Sequence[str] | None = None,
    tap_script: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build the subprocess command and environment for the patched gateway."""
    command_env = dict(os.environ if env is None else env)
    pythonpath_parts = [str(patched.src_dir)]
    tap = Path(tap_script).resolve() if tap_script else None
    if tap is not None:
        pythonpath_parts.insert(0, str(tap.parent))
    existing_pythonpath = command_env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    command_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    if tap is not None:
        cmd = [sys.executable, str(tap)]
    else:
        cmd = [sys.executable, "-m", "pearl_gateway.cli", *(gateway_args or ["start"])]
    return cmd, command_env


def launch_patched_gateway(
    gateway_args: Sequence[str] | None = None,
    *,
    source_dir: Path | str = DEFAULT_SOURCE,
    patch_file: Path | str = DEFAULT_PATCH,
    copy_parent: Path | str | None = None,
    tap_script: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.Popen:
    """Start a patched gateway subprocess and return the Popen handle."""
    patched = patch_gateway_copy(source_dir, patch_file, copy_parent)
    tap = tap_script or (env or os.environ).get("PMK_GATEWAY_TAP_SCRIPT")
    cmd, command_env = build_gateway_command(patched, gateway_args, tap, env)
    command_env["PMK_GATEWAY_PATCHED_COPY"] = str(patched.root_dir)
    process = subprocess.Popen(cmd, cwd=patched.root_dir, env=command_env)
    process.pmk_gateway_copy = patched
    return process


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a patched runtime copy of pearl-gateway")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--copy-parent", type=Path, default=None)
    parser.add_argument("--tap-script", type=Path, default=None)
    parser.add_argument("gateway_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.gateway_args and args.gateway_args[0] == "--":
        args.gateway_args = args.gateway_args[1:]
    if not args.gateway_args:
        args.gateway_args = ["start"]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    process = launch_patched_gateway(
        args.gateway_args,
        source_dir=args.source,
        patch_file=args.patch,
        copy_parent=args.copy_parent,
        tap_script=args.tap_script,
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return process.wait()
    finally:
        process.pmk_gateway_copy.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
