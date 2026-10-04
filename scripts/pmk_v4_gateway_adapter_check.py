#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Apply and inspect the permanent PMK cert-v4 gateway adapter patch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(
    os.environ.get("PMK_V4_ROOT")
    or Path(__file__).resolve().parents[1]
).resolve()
MINER = ROOT / "miner"
if str(MINER) not in sys.path:
    sys.path.insert(0, str(MINER))

from pmk_miner.gateway_launcher import patch_gateway_copy  # noqa: E402

DEFAULT_SOURCE = ROOT / "vendor" / "pearl-fp8" / "miner" / "pearl-gateway"
DEFAULT_PATCH = ROOT / "miner" / "gateway_patches" / "0002-b9-v4-safe-async-proving.patch"


class AdapterCheckError(RuntimeError):
    pass


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _require(text: str, needles: Iterable[str], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        formatted = "\n".join(f"  - {needle}" for needle in missing)
        raise AdapterCheckError(f"{label} missing expected adapter text:\n{formatted}")


def check_patched_gateway(root: Path) -> None:
    dataclasses = _read(root, "src/pearl_gateway/comm/dataclasses.py")
    server = _read(root, "src/pearl_gateway/miner_rpc/server.py")
    submission = _read(root, "src/pearl_gateway/submission_service.py")
    config = _read(root, "src/pearl_gateway/config.py")
    client = _read(root, "src/pearl_gateway/pearl_client.py")

    _require(
        dataclasses,
        (
            "def merkle_branch_for_index",
            "coinbase_merkle_branch: list[bytes]",
            "coinbase_tx: bytes = b\"\"",
            '"coinbase_tx": b64_encode(self.coinbase_tx)',
            '"ancestor_headers": [b64_encode(header) for header in self.ancestor_headers]',
            "submission_id: str | None = None",
        ),
        "dataclasses",
    )
    _require(
        server,
        (
            "PMK_GATEWAY_PROVING_QUEUE_SIZE",
            "await self._try_admit_proving_job()",
            '"proving_queue_full"',
            'claimed_submission_id = _validate_submission_id',
            "PlainProofV4",
            "CERT_VERSION_PLAIN_FP8",
            "proof_type.from_base64(plain_proof_base64)",
            '{"status": "submitted", "submission_id": identity["submission_id"]}',
            "self.handle_submit_plain_proof(plain_proof, mining_job, identity)",
            "stale_plain_proof",
            "block_submission_result",
        ),
        "server",
    )
    if server.index("await self._try_admit_proving_job()") > server.index("MiningJob.from_dict"):
        raise AdapterCheckError("server admits proving after MiningJob decode; admission must come first")
    if server.index("asyncio.create_task(") > server.index('{"status": "submitted"'):
        raise AdapterCheckError("server receipt is returned before task creation marker check failed")
    _require(
        submission,
        (
            "ProofPool | None",
            "PlainProofV4",
            "plain_proof.to_base64()",
            "ProofGenerator.build_block",
            "block_accepted",
            "block_rejected",
            "block_submission_error",
            "stage=stage",
            "phase=stage",
            "classification=classification",
            "error_type=_error_type(e)",
        ),
        "submission_service",
    )
    if "ProcessPoolExecutor" in submission:
        raise AdapterCheckError("V4 adapter must preserve the upstream ProofPool path")
    _require(config, ("Miner RPC TCP host must be loopback only", "UDS socket path must not be a symlink"), "config")
    _require(client, ("_redact_value", "_redact_url", "RPC params redacted"), "pearl_client")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument(
        "--copy-parent",
        type=Path,
        default=None,
        help="Optional parent for the derived patched copy; defaults to a temporary directory.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep and print the derived source copy.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    patched = patch_gateway_copy(args.source, args.patch, args.copy_parent)
    try:
        check_patched_gateway(patched.root_dir)
        print(f"v4_gateway_adapter_ok root={patched.root_dir}")
        if args.keep:
            print("kept=1")
            return 0
        return 0
    finally:
        if not args.keep:
            patched.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
