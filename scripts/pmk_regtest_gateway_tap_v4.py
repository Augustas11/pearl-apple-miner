#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""V4 pearl-gateway tap that records submitblock calls and node rejections."""

from __future__ import annotations

import json
import os
import sys
import time

from pearl_gateway import pearl_client as pc

_LOG = os.environ["RTAP_LOG"]
_CORRUPT_FIRST = os.environ.get("RTAP_CORRUPT_FIRST") == "1"
_orig_submit_block = pc.PearlNodeClient.submit_block
_state = {"n": 0}


def _log(msg: str) -> None:
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f} {msg}\n")


def _log_event(event: str, **fields) -> None:
    row = {"time": time.time(), "event": event, **fields}
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _flip_v4_certificate(block_hex: str, where: str) -> str:
    block = bytearray.fromhex(block_hex)
    if len(block) < 44:
        raise ValueError("block too short for certificate preamble")
    cert_version = int.from_bytes(block[:4], "little")
    if cert_version != 4:
        raise ValueError(f"expected v4 certificate, got version {cert_version}")
    public_len = int.from_bytes(block[36:40], "little")
    public_off = 40
    proof_len_off = public_off + public_len
    if proof_len_off + 4 > len(block):
        raise ValueError("block too short for v4 proof length")
    proof_len = int.from_bytes(block[proof_len_off : proof_len_off + 4], "little")
    proof_off = proof_len_off + 4
    proof_end = proof_off + proof_len
    if proof_end > len(block):
        raise ValueError("block too short for v4 proof data")
    if where == "proof":
        if proof_len == 0:
            raise ValueError("empty v4 proof")
        off = proof_off + proof_len // 2
    elif where == "public_data":
        if public_len == 0:
            raise ValueError("empty v4 public data")
        off = public_off + public_len // 2
    else:
        raise ValueError(f"unknown corruption target: {where}")
    block[off] ^= 0x01
    return block.hex()


def _rejection_text(exc: Exception) -> str:
    text = str(exc)
    if "Pearl RPC error" in text:
        return f"rejected: {text}"
    return f"error: {type(exc).__name__}"


async def submit_block(self, block_hex: str) -> str:
    _state["n"] += 1
    n = _state["n"]
    if _CORRUPT_FIRST and n == 1:
        for where in ("proof", "public_data"):
            try:
                result = await _orig_submit_block(self, _flip_v4_certificate(block_hex, where))
            except ValueError as exc:
                result = _rejection_text(exc)
            except Exception as exc:  # pragma: no cover - diagnostic only
                result = f"error: {type(exc).__name__}"
            _log(f"NEGATIVE corrupt_{where} verdict={result}")
            _log_event(f"negative_corrupt_{where}", result=result)
    _log(f"SUBMIT n={n} start")
    result = await _orig_submit_block(self, block_hex)
    _log(f"SUBMIT n={n} verdict={result}")
    event = "block_accepted" if result == "accepted" else "block_rejected"
    _log_event(event, result=result)
    return result


pc.PearlNodeClient.submit_block = submit_block

from pearl_gateway.submission_service import SubmissionService  # noqa: E402

_orig_submit_plain_proof = SubmissionService.submit_plain_proof


async def submit_plain_proof(self, plain_proof, template, *args, **kwargs):
    _log(
        "FOUND "
        f"height={template.height} cert={int(template.required_cert_version)} "
        f"ancestor_headers={len(template.ancestor_headers)}"
    )
    _log_event(
        "plain_proof_found",
        height=template.height,
        cert=int(template.required_cert_version),
        ancestor_headers=len(template.ancestor_headers),
    )
    return await _orig_submit_plain_proof(self, plain_proof, template, *args, **kwargs)


SubmissionService.submit_plain_proof = submit_plain_proof

from pearl_gateway.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["pearl-gateway", "start", "--debug"]
    main()
