#!/usr/bin/env python3
"""B4 gateway tap with corrupt-cert controls and P6 header correlation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from contextvars import ContextVar

from pearl_gateway import pearl_client as pc


RTAP_LOG = os.environ["RTAP_LOG"]
P6_LOG = os.environ.get("PMK_B4_P6_LOG")
CORRUPT_FIRST = os.environ.get("RTAP_CORRUPT_FIRST") == "1"
_orig_submit_block = pc.PearlNodeClient.submit_block
_state = {"n": 0}
_submit_context: ContextVar[dict | None] = ContextVar("b4_submit_context", default=None)


def _append(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _log(msg: str) -> None:
    _append(RTAP_LOG, f"{time.time():.6f} {msg}")


def _p6(**fields) -> None:
    if P6_LOG:
        _append(P6_LOG, json.dumps({"time": time.time(), **fields}, sort_keys=True))


def _flip(block_hex: str, where: str) -> str:
    b = bytearray.fromhex(block_hex)
    publen = int.from_bytes(b[36:40], "little")
    pub_off = 40
    proflen = int.from_bytes(b[pub_off + publen : pub_off + publen + 4], "little")
    proof_off = pub_off + publen + 4
    off = proof_off + proflen // 2 if where == "proof" else pub_off + publen // 2
    b[off] ^= 0x01
    return b.hex()


async def submit_block(self, block_hex: str) -> str:
    _state["n"] += 1
    n = _state["n"]
    if CORRUPT_FIRST and n == 1:
        for where in ("proof", "public_data"):
            try:
                result = await _orig_submit_block(self, _flip(block_hex, where))
            except Exception:
                result = "RPC-ERROR redacted"
            _log(f"NEGATIVE corrupt_{where} verdict={result}")
    _log(f"SUBMIT n={n} start")
    result = await _orig_submit_block(self, block_hex)
    verdict_time = time.time()
    _log(f"SUBMIT n={n} verdict={result}")
    context = _submit_context.get() or {}
    _p6(event="submit_verdict", submit_n=n, verdict=result, verdict_time=verdict_time, **context)
    return result


pc.PearlNodeClient.submit_block = submit_block

from pearl_gateway.submission_service import SubmissionService  # noqa: E402

_orig_submit_plain_proof = SubmissionService.submit_plain_proof


def _header_bytes(template) -> bytes:
    return template.header.serialize_without_proof_commitment()


async def submit_plain_proof(self, plain_proof, template, submission_identity=None):
    header = _header_bytes(template)
    header_hash = hashlib.blake2s(header, digest_size=16).hexdigest()
    proof_text = plain_proof.to_base64()
    proof_digest = hashlib.sha256(proof_text.encode("utf-8")).hexdigest()
    _log(f"FOUND header_hash={header_hash} template_time={template.header.timestamp}")
    _p6(
        event="proof_handoff",
        header_hash=header_hash,
        proof_digest=proof_digest,
        submission_identity=submission_identity,
        header_prefix=header[:8].hex(),
        handoff_time=time.time(),
        template_time=template.header.timestamp,
    )
    token = _submit_context.set({
        "header_hash": header_hash, "proof_digest": proof_digest,
        "header_fields": {
            "version": int.from_bytes(header[:4], "little"),
            "previousblockhash": header[4:36][::-1].hex(),
            "merkleroot": header[36:68][::-1].hex(),
            "time": int.from_bytes(header[68:72], "little"),
            "bits": f"{int.from_bytes(header[72:76], 'little'):08x}",
        },
        "submission_identity": submission_identity,
    })
    try:
        return await _orig_submit_plain_proof(self, plain_proof, template, submission_identity)
    finally:
        _submit_context.reset(token)


SubmissionService.submit_plain_proof = submit_plain_proof

from pearl_gateway.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["pearl-gateway", "start", "--debug"]
    main()
