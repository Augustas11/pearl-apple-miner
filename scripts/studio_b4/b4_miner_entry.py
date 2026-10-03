#!/usr/bin/env python3
"""Runtime pmk_miner entrypoint for B4.

This keeps lab-window policy out of the package source: an outer wrapper
owns the machine lock, so the repo GPU-lock context inherits that ownership. The P6
hook logs an exact header-hash correlation at the first point a GPU result is
known and eligible for block submission.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pmk_miner.__main__ as miner_main
from pmk_miner.runtime import LabSession
from pmk_miner.pipeline import Pipeline, Record


P6_LOG = Path(os.environ["PMK_B4_P6_LOG"]) if os.environ.get("PMK_B4_P6_LOG") else None
_orig_run = Pipeline._run
_orig_move = Record.move
_orig_lab_session = miner_main.LabSession


def _p6_log(**fields):
    if P6_LOG is None:
        return
    P6_LOG.parent.mkdir(parents=True, exist_ok=True)
    with P6_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), **fields}, sort_keys=True) + "\n")


async def _run_with_p6(self, index, record, share_target, share_nbits, submit, is_current):
    async def submit_with_p6(job, proof):
        header_hash = hashlib.blake2s(job.header, digest_size=16).hexdigest()
        proof_text = base64.b64encode(proof).decode("ascii") if isinstance(proof, bytes) else str(proof)
        _p6_log(
            event="gpu_find",
            job_id=record.job_id,
            header_hash=header_hash,
            header_prefix=job.header[:8].hex(),
            proof_digest=hashlib.sha256(proof_text.encode("utf-8")).hexdigest(),
            gpu_find_time=getattr(record, "b4_find_time", None),
            submit_callback_time=time.time(),
            stages=record.stage_seconds,
        )
        return await submit(job, proof)

    return await _orig_run(self, index, record, share_target, share_nbits, submit_with_p6, is_current)


def _move_with_find_time(self, state):
    result = _orig_move(self, state)
    if state == "scanned" and not hasattr(self, "b4_find_time"):
        self.b4_find_time = time.time()
    return result


class B4LabSession(LabSession):
    def finish(self):
        if os.environ.get("PMK_B4_CONTROLLER_LAB_SESSION") == str(os.getppid()):
            _p6_log(event="provider_resume_deferred_to_b4_window")
            return None
        return super().finish()


def main() -> int:
    miner_main.LabSession = B4LabSession
    Pipeline._run = _run_with_p6
    Record.move = _move_with_find_time
    try:
        return miner_main.main()
    finally:
        miner_main.LabSession = _orig_lab_session


if __name__ == "__main__":
    raise SystemExit(main())
