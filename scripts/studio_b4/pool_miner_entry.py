#!/usr/bin/env python3
"""Runtime pmk_miner entrypoint for the pool window.

The outer wrapper owns resume of other GPU workloads. This shim lets pmk validate the lab lock
and pause token, while making LabSession.finish a no-op for this child process.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pmk_miner.__main__ as miner_main
from pmk_miner.runtime import LabSession


POOL_LOG = Path(os.environ["PMK_POOL_WINDOW_LOG"]) if os.environ.get("PMK_POOL_WINDOW_LOG") else None
_orig_lab_session = miner_main.LabSession


def _pool_log(**fields):
    if POOL_LOG is None:
        return
    POOL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with POOL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), **fields}, sort_keys=True) + "\n")


class PoolLabSession(LabSession):
    def finish(self):
        _pool_log(event="provider_resume_deferred_to_lead")
        return None


def main() -> int:
    miner_main.LabSession = PoolLabSession
    try:
        return miner_main.main()
    finally:
        miner_main.LabSession = _orig_lab_session


if __name__ == "__main__":
    raise SystemExit(main())
