#!/usr/bin/env python3
"""Exercise production-shape Native + Pipeline startup and one real GPU job."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "miner"))

import pearl_mining as pm

from pmk_miner.monitor import bits_to_target
from pmk_miner.native import Native
from pmk_miner.pipeline import Pipeline, Shape
from pmk_miner.runtime import gpu_lock
from pmk_miner.transport import GatewayJob


BITS = 0x1E010000
TARGET = 1 << 232


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"time": time.time(), "event": event, **fields}, separators=(",", ":")), flush=True)


def synthetic_job() -> GatewayJob:
    header = (
        (1).to_bytes(4, "little")
        + bytes(32)
        + bytes(32)
        + (1).to_bytes(4, "little")
        + BITS.to_bytes(4, "little")
    )
    assert len(header) == 76
    assert bits_to_target(BITS) == TARGET
    return GatewayJob(header, TARGET, 3)


async def smoke(slots: int) -> dict[str, Any]:
    shape = Shape(8192, 8192, 4096, slots)
    memory_estimate = shape.validate()
    source = synthetic_job()
    native = Native()
    poll_count = 0
    polled: list[dict[str, Any]] = []
    original_poll = native.poll

    def observed_poll(handle: Any) -> Any:
        nonlocal poll_count
        result = original_poll(handle)
        poll_count += 1
        polled.append(
            {
                "status": int(result.status),
                "block_count": int(result.block_count),
                "block_stored": int(result.block_stored),
                "share_count": int(result.share_count),
                "share_stored": int(result.share_stored),
                "overflow": bool(result.overflow),
            }
        )
        return result

    native.poll = observed_poll
    events: list[dict[str, Any]] = []
    verified_submissions = 0
    pipeline: Pipeline | None = None

    def capture(event: str, **fields: Any) -> None:
        row = {"event": event, **fields}
        events.append(row)
        emit(event, **fields)

    started = time.monotonic()
    try:
        pipeline = Pipeline(native, shape, capture)
        pipeline.set_template(source)

        async def submit(job: GatewayJob, encoded: str) -> None:
            nonlocal verified_submissions
            proof = pm.PlainProof.from_base64(encoded)
            valid, _message = pm.verify_plain_proof_for_cert_version(
                3, pm.IncompleteBlockHeader.from_bytes(job.header), proof
            )
            if not valid:
                raise AssertionError("submitted production-shape proof did not verify")
            verified_submissions += 1
            # The easy regtest target produces many finds. One verified handoff
            # proves the full path, so prevent further proof construction.
            pipeline.cancel()

        record = await pipeline.run(0, TARGET, BITS, submit, lambda job: job is source)
        completed = [row for row in events if row.get("event") == "completed"]
        assert record.state == "released", record.history
        assert poll_count == 1, poll_count
        assert len(polled) == 1 and polled[0]["status"] == 0, polled
        assert not polled[0]["overflow"], polled
        assert polled[0]["block_count"] == polled[0]["block_stored"], polled
        assert polled[0]["share_count"] == polled[0]["share_stored"], polled
        assert completed and not completed[-1]["overflow"], completed
        assert verified_submissions >= 1
        return {
            "event": "production_shape_smoke",
            "passed": True,
            "shape": [shape.m, shape.n, shape.k],
            "slots": slots,
            "memory_estimate_bytes": memory_estimate,
            "poll_count": poll_count,
            "poll_result": polled[0],
            "verified_submissions": verified_submissions,
            "record_history": record.history,
            "elapsed_seconds": time.monotonic() - started,
            "probe_key": native.probe_key,
        }
    finally:
        if pipeline is not None:
            pipeline.cancel()
        native.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    with gpu_lock(emit, inherited=os.environ.get("PMK_GPU_LOCK_HELD") == "1"):
        result = asyncio.run(smoke(args.slots))
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
