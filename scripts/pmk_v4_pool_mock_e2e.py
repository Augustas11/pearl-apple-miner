#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Local cert-v4 pool wire smoke for the real pmk miner pipeline.

Default mode is CPU/read-only: print the planned loopback run and, with
``--cpu-check``, validate that the fixture-derived v4 header/ancestor can build a
pmkcore Grid-B200 job. The real GPU smoke requires ``--run-gpu`` and an inherited
``PMK_GPU_LOCK_HELD=1``. It connects the current PoolClient to a loopback object
pool, switches from a v3 notify to a v4 notify, checks stale/invalid submit
classification, then mines a real V3 proof locally, switches the same Pipeline to V4 without restart, submits the captured V3 proof as stale, and mines real V4 shares through Pipeline+Native.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import ctypes as C
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "miner"))

from pmk_miner.monitor import DIFF1_TARGET  # noqa: E402
from pmk_miner.native import Native  # noqa: E402
from pmk_miner.pipeline import Pipeline, PoolSubmissionPolicy, Shape  # noqa: E402
from pmk_miner.pool import PoolClient, PoolSubmitOutcome, pool_target_for_difficulty  # noqa: E402
from pmk_miner.v4_admission import validate_v4_g3_admission_file  # noqa: E402

U8 = C.c_uint8
U32 = C.c_uint32
U64 = C.c_uint64
PTR = C.c_void_p
PIN = "f696760b259500ecb608469ea3953aeabbe78948"
DEFAULT_FIXTURE = ROOT / "bench/v4_emulation/vectors/b9_integration/32x64_k1024_s1/metadata.json"
DEFAULT_OUTPUT = ROOT / "bench/evidence/b9_v5_pool_mock_e2e.txt"
DEFAULT_M = 2048
DEFAULT_N = 2048
DEFAULT_K = 4096
WALLET = "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
WORKER = "v4-loopback"
POOL_BITS = 0x1D00FFFF
POOL_DIFFICULTY = 1
POOL_TARGET = pool_target_for_difficulty(POOL_DIFFICULTY)


def frame(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode() + b"\n"


def load_fixture(path: Path) -> tuple[bytes, bytes, int, int, int]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    header = bytearray(base64.b64decode(meta["proposed_header_b64"], validate=True))
    ancestor = base64.b64decode(meta["ancestor_header_b64"], validate=True)
    if len(header) != 76 or len(ancestor) != 108:
        raise ValueError("fixture must contain a 76-byte proposed header and 108-byte ancestor")
    header[72:76] = POOL_BITS.to_bytes(4, "little")
    return bytes(header), ancestor, int(meta["m"]), int(meta["n"]), int(meta["k"])


def notify_params(job_id: str, header: bytes, *, cert_version: int, ancestor: bytes | None = None) -> dict[str, Any]:
    params = {
        "job_id": job_id,
        "header": header.hex(),
        "target": f"{POOL_TARGET:064x}",
        "height": 909001,
        "cert_version": cert_version,
    }
    if cert_version == 4:
        if ancestor is None:
            raise ValueError("v4 notify requires ancestor")
        params["ancestor_headers"] = [base64.b64encode(ancestor).decode("ascii")]
    return params


class CoreVerifier:
    def __init__(self, path: Path):
        self.path = path
        self.lib = C.CDLL(str(path))
        self.lib.pmkcore_v4_init.argtypes = [U32]
        self.lib.pmkcore_v4_init.restype = C.c_int32
        self.lib.pmkcore_v4_job_create_grid_b200.argtypes = [PTR, PTR, PTR, U64, U32, U32, U32, C.POINTER(PTR)]
        self.lib.pmkcore_v4_job_create_grid_b200.restype = C.c_int32
        self.lib.pmkcore_v4_job_free.argtypes = [PTR]
        self.lib.pmkcore_v4_job_free.restype = None
        self.lib.pmkcore_v4_strerror.argtypes = [C.c_int32]
        self.lib.pmkcore_v4_strerror.restype = C.c_char_p
        self.lib.pmkcore_v4_verify_plain_proof.argtypes = [PTR, PTR, U64, PTR, C.POINTER(U8)]
        self.lib.pmkcore_v4_verify_plain_proof.restype = C.c_int32
        rc = self.lib.pmkcore_v4_init(0)
        if rc not in (0, -9):
            self.check(rc, "pmkcore_v4_init")

    def check(self, rc: int, what: str) -> None:
        if rc == 0:
            return
        msg = self.lib.pmkcore_v4_strerror(rc)
        detail = msg.decode(errors="replace") if msg else ""
        raise RuntimeError(f"{what} failed rc={rc}: {detail}")

    def create_job_check(self, header: bytes, ancestor: bytes, m: int, n: int, k: int) -> None:
        out = PTR()
        self.check(
            self.lib.pmkcore_v4_job_create_grid_b200(buf(header), buf(ancestor), None, 0, m, n, k, C.byref(out)),
            "pmkcore_v4_job_create_grid_b200",
        )
        self.lib.pmkcore_v4_job_free(out)

    def verify_plain_proof(self, header: bytes, proof_b64: str, share_nbits: int) -> bool:
        try:
            proof = base64.b64decode(proof_b64, validate=True)
        except (ValueError, binascii.Error):
            return False
        accepted = U8()
        nbits = (U8 * 4).from_buffer_copy(int(share_nbits).to_bytes(4, "little"))
        rc = self.lib.pmkcore_v4_verify_plain_proof(buf(header), buf(proof), len(proof), nbits, C.byref(accepted))
        if rc != 0:
            return False
        return bool(accepted.value)


def buf(data: bytes) -> C.Array[U8]:
    return (U8 * len(data)).from_buffer_copy(data)


class LoopbackPool:
    def __init__(self, verifier: CoreVerifier, *, header: bytes, ancestor: bytes, target_accepted: int):
        self.verifier = verifier
        self.header = header
        self.ancestor = ancestor
        self.target_accepted = target_accepted
        self.server: asyncio.AbstractServer | None = None
        self.port = 0
        self.send_v4 = asyncio.Event()
        self.done = asyncio.Event()
        self.accepted = 0
        self.invalid = 0
        self.stale = 0
        self.submissions: list[dict[str, Any]] = []
        self.current_job_id = "v3-before-switch"
        self.current_cert_version = 3
        self.v4_share_nbits: int | None = None

    async def start(self) -> None:
        # PlainProof payloads exceed asyncio's 64 KiB default; match the v3 mock.
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=8 * 1024 * 1024)
        assert self.server.sockets is not None
        self.port = int(self.server.sockets[0].getsockname()[1])

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            authorize = json.loads(await reader.readline())
            writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params("v3-before-switch", self.header, cert_version=3)}))
            writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
            await writer.drain()
            await self.send_v4.wait()
            self.current_job_id = "v4-current"
            self.current_cert_version = 4
            writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params("v4-current", self.header, cert_version=4, ancestor=self.ancestor)}))
            await writer.drain()
            while True:
                raw = await reader.readline()
                if not raw:
                    return
                request = json.loads(raw)
                if request.get("method") != "mining.submit":
                    writer.write(frame({"id": request.get("id"), "result": False, "error": "unsupported method"}))
                    await writer.drain()
                    continue
                params = request.get("params") or {}
                outcome, error = self._classify_submit(str(params.get("job_id", "")), str(params.get("plain_proof", "")))
                writer.write(frame({"id": request.get("id"), "result": outcome == "accepted", "error": error}))
                await writer.drain()
                if self.accepted >= self.target_accepted:
                    self.done.set()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _classify_submit(self, job_id: str, proof: str) -> tuple[str, str | None]:
        row = {"job_id": job_id, "proof_prefix": proof[:16]}
        if job_id != self.current_job_id:
            self.stale += 1
            row["classification"] = PoolSubmitOutcome.STALE.value
            self.submissions.append(row)
            return PoolSubmitOutcome.STALE.value, "stale share"
        assert self.v4_share_nbits is not None
        if self.verifier.verify_plain_proof(self.header, proof, self.v4_share_nbits):
            self.accepted += 1
            row["classification"] = PoolSubmitOutcome.ACCEPTED.value
            self.submissions.append(row)
            return PoolSubmitOutcome.ACCEPTED.value, None
        self.invalid += 1
        row["classification"] = PoolSubmitOutcome.INVALID.value
        self.submissions.append(row)
        return PoolSubmitOutcome.INVALID.value, "Jackpot condition not satisfied"


async def wait_for(predicate, timeout: float, label: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise TimeoutError(label)


async def run_gpu_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PMK_GPU_LOCK_HELD") != "1":
        raise SystemExit("Refusing V4 pool smoke without PMK_GPU_LOCK_HELD=1 from root/coordinator")
    validate_v4_g3_admission_file()
    header, ancestor, fixture_m, fixture_n, fixture_k = load_fixture(args.fixture)
    selected = selected_shape(args)
    shape = Shape(selected["m"], selected["n"], selected["k"], selected["slots"])
    verifier = CoreVerifier(args.pmkcore_v4)
    verifier.create_job_check(header, ancestor, shape.m, shape.n, shape.k)
    pool = LoopbackPool(verifier, header=header, ancestor=ancestor, target_accepted=args.accepted)
    await pool.start()
    logs: list[dict[str, Any]] = []
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{pool.port}",
        WALLET,
        WORKER,
        difficulty_floor=POOL_DIFFICULTY,
        reply_timeout=args.reply_timeout,
        notify_interval=0.0,
        log=lambda event, **kw: logs.append({"event": event, **kw}),
    )
    stop = asyncio.Event()
    client_task = asyncio.create_task(client.run(stop))
    native = None
    try:
        v3_job = await wait_for(lambda: client.latest if client.latest and client.latest.cert_version == 3 else None, args.reply_timeout, "v3 notify")
        native = await asyncio.to_thread(Native)
        pipeline = Pipeline(native, shape, lambda event, **kw: logs.append({"event": event, **kw}), submission_policy=PoolSubmissionPolicy())
        native_identity = {"probe_key": native.probe_key, "native_object_id": id(native), "pipeline_object_id": id(pipeline)}
        await asyncio.to_thread(pipeline.set_template, v3_job)
        captured_v3: dict[str, Any] = {}

        async def capture_v3(job, proof):
            captured_v3.update({"job": job, "proof": proof, "pool_job_id": job.pool_job_id, "cert_version": job.cert_version})
            logs.append({"event": "captured_v3_verified_proof", "pool_job_id": job.pool_job_id, "proof_size_b64": len(proof)})

        def is_v3_current(job):
            return client.latest is not None and job.template_identity == client.latest.template_identity and job.cert_version == 3

        v3_jobs = 0
        while not captured_v3 and v3_jobs < args.v3_max_jobs:
            await pipeline.run(v3_jobs % shape.slots, v3_job.target, v3_job.share_nbits, capture_v3, is_v3_current)
            v3_jobs += 1
        if not captured_v3:
            raise RuntimeError(f"captured no locally verified v3 proof after {v3_jobs} jobs")
        pool.send_v4.set()
        v4_job = await wait_for(lambda: client.latest if client.latest and client.latest.cert_version == 4 else None, args.reply_timeout, "v4 notify")
        pool.v4_share_nbits = v4_job.share_nbits
        await asyncio.to_thread(pipeline.set_template, v4_job)
        stale = await client.submit(captured_v3["job"], captured_v3["proof"])
        corrupt = await client.submit(v4_job, base64.b64encode(b"not-a-valid-v4-proof").decode("ascii"))
        if stale != PoolSubmitOutcome.STALE.value:
            raise RuntimeError(f"late v3 share classified {stale}, expected stale")
        if corrupt != PoolSubmitOutcome.INVALID.value:
            raise RuntimeError(f"corrupt v4 proof classified {corrupt}, expected invalid")

        async def submit(job, proof):
            outcome = await client.submit(job, proof)
            logs.append({"event": "smoke_submit_outcome", "outcome": outcome, "pool_job_id": job.pool_job_id})
            if outcome != PoolSubmitOutcome.ACCEPTED.value and pool.accepted < args.accepted:
                raise RuntimeError(f"unexpected mined share outcome {outcome}")

        def is_current(job):
            return client.latest is not None and job.template_identity == client.latest.template_identity and job.cert_version == 4

        index = 0
        jobs = 0
        while pool.accepted < args.accepted and jobs < args.max_jobs:
            await pipeline.run(index, v4_job.target, v4_job.share_nbits, submit, is_current)
            jobs += 1
            index = (index + 1) % shape.slots
        if pool.accepted < args.accepted:
            raise RuntimeError(f"only accepted {pool.accepted}/{args.accepted} shares after {jobs} jobs")
        result = {
            "schema": "pmk-v4-pool-mock-e2e-v1",
            "passed": True,
            "shape": {"m": shape.m, "n": shape.n, "k": shape.k, "slots": shape.slots},
            "accepted": pool.accepted,
            "invalid": pool.invalid,
            "stale": pool.stale,
            "late_v3_outcome": stale,
            "corrupt_v4_outcome": corrupt,
            "v3_gpu_pass": True,
            "v3proof_accepted_locally": True,
            "stale_actual_v3": stale == PoolSubmitOutcome.STALE.value,
            "v4_upstream_accepted": pool.accepted,
            "v3_jobs": v3_jobs,
            "v4_jobs": jobs,
            "native_identity_same": native_identity,
            "submissions": pool.submissions,
            "events": logs,
        }
        write_result(args.output, result)
        return result
    finally:
        stop.set()
        await client.close()
        client_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await client_task
        if native is not None:
            native.close()
        await pool.close()


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def selected_shape(args: argparse.Namespace) -> dict[str, int]:
    return {"m": args.m or DEFAULT_M, "n": args.n or DEFAULT_N, "k": args.k or DEFAULT_K, "slots": args.slots}


def dry_plan(args: argparse.Namespace) -> dict[str, Any]:
    header, ancestor, fixture_m, fixture_n, fixture_k = load_fixture(args.fixture)
    shape = selected_shape(args)
    plan = {
        "schema": "pmk-v4-pool-mock-e2e-plan-v1",
        "run_gpu_required_env": "PMK_GPU_LOCK_HELD=1",
        "requires_real_v4_g3_admission_file": True,
        "real_pool": False,
        "loopback_only": True,
        "fixture": str(args.fixture),
        "header_hex": header.hex(),
        "ancestor_header_base64_len": len(base64.b64encode(ancestor).decode("ascii")),
        "shape": shape,
        "pool_difficulty": POOL_DIFFICULTY,
        "pool_target_hex": f"{POOL_TARGET:064x}",
        "target_accepted_shares": args.accepted,
        "checks": [
            "create one Native+Pipeline before v3 notify switch",
            "set v3 template and capture real locally verified v3 plain_proof without sending before v4",
            "set v4 template on the same Pipeline after v3 drain without restart",
            "late real v3 share returns stale over PoolClient.submit",
            "corrupt v4 proof returns invalid",
            "real Pipeline+Native V4 GPU job submits plain_proof base64",
            "loopback server verifies each submitted proof with pmkcore_v4_verify_plain_proof",
        ],
        "run_command_after_g3_gpu_lock": [
            "PMK_GPU_LOCK_HELD=1",
            str(ROOT / ".venv/bin/python"),
            str(Path(__file__).resolve()),
            "--run-gpu",
            "--output",
            str(args.output),
        ],
    }
    return plan


def cpu_check(args: argparse.Namespace) -> dict[str, Any]:
    header, ancestor, fixture_m, fixture_n, fixture_k = load_fixture(args.fixture)
    shape = selected_shape(args)
    verifier = CoreVerifier(args.pmkcore_v4)
    verifier.create_job_check(header, ancestor, shape["m"], shape["n"], shape["k"])
    return {"event": "cpu_check", "pmkcore_grid_b200_job_create": True, "shape": shape, "pool_difficulty": POOL_DIFFICULTY, "fixture_shape": {"m": fixture_m, "n": fixture_n, "k": fixture_k}, "admission_file_fabricated": False}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--pmkcore-v4", type=Path, default=ROOT / "pmkcore/v4/target/release/libpmkcore_v4.dylib")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--accepted", type=int, default=3)
    p.add_argument("--max-jobs", type=int, default=8)
    p.add_argument("--v3-max-jobs", type=int, default=8)
    p.add_argument("--reply-timeout", type=float, default=30.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--cpu-check", action="store_true", help="load pmkcore and validate fixture-derived job creation; no Metal/GPU")
    p.add_argument("--run-gpu", action="store_true", help="run real Pipeline+Native V4 GPU smoke; requires PMK_GPU_LOCK_HELD=1")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.accepted < 3:
        raise SystemExit("--accepted must be at least 3")
    if args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    if args.v3_max_jobs < 1:
        raise SystemExit("--v3-max-jobs must be positive")
    if args.slots not in (2, 3):
        raise SystemExit("--slots must be 2 or 3")
    for name in ("m", "n"):
        value = getattr(args, name)
        if value and (value <= 0 or value % 64):
            raise SystemExit(f"--{name} must be a positive multiple of 64 for the v3/v4 no-restart intersection")
    if args.k and args.k != 4096:
        raise SystemExit("--k must be 4096 for the v3/v4 no-restart intersection")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.run_gpu:
        result = asyncio.run(run_gpu_smoke(args))
        print(json.dumps(result, sort_keys=True))
        return 0
    plan = dry_plan(args)
    print(json.dumps(plan, sort_keys=True))
    if args.cpu_check:
        print(json.dumps(cpu_check(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
