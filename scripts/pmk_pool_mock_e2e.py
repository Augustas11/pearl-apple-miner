#!/usr/bin/env python3
"""Loopback object-dialect pool harness for pmk_miner pool mode."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
DIFF1_TARGET = 0xFFFF << 208
DEFAULT_WALLET = "prl1pmockpoolwallet000000000000000000000000000000000000000000"


def secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def log(message: str) -> None:
    print(f"[pmk-pool-e2e] {message}", flush=True)


def redact_wallet(wallet: str) -> str:
    if len(wallet) <= 12:
        return "<wallet>"
    return f"{wallet[:6]}...{wallet[-6:]}"


def poisson_interval(lam: float, *, alpha: float = 0.001) -> tuple[int, int]:
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if lam == 0:
        return (0, 0)
    tail = alpha / 2.0
    mode = max(0, int(math.floor(lam)))
    mode_pmf = math.exp(mode * math.log(lam) - lam - math.lgamma(mode + 1))
    lower_probs: list[tuple[int, float]] = [(mode, mode_pmf)]
    mass = mode_pmf
    k = mode
    while k > 0:
        mass *= k / lam
        k -= 1
        lower_probs.append((k, mass))
    cdf = 0.0
    lower = 0
    for k, mass in reversed(lower_probs):
        cdf += mass
        if cdf > tail:
            lower = k
            break
    cdf = sum(mass for _, mass in lower_probs)
    mass = mode_pmf
    upper = mode
    hard_limit = int(math.ceil(lam + 20 * math.sqrt(lam) + 100))
    for k in range(mode + 1, hard_limit + 1):
        mass *= lam / k
        cdf += mass
        upper = k
        if cdf >= 1.0 - tail:
            break
    return lower, upper


def target_to_bits_floor(target: int) -> int:
    if target <= 0:
        raise ValueError("target must be positive")
    raw = target.to_bytes(max(1, (target.bit_length() + 7) // 8), "big")
    exponent = len(raw)
    if raw[0] & 0x80:
        mantissa = int.from_bytes(b"\x00" + raw[:2], "big")
        exponent += 1
    else:
        mantissa = int.from_bytes(raw[:3].ljust(3, b"\x00"), "big")
    bits = (exponent << 24) | mantissa
    while bits_to_target(bits) > target:
        mantissa -= 1
        if mantissa <= 0:
            exponent -= 1
            mantissa = 0xFFFF
        bits = (exponent << 24) | mantissa
    return bits


def bits_to_target(bits: int) -> int:
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def make_header(sequence: int, *, block_difficulty: int, block_nbits: int | None = None) -> bytes:
    import pearl_mining as pm

    nbits = block_nbits if block_nbits is not None else target_to_bits_floor(DIFF1_TARGET // block_difficulty)
    prev = sequence.to_bytes(32, "little")
    merkle = secrets.token_bytes(32)
    header = pm.IncompleteBlockHeader(1, prev, merkle, int(time.time()) + sequence, nbits)
    return bytes(header.to_bytes())


@dataclass(slots=True)
class PoolJob:
    job_id: str
    header: bytes
    target: int
    height: int
    cert_version: int = 3


@dataclass(slots=True)
class PoolStats:
    accepted: int = 0
    stale: int = 0
    duplicate: int = 0
    low_difficulty: int = 0
    invalid: int = 0
    wrong_job_id: int = 0
    submits: int = 0
    authorized: int = 0
    proofs_seen: set[str] = field(default_factory=set)
    accepted_records: list[tuple[str, str]] = field(default_factory=list)


class MockPool:
    def __init__(
        self,
        *,
        difficulty: int,
        rotate_seconds: float,
        block_difficulty: int,
        block_nbits: int | None = None,
        reject_first_share: bool = False,
        malicious_cert: bool = False,
        malicious_target: bool = False,
    ) -> None:
        self.difficulty = difficulty
        self.target = DIFF1_TARGET // difficulty
        self.share_nbits = target_to_bits_floor(self.target)
        self.rotate_seconds = rotate_seconds
        self.block_difficulty = block_difficulty
        self.block_nbits = block_nbits
        self.reject_first_share = reject_first_share
        self.malicious_cert = malicious_cert
        self.malicious_target = malicious_target
        self.stats = PoolStats()
        self.jobs: dict[str, PoolJob] = {}
        self._sequence = 0
        self._server: asyncio.base_events.Server | None = None
        self.host = "127.0.0.1"
        self.port: int | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._rotator: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, 0, limit=8 * 1024 * 1024)
        sock = self._server.sockets[0]
        self.port = int(sock.getsockname()[1])
        self._rotator = asyncio.create_task(self._rotate_loop())

    async def stop(self) -> None:
        if self._rotator is not None:
            self._rotator.cancel()
            await asyncio.gather(self._rotator, return_exceptions=True)
        for writer in list(self._clients):
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in list(self._clients)), return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def url(self) -> str:
        if self.port is None:
            raise RuntimeError("pool not started")
        return f"stratum+tcp://{self.host}:{self.port}"

    def current_job(self) -> PoolJob:
        if not self.jobs:
            self._new_job()
        return next(reversed(self.jobs.values()))

    def _new_job(self) -> PoolJob:
        self._sequence += 1
        target = 0 if self.malicious_target else self.target
        job = PoolJob(
            job_id=f"{self._sequence:08x}_{self.difficulty}",
            header=make_header(self._sequence, block_difficulty=self.block_difficulty, block_nbits=self.block_nbits),
            target=target,
            height=100_000 + self._sequence,
            cert_version=4 if self.malicious_cert else 3,
        )
        self.jobs[job.job_id] = job
        return job

    async def _rotate_loop(self) -> None:
        self._new_job()
        while True:
            await asyncio.sleep(self.rotate_seconds)
            job = self._new_job()
            await self._broadcast_notify(job)

    async def _broadcast_notify(self, job: PoolJob) -> None:
        line = self._notify(job)
        for writer in list(self._clients):
            try:
                writer.write(line)
                await writer.drain()
            except OSError:
                self._clients.discard(writer)

    def _notify(self, job: PoolJob) -> bytes:
        payload = {
            "id": None,
            "method": "mining.notify",
            "params": {
                "job_id": job.job_id,
                "header": job.header.hex(),
                "target": f"{job.target:064x}",
                "height": job.height,
                "cert_version": job.cert_version,
            },
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients.add(writer)
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    return
                if len(line) > 8 * 1024 * 1024:
                    return
                try:
                    request = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return
                if not isinstance(request, dict):
                    await self._reply(writer, request.get("id") if isinstance(request, dict) else None, False, {"code": 20, "msg": "request must be an object"})
                    continue
                method = request.get("method")
                if method == "mining.authorize":
                    await self._authorize(writer, request)
                elif method == "mining.submit":
                    await self._submit(writer, request)
                else:
                    await self._reply(writer, request.get("id"), False, {"code": 21, "msg": "unknown method"})
        finally:
            self._clients.discard(writer)
            writer.close()
            await writer.wait_closed()

    async def _authorize(self, writer: asyncio.StreamWriter, request: dict[str, Any]) -> None:
        params = request.get("params")
        if not isinstance(params, dict):
            await self._reply(writer, request.get("id"), False, {"code": 20, "msg": "params must be an object"})
            return
        if not params.get("wallet") or "worker" not in params:
            await self._reply(writer, request.get("id"), False, {"code": 24, "msg": "Wallet is missing"})
            return
        self.stats.authorized += 1
        writer.write(self._notify(self.current_job()))
        await writer.drain()
        await self._reply(writer, request.get("id"), True, None)

    async def _submit(self, writer: asyncio.StreamWriter, request: dict[str, Any]) -> None:
        params = request.get("params")
        if not isinstance(params, dict):
            await self._reply(writer, request.get("id"), False, {"code": 20, "msg": "params must be an object"})
            return
        job_id = params.get("job_id")
        proof_b64 = params.get("plain_proof")
        self.stats.submits += 1
        if not isinstance(job_id, str) or job_id not in self.jobs:
            self.stats.wrong_job_id += 1
            await self._reply(writer, request.get("id"), False, {"code": 25, "msg": "unknown job_id"})
            return
        if not isinstance(proof_b64, str):
            self.stats.invalid += 1
            await self._reply(writer, request.get("id"), False, {"code": 26, "msg": "plain_proof missing"})
            return
        try:
            proof_digest = hashlib.sha256(base64.b64decode(proof_b64, validate=True)).hexdigest()
        except Exception:
            self.stats.invalid += 1
            await self._reply(writer, request.get("id"), False, {"code": 26, "msg": "plain_proof invalid"})
            return
        if proof_digest in self.stats.proofs_seen:
            self.stats.duplicate += 1
            await self._reply(writer, request.get("id"), False, {"code": 27, "msg": "duplicate share"})
            return
        self.stats.proofs_seen.add(proof_digest)
        if self.reject_first_share and self.stats.accepted == 0:
            self.stats.invalid += 1
            await self._reply(writer, request.get("id"), False, {"code": 28, "msg": "Jackpot condition not satisfied"})
            return
        ok = self._verify(self.jobs[job_id], proof_b64)
        if ok:
            self.stats.accepted += 1
            self.stats.accepted_records.append((job_id, proof_b64))
            await self._reply(writer, request.get("id"), True, None)
        else:
            self.stats.low_difficulty += 1
            await self._reply(writer, request.get("id"), False, {"code": 29, "msg": "low difficulty share"})

    def _verify(self, job: PoolJob, proof_b64: str) -> bool:
        import pearl_mining as pm

        try:
            proof = pm.PlainProof.from_base64(proof_b64)
            ok, _message = pm.verify_plain_proof_for_cert_version(
                3,
                pm.IncompleteBlockHeader.from_bytes(job.header),
                proof,
                nbits_override=self.share_nbits,
            )
            return bool(ok)
        except Exception:
            return False

    async def _reply(self, writer: asyncio.StreamWriter, request_id: Any, result: Any, error: Any) -> None:
        writer.write(json.dumps({"id": request_id, "result": result, "error": error}, separators=(",", ":")).encode("utf-8") + b"\n")
        await writer.drain()


def corrupted_proof_gate_probe(pool: MockPool) -> bool:
    """Corrupt an actual accepted bincode proof and run the production local gate."""
    import pearl_mining as pm
    from pmk_miner.pipeline import FatalDeviceError, verify_gate
    if not pool.stats.accepted_records:
        return False
    job_id, encoded = pool.stats.accepted_records[0]
    original = pm.PlainProof.from_base64(encoded)
    cfg = pm.MiningConfiguration(
        original.k, original.noise_rank, pm.MMAType.Int7xInt7ToInt32,
        pm.PeriodicPattern.from_list([int(x) - int(original.a.row_indices[0]) for x in original.a.row_indices]),
        pm.PeriodicPattern.from_list([int(x) - int(original.bt.row_indices[0]) for x in original.bt.row_indices]),
        None,
    )
    header = pool.jobs[job_id].header
    config = bytes(cfg.to_bytes())
    verify_gate(header, original, config, pool.share_nbits)
    raw = bytearray(base64.b64decode(encoded, validate=True))
    # Change the committed A root, leaving bincode lengths and the MoE tag intact.
    root = bytes(original.a.root)
    if raw.count(root) != 1:
        raise RuntimeError("accepted proof does not contain one unambiguous A root")
    raw[raw.index(root)] ^= 1
    damaged = pm.PlainProof.from_base64(base64.b64encode(raw).decode('ascii'))
    try:
        verify_gate(header, damaged, config, pool.share_nbits)
    except FatalDeviceError:
        return True
    return False


async def direct_wrong_job_probe(pool: MockPool) -> bool:
    reader, writer = await asyncio.open_connection(pool.host, pool.port)
    try:
        writer.write(json.dumps({"id": 1, "method": "mining.authorize", "params": {"wallet": DEFAULT_WALLET, "worker": "probe", "pass": "x", "agent": "pmk/probe"}}, separators=(",", ":")).encode() + b"\n")
        await writer.drain()
        for _ in range(2):
            line = await asyncio.wait_for(reader.readline(), 5)
            if json.loads(line.decode()).get("id") == 1:
                break
        writer.write(json.dumps({"id": 2, "method": "mining.submit", "params": {"job_id": "not-a-real-job", "plain_proof": base64.b64encode(b"bad").decode()}}, separators=(",", ":")).encode() + b"\n")
        await writer.drain()
        reply = json.loads((await asyncio.wait_for(reader.readline(), 5)).decode())
        return reply.get("result") is False and (reply.get("error") or {}).get("code") == 25
    finally:
        writer.close()
        await writer.wait_closed()


async def direct_mismatched_existing_job_probe(pool: MockPool) -> bool:
    if not pool.stats.accepted_records:
        return False
    original_job_id, proof_b64 = pool.stats.accepted_records[0]
    mismatched = next((job_id for job_id in pool.jobs if job_id != original_job_id), None)
    if mismatched is None:
        # A shortened run can find its first share before the rotation timer.
        # Retain a second genuine header to test job binding deterministically.
        mismatched = pool._new_job().job_id
    # This probe verifies that a real proof accepted for one retained job is not
    # accepted under another retained job ID. Remove the proof from the mock
    # duplicate cache so the duplicate-share guard does not mask the header/job
    # binding check.
    try:
        proof_digest = hashlib.sha256(base64.b64decode(proof_b64, validate=True)).hexdigest()
        pool.stats.proofs_seen.discard(proof_digest)
    except Exception:
        return False
    reader, writer = await asyncio.open_connection(pool.host, pool.port)
    try:
        writer.write(json.dumps({"id": 1, "method": "mining.authorize", "params": {"wallet": DEFAULT_WALLET, "worker": "probe", "pass": "x", "agent": "pmk/probe"}}, separators=(",", ":")).encode() + b"\n")
        await writer.drain()
        for _ in range(2):
            line = await asyncio.wait_for(reader.readline(), 5)
            if json.loads(line.decode()).get("id") == 1:
                break
        writer.write(json.dumps({"id": 2, "method": "mining.submit", "params": {"job_id": mismatched, "plain_proof": proof_b64}}, separators=(",", ":")).encode() + b"\n")
        await writer.drain()
        reply = json.loads((await asyncio.wait_for(reader.readline(), 5)).decode())
        return reply.get("result") is False and (reply.get("error") or {}).get("code") in {28, 29}
    finally:
        writer.close()
        await writer.wait_closed()


def make_run_root(parent: Path | None) -> Path:
    base = parent.expanduser() if parent is not None else Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="pmk-pool-", dir=base.resolve(strict=False)))
    run.chmod(0o700)
    return run


def make_config(path: Path, *, state_dir: Path, summary_file: Path, max_accepted: int, max_jobs: int, max_seconds: int, difficulty_floor: int, m: int, n: int, k: int, slots: int, kernel: str = "auto") -> None:
    write_text(
        path,
        f"""m = {m}
n = {n}
k = {k}
slots = {slots}

[run]
max_accepted = {max_accepted}
max_jobs = {max_jobs}
max_seconds = {max_seconds}
state_dir = "{state_dir}"
kernel = "{kernel}"

[pool]
difficulty_floor = {difficulty_floor}
summary_file = "{summary_file}"
""",
    )


def miner_command(pool_url: str, wallet_file: Path, allowlist: Path, worker: str, config: Path, kernel: str) -> list[str]:
    cmd = [
        str(PYTHON),
        "-m",
        "pmk_miner",
        "--mode",
        "pool",
        "--pool-url",
        pool_url,
        "--wallet-file",
        str(wallet_file),
        "--wallet-allowlist",
        str(allowlist),
        "--worker",
        worker,
        "--config",
        str(config),
    ]
    if kernel != "auto":
        cmd.extend(["--kernel", kernel])
    return cmd


def parse_json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            row = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def bound_trace_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for row in events:
        event = str(row.get("event", ""))
        if "bound" in event or any(key in row for key in ("share_nbits", "block_target", "native_function", "native_code")):
            trace.append(row)
    return trace


def signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
        return
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    try:
        process.send_signal(sig)
    except (ProcessLookupError, PermissionError):
        pass


def terminate(process: subprocess.Popen[str]) -> None:
    signal_process_group(process, signal.SIGTERM)
    deadline = time.time() + 10
    while process.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    signal_process_group(process, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        signal_process_group(process, signal.SIGKILL)
        process.wait(timeout=2)


async def run_miner_case(args: argparse.Namespace, *, reject_first: bool = False, malicious_cert: bool = False, malicious_target: bool = False) -> dict[str, Any]:
    pool = MockPool(
        difficulty=args.difficulty,
        rotate_seconds=args.rotate_seconds,
        block_difficulty=args.block_difficulty,
        block_nbits=args.block_nbits,
        reject_first_share=reject_first,
        malicious_cert=malicious_cert,
        malicious_target=malicious_target,
    )
    await pool.start()
    run = make_run_root(args.run_root)
    secure_dir(run / "logs")
    wallet_file = run / "wallet.txt"
    allowlist = run / "wallet-allowlist.txt"
    config = run / "pmk_pool.toml"
    summary_file = run / "pool_summary.json"
    miner_log = run / "logs" / "miner.log"
    write_text(wallet_file, args.wallet + "\n")
    write_text(allowlist, args.wallet + "\n")
    make_config(
        config,
        state_dir=run / "state",
        summary_file=summary_file,
        max_accepted=args.target_shares,
        max_jobs=args.target_completed_jobs if args.dispatch_only else 0,
        max_seconds=args.max_seconds,
        difficulty_floor=args.difficulty_floor,
        m=args.m,
        n=args.n,
        k=args.k,
        slots=args.slots,
        kernel=args.kernel,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'miner'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.pop("PMK_GPU_LOCK_HELD", None)
    proc: subprocess.Popen[str] | None = None
    try:
        cmd = miner_command(pool.url(), wallet_file, allowlist, args.worker, config, args.kernel)
        with miner_log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        deadline = time.time() + args.max_seconds + 45
        while time.time() < deadline and proc.poll() is None:
            await asyncio.sleep(0.5)
            if malicious_cert or malicious_target or reject_first:
                events = parse_json_events(miner_log)
                if any(row.get("event") == "fatal" for row in events):
                    break
        if proc.poll() is None:
            if not (malicious_cert or malicious_target or reject_first):
                raise RuntimeError(f"miner did not exit before timeout; log tail:\n{miner_log.read_text(errors='replace')[-8000:]}")
            terminate(proc)
        rc = proc.wait(timeout=10)
        events = parse_json_events(miner_log)
        summary = {}
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        elif events:
            summaries = [row for row in events if row.get("event") == "pool_summary"]
            summary = summaries[-1] if summaries else {}
        job = pool.current_job()
        result = {
            "returncode": rc,
            "run_root": str(run),
            "miner_log": str(miner_log),
            "run_config": {
                "difficulty": args.difficulty,
                "target": f"{pool.target:064x}",
                "share_nbits": pool.share_nbits,
                "block_nbits": int.from_bytes(job.header[72:76], "little"),
                "block_nbits_hex": f"0x{int.from_bytes(job.header[72:76], 'little'):08x}",
                "job_id": job.job_id,
                "height": job.height,
                "cert_version": job.cert_version,
                "shape": {"m": args.m, "n": args.n, "k": args.k, "slots": args.slots},
                "kernel": args.kernel,
            },
            "pool": {
                "accepted": pool.stats.accepted,
                "stale": pool.stats.stale,
                "duplicate": pool.stats.duplicate,
                "low_difficulty": pool.stats.low_difficulty,
                "invalid": pool.stats.invalid,
                "wrong_job_id": pool.stats.wrong_job_id,
                "submits": pool.stats.submits,
                "authorized": pool.stats.authorized,
            },
            "summary": summary,
            "fatal_events": [row for row in events if row.get("event") == "fatal"],
            "alert_events": [row for row in events if row.get("event") == "alert"],
            "work_completed_events": [row for row in events if row.get("event") == "work_completed"],
            "native_bound_trace": bound_trace_events(events),
            "mismatched_existing_job_rejected": False,
            "corrupted_proof_gate_caught": False,
        }
        if not (reject_first or malicious_cert or malicious_target):
            result["corrupted_proof_gate_caught"] = corrupted_proof_gate_probe(pool)
            result["mismatched_existing_job_rejected"] = await direct_mismatched_existing_job_probe(pool)
        # Preserve failed-process logs, including crashes after a valid summary.
        if not args.keep_run and rc == 0:
            shutil.rmtree(run, ignore_errors=True)
        return result
    finally:
        if proc is not None and proc.poll() is None:
            terminate(proc)
        await pool.stop()


def assert_positive(result: dict[str, Any], target_shares: int) -> None:
    summary = result.get("summary") or {}
    pool = result["pool"]
    required = {'accepted', 'submitted', 'stale', 'rejected', 'gate_failures',
                'expected', 'observed', 'lower', 'upper', 'completed_macs', 'poisson_ok', 'failed'}
    if not required <= summary.keys():
        raise RuntimeError('miner omitted required pool summary evidence')
    if summary['failed'] or not summary['poisson_ok'] or summary['completed_macs'] <= 0:
        raise RuntimeError('miner summary failed correctness/Poisson criteria')
    if pool['submits'] != pool['accepted'] or summary['submitted'] != pool['accepted']:
        raise RuntimeError('not every positive-run submission was accepted')
    accepted = int(summary.get("accepted", pool["accepted"]))
    stale = int(summary.get("stale", pool["stale"]))
    rejected = int(summary.get("rejected", pool["invalid"] + pool["low_difficulty"]))
    gate_failures = int(summary.get("gate_failures", 0))
    expected = float(summary.get("expected_shares", summary.get("expected", accepted)))
    fallback_lower, fallback_upper = poisson_interval(expected)
    lower = int(summary.get("poisson_lower", summary.get("lower", fallback_lower)))
    upper = int(summary.get("poisson_upper", summary.get("upper", fallback_upper)))
    observed = int(summary.get("observed_shares", summary.get("observed", accepted + stale + rejected)))
    if result["returncode"] != 0:
        raise RuntimeError(f"miner exited {result['returncode']}; log={result['miner_log']}")
    if accepted < target_shares:
        raise RuntimeError(f"accepted shares {accepted} < {target_shares}")
    if stale != 0 or rejected != 0 or gate_failures != 0:
        raise RuntimeError(f"unexpected outcomes stale={stale} rejected={rejected} gate={gate_failures}")
    if pool["accepted"] != accepted or pool["low_difficulty"] or pool["invalid"]:
        raise RuntimeError(f"mock pool verdict mismatch: {pool}")
    if not lower <= observed <= upper:
        raise RuntimeError(f"Poisson check failed expected={expected} observed={observed} interval=[{lower},{upper}]")


def assert_dispatch(result: dict[str, Any], target_completed_jobs: int) -> None:
    summary = result.get("summary") or {}
    completed = int(summary.get("completed_jobs", 0))
    if result["returncode"] != 0:
        raise RuntimeError(f"miner exited {result['returncode']}; log={result['miner_log']}")
    if result.get("fatal_events"):
        raise RuntimeError(f"dispatch run emitted fatal events: {result['fatal_events']}")
    gate_alerts = [row for row in result.get("alert_events", []) if row.get("reason") == "device_or_verifier_gate"]
    if gate_alerts:
        raise RuntimeError(f"dispatch run emitted device/verifier gate alerts: {gate_alerts}")
    if completed < target_completed_jobs:
        raise RuntimeError(f"completed jobs {completed} < {target_completed_jobs}")
    if int(summary.get("max_jobs", 0)) != target_completed_jobs:
        raise RuntimeError(f"summary max_jobs {summary.get('max_jobs')} != {target_completed_jobs}")
    if int(summary.get("completed_ops", 0)) <= 0:
        raise RuntimeError("dispatch summary omitted completed work")


async def async_main(args: argparse.Namespace) -> int:
    if not PYTHON.exists():
        raise SystemExit(f"missing {PYTHON}; run scripts/setup_env.sh first")
    positive = await run_miner_case(args)
    if positive['returncode'] != 0:
        log('failed_run=' + json.dumps(positive, sort_keys=True))
    if args.dispatch_only:
        assert_dispatch(positive, args.target_completed_jobs)
        log(
            "RESULT dispatch_completed_jobs="
            f"{(positive.get('summary') or {}).get('completed_jobs', 0)} "
            f"completed_ops={(positive.get('summary') or {}).get('completed_ops', 0)} "
            f"difficulty={(positive.get('run_config') or {}).get('difficulty')} "
            f"block_nbits={(positive.get('run_config') or {}).get('block_nbits_hex')} "
            f"fatal_events={len(positive.get('fatal_events') or [])} "
            f"device_or_verifier_alerts={sum(1 for row in positive.get('alert_events', []) if row.get('reason') == 'device_or_verifier_gate')}"
        )
        return 0
    assert_positive(positive, args.target_shares)
    if not positive.get("corrupted_proof_gate_caught"):
        raise RuntimeError("corrupted actual proof passed the local verifier gate")
    if not positive.get("mismatched_existing_job_rejected"):
        raise RuntimeError("mock pool did not reject a valid proof under a different retained job_id")

    if args.positive_only:
        log(
            "RESULT accepted="
            f"{positive['pool']['accepted']} stale={positive['pool']['stale']} rejected="
            f"{positive['pool']['invalid'] + positive['pool']['low_difficulty']} "
            f"corrupted_proof_gate_caught={positive.get('corrupted_proof_gate_caught')} "
            f"mismatched_existing_job_rejected={positive.get('mismatched_existing_job_rejected')}"
        )
        return 0

    wrong_pool = MockPool(difficulty=args.difficulty, rotate_seconds=args.rotate_seconds,
                          block_difficulty=args.block_difficulty, block_nbits=args.block_nbits)
    await wrong_pool.start()
    try:
        wrong_job_rejected = await direct_wrong_job_probe(wrong_pool)
    finally:
        await wrong_pool.stop()
    if not wrong_job_rejected:
        raise RuntimeError("mock pool did not reject a mismatched job_id")

    first_reject = await run_miner_case(args, reject_first=True)
    first_reject_reasons = [str(row.get("reason", "")) for row in first_reject.get("alert_events", [])]
    first_reject_stopped = (
        first_reject["returncode"] != 0
        and first_reject["pool"]["invalid"] >= 1
        and "pool rejected SG config; try next pool" in first_reject_reasons
    )
    cert_refused = await run_miner_case(args, malicious_cert=True)
    target_refused = await run_miner_case(args, malicious_target=True)
    if not first_reject_stopped:
        raise RuntimeError(f"first-share rejection did not stop miner: {first_reject}")
    if cert_refused["returncode"] == 0:
        raise RuntimeError(f"malicious cert_version was not refused: {cert_refused}")
    if target_refused["returncode"] == 0:
        raise RuntimeError(f"malicious target was not refused: {target_refused}")

    evidence = {
        "positive": positive,
        "negative": {
            "wrong_job_id_rejected": wrong_job_rejected,
            "corrupted_proof_gate_caught": positive.get("corrupted_proof_gate_caught"),
            "mismatched_existing_job_rejected": positive.get("mismatched_existing_job_rejected"),
            "first_reject": first_reject,
            "malicious_cert": cert_refused,
            "malicious_target": target_refused,
        },
        "wallet": redact_wallet(args.wallet),
        "difficulty": args.difficulty,
        "block_nbits": args.block_nbits,
        "target_shares": args.target_shares,
    }
    if args.evidence_dir is not None:
        evidence_path = args.evidence_dir / "b6_t1_summary.json"
        write_text(evidence_path, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    log(
        "RESULT accepted="
        f"{positive['pool']['accepted']} stale={positive['pool']['stale']} rejected="
        f"{positive['pool']['invalid'] + positive['pool']['low_difficulty']} "
        f"corrupted_proof_gate_caught={positive.get('corrupted_proof_gate_caught')} "
        f"wrong_job_id_rejected={wrong_job_rejected} first_reject_stopped={first_reject_stopped} "
        f"mismatched_existing_job_rejected={positive.get('mismatched_existing_job_rejected')} "
        f"malicious_cert_refused={cert_refused['returncode'] != 0} malicious_target_refused={target_refused['returncode'] != 0}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-shares", type=int, default=int(os.environ.get("PMK_POOL_T1_TARGET_SHARES", "50")))
    parser.add_argument("--max-seconds", type=int, default=int(os.environ.get("PMK_POOL_T1_MAX_SECONDS", "900")))
    parser.add_argument("--difficulty", type=int, default=int(os.environ.get("PMK_POOL_T1_DIFFICULTY", "1000")))
    parser.add_argument("--difficulty-floor", type=int, default=int(os.environ.get("PMK_POOL_T1_DIFFICULTY_FLOOR", "1000")))
    parser.add_argument("--block-difficulty", type=int, default=int(os.environ.get("PMK_POOL_T1_BLOCK_DIFFICULTY", "1000000000")))
    parser.add_argument("--block-nbits", type=lambda value: int(value, 0),
                        default=int(os.environ["PMK_POOL_T1_BLOCK_NBITS"], 0) if os.environ.get("PMK_POOL_T1_BLOCK_NBITS") else None)
    parser.add_argument("--rotate-seconds", type=float, default=float(os.environ.get("PMK_POOL_T1_ROTATE_SECONDS", "15")))
    parser.add_argument("--wallet", default=os.environ.get("PMK_POOL_T1_WALLET", DEFAULT_WALLET))
    parser.add_argument("--worker", default=os.environ.get("PMK_POOL_T1_WORKER", "pmk-t1"))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ["PMK_POOL_T1_RUN_ROOT"]) if os.environ.get("PMK_POOL_T1_RUN_ROOT") else None)
    parser.add_argument("--evidence-dir", type=Path, default=Path(os.environ["PMK_POOL_T1_EVIDENCE_DIR"]) if os.environ.get("PMK_POOL_T1_EVIDENCE_DIR") else None)
    parser.add_argument("--keep-run", action="store_true")
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--target-completed-jobs", type=int, default=int(os.environ.get("PMK_POOL_T1_TARGET_COMPLETED_JOBS", "0")))
    parser.add_argument("--m", type=int, default=int(os.environ.get("PMK_POOL_T1_M", "4096")))
    parser.add_argument("--n", type=int, default=int(os.environ.get("PMK_POOL_T1_N", "4096")))
    parser.add_argument("--k", type=int, default=int(os.environ.get("PMK_POOL_T1_K", "4096")))
    parser.add_argument("--slots", type=int, default=int(os.environ.get("PMK_POOL_T1_SLOTS", "2")))
    parser.add_argument("--kernel", choices=("auto", "sg", "na"), default=os.environ.get("PMK_KERNEL", "auto"),
                        help="miner kernel override for cert-v3 pool runs")
    args = parser.parse_args()
    if args.dispatch_only and args.target_completed_jobs <= 0:
        parser.error("--dispatch-only requires --target-completed-jobs > 0")
    return args


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
