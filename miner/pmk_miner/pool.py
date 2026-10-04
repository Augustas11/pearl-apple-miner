# SPDX-License-Identifier: Apache-2.0
"""Object-dialect Pearl pool transport for pmk pool mode."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import random
import re
import socket
import ssl
import time
import urllib.parse
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from .monitor import DIFF1_TARGET, bits_to_target, target_to_bits_floor
from .v4_admission import validate_v4_g3_admission_file


CERT_VERSION_ZK_V3 = 3
CERT_VERSION_FP8_V4 = 4
HEADER_LEN = 76
ANCESTOR_HEADER_LEN = 108
MAX_U256 = (1 << 256) - 1
MAX_POOL_LINE = 64 * 1024
DEFAULT_REPLY_TIMEOUT_SECONDS = 30.0
DEFAULT_SILENCE_TIMEOUT_SECONDS = 180.0
MIN_SILENCE_TIMEOUT_SECONDS = 30.0
MAX_SILENCE_TIMEOUT_SECONDS = 1800.0
DEFAULT_DIFFICULTY_FLOOR = 10_000
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_sleep = asyncio.sleep


class PoolProtocolError(RuntimeError):
    """Raised when the pool sends a malformed or unsafe message."""


class PoolTransportError(RuntimeError):
    """Raised for pool transport failures."""


class PoolSubmitOutcome(StrEnum):
    ACCEPTED = "accepted"
    STALE = "stale"
    DUPLICATE = "duplicate"
    LOW_DIFFICULTY = "low-difficulty"
    INVALID = "invalid"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class PoolEndpoint:
    host: str
    port: int
    tls: bool = False

    @classmethod
    def parse(cls, value: str) -> "PoolEndpoint":
        parsed = urllib.parse.urlparse(value)
        scheme = parsed.scheme.lower()
        if scheme == "stratum+tcp":
            tls = False
        elif scheme in {"stratum+ssl", "stratum+tls", "tls"}:
            tls = True
        else:
            raise ValueError("pool URL must use stratum+tcp, stratum+ssl, stratum+tls or tls")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("pool URL must contain only scheme, host and port")
        if not parsed.hostname:
            raise ValueError("pool URL must include a host")
        if parsed.port is None or not 1 <= parsed.port <= 65535:
            raise ValueError("pool URL must include a port")
        return cls(parsed.hostname, parsed.port, tls)


@dataclass(frozen=True, slots=True)
class PoolJob:
    pool_job_id: str
    header: bytes
    target: int
    block_target: int
    cert_version: int
    height: int | None
    share_nbits: int
    session_id: int
    ancestor_headers: tuple[bytes, ...] = ()

    @property
    def bits(self) -> int:
        return int.from_bytes(self.header[72:76], "little")

    @property
    def header_hex(self) -> str:
        return self.header.hex()

    @property
    def template_identity(self) -> str:
        if not self.ancestor_headers:
            return self.header.hex()
        import hashlib
        ancestors = hashlib.blake2s(b"".join(self.ancestor_headers), digest_size=16).hexdigest()
        return f"{self.cert_version}:{ancestors}:{self.header.hex()}"


@dataclass(frozen=True, slots=True)
class PoolMessage:
    id: int | str | None
    method: str | None
    params: dict[str, Any] | None
    result: Any = None
    error: Any = None


def mask_wallet(wallet: str) -> str:
    if len(wallet) <= 12:
        return "<redacted>"
    return f"{wallet[:6]}...{wallet[-6:]}"


def pool_target_for_difficulty(difficulty: int) -> int:
    if difficulty <= 0:
        raise ValueError("difficulty must be positive")
    return DIFF1_TARGET // difficulty


def target_to_share_nbits(target: int) -> int:
    bits = target_to_bits_floor(target)
    if bits_to_target(bits) > target:
        raise AssertionError("compact share target is easier than pool target")
    return bits


def parse_pool_message(line: bytes, *, max_line: int = MAX_POOL_LINE) -> PoolMessage:
    if len(line) > max_line:
        raise PoolProtocolError("pool line exceeds 64 KiB")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PoolProtocolError("pool line is not UTF-8") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PoolProtocolError("pool line is not JSON") from exc
    if not isinstance(data, dict):
        raise PoolProtocolError("pool message must be a JSON object")
    msg_id = data.get("id")
    if msg_id is not None and (
        isinstance(msg_id, bool) or not isinstance(msg_id, (int, str))
    ):
        raise PoolProtocolError("pool message id must be null, string or integer")
    method = data.get("method")
    if method is not None and not isinstance(method, str):
        raise PoolProtocolError("pool message method must be a string")
    params = data.get("params")
    if params is not None and not isinstance(params, dict):
        raise PoolProtocolError("pool message params must be an object")
    return PoolMessage(
        id=msg_id,
        method=method,
        params=params,
        result=data.get("result"),
        error=data.get("error"),
    )


def validate_pool_notify(
    params: dict[str, Any],
    *,
    session_id: int,
    difficulty_floor: int = DEFAULT_DIFFICULTY_FLOOR,
    log: Callable[..., None] | None = None,
) -> PoolJob:
    if isinstance(difficulty_floor, bool) or not isinstance(difficulty_floor, int):
        raise ValueError("difficulty_floor must be a positive integer")
    if difficulty_floor <= 0:
        raise ValueError("difficulty_floor must be a positive integer")
    if not isinstance(params, dict):
        raise PoolProtocolError("notify params must be an object")
    forbidden_config_keys = {"m", "n", "k", "rank", "rows_pattern", "cols_pattern", "config"}
    if forbidden_config_keys.intersection(params):
        raise PoolProtocolError("pool notify must not include mining configuration")
    header_hex = params.get("header")
    target_hex = params.get("target")
    pool_job_id = params.get("job_id")
    cert_version = params.get("cert_version")
    ancestor_headers_raw = params.get("ancestor_headers")
    height = params.get("height")
    if not isinstance(header_hex, str) or len(header_hex) != HEADER_LEN * 2:
        raise PoolProtocolError("notify header must be exactly 152 hex characters")
    if not _HEX_RE.fullmatch(header_hex):
        raise PoolProtocolError("notify header must be hex")
    if not isinstance(target_hex, str) or len(target_hex) != 64:
        raise PoolProtocolError("notify target must be exactly 64 hex characters")
    if not _HEX_RE.fullmatch(target_hex):
        raise PoolProtocolError("notify target must be hex")
    if not isinstance(pool_job_id, str) or len(pool_job_id) > 64:
        raise PoolProtocolError("notify job_id must be at most 64 printable ASCII characters")
    if not all(0x20 <= ord(ch) <= 0x7E for ch in pool_job_id):
        raise PoolProtocolError("notify job_id must be printable ASCII")
    if isinstance(cert_version, bool) or cert_version not in (CERT_VERSION_ZK_V3, CERT_VERSION_FP8_V4):
        raise PoolProtocolError("pool job requires cert_version=3 or cert_version=4")
    ancestor_headers: tuple[bytes, ...] = ()
    if cert_version == CERT_VERSION_FP8_V4:
        try:
            validate_v4_g3_admission_file()
        except ValueError as exc:
            raise PoolProtocolError(str(exc)) from exc
        if not isinstance(ancestor_headers_raw, list) or not ancestor_headers_raw:
            raise PoolProtocolError("cert_version=4 pool job requires ancestor_headers")
        try:
            ancestor_headers = tuple(
                base64.b64decode(item, validate=True) for item in ancestor_headers_raw
            )
        except (TypeError, ValueError) as exc:
            raise PoolProtocolError("ancestor_headers must be base64 complete headers") from exc
        if len(ancestor_headers) > 4 or any(len(item) != ANCESTOR_HEADER_LEN for item in ancestor_headers):
            raise PoolProtocolError("ancestor_headers must contain 108-byte complete headers")
    elif ancestor_headers_raw is not None:
        raise PoolProtocolError("ancestor_headers are only valid for cert_version=4")
    if height is not None and (isinstance(height, bool) or not isinstance(height, int)):
        raise PoolProtocolError("notify height must be an integer")

    header = bytes.fromhex(header_hex)
    target = int(target_hex, 16)
    if target <= 0:
        raise PoolProtocolError("notify target must be nonzero")
    if target > MAX_U256 or target > DIFF1_TARGET:
        raise PoolProtocolError("notify target overflows uint256")
    if DIFF1_TARGET < difficulty_floor * target:
        raise PoolProtocolError("pool target is below configured difficulty floor")

    block_nbits = int.from_bytes(header[72:76], "little")
    if not _valid_compact_bits(block_nbits):
        raise PoolProtocolError("notify header contains invalid nbits")
    block_target = bits_to_target(block_nbits)
    if block_target <= 0 or block_target > DIFF1_TARGET:
        raise PoolProtocolError("notify header contains invalid nbits")
    if target < block_target and log is not None:
        log("pool_target_tighter_than_block_target", pool_job_id=pool_job_id)

    share_nbits = target_to_share_nbits(target)
    return PoolJob(
        pool_job_id=pool_job_id,
        header=header,
        target=target,
        block_target=block_target,
        cert_version=cert_version,
        height=height,
        share_nbits=share_nbits,
        session_id=session_id,
        ancestor_headers=ancestor_headers,
    )


def classify_submit_response(message: PoolMessage) -> PoolSubmitOutcome:
    text = f"{message.result!r} {message.error!r}".lower()
    if message.error is None and message.result is True:
        return PoolSubmitOutcome.ACCEPTED
    if any(term in text for term in ("stale", "old job", "unknown job", "job not found", "expired job")):
        return PoolSubmitOutcome.STALE
    if any(term in text for term in ("duplicate", "already submitted", "already received")):
        return PoolSubmitOutcome.DUPLICATE
    if any(term in text for term in ("low difficulty", "low-difficulty", "difficulty too low", "insufficient difficulty")):
        return PoolSubmitOutcome.LOW_DIFFICULTY
    if "timeout" in text:
        return PoolSubmitOutcome.TIMEOUT
    if "transport" in text or "connection" in text:
        return PoolSubmitOutcome.TRANSPORT
    return PoolSubmitOutcome.INVALID


class PoolClient:
    """Async object-dialect Stratum client.

    The mining pipeline should always submit the immutable PoolJob it mined
    against; submit() uses that job's pool_job_id and never rewrites it to the
    newest notify.
    """

    def __init__(
        self,
        url: str,
        wallet: str,
        worker: str,
        *,
        difficulty_floor: int = DEFAULT_DIFFICULTY_FLOOR,
        agent: str = "pmk/0.1.0",
        log: Callable[..., None] | None = None,
        reply_timeout: float = DEFAULT_REPLY_TIMEOUT_SECONDS,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT_SECONDS,
        notify_interval: float = 1.0,
    ) -> None:
        if not wallet:
            raise ValueError("wallet must be nonempty")
        if not worker:
            raise ValueError("worker must be nonempty")
        if (
            isinstance(silence_timeout, bool)
            or not isinstance(silence_timeout, (int, float))
            or not MIN_SILENCE_TIMEOUT_SECONDS <= silence_timeout <= MAX_SILENCE_TIMEOUT_SECONDS
        ):
            raise ValueError("silence_timeout must be between 30 and 1800 seconds")
        self.endpoint = PoolEndpoint.parse(url)
        self.wallet = wallet
        self.worker = worker
        self.difficulty_floor = difficulty_floor
        self.agent = agent
        self.log = log or (lambda *args, **kwargs: None)
        self.reply_timeout = reply_timeout
        self.silence_timeout = float(silence_timeout)
        self._silence_timeout_seconds = self.silence_timeout
        self._last_valid_inbound = 0.0
        self.notify_interval = notify_interval
        self.latest: PoolJob | None = None
        self.session_id = 0
        self.fatal: BaseException | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._coalesce_task: asyncio.Task[None] | None = None
        self._pending: dict[int | str, asyncio.Future[PoolMessage]] = {}
        self._request_id = 0
        self._pending_notify: PoolJob | None = None
        self._last_publish = 0.0
        self._submit_semaphore = asyncio.Semaphore(4)
        self._submit_rate_lock = asyncio.Lock()
        self._submit_times: deque[float] = deque()
        self._accepted_or_rejected = 0
        self._authorized = False
        self._prev_by_height: OrderedDict[int, bytes] = OrderedDict()
        self._session_invalidated = asyncio.Event()

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            self._disconnect_session(invalidate=False)
            try:
                await self._connect_authorize()
                assert self._reader_task is not None
                stop_task = asyncio.create_task(stop.wait())
                try:
                    done, _ = await asyncio.wait(
                        {stop_task, self._reader_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if self._reader_task in done and not stop.is_set():
                        await self._reader_task
                finally:
                    stop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stop_task
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(exc, PoolProtocolError):
                    self.fatal = exc
                    raise
                self._log("pool_transport_error", error=type(exc).__name__)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=_jitter(backoff))
                except asyncio.TimeoutError:
                    pass
                backoff = min(60.0, backoff * 2.0)
            finally:
                await self.close()

    async def close(self) -> None:
        if self._reader_task is not None:
            if not self._reader_task.done():
                self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if self.fatal is None and not isinstance(exc, PoolTransportError):
                    self.fatal = exc
            self._reader_task = None
        if self._coalesce_task is not None:
            self._coalesce_task.cancel()
            try:
                await self._coalesce_task
            except asyncio.CancelledError:
                pass
            self._coalesce_task = None
        writer = self._writer
        self._writer = None
        self._reader = None
        self._authorized = False
        if writer is not None:
            writer.close()
            with contextlib.suppress(asyncio.TimeoutError, OSError):
                await asyncio.wait_for(writer.wait_closed(), timeout=min(5.0, self.reply_timeout))
        self._disconnect_session(invalidate=True)

    async def submit(self, job: PoolJob, proof: str | bytes) -> str:
        if job.session_id != self.session_id:
            return PoolSubmitOutcome.TRANSPORT.value
        async with self._submit_semaphore:
            if not await self._wait_submit_rate_limit(job.session_id):
                return PoolSubmitOutcome.TRANSPORT.value
            if job.session_id != self.session_id:
                return PoolSubmitOutcome.TRANSPORT.value
            try:
                message = await self._rpc(
                    "mining.submit",
                    {
                        "job_id": job.pool_job_id,
                        "plain_proof": _proof_text(proof),
                    },
                )
            except asyncio.TimeoutError:
                return PoolSubmitOutcome.TIMEOUT.value
            except PoolTransportError:
                return PoolSubmitOutcome.TRANSPORT.value
            outcome = classify_submit_response(message)
            self._accepted_or_rejected += 1
            return outcome.value

    async def _connect_authorize(self) -> None:
        ssl_ctx = ssl.create_default_context() if self.endpoint.tls else None
        self.session_id += 1
        self._session_invalidated = asyncio.Event()
        self.latest = None
        self._pending_notify = None
        self._authorized = False
        self._prev_by_height.clear()
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(
                self.endpoint.host,
                self.endpoint.port,
                ssl=ssl_ctx,
                limit=MAX_POOL_LINE + 1,
            ),
            timeout=self.reply_timeout,
        )
        self._configure_socket_keepalive()
        self._last_valid_inbound = time.monotonic()
        self._reader_task = asyncio.create_task(self._read_loop())
        reply = await self._rpc(
            "mining.authorize",
            {
                "wallet": self.wallet,
                "worker": self.worker,
                "pass": "x",
                "agent": self.agent,
            },
        )
        if reply.error is not None or reply.result is not True:
            raise PoolTransportError("pool authorization failed")
        self._authorized = True
        self._log("pool_authorized", wallet=mask_wallet(self.wallet), worker=self.worker)
        self._publish_pending_notify()

    async def _rpc(self, method: str, params: dict[str, Any]) -> PoolMessage:
        writer = self._writer
        if writer is None:
            raise PoolTransportError("pool is not connected")
        self._request_id += 1
        request_id = self._request_id
        future: asyncio.Future[PoolMessage] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = {
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            try:
                async with asyncio.timeout(self.reply_timeout):
                    writer.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
                    await writer.drain()
                    return await future
            except OSError as exc:
                raise PoolTransportError("pool write failed") from exc
        finally:
            self._pending.pop(request_id, None)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                try:
                    remaining = self._silence_timeout_seconds - (
                        time.monotonic() - self._last_valid_inbound
                    )
                    if remaining <= 0:
                        raise PoolTransportError("pool silence timeout")
                    raw = await asyncio.wait_for(
                        self._reader.readuntil(b"\n"), timeout=remaining
                    )
                except asyncio.TimeoutError as exc:
                    raise PoolTransportError("pool silence timeout") from exc
                except asyncio.LimitOverrunError as exc:
                    raise PoolTransportError("pool line exceeds 64 KiB") from exc
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        raise PoolTransportError("pool closed mid-frame") from exc
                    raise PoolTransportError("pool closed connection") from exc
                try:
                    message = parse_pool_message(raw)
                except PoolProtocolError as exc:
                    raise PoolTransportError(str(exc)) from exc
                if message.method in {
                    "pearl.set_mining_params",
                    "mining.set_difficulty",
                    "mining.set_target",
                }:
                    self.fatal = PoolProtocolError("pool attempted to configure mining parameters")
                    raise self.fatal
                if message.method == "mining.notify":
                    try:
                        self._handle_notify(message)
                    except PoolProtocolError as exc:
                        if "cert_version" in str(exc):
                            self._log("pool_job_rejected", reason="R-A6_cert_version")
                        self.fatal = exc
                        raise
                    self._last_valid_inbound = time.monotonic()
                    continue
                if message.id in self._pending:
                    future = self._pending[message.id]
                    if not future.done():
                        self._last_valid_inbound = time.monotonic()
                        future.set_result(message)
        except Exception as exc:
            self._invalidate_current_session()
            self._fail_pending(exc)
            raise

    def _handle_notify(self, message: PoolMessage) -> None:
        if message.params is None:
            raise PoolProtocolError("notify params must be an object")
        job = validate_pool_notify(
            message.params,
            session_id=self.session_id,
            difficulty_floor=self.difficulty_floor,
            log=self._log,
        )
        self._record_prev_hash(job)
        self._pending_notify = job
        if not self._authorized:
            return
        now = time.monotonic()
        if now - self._last_publish >= self.notify_interval:
            self._publish_pending_notify()
        elif self._coalesce_task is None or self._coalesce_task.done():
            self._coalesce_task = asyncio.create_task(
                self._publish_after(self.notify_interval - (now - self._last_publish))
            )

    async def _publish_after(self, delay: float) -> None:
        await _sleep(max(0.0, delay))
        self._publish_pending_notify()

    def _publish_pending_notify(self) -> None:
        if self._pending_notify is None:
            return
        self.latest = self._pending_notify
        self._pending_notify = None
        self._last_publish = time.monotonic()
        self._log("pool_job", pool_job_id=self.latest.pool_job_id, height=self.latest.height)

    def _disconnect_session(self, *, invalidate: bool) -> None:
        self.latest = None
        self._pending_notify = None
        self._authorized = False
        self._prev_by_height.clear()
        if invalidate:
            self.session_id += 1
        self._session_invalidated.set()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(PoolTransportError("pool disconnected"))
        self._pending.clear()

    def _invalidate_current_session(self) -> None:
        self.latest = None
        self._pending_notify = None
        self._authorized = False
        self._prev_by_height.clear()
        self.session_id += 1
        self._session_invalidated.set()
        if self._coalesce_task is not None and not self._coalesce_task.done():
            self._coalesce_task.cancel()

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)

    async def _wait_submit_rate_limit(self, session_id: int) -> bool:
        async with self._submit_rate_lock:
            while True:
                if session_id != self.session_id or self._session_invalidated.is_set():
                    return False
                now = time.monotonic()
                while self._submit_times and now - self._submit_times[0] >= 60.0:
                    self._submit_times.popleft()
                if len(self._submit_times) < 30:
                    self._submit_times.append(now)
                    return True
                delay = max(0.0, 60.0 - (now - self._submit_times[0]))
                sleep_task = asyncio.create_task(_sleep(delay))
                invalidated_task = asyncio.create_task(self._session_invalidated.wait())
                try:
                    done, pending = await asyncio.wait(
                        {sleep_task, invalidated_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    if invalidated_task in done or session_id != self.session_id:
                        return False
                finally:
                    for task in (sleep_task, invalidated_task):
                        if not task.done():
                            task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

    def _record_prev_hash(self, job: PoolJob) -> None:
        if job.height is None:
            return
        prev = job.header[4:36]
        prior = self._prev_by_height.get(job.height)
        if prior is None:
            self._prev_by_height[job.height] = prev
            self._prev_by_height.move_to_end(job.height)
            while len(self._prev_by_height) > 128:
                self._prev_by_height.popitem(last=False)
        elif prior != prev:
            self._log("pool_prev_block_inconsistent", height=job.height)

    def _configure_socket_keepalive(self) -> None:
        writer = self._writer
        if writer is None:
            return
        sock = writer.get_extra_info("socket")
        if sock is None:
            return
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except (AttributeError, OSError):
            return

        idle_seconds = max(1, min(60, int(self.silence_timeout / 3)))
        interval_seconds = max(1, min(10, int(self.silence_timeout / 6)))
        optional_options = (
            (getattr(socket, "TCP_KEEPALIVE", None), idle_seconds),
            (getattr(socket, "TCP_KEEPIDLE", None), idle_seconds),
            (getattr(socket, "TCP_KEEPINTVL", None), interval_seconds),
            (getattr(socket, "TCP_KEEPCNT", None), 3),
        )
        configured: set[int] = set()
        for option, value in optional_options:
            if option is None or option in configured:
                continue
            configured.add(option)
            with contextlib.suppress(AttributeError, OSError):
                sock.setsockopt(socket.IPPROTO_TCP, option, value)

    def _log(self, event: str, **kwargs: Any) -> None:
        self.log(event, **{key: self._sanitize_log_value(value) for key, value in kwargs.items()})

    def _sanitize_log_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self.wallet, mask_wallet(self.wallet))
        return value


def _proof_text(proof: str | bytes) -> str:
    if isinstance(proof, bytes):
        return base64.b64encode(proof).decode("ascii")
    return proof


def _valid_compact_bits(bits: int) -> bool:
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent == 0 or mantissa == 0:
        return False
    return (mantissa & 0x800000) == 0


def _jitter(backoff: float) -> float:
    return min(60.0, max(1.0, max(1.0, backoff) * random.uniform(0.75, 1.25)))
