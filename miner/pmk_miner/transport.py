# SPDX-License-Identifier: Apache-2.0
"""Gateway and node transport for pmk solo mining.

This module deliberately keeps the wire contracts small: the gateway speaks
newline-framed JSON-RPC, and the node confirmation path speaks pearld's HTTP
JSON-RPC through the read-only methods used for submission tracking.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .scheme import scheme_for_cert_version
from .v4_admission import validate_v4_g3_admission_file


CERT_VERSION_ZK_V3 = 3
HEADER_LEN = 76
MAX_U256 = (1 << 256) - 1
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_NODE_RPC_ALLOWLIST = frozenset({"getbestblockhash", "getblock"})
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TransportError(RuntimeError):
    """Raised for gateway or node transport failures."""


class FatalCertVersionError(RuntimeError):
    """Raised when a job requires an unsupported certificate version."""


class SubmissionOutcome(StrEnum):
    ACCEPTED = "accepted"
    CONSENSUS_INVALID = "consensus-invalid"
    STALE = "stale"
    DUPLICATE = "duplicate"
    LOW_DIFFICULTY = "low-difficulty"
    INVALID = "invalid"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    PROVING_ERROR = "proving-error"
    UNKNOWN_SUBMISSION = "unknown-submission"


class JobSource(Protocol):
    """Gateway-like source for solo jobs; pool mode can implement this later."""

    async def get_job(self) -> "GatewayJob": ...

    async def submit(self, job: "GatewayJob", proof_b64: str | bytes) -> "SubmitAck": ...


@dataclass(frozen=True, slots=True)
class GatewayEndpoint:
    """Loopback-only miner gateway endpoint."""

    host: str | None = None
    port: int | None = None
    socket_path: str | None = None

    @classmethod
    def parse(cls, value: str) -> "GatewayEndpoint":
        if value.startswith("unix:"):
            path = value.removeprefix("unix:")
            if not path:
                raise ValueError("empty unix socket path")
            return cls(socket_path=path)

        if ":" not in value:
            raise ValueError("gateway must be host:port or unix:/path")
        host, port_text = value.rsplit(":", 1)
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(f"gateway host must be loopback, got {host!r}")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError(f"invalid gateway port {port_text!r}") from exc
        if not (0 < port < 65536):
            raise ValueError(f"invalid gateway port {port}")
        return cls(host=host, port=port)


@dataclass(frozen=True, slots=True)
class GatewayJob:
    """Immutable mining job received from pearl-gateway."""

    header: bytes
    target: int
    cert_version: int
    coinbase_tx: bytes | None = None
    coinbase_merkle_branch: tuple[bytes, ...] = ()
    coinbase_index: int = 0
    submission_id: str | None = None
    ancestor_headers: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if len(self.header) != HEADER_LEN:
            raise ValueError(f"incomplete header must be {HEADER_LEN} bytes")
        try:
            scheme_for_cert_version(self.cert_version)
        except ValueError:
            raise FatalCertVersionError(
                f"gateway job requires unsupported cert_version={self.cert_version}"
            ) from None
        if type(self.target) is not int:
            raise ValueError("target must be a strict integer")
        if self.target <= 0 or self.target > MAX_U256:
            raise ValueError("target must be in 1..2^256-1")
        if self.target != _bits_to_target(self.bits):
            raise ValueError("gateway job target does not match header bits")
        if self.coinbase_tx is not None:
            object.__setattr__(self, "coinbase_tx", bytes(self.coinbase_tx))
            object.__setattr__(
                self, "coinbase_merkle_branch", tuple(bytes(x) for x in self.coinbase_merkle_branch)
            )
            if self.coinbase_index < 0:
                raise ValueError("coinbase index must be non-negative")
        object.__setattr__(self, "ancestor_headers", tuple(bytes(x) for x in self.ancestor_headers))
        if self.cert_version == 4:
            validate_v4_g3_admission_file()
            if not self.ancestor_headers:
                raise ValueError("v4 gateway job is missing ancestor_headers")
            if len(self.ancestor_headers) > 4 or any(len(x) != 108 for x in self.ancestor_headers):
                raise ValueError("v4 ancestor_headers must be 108-byte complete headers")
        elif self.ancestor_headers:
            raise ValueError("ancestor_headers are only valid for cert_version=4")
        if self.submission_id is not None and not re.fullmatch(r"[0-9a-f]{32}", self.submission_id):
            raise ValueError("submission_id must be 32 lowercase hex characters")

    @property
    def version(self) -> int:
        return int.from_bytes(self.header[0:4], "little")

    @property
    def prev_hash(self) -> str:
        return self.header[4:36][::-1].hex()

    @property
    def merkle_root(self) -> str:
        return self.header[36:68][::-1].hex()

    @property
    def timestamp(self) -> int:
        return int.from_bytes(self.header[68:72], "little")

    @property
    def bits(self) -> int:
        return int.from_bytes(self.header[72:76], "little")

    @property
    def bits_hex(self) -> str:
        return f"{self.bits:08x}"

    @property
    def template_identity(self) -> str:
        ancestors = hashlib.blake2s(b"".join(self.ancestor_headers), digest_size=16).hexdigest() if self.ancestor_headers else ""
        return f"{self.prev_hash}:{self.bits_hex}:{self.cert_version}:{ancestors}:{self.header.hex()}"

    @property
    def header_hex(self) -> str:
        return self.header.hex()

    @property
    def header_hash(self) -> str:
        return hashlib.blake2s(self.header, digest_size=16).hexdigest()

    def to_gateway_dict(self) -> dict[str, Any]:
        data = {
            "incomplete_header_bytes": base64.b64encode(self.header).decode("ascii"),
            "target": self.target,
            "cert_version": self.cert_version,
        }
        if self.coinbase_tx is not None:
            data["coinbase_tx"] = base64.b64encode(self.coinbase_tx).decode("ascii")
            data["coinbase_merkle_branch"] = [
                base64.b64encode(h).decode("ascii") for h in self.coinbase_merkle_branch
            ]
            data["coinbase_index"] = self.coinbase_index
        if self.submission_id is not None:
            data["submission_id"] = self.submission_id
        if self.ancestor_headers:
            data["ancestor_headers"] = [
                base64.b64encode(h).decode("ascii") for h in self.ancestor_headers
            ]
        return data

    @classmethod
    def from_gateway_dict(cls, data: dict[str, Any]) -> "GatewayJob":
        try:
            header = base64.b64decode(data["incomplete_header_bytes"], validate=True)
            target_raw = data["target"]
            cert_raw = data["cert_version"]
            if not _strict_int(target_raw):
                raise ValueError("target is not a strict integer")
            if not _strict_int(cert_raw):
                raise ValueError("cert_version is not a strict integer")
            target = int(target_raw)
            cert_version = int(cert_raw)
            coinbase_tx = None
            coinbase_merkle_branch: tuple[bytes, ...] = ()
            coinbase_index = 0
            if "coinbase_tx" in data:
                coinbase_tx = base64.b64decode(data["coinbase_tx"], validate=True)
                branch_raw = data.get("coinbase_merkle_branch", [])
                if not isinstance(branch_raw, list):
                    raise ValueError("coinbase_merkle_branch is not a list")
                coinbase_merkle_branch = tuple(
                    base64.b64decode(item, validate=True) for item in branch_raw
                )
                index_raw = data.get("coinbase_index", data.get("coinbase_merkle_index", 0))
                if not _strict_int(index_raw):
                    raise ValueError("coinbase_index is not a strict integer")
                coinbase_index = int(index_raw)
            ancestor_headers = ()
            if "ancestor_headers" in data:
                raw_ancestors = data["ancestor_headers"]
                if not isinstance(raw_ancestors, list):
                    raise ValueError("ancestor_headers is not a list")
                ancestor_headers = tuple(
                    base64.b64decode(item, validate=True) for item in raw_ancestors
                )
            submission_id = data.get("submission_id")
            if submission_id is not None and not isinstance(submission_id, str):
                raise ValueError("submission_id is not a string")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed gateway mining job") from exc
        return cls(
            header=header,
            target=target,
            cert_version=cert_version,
            coinbase_tx=coinbase_tx,
            coinbase_merkle_branch=coinbase_merkle_branch,
            coinbase_index=coinbase_index,
            submission_id=submission_id,
            ancestor_headers=ancestor_headers,
        )

    def with_submission_id(self, submission_id: str) -> "GatewayJob":
        return GatewayJob(
            header=self.header,
            target=self.target,
            cert_version=self.cert_version,
            coinbase_tx=self.coinbase_tx,
            coinbase_merkle_branch=self.coinbase_merkle_branch,
            coinbase_index=self.coinbase_index,
            submission_id=submission_id,
            ancestor_headers=self.ancestor_headers,
        )

    def authorize_coinbase(self, *, approved_script_hex: str) -> None:
        """Fail closed unless this job commits to an approved coinbase output."""

        if self.coinbase_tx is None:
            raise ValueError("gateway job is missing coinbase authorization data")
        authorize_coinbase_job(
            self.header,
            self.coinbase_tx,
            self.coinbase_merkle_branch,
            self.coinbase_index,
            approved_script_hex=approved_script_hex,
        )


@dataclass(frozen=True, slots=True)
class SubmitAck:
    result: str
    submission_id: str | None = None


class GatewayClient:
    """Async client for pearl-gateway's miner JSON-RPC endpoint."""

    def __init__(self, endpoint: GatewayEndpoint | str, *, timeout: float = 10.0) -> None:
        self.endpoint = GatewayEndpoint.parse(endpoint) if isinstance(endpoint, str) else endpoint
        self.timeout = timeout
        self._request_id = 0

    async def get_job(self) -> GatewayJob:
        result = await self._rpc("getMiningInfo", {})
        if not isinstance(result, dict):
            raise TransportError("getMiningInfo returned non-object result")
        return GatewayJob.from_gateway_dict(result)

    async def submit_plain_proof(self, job: GatewayJob, plain_proof: str | bytes) -> SubmitAck:
        if isinstance(plain_proof, bytes):
            proof_text = base64.b64encode(plain_proof).decode("ascii")
        else:
            proof_text = plain_proof
        result = await self._rpc(
            "submitPlainProof",
            {"plain_proof": proof_text, "mining_job": job.to_gateway_dict()},
        )
        submission_id = None
        if isinstance(result, dict):
            status = result.get("status")
            raw_submission_id = result.get("submission_id")
            if raw_submission_id is not None and not isinstance(raw_submission_id, str):
                raise TransportError("submitPlainProof returned non-string submission_id")
            submission_id = raw_submission_id
        else:
            status = result
        if status != "submitted":
            raise TransportError("submitPlainProof returned unexpected result")
        if job.submission_id is not None and submission_id != job.submission_id:
            raise TransportError("submitPlainProof submission_id mismatch")
        return SubmitAck(result=status, submission_id=submission_id)

    async def submit(self, job: GatewayJob, proof_b64: str | bytes) -> SubmitAck:
        return await self.submit_plain_proof(job, proof_b64)

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        self._request_id += 1
        request_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }
        line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"

        try:
            if self.endpoint.socket_path is not None:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(self.endpoint.socket_path), self.timeout
                )
            else:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.endpoint.host, self.endpoint.port), self.timeout
                )
        except OSError as exc:
            raise TransportError(f"gateway connection failed: {redact_credentials(str(exc))}") from None

        try:
            writer.write(line)
            await asyncio.wait_for(writer.drain(), self.timeout)
            response_line = await asyncio.wait_for(reader.readline(), self.timeout)
        finally:
            writer.close()
            await writer.wait_closed()

        if not response_line:
            raise TransportError("gateway closed connection without a response")
        try:
            response = json.loads(response_line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TransportError("gateway returned malformed JSON") from None
        if response.get("id") != request_id:
            raise TransportError("gateway response id mismatch")
        if response.get("error") is not None:
            raise TransportError(f"gateway JSON-RPC error: {_safe_rpc_error(response['error'])}")
        return response.get("result")


@dataclass(frozen=True, slots=True)
class NodeRpcConfig:
    rpc_url: str
    rpc_user: str
    rpc_password: str
    mining_address: str | None = None

    @classmethod
    def from_sources(
        cls,
        *,
        toml_path: str | os.PathLike[str] | None = None,
        env_path: str | os.PathLike[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "NodeRpcConfig":
        values: dict[str, Any] = {}
        if toml_path is not None:
            values.update(_read_toml_config(Path(toml_path)))
        if env_path is not None:
            values.update(_read_env_file(Path(env_path)))
        merged_env = os.environ if env is None else env
        for key in (
            "PEARLD_RPC_URL",
            "PEARLD_RPC_USER",
            "PEARLD_RPC_PASSWORD",
            "PEARLD_MINING_ADDRESS",
        ):
            if key in merged_env:
                values[key] = merged_env[key]

        try:
            return cls(
                rpc_url=str(values["PEARLD_RPC_URL"]),
                rpc_user=str(values["PEARLD_RPC_USER"]),
                rpc_password=str(values["PEARLD_RPC_PASSWORD"]),
                mining_address=(
                    str(values["PEARLD_MINING_ADDRESS"])
                    if values.get("PEARLD_MINING_ADDRESS") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ValueError(f"missing node RPC config value {exc.args[0]}") from exc


def _read_toml_config(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text())
    flat: dict[str, Any] = {}
    pearl = data.get("pearl", {})
    node = data.get("node_rpc", {})
    for source in (pearl, node, data):
        if not isinstance(source, dict):
            continue
        mapping = {
            "rpc_url": "PEARLD_RPC_URL",
            "rpc_user": "PEARLD_RPC_USER",
            "rpc_password": "PEARLD_RPC_PASSWORD",
            "mining_address": "PEARLD_MINING_ADDRESS",
        }
        for key, env_key in mapping.items():
            if key in source:
                flat[env_key] = source[key]
    return flat


def _read_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        result[key.strip()] = value
    return result


class NodeRpcClient:
    """Read-only pearld JSON-RPC client used by submission tracking."""

    def __init__(self, config: NodeRpcConfig, *, timeout: float = 10.0) -> None:
        self.config = config
        self.timeout = timeout
        self._request_id = 0
        parsed = urllib.parse.urlparse(config.rpc_url)
        if parsed.scheme != "http":
            raise ValueError("node RPC URL must use http on loopback")
        if parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError(f"node RPC URL must be loopback, got {parsed.hostname!r}")
        if parsed.username or parsed.password:
            raise ValueError("node RPC credentials must come from config, not URL userinfo")
        self._opener = urllib.request.build_opener(_NoRedirectHandler, urllib.request.ProxyHandler({}))

    async def call(self, method: str, params: list[Any] | None = None) -> Any:
        if method not in _NODE_RPC_ALLOWLIST:
            raise TransportError(f"node RPC method not allowlisted: {method}")
        return await asyncio.to_thread(self._call_sync, method, params or [])

    async def get_best_block_hash(self) -> str:
        return str(await self.call("getbestblockhash"))

    async def get_block(self, block_hash: str, verbosity: int = 1) -> dict[str, Any]:
        result = await self.call("getblock", [block_hash, verbosity])
        if not isinstance(result, dict):
            raise TransportError("getblock returned non-object result")
        return result

    def _call_sync(self, method: str, params: list[Any]) -> Any:
        self._request_id += 1
        payload = json.dumps(
            {"jsonrpc": "1.0", "id": self._request_id, "method": method, "params": params}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.rpc_url,
            data=payload,
            headers={"content-type": "text/plain"},
            method="POST",
        )
        token = base64.b64encode(
            f"{self.config.rpc_user}:{self.config.rpc_password}".encode("utf-8")
        ).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read()
        except Exception as exc:
            raise TransportError(f"node RPC transport failed: {type(exc).__name__}") from None
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TransportError("node RPC returned malformed JSON") from None
        if not isinstance(data, dict):
            raise TransportError("node RPC returned malformed envelope")
        if data.get("error") is not None:
            raise TransportError(
                f"node RPC error: {_safe_rpc_error(data['error'], self.config)}"
            ) from None
        return data.get("result")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(slots=True)
class GatewayLogTail:
    """Small gateway log classifier with offset tracking."""

    path: Path
    offset: int | None = None
    inode: int | None = None
    dev: int | None = None
    partial: str = ""

    def __post_init__(self) -> None:
        if self.path.exists():
            st = self.path.stat()
            self.inode = st.st_ino
            self.dev = st.st_dev
            if self.offset is None:
                self.offset = st.st_size
        elif self.offset is None:
            self.offset = 0

    def begin_job(self) -> int:
        if self.path.exists():
            st = self.path.stat()
            self.inode = st.st_ino
            self.dev = st.st_dev
            self.offset = st.st_size
        else:
            self.inode = None
            self.dev = None
            self.offset = 0
        return self.offset

    def read_events(self) -> list[str]:
        if not self.path.exists():
            return []
        st = self.path.stat()
        assert self.offset is not None
        chunks: list[str] = []
        if self.inode != st.st_ino or self.dev != st.st_dev or st.st_size < self.offset:
            chunks.extend(self._read_rotated_events())
            self.offset = 0
            self.inode = st.st_ino
            self.dev = st.st_dev
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self.offset)
            chunks.append(fh.read())
            self.offset = fh.tell()
        return self._complete_lines("".join(chunks))

    def _read_rotated_events(self) -> list[str]:
        if self.inode is None or self.dev is None or self.offset is None:
            return []
        chunks: list[str] = []
        for candidate in self.path.parent.glob(f"{self.path.name}*"):
            if candidate == self.path:
                continue
            try:
                st = candidate.stat()
            except OSError:
                continue
            if st.st_ino != self.inode or st.st_dev != self.dev:
                continue
            try:
                with candidate.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(min(self.offset, st.st_size))
                    chunks.append(fh.read())
            except OSError:
                continue
        return chunks

    def _complete_lines(self, text: str) -> list[str]:
        if not text:
            return []
        text = self.partial + text
        if text.endswith("\n"):
            self.partial = ""
            return text.splitlines()
        lines = text.splitlines()
        if not lines:
            self.partial = text
            return []
        self.partial = lines.pop()
        return lines


class SubmissionTracker:
    """Confirm and classify a submitted proof after gateway acknowledgement."""

    def __init__(
        self,
        node: NodeRpcClient,
        *,
        gateway_log: GatewayLogTail | None = None,
        timeout_seconds: float = 120.0,
        poll_seconds: float = 2.0,
        max_chain_walk: int = 64,
    ) -> None:
        self.node = node
        self.gateway_log = gateway_log
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.max_chain_walk = max_chain_walk

    async def confirm(
        self,
        job: GatewayJob,
        *,
        submitted_block_hash: str | None = None,
    ) -> SubmissionOutcome:
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return self._final_log_outcome(job) or SubmissionOutcome.UNKNOWN_SUBMISSION
            log_outcome = self._classify_gateway_logs(job)
            if log_outcome in {
                SubmissionOutcome.CONSENSUS_INVALID,
                SubmissionOutcome.DUPLICATE,
                SubmissionOutcome.PROVING_ERROR,
                SubmissionOutcome.STALE,
                SubmissionOutcome.TRANSPORT,
                SubmissionOutcome.ACCEPTED,
            }:
                return log_outcome
            try:
                if submitted_block_hash is not None:
                    block = await asyncio.wait_for(
                        self.node.get_block(submitted_block_hash, 1), timeout=remaining
                    )
                    if _block_matches_job(block, job):
                        return SubmissionOutcome.ACCEPTED
                else:
                    if await asyncio.wait_for(self._chain_contains_job(job), timeout=remaining):
                        return SubmissionOutcome.ACCEPTED
            except asyncio.TimeoutError:
                return self._final_log_outcome(job) or SubmissionOutcome.UNKNOWN_SUBMISSION
            except TransportError:
                if asyncio.get_running_loop().time() >= deadline:
                    return self._final_log_outcome(job) or SubmissionOutcome.UNKNOWN_SUBMISSION
            if asyncio.get_running_loop().time() >= deadline:
                return self._final_log_outcome(job) or log_outcome or SubmissionOutcome.UNKNOWN_SUBMISSION
            await asyncio.sleep(self.poll_seconds)

    async def track(
        self,
        job: GatewayJob,
        *,
        submitted_block_hash: str | None = None,
    ) -> SubmissionOutcome:
        return await self.confirm(job, submitted_block_hash=submitted_block_hash)

    async def _chain_contains_job(self, job: GatewayJob) -> bool:
        block_hash = await self.node.get_best_block_hash()
        for _ in range(self.max_chain_walk):
            block = await self.node.get_block(block_hash, 1)
            if _block_matches_job(block, job):
                return True
            previous = block.get("previousblockhash")
            if not previous:
                return False
            block_hash = str(previous)
        return False

    def _classify_gateway_logs(self, job: GatewayJob) -> SubmissionOutcome | None:
        if self.gateway_log is None:
            return None
        best: SubmissionOutcome | None = None
        for line in self.gateway_log.read_events():
            event = _parse_gateway_event(line)
            if event is None or not _event_matches_job(event, job):
                continue
            event_name = str(event.get("event", "")).lower()
            status = ""
            result = event.get("result")
            if isinstance(result, dict):
                status = str(result.get("status", "")).lower()
            elif result is not None:
                status = str(result).lower()
            outcome = None
            if event_name == "duplicate_block_submission" or status == "already_submitted":
                outcome = SubmissionOutcome.DUPLICATE
            elif event_name == "plain_proof_decode_error":
                outcome = SubmissionOutcome.PROVING_ERROR
            elif event_name == "proving_queue_full":
                outcome = SubmissionOutcome.TRANSPORT
            elif event_name == "stale_plain_proof":
                outcome = SubmissionOutcome.STALE
            elif event_name == "block_accepted" or status == "accepted":
                outcome = None
            if event_name in {"plain_proof_rejected", "block_submission_error"}:
                outcome = _classify_submission_error(event)
            elif event_name == "block_rejected" or status.startswith("rejected"):
                outcome = _classify_gateway_rejection(event, status)
            best = _prefer_outcome(best, outcome)
        return best

    def _final_log_outcome(self, job: GatewayJob) -> SubmissionOutcome | None:
        outcome = self._classify_gateway_logs(job)
        if outcome is None:
            return None
        return outcome


def _block_matches_job(block: dict[str, Any], job: GatewayJob) -> bool:
    try:
        return (
            int(block.get("version", -1)) == job.version
            and str(block.get("previousblockhash", "")).lower() == job.prev_hash
            and str(block.get("bits", "")).lower() == job.bits_hex
            and str(block.get("merkleroot", "")).lower() == job.merkle_root
            and int(block.get("time", -1)) == job.timestamp
        )
    except (TypeError, ValueError):
        return False


def _parse_gateway_event(line: str) -> dict[str, Any] | None:
    clean = _ANSI_RE.sub("", line)
    start = clean.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(clean[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _event_matches_job(event: dict[str, Any], job: GatewayJob) -> bool:
    submission_id = event.get("submission_id")
    job_submission_id = getattr(job, "submission_id", None)
    if job_submission_id is not None:
        return submission_id == job_submission_id
    header_hex = event.get("header_hex")
    if isinstance(header_hex, str):
        return header_hex.lower() == job.header_hex
    header_hash = event.get("header_hash")
    if isinstance(header_hash, str):
        return header_hash.lower() == job.header_hash
    return False


def _classify_rejection_text(text: str) -> SubmissionOutcome:
    low = text.lower()
    if "duplicate" in low or "already" in low:
        return SubmissionOutcome.DUPLICATE
    if "stale" in low or "old header" in low or "orphan" in low:
        return SubmissionOutcome.STALE
    return SubmissionOutcome.CONSENSUS_INVALID


def _classify_gateway_rejection(event: dict[str, Any], status: str) -> SubmissionOutcome | None:
    classification = _event_text(event, "classification")
    if classification in {outcome.value for outcome in SubmissionOutcome}:
        return SubmissionOutcome(classification)
    stage = _event_text(event, "stage") or _event_text(event, "phase")
    error_type = _event_text(event, "error_type")
    text = status or _event_text(event, "error") or _event_text(event, "message")
    if "duplicate" in text or "already" in text:
        return SubmissionOutcome.DUPLICATE
    if "stale" in text or "old header" in text or "orphan" in text:
        return SubmissionOutcome.STALE
    if (
        stage in {"node", "submitblock", "node_rpc"}
        and error_type in {"rpc-rejection", "node-verdict", "consensus"}
        and ("rejected:" in text or "bad-certificate" in text)
    ):
        return SubmissionOutcome.CONSENSUS_INVALID
    if classification == SubmissionOutcome.CONSENSUS_INVALID.value:
        return SubmissionOutcome.CONSENSUS_INVALID
    return None


def _prefer_outcome(
    current: SubmissionOutcome | None, candidate: SubmissionOutcome | None
) -> SubmissionOutcome | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    priority = {
        SubmissionOutcome.CONSENSUS_INVALID: 6,
        SubmissionOutcome.UNKNOWN_SUBMISSION: 5,
        SubmissionOutcome.TRANSPORT: 4,
        SubmissionOutcome.PROVING_ERROR: 3,
        SubmissionOutcome.STALE: 2,
        SubmissionOutcome.DUPLICATE: 1,
        SubmissionOutcome.ACCEPTED: 0,
    }
    return candidate if priority[candidate] > priority[current] else current


def _classify_submission_error(event: dict[str, Any]) -> SubmissionOutcome:
    classification = _event_text(event, "classification")
    if classification in {outcome.value for outcome in SubmissionOutcome}:
        return SubmissionOutcome(classification)

    stage = _event_text(event, "stage") or _event_text(event, "phase")
    error_type = _event_text(event, "error_type")
    error_text = _event_text(event, "error") or _event_text(event, "message")
    result = event.get("result")
    if isinstance(result, dict):
        error_text = error_text or str(result.get("status", "")).lower()
    elif result is not None:
        error_text = error_text or str(result).lower()

    if stage in {"node", "submitblock", "node_rpc"}:
        if error_type in {"transport", "timeout", "connection"}:
            return SubmissionOutcome.TRANSPORT
        if error_type in {"rpc-rejection", "node-verdict", "consensus"} and (
            "rejected:" in error_text or "bad-certificate" in error_text
        ):
            return _classify_rejection_text(error_text)
        if "connection" in error_text or "timeout" in error_text or "transport" in error_text:
            return SubmissionOutcome.TRANSPORT
        return SubmissionOutcome.TRANSPORT

    return SubmissionOutcome.PROVING_ERROR


def _event_text(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return str(value).lower() if value is not None else ""


def _strict_int(value: Any) -> bool:
    return type(value) is int


def _bits_to_target(bits: int) -> int:
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent == 0 or mantissa & 0x800000:
        return 0
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        shift = 8 * (exponent - 3)
        if mantissa.bit_length() + shift > 256:
            raise ValueError("compact target overflows U256")
        target = mantissa << shift
    if target > MAX_U256:
        raise ValueError("compact target overflows U256")
    return target


def _safe_rpc_error(error: Any, config: NodeRpcConfig | None = None) -> str:
    if isinstance(error, dict):
        code = error.get("code", "unknown")
        message = redact_credentials(str(error.get("message", "")), config)
        return f"code={code} message={message[:160]}"
    return redact_credentials(str(error), config)[:160]


def redact_secret(value: str, *, prefix: int = 4, suffix: int = 4) -> str:
    return "<redacted>"


_CRED_RE = re.compile(
    r"(?P<key>rpc_password|rpcpass|password|pass|token|secret)(?P<sep>['\"]?\s*[:=]\s*['\"]?)(?P<val>[^,'\"\s]+)",
    re.IGNORECASE,
)
_WALLET_RE = re.compile(r"\bprl1[0-9a-zA-Z]{8,}\b")


def redact_credentials(text: str, config: NodeRpcConfig | None = None) -> str:
    redacted = _CRED_RE.sub(
        lambda match: f"{match.group('key')}{match.group('sep')}{redact_secret(match.group('val'))}",
        text,
    )
    redacted = _WALLET_RE.sub("<redacted>", redacted)
    if config is not None:
        for secret in (config.rpc_user, config.rpc_password):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
    return redacted


@dataclass(frozen=True, slots=True)
class SubmissionEntry:
    submission_id: str
    template_identity: str
    header_hex: str
    proof_hash: str
    outcome: SubmissionOutcome | None = None
    source: str = "solo"
    pool_job_id: str | None = None
    target: int | None = None
    share_nbits: int | None = None
    cfg_hex: str | None = None
    session_id: str | None = None
    cert_version: int = CERT_VERSION_ZK_V3
    ancestor_headers: tuple[bytes, ...] = ()


@dataclass(frozen=True, slots=True)
class _TxInput:
    previous_hash: bytes
    previous_index: int
    sequence: int


@dataclass(frozen=True, slots=True)
class _TxOutput:
    value: int
    script: bytes


@dataclass(frozen=True, slots=True)
class _Transaction:
    inputs: tuple[_TxInput, ...]
    outputs: tuple[_TxOutput, ...]


class SubmissionLedger:
    """Append-only JSONL submission ledger for restart-safe confirmation."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            _fsync_dir(self.path.parent)
        else:
            os.chmod(self.path, 0o600)
        self._entries = self._load()

    def prepare(
        self,
        job: object,
        proof: str | bytes,
        *,
        source: str | None = None,
        share_nbits: int | None = None,
        cfg: bytes | str | None = None,
        session_id: str | None = None,
    ) -> SubmissionEntry:
        proof_hash = _proof_hash(proof)
        existing = self.find(job, proof)
        if existing is not None:
            return existing
        submission_id = secrets.token_hex(16)
        metadata = _submission_metadata(
            job,
            source=source,
            share_nbits=share_nbits,
            cfg=cfg,
            session_id=session_id,
        )
        entry = SubmissionEntry(
            submission_id=submission_id,
            template_identity=str(job.template_identity),
            header_hex=_job_header_hex(job),
            proof_hash=proof_hash,
            **metadata,
        )
        self._append(
            {
                "event": "prepared",
                "submission_id": submission_id,
                "template_identity": entry.template_identity,
                "header_hex": entry.header_hex,
                "proof_hash": proof_hash,
                **_entry_metadata_row(entry),
            }
        )
        self._entries[submission_id] = entry
        return entry

    def finish(self, submission_id: str, outcome: SubmissionOutcome | str) -> SubmissionEntry:
        if submission_id not in self._entries:
            raise KeyError(f"unknown submission_id {submission_id}")
        parsed = outcome if isinstance(outcome, SubmissionOutcome) else SubmissionOutcome(str(outcome))
        old = self._entries[submission_id]
        if old.outcome is not None:
            if old.outcome == parsed:
                return old
            raise TransportError(
                f"submission {submission_id} already has terminal outcome {old.outcome.value}"
            )
        entry = SubmissionEntry(
            submission_id=old.submission_id,
            template_identity=old.template_identity,
            header_hex=old.header_hex,
            proof_hash=old.proof_hash,
            outcome=parsed,
            source=old.source,
            pool_job_id=old.pool_job_id,
            target=old.target,
            share_nbits=old.share_nbits,
            cfg_hex=old.cfg_hex,
            session_id=old.session_id,
            cert_version=old.cert_version,
            ancestor_headers=old.ancestor_headers,
        )
        self._append({"event": "finished", "submission_id": submission_id, "outcome": parsed.value})
        self._entries[submission_id] = entry
        return entry

    def outstanding(self) -> list[SubmissionEntry]:
        return [entry for entry in self._entries.values() if entry.outcome is None]

    def entries(self) -> dict[str, SubmissionEntry]:
        return dict(self._entries)

    def fail_closed_entries(self) -> list[SubmissionEntry]:
        return [
            entry
            for entry in self._entries.values()
            if entry.outcome
            in {SubmissionOutcome.CONSENSUS_INVALID, SubmissionOutcome.UNKNOWN_SUBMISSION}
        ]

    def find(self, job: object, proof: str | bytes) -> SubmissionEntry | None:
        proof_hash = _proof_hash(proof)
        for entry in self._entries.values():
            if entry.template_identity == str(job.template_identity) and entry.proof_hash == proof_hash:
                return entry
        return None

    def seen_proof(self, job: object, proof: str | bytes) -> bool:
        return self.find(job, proof) is not None

    def _load(self) -> dict[str, SubmissionEntry]:
        entries: dict[str, SubmissionEntry] = {}
        for line_no, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                raise TransportError(f"submission ledger is malformed at line {line_no}") from None
            submission_id = row.get("submission_id")
            if not isinstance(submission_id, str) or not re.fullmatch(r"[0-9a-f]{32}", submission_id):
                raise TransportError(f"submission ledger has invalid submission_id at line {line_no}")
            if row.get("event") == "prepared":
                try:
                    template_identity = str(row["template_identity"])
                    header_hex = str(row["header_hex"])
                    proof_hash = str(row["proof_hash"])
                    source = str(row.get("source", "solo"))
                    if source not in {"solo", "pool"}:
                        raise ValueError("invalid source")
                    if not _valid_template_identity(template_identity):
                        raise ValueError("invalid template identity")
                    if not re.fullmatch(r"[0-9a-f]{152}", header_hex):
                        raise ValueError("invalid header hex")
                    if _identity_header_hex(template_identity) != header_hex:
                        raise ValueError("template identity/header mismatch")
                    if not re.fullmatch(r"[0-9a-f]{64}", proof_hash):
                        raise ValueError("invalid proof hash")
                    metadata = _load_entry_metadata(row, source)
                    entry = SubmissionEntry(
                        submission_id=submission_id,
                        template_identity=template_identity,
                        header_hex=header_hex,
                        proof_hash=proof_hash,
                        **metadata,
                    )
                    existing = entries.get(submission_id)
                    if existing is not None and existing != entry:
                        raise ValueError("conflicting prepared row")
                    entries[submission_id] = entry
                except (KeyError, ValueError):
                    raise TransportError(f"submission ledger has invalid prepared row at line {line_no}") from None
            elif row.get("event") == "finished" and submission_id in entries:
                try:
                    parsed = SubmissionOutcome(str(row["outcome"]))
                    current = entries[submission_id]
                    if current.outcome is not None:
                        if current.outcome == parsed:
                            continue
                        raise ValueError("conflicting terminal outcome")
                    entries[submission_id] = self._finished_entry(current, parsed)
                except (KeyError, ValueError):
                    raise TransportError(f"submission ledger has invalid finished row at line {line_no}") from None
            else:
                raise TransportError(f"submission ledger has invalid event order at line {line_no}")
        return entries

    def _finished_entry(
        self, entry: SubmissionEntry, outcome: SubmissionOutcome
    ) -> SubmissionEntry:
        return SubmissionEntry(
            submission_id=entry.submission_id,
            template_identity=entry.template_identity,
            header_hex=entry.header_hex,
            proof_hash=entry.proof_hash,
            outcome=outcome,
            source=entry.source,
            pool_job_id=entry.pool_job_id,
            target=entry.target,
            share_nbits=entry.share_nbits,
            cfg_hex=entry.cfg_hex,
            session_id=entry.session_id,
            cert_version=entry.cert_version,
            ancestor_headers=entry.ancestor_headers,
        )

    def _append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _job_header_hex(job: object) -> str:
    header_hex = getattr(job, "header_hex", None)
    if isinstance(header_hex, str):
        value = header_hex.lower()
    else:
        header = getattr(job, "header", None)
        if not isinstance(header, (bytes, bytearray)):
            raise ValueError("submission job is missing header bytes")
        value = bytes(header).hex()
    if not re.fullmatch(r"[0-9a-f]{152}", value):
        raise ValueError("submission job header must be 76 bytes")
    return value


def _submission_metadata(
    job: object,
    *,
    source: str | None,
    share_nbits: int | None,
    cfg: bytes | str | None,
    session_id: str | None,
) -> dict[str, Any]:
    entry_source = source or getattr(job, "source", None)
    if entry_source is None:
        entry_source = "pool" if hasattr(job, "pool_job_id") else "solo"
    entry_source = str(entry_source)
    if entry_source not in {"solo", "pool"}:
        raise ValueError("submission source must be solo or pool")

    cfg_hex = _cfg_hex(cfg if cfg is not None else getattr(job, "cfg", None))
    metadata = {
        "source": entry_source,
        "pool_job_id": None,
        "target": None,
        "share_nbits": None,
        "cfg_hex": cfg_hex,
        "session_id": session_id if session_id is not None else getattr(job, "session_id", None),
        "cert_version": getattr(job, "cert_version", CERT_VERSION_ZK_V3),
        "ancestor_headers": tuple(getattr(job, "ancestor_headers", ()) or ()),
    }
    if type(metadata["cert_version"]) is not int or metadata["cert_version"] not in (3,4):
        raise ValueError("submission ledger requires cert_version 3 or 4")
    metadata["ancestor_headers"] = tuple(bytes(x) for x in metadata["ancestor_headers"])
    if metadata["cert_version"] == 4:
        if not metadata["ancestor_headers"] or any(len(x) != 108 for x in metadata["ancestor_headers"]):
            raise ValueError("v4 ledger entry requires complete ancestor_headers")
    elif metadata["ancestor_headers"]:
        raise ValueError("ancestor_headers are only valid for cert_version=4")
    if metadata["session_id"] is not None:
        metadata["session_id"] = str(metadata["session_id"])

    if entry_source == "pool":
        pool_job_id = getattr(job, "pool_job_id", None)
        if not isinstance(pool_job_id, str):
            raise ValueError("pool ledger entry requires pool_job_id")
        if len(pool_job_id) > 64 or any(ord(ch) < 32 or ord(ch) > 126 for ch in pool_job_id):
            raise ValueError("pool_job_id must be printable ASCII <= 64 chars")
        target = getattr(job, "target", None)
        if type(target) is not int or target <= 0 or target > MAX_U256:
            raise ValueError("pool ledger entry requires target")
        nbits = share_nbits if share_nbits is not None else getattr(job, "share_nbits", None)
        if type(nbits) is not int:
            raise ValueError("pool ledger entry requires share_nbits")
        if cfg_hex is None:
            raise ValueError("pool ledger entry requires cfg")
        if metadata["session_id"] is None:
            raise ValueError("pool ledger entry requires session_id")
        metadata.update(pool_job_id=pool_job_id, target=target, share_nbits=nbits)
    return metadata


def _cfg_hex(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    text = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]*", text) or len(text) % 2:
        raise ValueError("cfg must be bytes or even-length lowercase hex")
    return text


def _entry_metadata_row(entry: SubmissionEntry) -> dict[str, Any]:
    row: dict[str, Any] = {"source": entry.source}
    if entry.pool_job_id is not None:
        row["pool_job_id"] = entry.pool_job_id
    if entry.target is not None:
        row["target"] = entry.target
    if entry.share_nbits is not None:
        row["share_nbits"] = entry.share_nbits
    if entry.cfg_hex is not None:
        row["cfg_hex"] = entry.cfg_hex
    row["cert_version"] = entry.cert_version
    if entry.ancestor_headers:
        row["ancestor_headers"] = [base64.b64encode(h).decode("ascii") for h in entry.ancestor_headers]
    if entry.session_id is not None:
        row["session_id"] = entry.session_id
    return row


def _valid_template_identity(template_identity: str) -> bool:
    return bool(
        re.fullmatch(r"[0-9a-f]+:[0-9a-f]{8}:[0-9a-f]{152}", template_identity)
        or re.fullmatch(r"[0-9a-f]+:[0-9a-f]{8}:(3|4):([0-9a-f]{32})?:[0-9a-f]{152}", template_identity)
        or re.fullmatch(r"[0-9a-f]{152}", template_identity)
    )


def _identity_header_hex(template_identity: str) -> str:
    if ":" in template_identity:
        return template_identity.rsplit(":", 1)[1]
    return template_identity


def _load_entry_metadata(row: dict[str, Any], source: str) -> dict[str, Any]:
    pool_job_id = row.get("pool_job_id")
    target = row.get("target")
    share_nbits = row.get("share_nbits")
    cfg_hex = row.get("cfg_hex")
    session_id = row.get("session_id")
    cert_version = row.get("cert_version", CERT_VERSION_ZK_V3)
    ancestor_headers_raw = row.get("ancestor_headers", [])
    if pool_job_id is not None and (
        not isinstance(pool_job_id, str)
        or len(pool_job_id) > 64
        or any(ord(ch) < 32 or ord(ch) > 126 for ch in pool_job_id)
    ):
        raise ValueError("invalid pool_job_id")
    if target is not None and (type(target) is not int or target <= 0 or target > MAX_U256):
        raise ValueError("invalid target")
    if share_nbits is not None and type(share_nbits) is not int:
        raise ValueError("invalid share_nbits")
    if cfg_hex is not None and (
        not isinstance(cfg_hex, str) or not re.fullmatch(r"[0-9a-f]*", cfg_hex) or len(cfg_hex) % 2
    ):
        raise ValueError("invalid cfg_hex")
    if session_id is not None and not isinstance(session_id, str):
        raise ValueError("invalid session_id")
    if type(cert_version) is not int or cert_version not in (3,4):
        raise ValueError("invalid cert_version")
    if not isinstance(ancestor_headers_raw, list):
        raise ValueError("invalid ancestor_headers")
    try:
        ancestor_headers = tuple(base64.b64decode(item, validate=True) for item in ancestor_headers_raw)
    except (TypeError, ValueError):
        raise ValueError("invalid ancestor_headers") from None
    if cert_version == 4:
        if not ancestor_headers or any(len(item) != 108 for item in ancestor_headers):
            raise ValueError("invalid ancestor_headers")
    elif ancestor_headers:
        raise ValueError("invalid ancestor_headers")
    if source == "pool" and (
        pool_job_id is None
        or target is None
        or share_nbits is None
        or cfg_hex is None
        or session_id is None
    ):
        raise ValueError("pool prepared row missing immutable metadata")
    return {
        "source": source,
        "pool_job_id": pool_job_id,
        "target": target,
        "share_nbits": share_nbits,
        "cfg_hex": cfg_hex,
        "session_id": session_id,
        "cert_version": cert_version,
        "ancestor_headers": ancestor_headers,
    }


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _proof_hash(proof: str | bytes) -> str:
    if isinstance(proof, str):
        data = base64.b64decode(proof, validate=True)
    else:
        data = bytes(proof)
    return hashlib.blake2s(data, digest_size=32).hexdigest()


def authorize_coinbase_job(
    header: bytes,
    coinbase_tx: bytes,
    merkle_branch: tuple[bytes, ...] | list[bytes],
    coinbase_index: int,
    *,
    approved_script_hex: str,
) -> None:
    if len(header) != HEADER_LEN:
        raise ValueError("incomplete header length invalid")
    if coinbase_index != 0:
        raise ValueError("job coinbase must be transaction index 0")
    approved_script = bytes.fromhex(approved_script_hex)
    tx = _parse_transaction(coinbase_tx)
    if len(tx.inputs) != 1:
        raise ValueError("job coinbase must have exactly one input")
    txin = tx.inputs[0]
    if txin.previous_hash != b"\x00" * 32 or txin.previous_index != 0xFFFFFFFF:
        raise ValueError("job transaction is not a coinbase")
    approved_outputs = [output for output in tx.outputs if output.script == approved_script]
    if len(approved_outputs) != 1 or approved_outputs[0].value <= 0:
        raise ValueError("job coinbase does not pay exactly one positive approved script")
    for output in tx.outputs:
        if output.script == approved_script:
            continue
        if output.value != 0 or not _is_op_return(output.script):
            raise ValueError("job coinbase has unauthorized nonzero or spendable output")
    root = _merkle_root_from_branch(_txid(coinbase_tx), merkle_branch, coinbase_index)
    header_root = header[36:68]
    if root != header_root:
        raise ValueError("job coinbase is not committed by header merkle root")


def _txid(tx: bytes) -> bytes:
    stripped = _strip_witness(tx)
    return hashlib.sha256(hashlib.sha256(stripped).digest()).digest()


def _merkle_root_from_branch(leaf: bytes, branch: tuple[bytes, ...] | list[bytes], index: int) -> bytes:
    root = leaf
    for sibling in branch:
        if len(sibling) != 32:
            raise ValueError("merkle branch hash must be 32 bytes")
        pair = root + sibling if index % 2 == 0 else sibling + root
        root = hashlib.sha256(hashlib.sha256(pair).digest()).digest()
        index //= 2
    if index:
        raise ValueError("merkle branch too short for coinbase index")
    return root


def _parse_transaction(tx: bytes) -> _Transaction:
    view = memoryview(tx)
    pos = 4
    if len(view) < pos:
        raise ValueError("coinbase tx too short")
    segwit = len(view) > pos + 1 and view[pos] == 0 and view[pos + 1] != 0
    if segwit:
        pos += 2
    input_count, pos = _read_varint(view, pos)
    inputs: list[_TxInput] = []
    for _ in range(input_count):
        txin, pos = _read_input(view, pos)
        inputs.append(txin)
    output_count, pos = _read_varint(view, pos)
    outputs: list[_TxOutput] = []
    for _ in range(output_count):
        if pos + 8 > len(view):
            raise ValueError("coinbase output truncated")
        value = int.from_bytes(view[pos : pos + 8], "little")
        pos += 8
        script_len, pos = _read_varint(view, pos)
        if pos + script_len > len(view):
            raise ValueError("coinbase output script truncated")
        outputs.append(_TxOutput(value=value, script=bytes(view[pos : pos + script_len])))
        pos += script_len
    if segwit:
        for _ in range(input_count):
            items, pos = _read_varint(view, pos)
            for _ in range(items):
                item_len, pos = _read_varint(view, pos)
                if pos + item_len > len(view):
                    raise ValueError("coinbase witness truncated")
                pos += item_len
    if pos + 4 > len(view):
        raise ValueError("coinbase locktime truncated")
    pos += 4
    if pos != len(view):
        raise ValueError("coinbase transaction has trailing data")
    return _Transaction(inputs=tuple(inputs), outputs=tuple(outputs))


def _strip_witness(tx: bytes) -> bytes:
    view = memoryview(tx)
    pos = 4
    if len(view) <= pos + 1 or view[pos] != 0 or view[pos + 1] == 0:
        return tx
    version = bytes(view[:4])
    pos += 2
    input_count_pos = pos
    input_count, pos = _read_varint(view, pos)
    for _ in range(input_count):
        _, pos = _read_input(view, pos)
    output_count_pos = pos
    output_count, pos = _read_varint(view, pos)
    for _ in range(output_count):
        if pos + 8 > len(view):
            raise ValueError("coinbase output truncated")
        pos += 8
        script_len, pos = _read_varint(view, pos)
        if pos + script_len > len(view):
            raise ValueError("coinbase output script truncated")
        pos += script_len
    outputs_end = pos
    for _ in range(input_count):
        items, pos = _read_varint(view, pos)
        for _ in range(items):
            item_len, pos = _read_varint(view, pos)
            pos += item_len
            if pos > len(view):
                raise ValueError("coinbase witness truncated")
    if pos + 4 > len(view):
        raise ValueError("coinbase locktime truncated")
    locktime = bytes(view[pos : pos + 4])
    pos += 4
    if pos != len(view):
        raise ValueError("coinbase transaction has trailing data")
    return (
        version
        + bytes(view[input_count_pos:output_count_pos])
        + bytes(view[output_count_pos:outputs_end])
        + locktime
    )


def _read_input(view: memoryview, pos: int) -> tuple[_TxInput, int]:
    if pos + 36 > len(view):
        raise ValueError("coinbase input truncated")
    previous_hash = bytes(view[pos : pos + 32])
    pos += 32
    previous_index = int.from_bytes(view[pos : pos + 4], "little")
    pos += 4
    script_len, pos = _read_varint(view, pos)
    if pos + script_len > len(view):
        raise ValueError("coinbase input script truncated")
    pos += script_len
    if pos + 4 > len(view):
        raise ValueError("coinbase sequence truncated")
    sequence = int.from_bytes(view[pos : pos + 4], "little")
    return _TxInput(previous_hash, previous_index, sequence), pos + 4


def _is_op_return(script: bytes) -> bool:
    return bool(script) and script[0] == 0x6A


def _read_varint(view: memoryview, pos: int) -> tuple[int, int]:
    if pos >= len(view):
        raise ValueError("varint truncated")
    first = view[pos]
    pos += 1
    if first < 0xFD:
        return first, pos
    if first == 0xFD:
        size = 2
    elif first == 0xFE:
        size = 4
    else:
        size = 8
    if pos + size > len(view):
        raise ValueError("varint truncated")
    return int.from_bytes(view[pos : pos + size], "little"), pos + size
