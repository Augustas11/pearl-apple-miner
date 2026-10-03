"""CPU mining-loop subprocess entry point.

Run with::

    python -m oj_pearl_mps._miner_loop_main \
        --gateway-host 127.0.0.1 --gateway-port 8337 \
        --m 256 --n 128 --k 2048 --rank 128

Connects to pearl-gateway, polls for work via ``getMiningInfo``, runs
``pearl_mining.mine()``, submits proofs via ``submitPlainProof``. Designed
to be killed via SIGTERM by the parent provider; no graceful shutdown
handshake -- Pearl's gateway tolerates client disconnects cleanly.

The parent OJ process spawns this module via ``python -m`` and never
imports it directly, keeping the parent's import graph free of
``pearl_mining`` (which is an optional dependency installed only with
``--extra mining-pearl-cpu``).

Modified for pearl-metal-miner (see NOTICE / docs/OPENJARVIS_UPGRADE.md):
ported to py-pearl-mining 0.3.1 + pearl-gateway at Pearl 7039e66f
(MoE fork, rank-penalty rule, salted-seed V3 fork).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from typing import Any

logger = logging.getLogger("oj_pearl_mps.miner_loop")

# Backoff after a failed mining round before retrying. Short enough that
# transient gateway errors don't stall mining; long enough not to spin.
_FAILURE_BACKOFF_SECONDS = 1.0
_CONNECT_RETRY_SECONDS = 0.25
_CONNECT_TIMEOUT_SECONDS = 30.0


def _make_request(
    method: str, params: dict[str, Any], request_id: int
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request envelope matching gateway's schema."""
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}


def _decode_mining_info(result: dict[str, Any]) -> tuple[bytes, int, int]:
    """Decode a getMiningInfo result into ``(incomplete_header_bytes, target, cert_version)``.

    Current pearl-gateway ``MiningJob.to_dict()`` carries ``cert_version`` (the
    template's ``requiredcertversion``); it selects the noise-seed derivation and
    must be echoed back in ``submitPlainProof``.
    """
    header_b64 = result["incomplete_header_bytes"]
    target = int(result["target"])
    cert_version = int(result["cert_version"])
    return base64.b64decode(header_b64), target, cert_version


def _mining_job_params(header_bytes: bytes, target: int, cert_version: int) -> dict[str, Any]:
    """``mining_job`` object for ``submitPlainProof`` (gateway schema requires cert_version)."""
    return {
        "incomplete_header_bytes": base64.b64encode(header_bytes).decode(),
        "target": target,
        "cert_version": cert_version,
    }


def _encode_plain_proof(plain_proof: Any) -> str:
    """Return the proof as a base64 string for ``submitPlainProof``.

    Pearl's ``PlainProof`` exposes ``to_base64()`` directly — no manual
    encoding required. (Re-verified on py-pearl-mining 0.3.1, macOS arm64.)
    """
    return plain_proof.to_base64()


async def _read_response(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one line of JSON-RPC response from the gateway socket."""
    line = await reader.readline()
    if not line:
        raise ConnectionError("gateway closed the connection")
    return json.loads(line)


async def _send_request(writer: asyncio.StreamWriter, request: dict[str, Any]) -> None:
    """Write one JSON-RPC request followed by newline."""
    writer.write(json.dumps(request).encode() + b"\n")
    await writer.drain()


async def _open_gateway_connection(
    host: str,
    port: int,
    *,
    timeout_seconds: float = _CONNECT_TIMEOUT_SECONDS,
    retry_seconds: float = _CONNECT_RETRY_SECONDS,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to pearl-gateway, retrying while its listener comes up."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: OSError | None = None
    while True:
        try:
            return await asyncio.open_connection(host, port)
        except OSError as exc:
            last_error = exc
            if asyncio.get_running_loop().time() >= deadline:
                raise ConnectionError(
                    f"timed out connecting to pearl-gateway at {host}:{port}"
                ) from last_error
            logger.info(
                "gateway %s:%d not ready yet (%s); retrying",
                host,
                port,
                exc,
            )
            await asyncio.sleep(retry_seconds)


def _build_mining_config(pearl_mining_module: Any, *, k: int, rank: int) -> Any:
    """Build the upstream MiningConfiguration with default patterns.

    py-pearl-mining >= 0.2 removed ``reserved=`` / ``MiningConfiguration.RESERVED``;
    the last argument is now ``moe`` (``None`` for dense mining).
    """
    from ._constants import (
        CPU_PEARL_DEFAULT_COLS_PATTERN,
        CPU_PEARL_DEFAULT_ROWS_PATTERN,
    )

    return pearl_mining_module.MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=pearl_mining_module.MMAType.Int7xInt7ToInt32,
        rows_pattern=pearl_mining_module.PeriodicPattern.from_list(
            list(CPU_PEARL_DEFAULT_ROWS_PATTERN)
        ),
        cols_pattern=pearl_mining_module.PeriodicPattern.from_list(
            list(CPU_PEARL_DEFAULT_COLS_PATTERN)
        ),
        moe=None,
    )


async def _mine_one_round(
    pearl_mining_module: Any,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    request_id: int,
    m: int,
    n: int,
    k: int,
    rank: int,
) -> bool:
    """Get work, mine, submit. Return True if the gateway accepted the proof."""
    # 1. Ask the gateway for work.
    await _send_request(writer, _make_request("getMiningInfo", {}, request_id))
    info_response = await _read_response(reader)
    if "error" in info_response:
        logger.warning("getMiningInfo error: %s", info_response["error"])
        return False
    header_bytes, target, cert_version = _decode_mining_info(info_response["result"])

    # 2. Reconstruct the IncompleteBlockHeader and run pearl_mining.mine().
    # APIs (py-pearl-mining 0.3.1):
    #   - IncompleteBlockHeader.from_bytes(bytes) -> IncompleteBlockHeader
    #   - mine(..., *, cert_version) — keyword-only, selects the seed derivation
    #   - PlainProof.to_base64() -> str  (used by _encode_plain_proof above)
    header = pearl_mining_module.IncompleteBlockHeader.from_bytes(header_bytes)
    mining_config = _build_mining_config(pearl_mining_module, k=k, rank=rank)
    plain_proof = pearl_mining_module.mine(
        m,
        n,
        k,
        header,
        mining_config,
        signal_range=None,
        wrong_jackpot_hash=False,
        cert_version=cert_version,
    )

    # 3. Submit the proof back to the gateway.
    submit_params = {
        "plain_proof": _encode_plain_proof(plain_proof),
        "mining_job": _mining_job_params(header_bytes, target, cert_version),
    }
    await _send_request(
        writer, _make_request("submitPlainProof", submit_params, request_id + 1)
    )
    submit_response = await _read_response(reader)
    if "error" in submit_response:
        logger.warning("submitPlainProof rejected: %s", submit_response["error"])
        return False
    # The gateway answers "submitted" and then proves/submits the block
    # asynchronously; node acceptance is only visible in the gateway log/chain.
    logger.info("submitPlainProof result: %s", submit_response.get("result"))
    return True


async def _main_loop(args: argparse.Namespace) -> None:
    import pearl_mining  # imported lazily so this module is itself import-safe

    reader, writer = await _open_gateway_connection(
        args.gateway_host,
        args.gateway_port,
        timeout_seconds=args.connect_timeout_seconds,
        retry_seconds=args.connect_retry_seconds,
    )
    request_id = 0
    try:
        while True:
            request_id += 2
            try:
                accepted = await _mine_one_round(
                    pearl_mining,
                    reader,
                    writer,
                    request_id=request_id,
                    m=args.m,
                    n=args.n,
                    k=args.k,
                    rank=args.rank,
                )
            except Exception:
                logger.exception("mining round failed; retrying after backoff")
                await asyncio.sleep(_FAILURE_BACKOFF_SECONDS)
                continue
            if accepted:
                logger.info("proof handed to gateway")
            else:
                await asyncio.sleep(_FAILURE_BACKOFF_SECONDS)
    finally:
        writer.close()
        await writer.wait_closed()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="oj_pearl_mps._miner_loop_main")
    p.add_argument("--gateway-host", default="127.0.0.1")
    p.add_argument("--gateway-port", type=int, default=8337)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    p.add_argument("--connect-retry-seconds", type=float, default=0.25)
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    try:
        asyncio.run(_main_loop(args))
    except KeyboardInterrupt:
        sys.exit(0)
