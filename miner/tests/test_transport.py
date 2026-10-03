from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
from pathlib import Path

import pytest

from pmk_miner.transport import (
    FatalCertVersionError,
    GatewayClient,
    GatewayEndpoint,
    GatewayJob,
    GatewayLogTail,
    NodeRpcConfig,
    SubmissionLedger,
    SubmissionOutcome,
    SubmissionTracker,
    TransportError,
    authorize_coinbase_job,
    redact_credentials,
)


def header(
    *,
    version: int = 2,
    prev: bytes = b"\x11" * 32,
    merkle: bytes = b"\x22" * 32,
    timestamp: int = 123456,
    bits: int = 0x1D00FFFF,
) -> bytes:
    return (
        version.to_bytes(4, "little")
        + prev
        + merkle
        + timestamp.to_bytes(4, "little")
        + bits.to_bytes(4, "little")
    )


def bits_to_target(bits: int) -> int:
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent == 0 or mantissa & 0x800000:
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    target = mantissa << (8 * (exponent - 3))
    if target > (1 << 256) - 1:
        raise ValueError("overflow")
    return target


async def start_gateway(handler):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        line = await reader.readline()
        request = json.loads(line)
        response = await handler(request)
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def gateway_job_dict(job_header: bytes | None = None, cert_version: int = 3) -> dict:
    job_header = job_header or header()
    return {
        "incomplete_header_bytes": base64.b64encode(job_header).decode(),
        "target": bits_to_target(int.from_bytes(job_header[72:76], "little")),
        "cert_version": cert_version,
    }


def tx_output(value: int, script: bytes) -> bytes:
    return value.to_bytes(8, "little") + bytes([len(script)]) + script


def coinbase_tx(script: bytes, *, outputs: list[tuple[int, bytes]] | None = None) -> bytes:
    outputs = outputs or [(50_00000000, script)]
    return (
        (1).to_bytes(4, "little")
        + b"\x01"
        + (b"\x00" * 32)
        + (0xFFFFFFFF).to_bytes(4, "little")
        + b"\x01\x51"
        + (0xFFFFFFFF).to_bytes(4, "little")
        + bytes([len(outputs)])
        + b"".join(tx_output(value, output_script) for value, output_script in outputs)
        + (0).to_bytes(4, "little")
    )


def txid(tx: bytes) -> bytes:
    import hashlib

    return hashlib.sha256(hashlib.sha256(tx).digest()).digest()


def test_endpoint_rejects_non_loopback():
    with pytest.raises(ValueError, match="loopback"):
        GatewayEndpoint.parse("192.0.2.10:8337")


def test_job_identity_uses_prev_bits_and_header():
    h1 = header(prev=b"\xaa" * 32, bits=0x1E010000)
    h2 = header(prev=b"\xbb" * 32, bits=0x1E010000)
    job1 = GatewayJob(h1, target=bits_to_target(0x1E010000), cert_version=3)
    job2 = GatewayJob(h2, target=bits_to_target(0x1E010000), cert_version=3)
    assert job1.header == h1
    assert job1.target == bits_to_target(0x1E010000)
    assert job1.cert_version == 3
    assert job1.prev_hash == (b"\xaa" * 32)[::-1].hex()
    assert job1.bits_hex == "1e010000"
    assert job1.template_identity != job2.template_identity
    assert job1.template_identity.endswith(h1.hex())


def test_cert_guard_is_fatal():
    with pytest.raises(FatalCertVersionError):
        GatewayJob(header(), target=bits_to_target(0x1D00FFFF), cert_version=4)


def test_job_rejects_target_mismatch_and_non_strict_cert():
    with pytest.raises(ValueError, match="target"):
        GatewayJob(header(), target=123, cert_version=3)
    with pytest.raises(FatalCertVersionError):
        GatewayJob(header(), target=bits_to_target(0x1D00FFFF), cert_version=3.0)
    with pytest.raises(ValueError, match="strict integer"):
        GatewayJob(header(), target=float(bits_to_target(0x1D00FFFF)), cert_version=3)
    with pytest.raises(ValueError, match="1..2\\^256-1"):
        GatewayJob(header(), target=(1 << 256), cert_version=3)
    data = gateway_job_dict()
    data["cert_version"] = 3.5
    with pytest.raises(ValueError, match="malformed"):
        GatewayJob.from_gateway_dict(data)
    data = gateway_job_dict()
    data["target"] = str(data["target"])
    with pytest.raises(ValueError, match="malformed"):
        GatewayJob.from_gateway_dict(data)


def test_job_rejects_invalid_compact_targets():
    with pytest.raises(ValueError, match="target"):
        GatewayJob(header(bits=0x1D800000), target=0, cert_version=3)
    with pytest.raises(ValueError, match="target"):
        GatewayJob(header(bits=0), target=0, cert_version=3)
    with pytest.raises(ValueError, match="overflows"):
        GatewayJob(header(bits=0x23010000), target=1, cert_version=3)


def test_gateway_job_authorizes_committed_coinbase():
    script = bytes.fromhex("5120" + "2d" * 32)
    tx = coinbase_tx(script)
    job = GatewayJob(
        header(merkle=txid(tx)),
        target=bits_to_target(0x1D00FFFF),
        cert_version=3,
        coinbase_tx=tx,
    )
    job.authorize_coinbase(approved_script_hex=script.hex())


def test_gateway_job_rejects_malicious_coinbase_script():
    approved = bytes.fromhex("5120" + "2d" * 32)
    attacker = bytes.fromhex("5120" + "99" * 32)
    tx = coinbase_tx(attacker)
    job = GatewayJob(header(merkle=txid(tx)), bits_to_target(0x1D00FFFF), 3, coinbase_tx=tx)
    with pytest.raises(ValueError, match="approved script"):
        job.authorize_coinbase(approved_script_hex=approved.hex())


def test_gateway_job_rejects_split_reward_attack():
    approved = bytes.fromhex("5120" + "2d" * 32)
    attacker = bytes.fromhex("5120" + "99" * 32)
    tx = coinbase_tx(approved, outputs=[(1, approved), (49_99999999, attacker)])
    job = GatewayJob(header(merkle=txid(tx)), bits_to_target(0x1D00FFFF), 3, coinbase_tx=tx)
    with pytest.raises(ValueError, match="unauthorized"):
        job.authorize_coinbase(approved_script_hex=approved.hex())


def test_gateway_job_allows_zero_value_op_return_extra():
    approved = bytes.fromhex("5120" + "2d" * 32)
    tx = coinbase_tx(approved, outputs=[(50_00000000, approved), (0, b"\x6a\x02pm")])
    job = GatewayJob(header(merkle=txid(tx)), bits_to_target(0x1D00FFFF), 3, coinbase_tx=tx)
    job.authorize_coinbase(approved_script_hex=approved.hex())


def test_gateway_job_rejects_noncoinbase_input():
    approved = bytes.fromhex("5120" + "2d" * 32)
    tx = bytearray(coinbase_tx(approved))
    tx[5:37] = b"\x44" * 32
    h = header(merkle=txid(bytes(tx)))
    job = GatewayJob(h, bits_to_target(0x1D00FFFF), 3, coinbase_tx=bytes(tx))
    with pytest.raises(ValueError, match="not a coinbase"):
        job.authorize_coinbase(approved_script_hex=approved.hex())


def test_gateway_job_rejects_nonzero_coinbase_index():
    approved = bytes.fromhex("5120" + "2d" * 32)
    tx = coinbase_tx(approved)
    h = header(merkle=txid(tx))
    job = GatewayJob(h, bits_to_target(0x1D00FFFF), 3, coinbase_tx=tx, coinbase_index=1)
    with pytest.raises(ValueError, match="index 0"):
        job.authorize_coinbase(approved_script_hex=approved.hex())


def test_gateway_job_rejects_malformed_and_trailing_coinbase():
    approved = bytes.fromhex("5120" + "2d" * 32)
    tx = coinbase_tx(approved)
    cases = (tx[:-1], tx + b"\x00")
    for bad in cases:
        job = GatewayJob(header(merkle=txid(bad)), bits_to_target(0x1D00FFFF), 3, coinbase_tx=bad)
        with pytest.raises(ValueError):
            job.authorize_coinbase(approved_script_hex=approved.hex())


def test_coinbase_authorization_rejects_uncommitted_merkle_root():
    script = bytes.fromhex("5120" + "2d" * 32)
    tx = coinbase_tx(script)
    with pytest.raises(ValueError, match="not committed"):
        authorize_coinbase_job(
            header(merkle=b"\x33" * 32),
            tx,
            [],
            0,
            approved_script_hex=script.hex(),
        )


def test_gateway_get_job_and_submit():
    asyncio.run(_test_gateway_get_job_and_submit())


def test_gateway_job_round_trips_submission_id():
    job = GatewayJob.from_gateway_dict(
        {**gateway_job_dict(), "submission_id": "a" * 32}
    )
    assert job.submission_id == "a" * 32
    assert job.to_gateway_dict()["submission_id"] == "a" * 32


async def _test_gateway_get_job_and_submit():
    seen_submit = {}

    async def handler(request: dict):
        if request["method"] == "getMiningInfo":
            return {"jsonrpc": "2.0", "result": gateway_job_dict(), "id": request["id"]}
        seen_submit.update(request)
        mining_job = request["params"]["mining_job"]
        if "submission_id" in mining_job:
            result = {"status": "submitted", "submission_id": mining_job["submission_id"]}
        else:
            result = "submitted"
        return {"jsonrpc": "2.0", "result": result, "id": request["id"]}

    server, port = await start_gateway(handler)
    try:
        client = GatewayClient(f"127.0.0.1:{port}")
        job = await client.get_job()
        ack = await client.submit(job, "cHJvb2Y=")
    finally:
        server.close()
        await server.wait_closed()

    assert ack.result == "submitted"
    assert seen_submit["method"] == "submitPlainProof"
    assert seen_submit["params"]["plain_proof"] == "cHJvb2Y="
    assert seen_submit["params"]["mining_job"]["cert_version"] == 3


def test_gateway_submit_requires_ack_submission_id_echo():
    asyncio.run(_test_gateway_submit_requires_ack_submission_id_echo())


async def _test_gateway_submit_requires_ack_submission_id_echo():
    async def handler(request: dict):
        return {
            "jsonrpc": "2.0",
            "result": {"status": "submitted", "submission_id": "b" * 32},
            "id": request["id"],
        }

    server, port = await start_gateway(handler)
    try:
        client = GatewayClient(f"127.0.0.1:{port}")
        job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3, submission_id="a" * 32)
        with pytest.raises(TransportError, match="submission_id mismatch"):
            await client.submit(job, "cHJvb2Y=")
    finally:
        server.close()
        await server.wait_closed()


def test_gateway_job_submission_id_is_sent_to_gateway():
    job = GatewayJob(
        header(),
        bits_to_target(0x1D00FFFF),
        3,
        submission_id="0123456789abcdef0123456789abcdef",
    )
    assert job.to_gateway_dict()["submission_id"] == "0123456789abcdef0123456789abcdef"
    with pytest.raises(ValueError, match="submission_id"):
        GatewayJob(header(), bits_to_target(0x1D00FFFF), 3, submission_id="bad")


def test_gateway_get_job_rejects_bad_cert():
    asyncio.run(_test_gateway_get_job_rejects_bad_cert())


async def _test_gateway_get_job_rejects_bad_cert():
    async def handler(request: dict):
        return {"jsonrpc": "2.0", "result": gateway_job_dict(cert_version=2), "id": request["id"]}

    server, port = await start_gateway(handler)
    try:
        client = GatewayClient(f"127.0.0.1:{port}")
        with pytest.raises(FatalCertVersionError):
            await client.get_job()
    finally:
        server.close()
        await server.wait_closed()


def test_gateway_client_does_not_leak_raw_error_results():
    asyncio.run(_test_gateway_client_does_not_leak_raw_error_results())


async def _test_gateway_client_does_not_leak_raw_error_results():
    secret = "rpc_password=supersecret"

    async def handler(request: dict):
        if request["method"] == "getMiningInfo":
            return {"jsonrpc": "2.0", "result": secret, "id": request["id"]}
        return {"jsonrpc": "2.0", "result": {"status": secret}, "id": request["id"]}

    server, port = await start_gateway(handler)
    try:
        client = GatewayClient(f"127.0.0.1:{port}")
        with pytest.raises(TransportError) as get_error:
            await client.get_job()
        assert "supersecret" not in str(get_error.value)
        job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
        with pytest.raises(TransportError) as submit_error:
            await client.submit(job, "proof")
        assert "supersecret" not in str(submit_error.value)
    finally:
        server.close()
        await server.wait_closed()


def test_node_rpc_config_from_toml_env_file_and_env_override(tmp_path: Path):
    toml = tmp_path / "gateway.toml"
    toml.write_text(
        """
[pearl]
rpc_url = "http://127.0.0.1:1"
rpc_user = "toml-user"
rpc_password = "toml-pass"
mining_address = "rprl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq2lzcqla"
""".strip()
    )
    env_file = tmp_path / "gateway.env"
    env_file.write_text('PEARLD_RPC_USER="env-user"\n')
    cfg = NodeRpcConfig.from_sources(
        toml_path=toml,
        env_path=env_file,
        env={"PEARLD_RPC_PASSWORD": "override-pass"},
    )
    assert cfg.rpc_url == "http://127.0.0.1:1"
    assert cfg.rpc_user == "env-user"
    assert cfg.rpc_password == "override-pass"


def test_node_rpc_client_rejects_unsafe_urls_and_methods():
    with pytest.raises(ValueError, match="loopback"):
        from pmk_miner.transport import NodeRpcClient

        NodeRpcClient(NodeRpcConfig("http://192.0.2.1:1", "u", "p"))
    with pytest.raises(ValueError, match="userinfo"):
        from pmk_miner.transport import NodeRpcClient

        NodeRpcClient(NodeRpcConfig("http://u:p@127.0.0.1:1", "u", "p"))
    with pytest.raises(TransportError, match="allowlisted"):
        from pmk_miner.transport import NodeRpcClient

        asyncio.run(NodeRpcClient(NodeRpcConfig("http://127.0.0.1:1", "u", "p")).call("submitblock"))


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class FakeOpener:
    def __init__(self, body: bytes | None = None, exc: Exception | None = None):
        self.body = body
        self.exc = exc

    def open(self, request, timeout):
        if self.exc is not None:
            raise self.exc
        assert self.body is not None
        return FakeResponse(self.body)


def test_node_rpc_redacts_credentials_on_every_error_path():
    cfg = NodeRpcConfig("http://127.0.0.1:1", "alice-secret", "pass-secret")
    cases = (
        FakeOpener(exc=urllib.error.URLError("pass-secret alice-secret")),
        FakeOpener(body=b"{not json pass-secret}"),
        FakeOpener(body=json.dumps(["pass-secret"]).encode()),
        FakeOpener(body=json.dumps({"error": {"message": "alice-secret pass-secret"}}).encode()),
        FakeOpener(body=json.dumps({"result": "pass-secret"}).encode()),
    )
    for opener in cases:
        from pmk_miner.transport import NodeRpcClient

        client = NodeRpcClient(cfg)
        client._opener = opener
        with pytest.raises(TransportError) as caught:
            asyncio.run(client.get_block("hash"))
        rendered = str(caught.value)
        assert "alice-secret" not in rendered
        assert "pass-secret" not in rendered
        assert caught.value.__cause__ is None


class FakeNode:
    def __init__(
        self,
        block: dict | None = None,
        fail: bool = False,
        blocks: dict | None = None,
        delay: float = 0.0,
    ):
        self.block = block
        self.fail = fail
        self.blocks = blocks or {}
        self.delay = delay

    async def get_best_block_hash(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            from pmk_miner.transport import TransportError

            raise TransportError("down")
        return "best"

    async def get_block(self, block_hash: str, verbosity: int = 1):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            from pmk_miner.transport import TransportError

            raise TransportError("down")
        if self.blocks:
            return self.blocks[block_hash]
        return self.block


def test_submission_tracker_accepts_matching_best_block():
    asyncio.run(_test_submission_tracker_accepts_matching_best_block())


async def _test_submission_tracker_accepts_matching_best_block():
    h = header(prev=bytes.fromhex("01" * 32), merkle=bytes.fromhex("02" * 32))
    job = GatewayJob(h, bits_to_target(0x1D00FFFF), 3)
    block = {
        "version": job.version,
        "previousblockhash": job.prev_hash,
        "merkleroot": job.merkle_root,
        "bits": job.bits_hex,
        "time": job.timestamp,
    }
    tracker = SubmissionTracker(FakeNode(block), timeout_seconds=0.01, poll_seconds=0.001)
    assert await tracker.track(job) == SubmissionOutcome.ACCEPTED


def test_submission_tracker_classifies_gateway_log(tmp_path: Path):
    asyncio.run(_test_submission_tracker_classifies_gateway_log(tmp_path))


async def _test_submission_tracker_classifies_gateway_log(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    log.write_text(
        "INFO Block rejected: unrelated negative control\n"
        + json.dumps(
                {
                    "event": "block_rejected",
                    "header_hex": job.header_hex,
                    "stage": "node",
                    "error_type": "node-verdict",
                    "result": "rejected: bad-certificate",
                }
        )
        + "\n"
    )
    tracker = SubmissionTracker(
        FakeNode(fail=True),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.01,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.CONSENSUS_INVALID


def test_submission_tracker_does_not_finalize_on_gateway_accepted_log(tmp_path: Path):
    asyncio.run(_test_submission_tracker_does_not_finalize_on_gateway_accepted_log(tmp_path))


async def _test_submission_tracker_does_not_finalize_on_gateway_accepted_log(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    log.write_text(json.dumps({"event": "block_accepted", "header_hex": job.header_hex}) + "\n")
    tracker = SubmissionTracker(
        FakeNode(block={"version": 99}),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.002,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.UNKNOWN_SUBMISSION


def test_submission_tracker_classifies_gateway_nonadmission_events(tmp_path: Path):
    asyncio.run(_test_submission_tracker_classifies_gateway_nonadmission_events(tmp_path))


async def _test_submission_tracker_classifies_gateway_nonadmission_events(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    cases = (
        ("plain_proof_decode_error", SubmissionOutcome.PROVING_ERROR),
        ("proving_queue_full", SubmissionOutcome.TRANSPORT),
    )
    for event_name, outcome in cases:
        log = tmp_path / f"{event_name}.log"
        log.write_text(json.dumps({"event": event_name, "header_hex": job.header_hex}) + "\n")
        tracker = SubmissionTracker(
            FakeNode(block={"version": 99}),
            gateway_log=GatewayLogTail(log, offset=0),
            timeout_seconds=0.01,
            poll_seconds=0.001,
        )
        assert await tracker.track(job) == outcome


def test_submission_tracker_prioritizes_late_consensus_invalid(tmp_path: Path):
    asyncio.run(_test_submission_tracker_prioritizes_late_consensus_invalid(tmp_path))


async def _test_submission_tracker_prioritizes_late_consensus_invalid(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    log.write_text(
        json.dumps({"event": "stale_plain_proof", "header_hex": job.header_hex})
        + "\n"
        + json.dumps(
            {
                "event": "block_submission_error",
                "header_hex": job.header_hex,
                "stage": "node",
                "error_type": "node-verdict",
                "error": "rejected: bad-certificate",
            }
        )
        + "\n"
    )
    tracker = SubmissionTracker(
        FakeNode(block={"version": 99}),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.01,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.CONSENSUS_INVALID


def test_submission_tracker_sees_consensus_invalid_appended_during_node_query(tmp_path: Path):
    asyncio.run(_test_submission_tracker_sees_consensus_invalid_appended_during_node_query(tmp_path))


async def _test_submission_tracker_sees_consensus_invalid_appended_during_node_query(tmp_path: Path):
    log = tmp_path / "gateway.log"
    log.write_text("")
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)

    class AppendingNode(FakeNode):
        async def get_best_block_hash(self):
            with log.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "event": "block_submission_error",
                            "header_hex": job.header_hex,
                            "stage": "node",
                            "error_type": "node-verdict",
                            "error": "rejected: bad-certificate",
                        }
                    )
                    + "\n"
                )
            raise TransportError("down")

    tracker = SubmissionTracker(
        AppendingNode(),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.001,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.CONSENSUS_INVALID


def test_submission_tracker_generic_block_rejected_is_not_consensus_invalid(tmp_path: Path):
    asyncio.run(_test_submission_tracker_generic_block_rejected_is_not_consensus_invalid(tmp_path))


async def _test_submission_tracker_generic_block_rejected_is_not_consensus_invalid(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    log.write_text(
        json.dumps(
            {
                "event": "block_rejected",
                "header_hex": job.header_hex,
                "result": "node exception while submitting block",
            }
        )
        + "\n"
    )
    tracker = SubmissionTracker(
        FakeNode(block={"version": 99}),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.002,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.UNKNOWN_SUBMISSION


def test_submission_tracker_classifies_correlated_expected_rejections(tmp_path: Path):
    asyncio.run(_test_submission_tracker_classifies_correlated_expected_rejections(tmp_path))


async def _test_submission_tracker_classifies_correlated_expected_rejections(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    cases = (
        ("rejected: duplicate block", SubmissionOutcome.DUPLICATE),
        ("rejected: stale block", SubmissionOutcome.STALE),
        ("rejected: orphan block", SubmissionOutcome.STALE),
    )
    for text, outcome in cases:
        log = tmp_path / f"{outcome}-{len(text)}.log"
        log.write_text(
            json.dumps(
                {
                    "event": "block_rejected",
                    "header_hex": job.header_hex,
                    "result": text,
                }
            )
            + "\n"
        )
        tracker = SubmissionTracker(
            FakeNode(fail=True),
            gateway_log=GatewayLogTail(log, offset=0),
            timeout_seconds=0.01,
            poll_seconds=0.001,
        )
        assert await tracker.track(job) == outcome


def test_submission_tracker_parses_realistic_loguru_json_line(tmp_path: Path):
    asyncio.run(_test_submission_tracker_parses_realistic_loguru_json_line(tmp_path))


async def _test_submission_tracker_parses_realistic_loguru_json_line(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    payload = json.dumps(
        {
            "event": "block_submission_error",
            "header_hex": job.header_hex,
            "error": "cannot pickle 'builtins.PlainProof' object",
        },
        separators=(",", ":"),
    )
    log.write_text(
        f"\x1b[32m2026-10-02 12:00:00.000\x1b[0m | ERROR | {payload} - submission_service.py:123\n"
    )
    tracker = SubmissionTracker(
        FakeNode(fail=True),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.01,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.PROVING_ERROR


def test_submission_tracker_classifies_node_submission_errors(tmp_path: Path):
    asyncio.run(_test_submission_tracker_classifies_node_submission_errors(tmp_path))


async def _test_submission_tracker_classifies_node_submission_errors(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    cases = (
        (
            {
                "event": "block_submission_error",
                "header_hex": job.header_hex,
                "stage": "node",
                "error_type": "node-verdict",
                "error": "Pearl RPC error: rejected: bad-certificate",
            },
            SubmissionOutcome.CONSENSUS_INVALID,
        ),
        (
            {
                "event": "block_submission_error",
                "header_hex": job.header_hex,
                "stage": "node",
                "error_type": "rpc",
                "error": "Pearl RPC exception: rejected: bad-certificate",
            },
            SubmissionOutcome.TRANSPORT,
        ),
        (
            {
                "event": "block_submission_error",
                "header_hex": job.header_hex,
                "phase": "node",
                "error_type": "transport",
                "error": "Connection reset by peer",
            },
            SubmissionOutcome.TRANSPORT,
        ),
        (
            {
                "event": "block_submission_error",
                "header_hex": job.header_hex,
                "classification": "transport",
                "error": "gateway supplied final classification",
            },
            SubmissionOutcome.TRANSPORT,
        ),
    )
    for idx, (payload, outcome) in enumerate(cases):
        log = tmp_path / f"node-error-{idx}.log"
        log.write_text(json.dumps(payload) + "\n")
        tracker = SubmissionTracker(
            FakeNode(fail=True),
            gateway_log=GatewayLogTail(log, offset=0),
            timeout_seconds=0.01,
            poll_seconds=0.001,
        )
        assert await tracker.track(job) == outcome


def test_submission_tracker_ignores_uncorrelated_log_rejection(tmp_path: Path):
    asyncio.run(_test_submission_tracker_ignores_uncorrelated_log_rejection(tmp_path))


async def _test_submission_tracker_ignores_uncorrelated_log_rejection(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    other = GatewayJob(header(timestamp=999), bits_to_target(0x1D00FFFF), 3)
    log.write_text(
        "INFO Block rejected: unrelated plain text\n"
        + json.dumps(
            {
                "event": "block_rejected",
                "header_hex": other.header_hex,
                "result": "rejected: negative-control",
            }
        )
        + "\n"
    )
    tracker = SubmissionTracker(
        FakeNode(fail=True),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.01,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.UNKNOWN_SUBMISSION


def test_submission_tracker_requires_submission_id_match_when_present(tmp_path: Path):
    asyncio.run(_test_submission_tracker_requires_submission_id_match_when_present(tmp_path))


async def _test_submission_tracker_requires_submission_id_match_when_present(tmp_path: Path):
    log = tmp_path / "gateway.log"
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3).with_submission_id("a" * 32)
    log.write_text(
        json.dumps(
            {
                "event": "block_rejected",
                "header_hex": job.header_hex,
                "result": "rejected: bad-certificate",
            }
        )
        + "\n"
    )
    tracker = SubmissionTracker(
        FakeNode(fail=True),
        gateway_log=GatewayLogTail(log, offset=0),
        timeout_seconds=0.01,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.UNKNOWN_SUBMISSION


def test_gateway_log_tail_handles_rotation_and_truncation(tmp_path: Path):
    log = tmp_path / "gateway.log"
    log.write_text("old\n")
    tail = GatewayLogTail(log)
    log.rename(tmp_path / "gateway.log.1")
    log.write_text("new\n")
    assert tail.read_events() == ["new"]
    assert tail.read_events() == []
    log.write_text("x\n")
    assert tail.read_events() == ["x"]


def test_gateway_log_tail_buffers_partial_lines(tmp_path: Path):
    log = tmp_path / "gateway.log"
    log.write_text("")
    tail = GatewayLogTail(log, offset=0)
    log.write_text('{"event":"block_rejected"')
    assert tail.read_events() == []
    with log.open("a") as fh:
        fh.write(',"result":"rejected"}\n')
    assert tail.read_events() == ['{"event":"block_rejected","result":"rejected"}']


def test_gateway_log_tail_drains_rotated_inode(tmp_path: Path):
    log = tmp_path / "gateway.log"
    log.write_text("old\n")
    tail = GatewayLogTail(log)
    rotated = tmp_path / "gateway.log.1"
    log.rename(rotated)
    with rotated.open("a") as fh:
        fh.write("late-old\n")
    log.write_text("new\n")
    assert tail.read_events() == ["late-old", "new"]


def test_submission_tracker_walks_back_from_best_block():
    asyncio.run(_test_submission_tracker_walks_back_from_best_block())


async def _test_submission_tracker_walks_back_from_best_block():
    h = header(prev=bytes.fromhex("01" * 32), merkle=bytes.fromhex("02" * 32))
    job = GatewayJob(h, bits_to_target(0x1D00FFFF), 3)
    unrelated = {
        "version": 1,
        "previousblockhash": "match",
        "merkleroot": "00",
        "bits": "1d00ffff",
        "time": 1,
    }
    matched = {
        "version": job.version,
        "previousblockhash": job.prev_hash,
        "merkleroot": job.merkle_root,
        "bits": job.bits_hex,
        "time": job.timestamp,
    }
    tracker = SubmissionTracker(
        FakeNode(blocks={"best": unrelated, "match": matched}),
        timeout_seconds=0.01,
        poll_seconds=0.001,
    )
    assert await tracker.track(job) == SubmissionOutcome.ACCEPTED


def test_submission_tracker_transport_timeout():
    asyncio.run(_test_submission_tracker_transport_timeout())


async def _test_submission_tracker_transport_timeout():
    tracker = SubmissionTracker(FakeNode(fail=True), timeout_seconds=0.001, poll_seconds=0.001)
    assert (
        await tracker.track(GatewayJob(header(), bits_to_target(0x1D00FFFF), 3))
        == SubmissionOutcome.UNKNOWN_SUBMISSION
    )


def test_submission_tracker_overall_deadline_bounds_slow_node_calls():
    asyncio.run(_test_submission_tracker_overall_deadline_bounds_slow_node_calls())


async def _test_submission_tracker_overall_deadline_bounds_slow_node_calls():
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    tracker = SubmissionTracker(FakeNode(delay=0.05), timeout_seconds=0.005, poll_seconds=0.001)
    assert await tracker.track(job) == SubmissionOutcome.UNKNOWN_SUBMISSION


def test_submission_tracker_unknown_submission_without_verdict():
    asyncio.run(_test_submission_tracker_unknown_submission_without_verdict())


async def _test_submission_tracker_unknown_submission_without_verdict():
    tracker = SubmissionTracker(
        FakeNode(block={"version": 99}), timeout_seconds=0.002, poll_seconds=0.001
    )
    assert (
        await tracker.track(GatewayJob(header(), bits_to_target(0x1D00FFFF), 3))
        == SubmissionOutcome.UNKNOWN_SUBMISSION
    )


def test_submission_ledger_persists_outstanding_and_deduplicates_restart(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    first = ledger.prepare(job, b"proof")
    assert ledger.seen_proof(job, "cHJvb2Y=")
    restarted = SubmissionLedger(path)
    assert restarted.prepare(job, "cHJvb2Y=").submission_id == first.submission_id
    assert [entry.submission_id for entry in restarted.outstanding()] == [first.submission_id]
    restarted.finish(first.submission_id, SubmissionOutcome.ACCEPTED)
    assert SubmissionLedger(path).outstanding() == []


def test_submission_ledger_rejects_conflicting_duplicate_prepared_id(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    entry = ledger.prepare(job, b"proof")
    row = {
        "event": "prepared",
        "submission_id": entry.submission_id,
        "template_identity": entry.template_identity,
        "header_hex": entry.header_hex,
        "proof_hash": "0" * 64,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(TransportError, match="invalid prepared"):
        SubmissionLedger(path)


def test_submission_ledger_rejects_template_header_mismatch(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    entry = ledger.prepare(job, b"proof")
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["header_hex"] = ("00" * 76)
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(TransportError, match="invalid prepared"):
        SubmissionLedger(path)


def test_submission_ledger_persists_fail_closed_terminal_outcome(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    entry = ledger.prepare(job, b"proof")
    ledger.finish(entry.submission_id, SubmissionOutcome.UNKNOWN_SUBMISSION)
    restarted = SubmissionLedger(tmp_path / "ledger.jsonl")
    assert restarted.entries()[entry.submission_id].outcome == SubmissionOutcome.UNKNOWN_SUBMISSION
    assert restarted.fail_closed_entries()[0].submission_id == entry.submission_id


def test_submission_ledger_terminal_outcome_is_immutable(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    entry = ledger.prepare(job, b"proof")
    ledger.finish(entry.submission_id, SubmissionOutcome.CONSENSUS_INVALID)
    assert ledger.finish(entry.submission_id, SubmissionOutcome.CONSENSUS_INVALID).outcome == SubmissionOutcome.CONSENSUS_INVALID
    with pytest.raises(TransportError, match="already has terminal outcome"):
        ledger.finish(entry.submission_id, SubmissionOutcome.ACCEPTED)
    assert SubmissionLedger(path).entries()[entry.submission_id].outcome == SubmissionOutcome.CONSENSUS_INVALID


def test_submission_ledger_rejects_rewritten_terminal_outcome_on_restart(tmp_path: Path):
    job = GatewayJob(header(), bits_to_target(0x1D00FFFF), 3)
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    entry = ledger.prepare(job, b"proof")
    ledger.finish(entry.submission_id, SubmissionOutcome.CONSENSUS_INVALID)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "finished", "submission_id": entry.submission_id,
                             "outcome": SubmissionOutcome.ACCEPTED.value},
                            sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(TransportError, match="invalid finished row"):
        SubmissionLedger(path)


def test_submission_ledger_persists_pool_metadata_immutably(tmp_path: Path):
    h = header(timestamp=654321)
    job = type("PoolJob", (), {})()
    job.header = h
    job.header_hex = h.hex()
    job.template_identity = h.hex()
    job.pool_job_id = "00000001_1000"
    job.target = 123456789
    job.share_nbits = 0x1A07FFF8
    job.session_id = "session-a"
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    entry = ledger.prepare(job, b"proof", cfg=b"cfg")
    assert entry.source == "pool"
    assert entry.pool_job_id == job.pool_job_id
    assert entry.target == job.target
    assert entry.share_nbits == job.share_nbits
    assert entry.cfg_hex == b"cfg".hex()
    assert entry.session_id == "session-a"
    restarted = SubmissionLedger(path)
    assert restarted.prepare(job, "cHJvb2Y=", cfg=b"cfg").submission_id == entry.submission_id
    restarted.finish(entry.submission_id, SubmissionOutcome.ACCEPTED)
    finished = SubmissionLedger(path).entries()[entry.submission_id]
    assert finished.source == "pool"
    assert finished.pool_job_id == job.pool_job_id
    assert finished.outcome == SubmissionOutcome.ACCEPTED


def test_submission_ledger_rejects_rewritten_pool_metadata_on_restart(tmp_path: Path):
    h = header(timestamp=654321)
    job = type("PoolJob", (), {})()
    job.header = h
    job.template_identity = h.hex()
    job.pool_job_id = "00000001_1000"
    job.target = 123456789
    job.share_nbits = 0x1A07FFF8
    path = tmp_path / "ledger.jsonl"
    ledger = SubmissionLedger(path)
    entry = ledger.prepare(job, b"proof", cfg=b"cfg", session_id="session-a")
    row = {
        "event": "prepared",
        "submission_id": entry.submission_id,
        "template_identity": entry.template_identity,
        "header_hex": entry.header_hex,
        "proof_hash": entry.proof_hash,
        "source": "pool",
        "pool_job_id": "00000002_1000",
        "target": entry.target,
        "share_nbits": entry.share_nbits,
        "cfg_hex": entry.cfg_hex,
        "session_id": entry.session_id,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(TransportError, match="invalid prepared"):
        SubmissionLedger(path)


def test_submission_ledger_malformed_rows_fail_closed(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(TransportError, match="malformed"):
        SubmissionLedger(path)


def test_redact_credentials():
    text = "rpc_user=alice rpc_password=supersecret token:abcdef123456"
    redacted = redact_credentials(text)
    assert "supersecret" not in redacted
    assert "abcdef123456" not in redacted
    assert "rpc_password=<redacted>" in redacted
    assert "token:<redacted>" in redacted
