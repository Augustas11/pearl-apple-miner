# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import json
import time
import contextlib
from collections import deque
from pathlib import Path

import pytest

import pmk_miner.pool as pool_mod
from pmk_miner.monitor import DIFF1_TARGET, bits_to_target
from pmk_miner.v4_admission import V4_G3_ADMISSION_ENV, V4_VENDOR_PEARL_FP8
from pmk_miner.pool import (
    PoolClient,
    PoolMessage,
    PoolProtocolError,
    PoolTransportError,
    PoolSubmitOutcome,
    classify_submit_response,
    parse_pool_message,
    pool_target_for_difficulty,
    target_to_share_nbits,
    validate_pool_notify,
)


def write_v4_admission(tmp_path, monkeypatch, **overrides):
    now=time.time()
    record={"schema":"pmk-v4-admission-v1","gpu_name":"Unit GPU","device_class":"Apple7-9",
        "os_build":"unit-os","cache_key":"unit-cache","kernel":"E","metal_language":"3.1",
        "library_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vendor_pearl_fp8":V4_VENDOR_PEARL_FP8,"v4_g3_passed":True,"exact_cells":100_000_000,
        "last_probe_unix":now-1,"valid_until_unix":now+3600}
    record.update(overrides)
    path=tmp_path / "v4-admission.json"
    path.write_text(json.dumps({"devices":[record]}), encoding="utf-8")
    monkeypatch.setenv(V4_G3_ADMISSION_ENV,str(path))
    return path


def header(*, bits: int = 0x1D00FFFF) -> bytes:
    return (
        (2).to_bytes(4, "little")
        + b"\x11" * 32
        + b"\x22" * 32
        + (123456).to_bytes(4, "little")
        + bits.to_bytes(4, "little")
    )


T0_DOCUMENTED_FIXTURE = Path(__file__).with_name("fixtures") / "pool_t0_documented.json"


def notify_params(**overrides):
    """Build synthetic notify parameters for validation and mock-pool tests.

    These values follow the documented object dialect, but they are generated
    test data rather than recorded pool traffic.
    """
    params = {
        "job_id": "00000000_2097152",
        "header": header(bits=0x1B00FFFF).hex(),
        "target": f"{pool_target_for_difficulty(2**21):064x}",
        "height": 101009,
        "cert_version": 3,
    }
    params.update(overrides)
    return params


def frame(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode() + b"\n"


def test_documented_t0_notify_fixture_has_honest_provenance_and_parses():
    fixture = json.loads(T0_DOCUMENTED_FIXTURE.read_text())
    provenance = fixture["provenance"]
    assert provenance["classification"] == "documented_protocol_example_with_synthetic_substitutions"
    assert provenance["recorded_traffic"] is False
    message = parse_pool_message(fixture["frames"]["valid_notify"].encode())
    assert message.method == "mining.notify"
    job = validate_pool_notify(message.params, session_id=7)
    assert job.pool_job_id == "00000000_2097152"
    assert job.height == 101009
    assert job.cert_version == 3
    assert job.share_nbits == 0x1A07FFF8


def test_valid_notify_builds_immutable_pool_job():
    job = validate_pool_notify(notify_params(), session_id=7)
    assert job.pool_job_id == "00000000_2097152"
    assert job.header == header(bits=0x1B00FFFF)
    assert job.target == pool_target_for_difficulty(2**21)
    assert job.block_target == bits_to_target(0x1B00FFFF)
    assert job.cert_version == 3
    assert job.height == 101009
    assert job.share_nbits == 0x1A07FFF8
    assert job.session_id == 7
    assert job.template_identity == job.header.hex()


def test_notify_rejects_missing_cert_version():
    params = notify_params()
    del params["cert_version"]
    with pytest.raises(PoolProtocolError, match="cert_version"):
        validate_pool_notify(params, session_id=1)


def test_v4_notify_accepts_explicit_ancestor_header_extension(tmp_path, monkeypatch):
    write_v4_admission(tmp_path,monkeypatch)
    ancestor = b"a" * 108
    job = validate_pool_notify(
        notify_params(cert_version=4, ancestor_headers=[base64.b64encode(ancestor).decode()]),
        session_id=1,
    )
    assert job.cert_version == 4
    assert job.ancestor_headers == (ancestor,)
    assert job.template_identity.endswith(job.header.hex())


def test_notify_rejects_unknown_cert_version():
    with pytest.raises(PoolProtocolError, match="cert_version=3 or cert_version=4"):
        validate_pool_notify(notify_params(cert_version=5), session_id=1)


@pytest.mark.parametrize("hex_len", [150, 151, 153, 154])
def test_notify_rejects_bad_header_lengths(hex_len):
    with pytest.raises(PoolProtocolError, match="152"):
        validate_pool_notify(notify_params(header="a" * hex_len), session_id=1)


def test_notify_rejects_zero_target():
    with pytest.raises(PoolProtocolError, match="nonzero"):
        validate_pool_notify(notify_params(target="0" * 64), session_id=1)


def test_array_params_rejected():
    with pytest.raises(PoolProtocolError, match="params"):
        parse_pool_message(frame({"id": None, "method": "mining.notify", "params": []}))


def test_oversized_and_non_utf8_lines_are_rejected():
    with pytest.raises(PoolProtocolError, match="64 KiB"):
        parse_pool_message(b"a" * (70 * 1024))
    with pytest.raises(PoolProtocolError, match="UTF-8"):
        parse_pool_message(b"\xff\n")


@pytest.mark.parametrize(
    ("difficulty", "expected_bits"),
    [
        (2**21, 0x1A07FFF8),
        (2_000_000, 0x1A086373),
        (50_000, 0x1B014F8A),
        (10_000, 0x1B068DB2),
    ],
)
def test_target_to_nbits_vectors_round_down(difficulty, expected_bits):
    target = DIFF1_TARGET // difficulty
    bits = target_to_share_nbits(target)
    assert bits == expected_bits
    assert bits_to_target(bits) <= target


def test_difficulty_floor_rejects_easy_target():
    with pytest.raises(PoolProtocolError, match="difficulty floor"):
        validate_pool_notify(notify_params(target=f"{pool_target_for_difficulty(9999):064x}"), session_id=1)


def test_notify_rejects_config_fields_and_non_strict_integers():
    with pytest.raises(PoolProtocolError, match="configuration"):
        validate_pool_notify(notify_params(k=4096), session_id=1)
    with pytest.raises(PoolProtocolError, match="cert_version"):
        validate_pool_notify(notify_params(cert_version=True), session_id=1)
    with pytest.raises(PoolProtocolError, match="height"):
        validate_pool_notify(notify_params(height=True), session_id=1)


def test_notify_rejects_noncanonical_compact_nbits():
    with pytest.raises(PoolProtocolError, match="invalid nbits"):
        validate_pool_notify(notify_params(header=header(bits=0x1D800000).hex()), session_id=1)


def test_tighter_than_block_target_logs_but_accepts():
    seen = []
    job = validate_pool_notify(
        notify_params(header=header(bits=0x1D00FFFF).hex()),
        session_id=1,
        log=lambda event, **kw: seen.append((event, kw)),
    )
    assert job.block_target == bits_to_target(0x1D00FFFF)
    assert seen[0][0] == "pool_target_tighter_than_block_target"


def test_submit_outcome_classification():
    accepted = parse_pool_message(frame({"id": 1, "result": True, "error": None}))
    stale = parse_pool_message(frame({"id": 1, "result": False, "error": "stale share"}))
    duplicate = parse_pool_message(frame({"id": 1, "error": {"message": "duplicate"}}))
    low = parse_pool_message(frame({"id": 1, "error": "low difficulty"}))
    invalid = parse_pool_message(frame({"id": 1, "result": False, "error": "Jackpot condition not satisfied"}))
    assert classify_submit_response(accepted) == PoolSubmitOutcome.ACCEPTED
    assert classify_submit_response(stale) == PoolSubmitOutcome.STALE
    assert classify_submit_response(duplicate) == PoolSubmitOutcome.DUPLICATE
    assert classify_submit_response(low) == PoolSubmitOutcome.LOW_DIFFICULTY
    assert classify_submit_response(invalid) == PoolSubmitOutcome.INVALID


def test_client_sanitizes_wallet_in_internal_logs():
    wallet = "secretwallet123456"
    logs = []
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        wallet,
        "rig",
        reply_timeout=0.01,
        notify_interval=0.0,
        log=lambda event, **kw: logs.append((event, kw)),
    )
    client.session_id = 1
    client._authorized = True
    client._handle_notify(
        PoolMessage(
            id=None,
            method="mining.notify",
            params=notify_params(
                job_id=f"{wallet}_job",
                header=header(bits=0x1D00FFFF).hex(),
            ),
        )
    )
    assert logs
    assert wallet not in json.dumps(logs)
    assert "secret...123456" in json.dumps(logs)


def test_prev_block_inconsistency_at_same_height_is_logged():
    logs = []
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
        log=lambda event, **kw: logs.append((event, kw)),
        notify_interval=0.0,
    )
    client.session_id = 1
    client._authorized = True
    client._handle_notify(PoolMessage(id=None, method="mining.notify", params=notify_params()))
    changed = header(bits=0x1B00FFFF)[:4] + b"\x33" * 32 + header(bits=0x1B00FFFF)[36:]
    client._handle_notify(
        PoolMessage(
            id=None,
            method="mining.notify",
            params=notify_params(job_id="other", header=changed.hex()),
        )
    )
    assert ("pool_prev_block_inconsistent", {"height": 101009}) in logs


def test_prev_block_tracking_is_session_bounded():
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
        notify_interval=0.0,
    )
    client.session_id = 1
    client._authorized = True
    for height in range(200):
        client._handle_notify(
            PoolMessage(
                id=None,
                method="mining.notify",
                params=notify_params(job_id=f"job-{height}", height=height),
            )
        )
    assert len(client._prev_by_height) == 128


def test_reply_id_interleaving_and_submit_uses_original_job_id():
    asyncio.run(_reply_id_interleaving_and_submit_uses_original_job_id())


async def _reply_id_interleaving_and_submit_uses_original_job_id():
    seen_requests = []
    submitted_job_ids = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        authorize = json.loads(await reader.readline())
        seen_requests.append(authorize)
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="job-old")}))
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        await writer.drain()

        submit = json.loads(await reader.readline())
        submitted_job_ids.append(submit["params"]["job_id"])
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="job-new")}))
        writer.write(frame({"id": submit["id"], "result": True, "error": None}))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.0,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        for _ in range(50):
            if client.latest is not None:
                break
            await asyncio.sleep(0.01)
        old_job = client.latest
        assert old_job is not None
        outcome = await client.submit(old_job, b"plain-proof")
        assert outcome == "accepted"
        assert submitted_job_ids == ["job-old"]
        assert seen_requests[0]["params"] == {
            "wallet": "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
            "worker": "rig1",
            "pass": "x",
            "agent": "pmk/0.1.0",
        }
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_coalesces_notifies_to_newest_job():
    asyncio.run(_coalesces_notifies_to_newest_job())


async def _coalesces_notifies_to_newest_job():
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="first")}))
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="second")}))
        await writer.drain()
        await asyncio.sleep(0.2)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.05,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.sleep(0.12)
        assert client.latest is not None
        assert client.latest.pool_job_id == "second"
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_bad_cert_notify_fails_run_and_invalidates_session():
    asyncio.run(_bad_cert_notify_fails_run_and_invalidates_session())


async def _bad_cert_notify_fails_run_and_invalidates_session():
    logs = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(cert_version=5)}))
        await writer.drain()
        await asyncio.sleep(0.1)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.0,
        log=lambda event, **kw: logs.append((event, kw)),
    )
    stop = asyncio.Event()
    with pytest.raises(PoolProtocolError, match="cert_version"):
        await client.run(stop)
    assert client.latest is None
    assert client.session_id > 1
    assert ("pool_job_rejected", {"reason": "R-A6_cert_version"}) in logs
    server.close()
    await server.wait_closed()


def test_notify_before_authorize_reply_is_not_published_until_authorized():
    asyncio.run(_notify_before_authorize_reply_is_not_published_until_authorized())


async def _notify_before_authorize_reply_is_not_published_until_authorized():
    got_authorize = asyncio.Event()
    release_authorize = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="early")}))
        await writer.drain()
        got_authorize.set()
        await release_authorize.wait()
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.0,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.wait_for(got_authorize.wait(), 1.0)
        await asyncio.sleep(0.02)
        assert client.latest is None
        release_authorize.set()
        for _ in range(50):
            if client.latest is not None:
                break
            await asyncio.sleep(0.01)
        assert client.latest is not None
        assert client.latest.pool_job_id == "early"
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_malformed_framing_reconnects_and_does_not_set_fatal(monkeypatch):
    asyncio.run(_malformed_framing_reconnects_and_does_not_set_fatal(monkeypatch))


async def _malformed_framing_reconnects_and_does_not_set_fatal(monkeypatch):
    monkeypatch.setattr(pool_mod, "_jitter", lambda backoff: 0.01)
    connections = 0

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        if connections == 1:
            writer.write(b"\xff\n")
        else:
            writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="after-reconnect")}))
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.0,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        for _ in range(100):
            if client.latest is not None:
                break
            await asyncio.sleep(0.01)
        assert client.latest is not None
        assert client.latest.pool_job_id == "after-reconnect"
        assert connections >= 2
        assert client.fatal is None
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_reader_failure_invalidates_current_job_immediately(monkeypatch):
    asyncio.run(_reader_failure_invalidates_current_job_immediately(monkeypatch))


async def _reader_failure_invalidates_current_job_immediately(monkeypatch):
    monkeypatch.setattr(pool_mod, "_jitter", lambda backoff: 5.0)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="will-drop")}))
        await writer.drain()
        await asyncio.sleep(0.02)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.0,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        for _ in range(50):
            if client.latest is not None:
                break
            await asyncio.sleep(0.01)
        old_session = client.session_id
        assert client.latest is not None
        for _ in range(50):
            if client.latest is None and client.session_id > old_session:
                break
            await asyncio.sleep(0.01)
        assert client.latest is None
        assert client.session_id > old_session
        assert client.fatal is None
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_submit_timeout_and_first_reject_do_not_set_transport_fatal():
    asyncio.run(_submit_timeout_and_first_reject_do_not_set_transport_fatal())


async def _submit_timeout_and_first_reject_do_not_set_transport_fatal():
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=0.03,
        notify_interval=0.0,
    )
    job = validate_pool_notify(notify_params(job_id="timeout-job"), session_id=1)
    client.session_id = 1

    async def timeout_rpc(method, params):
        raise asyncio.TimeoutError

    client._rpc = timeout_rpc
    assert await client.submit(job, b"proof") == "timeout"
    assert client.fatal is None

    async def invalid_rpc(method, params):
        return PoolMessage(id=1, method=None, params=None, result=False, error="Jackpot condition not satisfied")

    client._rpc = invalid_rpc
    assert await client.submit(job, b"proof") == "invalid"
    assert client.fatal is None


def test_submit_accepts_outgoing_proof_larger_than_inbound_frame_cap():
    asyncio.run(_submit_accepts_outgoing_proof_larger_than_inbound_frame_cap())


async def _submit_accepts_outgoing_proof_larger_than_inbound_frame_cap():
    seen_len = 0

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal seen_len
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        writer.write(frame({"id": None, "method": "mining.notify", "params": notify_params(job_id="big-proof")}))
        await writer.drain()
        submit = json.loads(await reader.readline())
        seen_len = len(submit["params"]["plain_proof"])
        writer.write(frame({"id": submit["id"], "result": True, "error": None}))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, limit=256 * 1024)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig1",
        reply_timeout=1.0,
        notify_interval=0.0,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        for _ in range(50):
            if client.latest is not None:
                break
            await asyncio.sleep(0.01)
        assert client.latest is not None
        assert await client.submit(client.latest, b"x" * (70 * 1024)) == "accepted"
        assert seen_len > 64 * 1024
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_rpc_writer_oserror_becomes_transport_error():
    asyncio.run(_rpc_writer_oserror_becomes_transport_error())


async def _rpc_writer_oserror_becomes_transport_error():
    class BadWriter:
        def write(self, data):
            pass

        async def drain(self):
            raise OSError("broken pipe")

    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
        reply_timeout=0.1,
    )
    client._writer = BadWriter()
    with pytest.raises(PoolTransportError, match="write failed"):
        await client._rpc("mining.submit", {})


def test_connect_authorize_obeys_reply_timeout(monkeypatch):
    asyncio.run(_connect_authorize_obeys_reply_timeout(monkeypatch))


async def _connect_authorize_obeys_reply_timeout(monkeypatch):
    original_sleep = asyncio.sleep

    async def slow_open_connection(*args, **kwargs):
        await original_sleep(1.0)

    monkeypatch.setattr(pool_mod.asyncio, "open_connection", slow_open_connection)
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
        reply_timeout=0.01,
    )
    with pytest.raises(asyncio.TimeoutError):
        await client._connect_authorize()


def test_submit_concurrency_cap_is_four():
    asyncio.run(_submit_concurrency_cap_is_four())


async def _submit_concurrency_cap_is_four():
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
        notify_interval=0.0,
    )
    job = validate_pool_notify(notify_params(), session_id=1)
    client.session_id = 1
    active = 0
    max_active = 0
    release = asyncio.Event()

    async def fake_rpc(method, params):
        nonlocal active, max_active
        assert method == "mining.submit"
        active += 1
        max_active = max(max_active, active)
        await release.wait()
        active -= 1
        return PoolMessage(id=1, method=None, params=None, result=True, error=None)

    client._rpc = fake_rpc
    tasks = [asyncio.create_task(client.submit(job, b"proof")) for _ in range(6)]
    try:
        for _ in range(50):
            if max_active == 4:
                break
            await asyncio.sleep(0.01)
        assert max_active == 4
        release.set()
        assert await asyncio.gather(*tasks) == ["accepted"] * 6
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


def test_submit_per_minute_rate_cap_waits(monkeypatch):
    asyncio.run(_submit_per_minute_rate_cap_waits(monkeypatch))


async def _submit_per_minute_rate_cap_waits(monkeypatch):
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
    )
    now = [100.0]
    sleeps = []
    client._submit_times = deque(41.0 + i for i in range(30))

    async def fake_sleep(delay):
        sleeps.append(delay)
        now[0] += delay + 0.001

    monkeypatch.setattr(pool_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pool_mod, "_sleep", fake_sleep)
    assert await client._wait_submit_rate_limit(client.session_id)
    assert sleeps and sleeps[0] > 0
    assert len(client._submit_times) == 30


def test_reconnect_jitter_is_clamped_to_minimum(monkeypatch):
    monkeypatch.setattr(pool_mod.random, "uniform", lambda low, high: 0.75)
    assert pool_mod._jitter(1.0) >= 1.0


def test_submit_on_invalidated_session_is_transport_not_stale():
    asyncio.run(_submit_on_invalidated_session_is_transport_not_stale())


async def _submit_on_invalidated_session_is_transport_not_stale():
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1",
        "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        "rig",
    )
    job = validate_pool_notify(notify_params(), session_id=1)
    client.session_id = 2
    assert await client.submit(job, b"proof") == "transport"


@pytest.mark.parametrize("url", ["stratum+tcp://user:secret@localhost:1", "stratum+tcp://localhost:0", "stratum+tcp://localhost:1/path", "stratum+tcp://localhost:1?wallet=secret"])
def test_endpoint_rejects_credential_or_path_urls(url):
    with pytest.raises(ValueError):
        pool_mod.PoolEndpoint.parse(url)


@pytest.mark.parametrize("message,outcome", [("config not allowed", "invalid"), ("unknown job_id", "stale"), ("Jackpot condition not satisfied", "invalid"), ("low difficulty share", "low-difficulty")])
def test_specific_rejection_messages(message, outcome):
    reply = PoolMessage(1, None, None, False, {"msg": message})
    assert str(classify_submit_response(reply)) == outcome
