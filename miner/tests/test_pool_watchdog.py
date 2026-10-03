from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import pytest

import pmk_miner.pool as pool_mod
from pmk_miner.pool import PoolClient


WALLET = "prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"


def frame(obj: object) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode() + b"\n"


def notify_params(job_id: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "header": (
            (2).to_bytes(4, "little")
            + b"\x11" * 32
            + b"\x22" * 32
            + (123456).to_bytes(4, "little")
            + (0x1B00FFFF).to_bytes(4, "little")
        ).hex(),
        "target": f"{pool_mod.pool_target_for_difficulty(2**21):064x}",
        "height": 101009,
        "cert_version": 3,
    }


@pytest.mark.parametrize("value", [False, 0, 29.999, 1800.001, "180"])
def test_silence_timeout_rejects_values_outside_library_range(value):
    with pytest.raises(ValueError, match="between 30 and 1800"):
        PoolClient("stratum+tcp://127.0.0.1:1", WALLET, "rig", silence_timeout=value)


@pytest.mark.parametrize("value", [30, 180, 1800])
def test_silence_timeout_accepts_documented_range_and_defaults(value):
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1", WALLET, "rig", silence_timeout=value
    )
    assert client.silence_timeout == float(value)
    assert PoolClient("stratum+tcp://127.0.0.1:1", WALLET, "rig").silence_timeout == 180.0


def test_socket_keepalive_enables_supported_tcp_probes(monkeypatch):
    calls: list[tuple[int, int, int]] = []

    class FakeSocket:
        def setsockopt(self, level, option, value):
            calls.append((level, option, value))

    class FakeWriter:
        def get_extra_info(self, name):
            assert name == "socket"
            return FakeSocket()

    monkeypatch.setattr(pool_mod.socket, "TCP_KEEPALIVE", 0x101, raising=False)
    monkeypatch.setattr(pool_mod.socket, "TCP_KEEPIDLE", 0x102, raising=False)
    monkeypatch.setattr(pool_mod.socket, "TCP_KEEPINTVL", 0x103, raising=False)
    monkeypatch.setattr(pool_mod.socket, "TCP_KEEPCNT", 0x104, raising=False)
    client = PoolClient(
        "stratum+tcp://127.0.0.1:1", WALLET, "rig", silence_timeout=180
    )
    client._writer = FakeWriter()

    client._configure_socket_keepalive()

    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in calls
    assert (socket.IPPROTO_TCP, 0x101, 60) in calls
    assert (socket.IPPROTO_TCP, 0x102, 60) in calls
    assert (socket.IPPROTO_TCP, 0x103, 10) in calls
    assert (socket.IPPROTO_TCP, 0x104, 3) in calls


def test_socket_keepalive_values_are_applied_to_real_loopback_socket():
    asyncio.run(_socket_keepalive_values_are_applied_to_real_loopback_socket())


async def _socket_keepalive_values_are_applied_to_real_loopback_socket():
    release = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        await release.wait()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}", WALLET, "rig", silence_timeout=180
    )
    client._writer = writer
    try:
        client._configure_socket_keepalive()
        sock = writer.get_extra_info("socket")
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
        expected = {
            "TCP_KEEPALIVE": 60,
            "TCP_KEEPIDLE": 60,
            "TCP_KEEPINTVL": 10,
            "TCP_KEEPCNT": 3,
        }
        checked = set()
        for name, value in expected.items():
            option = getattr(socket, name, None)
            if option is None or option in checked:
                continue
            checked.add(option)
            try:
                configured = sock.getsockopt(socket.IPPROTO_TCP, option)
            except OSError:
                continue
            assert configured == value
    finally:
        release.set()
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()


def test_mute_open_pool_times_out_and_reconnects(monkeypatch):
    asyncio.run(_mute_open_pool_times_out_and_reconnects(monkeypatch))


async def _mute_open_pool_times_out_and_reconnects(monkeypatch):
    monkeypatch.setattr(pool_mod, "_jitter", lambda backoff: 0.01)
    connections = 0
    second_connection = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        await reader.readline()
        if connections >= 2:
            second_connection.set()
        try:
            await asyncio.sleep(1.0)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        WALLET,
        "rig",
        reply_timeout=1.0,
        silence_timeout=30,
    )
    client._silence_timeout_seconds = 0.04
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.wait_for(second_connection.wait(), timeout=1.0)
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


def test_valid_reply_and_notify_each_extend_silence_deadline():
    asyncio.run(_valid_reply_and_notify_each_extend_silence_deadline())


async def _valid_reply_and_notify_each_extend_silence_deadline():
    connections = 0
    notify_received = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        authorize = json.loads(await reader.readline())
        await asyncio.sleep(0.06)
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        await writer.drain()
        await asyncio.sleep(0.06)
        writer.write(
            frame(
                {
                    "id": None,
                    "method": "mining.notify",
                    "params": notify_params("watchdog-job"),
                }
            )
        )
        await writer.drain()
        notify_received.set()
        await asyncio.sleep(1.0)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        WALLET,
        "rig",
        reply_timeout=1.0,
        silence_timeout=30,
        notify_interval=0.0,
    )
    client._silence_timeout_seconds = 0.1
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.wait_for(notify_received.wait(), timeout=1.0)
        await asyncio.sleep(0.03)
        assert connections == 1
        assert client.latest is not None
        assert client.latest.pool_job_id == "watchdog-job"
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()


def test_blackholed_loopback_proxy_reconnects_without_peer_eof(monkeypatch):
    asyncio.run(_blackholed_loopback_proxy_reconnects_without_peer_eof(monkeypatch))


async def _blackholed_loopback_proxy_reconnects_without_peer_eof(monkeypatch):
    """The proxy drops application bytes while both TCP legs remain open.

    Kernel keepalive probes still succeed against the local proxy. This proves
    the application silence watchdog bounds that blackholed route; it does not
    simulate a kernel-level packet blackhole.
    """
    monkeypatch.setattr(pool_mod, "_jitter", lambda backoff: 0.01)
    backend_connections = 0
    proxy_connections = 0
    blackhole_armed = asyncio.Event()
    submitted_job_ids: list[str] = []
    backend_writers: set[asyncio.StreamWriter] = set()

    async def backend(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal backend_connections
        backend_connections += 1
        connection_id = backend_connections
        backend_writers.add(writer)
        try:
            authorize = json.loads(await reader.readline())
            writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
            writer.write(
                frame(
                    {
                        "id": None,
                        "method": "mining.notify",
                        "params": notify_params(f"proxy-job-{connection_id}"),
                    }
                )
            )
            await writer.drain()
            while line := await reader.readline():
                message = json.loads(line)
                if message.get("method") == "mining.submit":
                    submitted_job_ids.append(message["params"]["job_id"])
                    writer.write(
                        frame({"id": message["id"], "result": True, "error": None})
                    )
                    await writer.drain()
        finally:
            backend_writers.discard(writer)
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    backend_server = await asyncio.start_server(backend, "127.0.0.1", 0)
    backend_port = backend_server.sockets[0].getsockname()[1]

    async def proxy(down_reader: asyncio.StreamReader, down_writer: asyncio.StreamWriter):
        nonlocal proxy_connections
        proxy_connections += 1
        connection_id = proxy_connections
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", backend_port)
        blackhole = asyncio.Event()

        async def toward_backend():
            while data := await down_reader.read(64 * 1024):
                if blackhole.is_set():
                    continue
                up_writer.write(data)
                await up_writer.drain()

        async def toward_client():
            while data := await up_reader.readline():
                if blackhole.is_set():
                    continue
                down_writer.write(data)
                await down_writer.drain()
                if connection_id == 1:
                    with contextlib.suppress(json.JSONDecodeError):
                        message = json.loads(data)
                        if message.get("method") == "mining.notify":
                            blackhole.set()
                            blackhole_armed.set()

        tasks = {
            asyncio.create_task(toward_backend()),
            asyncio.create_task(toward_client()),
        }
        try:
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                    await task
        finally:
            up_writer.close()
            down_writer.close()
            with contextlib.suppress(ConnectionError):
                await up_writer.wait_closed()
            with contextlib.suppress(ConnectionError):
                await down_writer.wait_closed()

    proxy_server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    proxy_port = proxy_server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{proxy_port}",
        WALLET,
        "rig",
        reply_timeout=1.0,
        silence_timeout=30,
        notify_interval=0.0,
    )
    client._silence_timeout_seconds = 0.1
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.wait_for(blackhole_armed.wait(), timeout=1.0)
        for _ in range(50):
            if client.latest is not None and client.latest.pool_job_id == "proxy-job-1":
                break
            await asyncio.sleep(0.005)
        old_job = client.latest
        assert old_job is not None
        assert old_job.pool_job_id == "proxy-job-1"
        old_session = old_job.session_id

        for _ in range(200):
            if client.latest is not None and client.latest.pool_job_id == "proxy-job-2":
                break
            await asyncio.sleep(0.005)
        assert proxy_connections >= 2
        assert client.latest is not None
        assert client.latest.pool_job_id == "proxy-job-2"
        assert client.latest.session_id > old_session
        assert await client.submit(old_job, b"stale-session-proof") == "transport"
        await asyncio.sleep(0)
        assert "proxy-job-1" not in submitted_job_ids
        assert client.fatal is None
    finally:
        stop.set()
        await client.close()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        proxy_server.close()
        backend_server.close()
        await proxy_server.wait_closed()
        await backend_server.wait_closed()
        for writer in tuple(backend_writers):
            writer.close()


def test_peer_eof_is_detected_and_reconnects(monkeypatch):
    asyncio.run(_peer_eof_is_detected_and_reconnects(monkeypatch))


async def _peer_eof_is_detected_and_reconnects(monkeypatch):
    """Loopback EOF covers peer-close detection, not a packet-blackhole simulation."""
    monkeypatch.setattr(pool_mod, "_jitter", lambda backoff: 0.01)
    connections = 0
    second_connection = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        authorize = json.loads(await reader.readline())
        writer.write(frame({"id": authorize["id"], "result": True, "error": None}))
        await writer.drain()
        if connections == 1:
            writer.close()
            await writer.wait_closed()
        else:
            second_connection.set()
            await asyncio.sleep(1.0)
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = PoolClient(
        f"stratum+tcp://127.0.0.1:{port}",
        WALLET,
        "rig",
        reply_timeout=1.0,
        silence_timeout=30,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(client.run(stop))
    try:
        await asyncio.wait_for(second_connection.wait(), timeout=1.0)
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
