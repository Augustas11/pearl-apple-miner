# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Replay a captured regtest job through the real CLI, native core, and GPU."""
from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path
import signal
import socketserver
import sys
import threading

import pytest

import pmk_miner.__main__ as cli
from pmk_miner.v4_admission import V4_G3_ADMISSION_ENV

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / 'fixtures/v4_regtest_gateway_job.json'
LIBPMK = ROOT / 'libpmk/.build/release/libpmk.dylib'
PAYOUT = '51202d6c74a5d8af133f1c65c0a9f99c433498dc807dce022e559c60a3a70e0223ea'
ADDRESS = 'rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2'


def write_test_admission(path: Path) -> None:
    """Create an exact-binary/device fixture without reading operator state."""
    lib = C.CDLL(str(LIBPMK))
    ptr, u64 = C.c_void_p, C.c_uint64
    lib.pmk_v4_init_diagnostic.argtypes = [C.POINTER(ptr), ptr, u64]
    lib.pmk_v4_init_diagnostic.restype = C.c_int32
    lib.pmk_v4_write_admission_record.argtypes = [ptr, C.c_char_p, u64, ptr, u64]
    lib.pmk_v4_write_admission_record.restype = C.c_int32
    lib.pmk_v4_destroy.argtypes = [ptr]
    context, error = ptr(), C.create_string_buffer(4096)
    assert lib.pmk_v4_init_diagnostic(C.byref(context), error, len(error)) == 0, error.value
    try:
        assert lib.pmk_v4_write_admission_record(
            context, os.fsencode(path), 100_000_000, error, len(error)) == 0, error.value
    finally:
        lib.pmk_v4_destroy(context)


@pytest.mark.parametrize('admission_mode', ['config', 'environment', 'missing'])
def test_real_solo_entry_dispatches_captured_v4_job_with_config_admission(
    tmp_path, monkeypatch, admission_mode, v3_g3_admission
):
    # Only the remote gateway is replayed. No miner, verifier, admission, or GPU
    # implementation is replaced. The full node acceptance gate is the regtest E2E.
    fixture = json.loads(FIXTURE.read_text())
    assert fixture['cert_version'] == 4
    requests = []

    class Replay(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            requests.append(request['method'])
            assert request['method'] == 'getMiningInfo'
            self.wfile.write((json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                                         'result': fixture}) + '\n').encode())

    # Generate an isolated exact-device/library record; never consume ~/.pmk.
    write_test_admission(tmp_path / 'admission.json')
    monkeypatch.delenv(V4_G3_ADMISSION_ENV, raising=False)
    if admission_mode == 'environment':
        monkeypatch.setenv(V4_G3_ADMISSION_ENV, str(tmp_path / 'admission.json'))
    for key in ('PEARLD_RPC_URL', 'PEARLD_RPC_USER', 'PEARLD_RPC_PASSWORD', 'PEARLD_MINING_ADDRESS'):
        monkeypatch.delenv(key, raising=False)
    gateway_log = tmp_path / 'gateway.log'
    gateway_log.touch()
    config = tmp_path / 'miner.toml'
    admission_config = '[v4]\nadmission_file = "admission.json"' if admission_mode == 'config' else ''
    config.write_text(f'''m = 128
n = 128
k = 4096
slots = 2
{admission_config}
[gateway]
log_file = "{gateway_log}"
[node_rpc]
rpc_url = "http://127.0.0.1:1"
rpc_user = "replay"
rpc_password = "replay"
mining_address = "{ADDRESS}"
[payout]
hrp = "rprl"
script = "{PAYOUT}"
[run]
state_dir = "{tmp_path / 'state'}"
''')
    events = []
    original_log = cli.log

    def record(event, **fields):
        events.append((event, fields))
        original_log(event, **fields)
        if event == 'gpu_dispatch':
            # Exercise the CLI's real graceful signal path after native dispatch.
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(cli, 'log', record)
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    watchdog = threading.Timer(60, lambda: os.kill(os.getpid(), signal.SIGINT))
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), Replay) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setattr(sys, 'argv', ['pmk_miner', '--mode', 'solo', '--gateway',
                                         f'127.0.0.1:{server.server_address[1]}', '--config', str(config)])
        watchdog.start()
        try:
            result = cli.main()
        finally:
            watchdog.cancel()
            server.shutdown()
            thread.join()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    assert requests and set(requests) == {'getMiningInfo'}
    if admission_mode == 'missing':
        assert result == 1, events
        assert not any(event == 'gpu_dispatch' for event, _ in events), events
        assert ('alert', {'severity': 'P0', 'reason': 'v4_admission'}) in events
        fatal = next(fields for event, fields in events if event == 'fatal')
        assert fatal['error_message'] == 'missing v4 G3 admission file'
        assert fatal['admission_env'] == V4_G3_ADMISSION_ENV
        assert fatal['admission_config'] == 'v4.admission_file'
    else:
        assert result == 0, events
        assert any(event == 'gpu_dispatch' for event, _ in events), events
        assert any(event == 'stopped' for event, _ in events), events
        assert not any(event in {'fatal', 'alert'} for event, _ in events), events
    if admission_mode == 'environment':
        assert os.environ[V4_G3_ADMISSION_ENV] == str(tmp_path / 'admission.json')
    else:
        assert V4_G3_ADMISSION_ENV not in os.environ
