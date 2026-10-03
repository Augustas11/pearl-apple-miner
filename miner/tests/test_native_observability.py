import ctypes as C
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pmk_miner.__main__ as cli
from pmk_miner.native import Native, NativeError
from pmk_miner.pipeline import FatalDeviceError


class FailingCall:
    def __init__(self, name, code):
        self.__name__ = name
        self.code = code

    def __call__(self, *_args):
        return self.code


def bare_native(core_message=b'buffer shorter than pmkcore_padded_len(rows, k)'):
    native = Native.__new__(Native)
    native.timing = SimpleNamespace()
    native.core = SimpleNamespace(pmkcore_strerror=lambda _code: core_message)
    native.context = C.c_void_p(123)
    return native


def test_native_error_uses_pmkcore_strerror_and_structured_fields():
    native = bare_native()

    try:
        native.call(FailingCall('pmkcore_template_init', -6))
    except NativeError as exc:
        assert exc.function == 'pmkcore_template_init'
        assert exc.code == -6
        assert exc.message == 'buffer shorter than pmkcore_padded_len(rows, k)'
        assert str(exc) == ('pmkcore_template_init failed (-6): '
                            'buffer shorter than pmkcore_padded_len(rows, k)')
    else:
        raise AssertionError('NativeError was not raised')


def test_libpmk_error_has_abi_message_and_init_diagnostic_wins():
    native = bare_native()
    with_message = C.create_string_buffer(b'DO NOT MINE: Metal probe mismatch')

    try:
        native.call(FailingCall('pmk_init', -103), error_buffer=with_message)
    except NativeError as exc:
        assert (exc.function, exc.code, exc.message) == (
            'pmk_init', -103, 'DO NOT MINE: Metal probe mismatch')
    else:
        raise AssertionError('NativeError was not raised')

    try:
        native.call(FailingCall('pmk_buffer_alloc', -102))
    except NativeError as exc:
        assert exc.message == 'resource budget or allocation limit exceeded'
    else:
        raise AssertionError('NativeError was not raised')


def test_libpmk_context_and_job_error_accessors_win():
    native = bare_native()
    context_error = C.create_string_buffer(b'job encode failed: no command buffer for k3')
    job_error = C.create_string_buffer(b'command buffer status=4: GPU fault')

    native.metal = SimpleNamespace(
        pmk_context_error=lambda _ctx, out, _cap: C.memmove(out, context_error, len(context_error)) and 0,
        pmk_job_error=lambda _job, out, _cap: C.memmove(out, job_error, len(job_error)) and 0,
    )

    try:
        native.call(FailingCall('pmk_run_job', -102), native.context)
    except NativeError as exc:
        assert exc.message == 'job encode failed: no command buffer for k3'
    else:
        raise AssertionError('NativeError was not raised')

    assert native.job_error(C.c_void_p(456)) == 'command buffer status=4: GPU fault'


def test_release_waits_for_foreign_callback_return_before_dropping_job():
    native = Native.__new__(Native)
    calls = []

    def wait(handle):
        calls.append(('wait', handle.value))
        return 0

    def release(handle):
        calls.append(('release', handle.value))
        return 0

    native.metal = SimpleNamespace(
        pmk_job_wait_callback=wait,
        pmk_job_release=release,
    )
    native.call = lambda function, *args: function(*args)
    native.release(C.c_void_p(456))

    assert calls == [('wait', 456), ('release', 456)]


def test_pmk_native_check_entrypoint_passes_logger_to_gpu_lock(monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'pmk_native_check.py'
    spec = importlib.util.spec_from_file_location('pmk_native_check_under_test', script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    calls = []

    @contextlib.contextmanager
    def fake_gpu_lock(logger):
        calls.append(('lock', logger is module.log))
        yield

    async def fake_run(args):
        calls.append(('run', args.jobs))

    monkeypatch.setattr(module, 'gpu_lock', fake_gpu_lock)
    monkeypatch.setattr(module, 'run', fake_run)

    assert module.main(['--jobs', '0']) is None
    assert calls == [('lock', True), ('run', 0)]


def test_cli_logs_native_diagnostics_but_not_arbitrary_exception_messages(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, 'version_info', (3, 12))
    monkeypatch.setattr(cli.argparse.ArgumentParser, 'parse_args', lambda _self: SimpleNamespace(
        config=SimpleNamespace(read_text=lambda: ''), lab_owner_token=None,
        resume_hook=None, resume_report=None, gateway='unused', mode='solo'))
    monkeypatch.setattr(cli, 'LabSession', lambda **_kwargs: SimpleNamespace(
        check=lambda: (_ for _ in ()).throw(
            NativeError('pmk_buffer_alloc', -102, 'resource budget or allocation limit exceeded'))))

    assert cli.main() == 1
    event = json.loads(capsys.readouterr().out)
    assert event['event'] == 'fatal'
    assert event['error_type'] == 'NativeError'
    assert event['native_function'] == 'pmk_buffer_alloc'
    assert event['native_code'] == -102
    assert event['error_message'] == 'resource budget or allocation limit exceeded'

    monkeypatch.setattr(cli, 'LabSession', lambda **_kwargs: SimpleNamespace(
        check=lambda: (_ for _ in ()).throw(RuntimeError('rpc_password=supersecret'))))
    assert cli.main() == 1
    event = json.loads(capsys.readouterr().out)
    assert event == {'time': event['time'], 'event': 'fatal', 'error_type': 'RuntimeError'}
    assert 'supersecret' not in json.dumps(event)


def test_cli_logs_fatal_device_error_gate_and_safe_diagnostics(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, 'version_info', (3, 12))
    monkeypatch.setattr(cli.argparse.ArgumentParser, 'parse_args', lambda _self: SimpleNamespace(
        config=SimpleNamespace(read_text=lambda: ''), lab_owner_token=None,
        resume_hook=None, resume_report=None, gateway='unused', mode='solo'))
    monkeypatch.setattr(cli, 'LabSession', lambda **_kwargs: SimpleNamespace(
        check=lambda: (_ for _ in ()).throw(FatalDeviceError.verifier(
            'v3 verifier gate failed for rpc_password=supersecret wallet=prl1qqqqqqqqqqqqqq'))))

    assert cli.main() == 1
    event = json.loads(capsys.readouterr().out)
    assert event['event'] == 'fatal'
    assert event['error_type'] == 'FatalDeviceError'
    assert event['gate'] == 'verifier'
    assert event['native_function'] == 'python.verifier_gate'
    assert event['native_code'] == -2002
    assert event['error_message'] == 'v3 verifier gate failed for rpc_password=<redacted> wallet=<redacted>'
    assert 'supersecret' not in json.dumps(event)
    assert 'prl1qqqq' not in json.dumps(event)


def test_memory_limits_use_one_physical_ram_read_for_budget_and_validation(monkeypatch):
    calls = []
    shape = SimpleNamespace(validate=lambda ram: calls.append(ram) or 1234)
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda command: (
        calls.append(command) or b'25769803776\n'))

    assert cli.memory_limits(shape) == (6 * 1024**3, 1234)
    assert calls == [['sysctl', '-n', 'hw.memsize'], 24 * 1024**3]
