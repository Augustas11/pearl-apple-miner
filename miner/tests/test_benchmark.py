import asyncio
from contextlib import contextmanager
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from pmk_miner import benchmark


def test_synthetic_job_uses_exact_tiny_target():
    job = benchmark.synthetic_job()
    assert job.target == 1
    assert job.bits == benchmark.BENCHMARK_BITS
    assert job.cert_version == 3
    assert len(job.header) == 76


def test_diff1_macs_matches_consensus_scaling():
    assert benchmark.DIFF1_MACS == benchmark.POW_DENOMINATOR / (2 * benchmark.DIFF1_TARGET)
    assert 2**32 < benchmark.DIFF1_MACS < 2**33


def test_admission_adapter_forces_full_public_g3(monkeypatch, tmp_path):
    module = benchmark._admission_module()
    destination = tmp_path / "g3-admission.json"
    calls = []

    class Quickstart:
        @staticmethod
        def ensure_admission(**kwargs):
            calls.append(kwargs)
            return destination

    monkeypatch.setattr(module, "_load_quickstart", lambda: Quickstart)
    monkeypatch.delenv("PMK_G3_ADMISSION_FILE", raising=False)
    assert module.ensure_benchmark_admission(lambda: False, inherited=True) == destination
    assert len(calls) == 1
    assert calls[0]["force"] is True and calls[0]["inherited"] is True
    assert callable(calls[0]["cancelled"])
    assert module.os.environ["PMK_G3_ADMISSION_FILE"] == str(destination)


def test_benchmark_uses_full_production_pipeline_and_reports_rates(monkeypatch, tmp_path, capsys):
    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

    clock = Clock()
    monkeypatch.setattr(benchmark, "_ensure_admission", lambda stop: _async_value(tmp_path / "g3.json"))
    monkeypatch.setattr(benchmark, "_host_fields", lambda: {
        "chip": "Test Chip", "gpu_cores": "10", "macos": "26.0", "pmk_version": "test",
    })

    class Native:
        def close(self):
            self.closed = True

    class Record:
        completed_ops = benchmark.PRODUCTION_SHAPE.ops
        gpu_seconds = 4.0

    class Pipeline:
        def __init__(self, native, shape, log, *, max_gpu_seconds, desktop):
            assert shape == benchmark.Shape(8192, 8192, 4096, 2)
            assert max_gpu_seconds is None
            assert desktop is controls
            self.calls = 0

        def set_template(self, source):
            assert source.target == 1

        async def run(self, index, target, nbits, submit, is_current):
            assert index in (0, 1) and target == 1 and nbits == benchmark.BENCHMARK_BITS
            self.calls += 1
            await asyncio.sleep(0)
            if self.calls == 2:
                clock.now = 112.0
            return Record()

        def cancel(self):
            pass

    class Controls:
        reads = 0

        @property
        def busy_seconds(self):
            self.reads += 1
            return 0.0 if self.reads == 1 else 3.0

        def power_source(self):
            return "ac"

    controls = Controls()
    monkeypatch.setattr(benchmark, "Native", Native)
    monkeypatch.setattr(benchmark, "Pipeline", Pipeline)
    monkeypatch.setattr(benchmark, "_install_signal_handlers", lambda event: (asyncio.get_running_loop(), []))
    events = []
    result = asyncio.run(benchmark.run_benchmark(
        SimpleNamespace(benchmark=10, difficulty=2**21, desktop=controls,
                        benchmark_clock=clock),
        lambda event, **fields: events.append((event, fields)),
    ))
    assert result["shape"] == {"m": 8192, "n": 8192, "k": 4096, "slots": 2}
    assert result["jobs"] == 2
    assert result["completed_ops"] == 2 * benchmark.PRODUCTION_SHAPE.ops
    assert result["pool_hashrate"] == result["macs_per_second"]
    assert result["gpu_busy_pct"] == 25.0
    assert result["expected_share_seconds"] == pytest.approx(
        2**21 * benchmark.DIFF1_MACS / result["macs_per_second"]
    )
    assert "8192x8192x4096" in result["paste_block"]
    assert "Test Chip" in capsys.readouterr().out
    assert [event for event, _ in events] == ["benchmark_admission", "benchmark_started", "benchmark_summary"]


def test_lane_error_drains_peer_before_native_close(monkeypatch, tmp_path):
    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

    clock = Clock()
    state = {"peer_drained": False}
    monkeypatch.setattr(benchmark, "_ensure_admission", lambda stop: _async_value(tmp_path / "g3.json"))
    monkeypatch.setattr(benchmark, "_install_signal_handlers", lambda event: (asyncio.get_running_loop(), []))

    class Native:
        def close(self):
            assert state["peer_drained"]

    class Pipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_template(self, source):
            self.source = source

        async def run(self, index, *_args):
            if index == 0:
                await asyncio.sleep(0)
                raise RuntimeError("lane failed")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            state["peer_drained"] = True
            clock.now = 111.0
            return SimpleNamespace(completed_ops=0, gpu_seconds=0.0)

        def cancel(self):
            pass

    monkeypatch.setattr(benchmark, "Native", Native)
    monkeypatch.setattr(benchmark, "Pipeline", Pipeline)
    with pytest.raises(RuntimeError, match="lane failed"):
        asyncio.run(benchmark.run_benchmark(
            SimpleNamespace(benchmark=10, difficulty=1, benchmark_clock=clock),
            lambda *_args, **_kwargs: None,
        ))


def test_deadline_is_part_of_pipeline_current_predicate(monkeypatch, tmp_path):
    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

    clock = Clock()
    predicate_results = []
    monkeypatch.setattr(benchmark, "_ensure_admission", lambda stop: _async_value(tmp_path / "g3.json"))
    monkeypatch.setattr(benchmark, "_install_signal_handlers", lambda event: (asyncio.get_running_loop(), []))
    monkeypatch.setattr(benchmark, "_host_fields", lambda: {
        "chip": "Test", "gpu_cores": "1", "macos": "Test", "pmk_version": "test",
    })

    class Native:
        def close(self):
            pass

    class Pipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_template(self, source):
            self.source = source

        async def run(self, _index, _target, _nbits, _submit, is_current):
            clock.now = 110.0
            predicate_results.append(is_current(self.source))
            return SimpleNamespace(completed_ops=0, gpu_seconds=0.0)

        def cancel(self):
            pass

    monkeypatch.setattr(benchmark, "Native", Native)
    monkeypatch.setattr(benchmark, "Pipeline", Pipeline)
    result = asyncio.run(benchmark.run_benchmark(
        SimpleNamespace(benchmark=10, difficulty=1, benchmark_clock=clock),
        lambda *_args, **_kwargs: None,
    ))
    assert predicate_results == [False]
    assert result["jobs"] == 0 and result["expected_share_seconds"] is None


def test_cancelled_admission_waits_for_worker_to_stop(monkeypatch, tmp_path):
    started = threading.Event()
    exited = threading.Event()

    class Admission:
        @staticmethod
        def ensure_benchmark_admission(cancelled, *, inherited):
            assert inherited is True
            started.set()
            while not cancelled():
                time.sleep(0.001)
            exited.set()
            return tmp_path / "g3.json"

    monkeypatch.setattr(benchmark, "_admission_module", lambda: Admission)

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(benchmark._ensure_admission(stop))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stop.is_set() and exited.is_set()

    asyncio.run(scenario())


async def _async_value(value):
    return value


@pytest.mark.parametrize("duration", (9, 601))
def test_benchmark_rejects_duration_outside_cli_contract(duration):
    with pytest.raises(ValueError, match="benchmark must be"):
        asyncio.run(benchmark.run_benchmark(
            SimpleNamespace(benchmark=duration, difficulty=2**21), lambda *a, **k: None,
        ))


def test_core_benchmark_cli_does_not_read_mining_credentials_or_network(monkeypatch):
    from pmk_miner import __main__ as cli
    calls = []

    @contextmanager
    def context(value=None):
        yield value

    async def fake_benchmark(args, log):
        calls.append((args.benchmark, args.difficulty, args.desktop))

    def forbidden(**_kwargs):
        raise AssertionError('offline benchmark read mining credentials')

    desktop = SimpleNamespace()
    monkeypatch.setattr(cli, 'gpu_lock', lambda _log: context())
    monkeypatch.setattr(cli, 'mining_session', lambda **_kwargs: context(desktop))
    monkeypatch.setattr(cli.NodeRpcConfig, 'from_sources', forbidden)
    monkeypatch.setattr(benchmark, 'run_benchmark', fake_benchmark)
    monkeypatch.setattr(sys, 'argv', ['pmk_miner', '--benchmark', '45', '--difficulty', '8192'])

    assert cli.main(standalone=True) == 0
    assert calls == [(45, 8192.0, desktop)]
