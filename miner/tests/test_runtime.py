import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import pmk_miner.runtime as runtime
from pmk_miner.runtime import LabSession, RotatingJsonlSink, RoutineTelemetry, RunState, gpu_lock


def test_restart_preserves_monitor_counters(tmp_path):
    path = tmp_path / 'state.json'
    state = RunState(path)
    state.save(completed_ops=123, window_expected=8.5, window_shares=9, daily=17)
    resumed = RunState(path)
    assert resumed.data == state.data
    assert path.stat().st_mode & 0o777 == 0o600


def test_run_state_cadences_routine_checkpoints_and_flushes(tmp_path):
    now = [0.0]
    path = tmp_path / 'state.json'
    state = RunState(path, clock=lambda:now[0])
    state.save_if_due(10, completed_ops=123, elapsed_seconds=7)
    persisted = json.loads(path.read_text())
    assert persisted['completed_ops'] == 0
    now[0] = 10.0
    assert state.save_if_due(10, window_shares=2) is True
    persisted = json.loads(path.read_text())
    assert persisted['completed_ops'] == 123
    assert persisted['window_shares'] == 2
    state.save_if_due(10, completed_ops=456)
    state.flush()
    assert json.loads(path.read_text())['completed_ops'] == 456


def test_routine_telemetry_bounds_per_job_events_and_flushes_alerts():
    now = [0.0]
    events = []
    telemetry = RoutineTelemetry(lambda event, **fields: events.append((event, fields)), seconds=5, clock=lambda:now[0])
    telemetry.log('completed', job_id=1, shares=0, blocks=0)
    telemetry.log('completed', job_id=2, shares=0, blocks=0)
    telemetry.log('python_overhead', job_id=2, wall_seconds=1.0)
    telemetry.log('alert', severity='P0', reason='device_or_verifier_gate')
    assert [event for event, _ in events] == ['completed', 'python_overhead', 'routine_telemetry', 'alert']
    assert events[2][1]['counts'] == {'completed': 2, 'python_overhead': 1}


def test_routine_telemetry_nonroutine_flush_does_not_resample_until_interval(tmp_path):
    now = [0.0]
    events = []
    sink = RotatingJsonlSink(tmp_path / 'routine.jsonl', max_bytes=160, backups=1)
    telemetry = RoutineTelemetry(lambda event, **fields: events.append((event, fields)),
                                 seconds=5, clock=lambda:now[0], routine_sink=sink)
    telemetry.log('completed', job_id=1)
    telemetry.log('outcome', classification='accepted')
    telemetry.log('completed', job_id=2)
    now[0] = 5.0
    telemetry.log('completed', job_id=3)
    assert [event for event, _ in events] == [
        'completed', 'routine_telemetry', 'outcome', 'routine_telemetry', 'completed'
    ]
    rows = [json.loads(line) for line in (tmp_path / 'routine.jsonl').read_text().splitlines()]
    assert all(row['event'] == 'routine_telemetry' for row in rows)


def test_routine_sink_rotates(tmp_path):
    sink = RotatingJsonlSink(tmp_path / 'routine.jsonl', max_bytes=1, backups=1)
    sink.write('routine_telemetry', counts={'completed': 1})
    sink.write('routine_telemetry', counts={'completed': 2})
    assert (tmp_path / 'routine.jsonl').exists()
    assert (tmp_path / 'routine.jsonl.1').exists()


def test_high_rate_checkpoints_and_telemetry_are_bounded(monkeypatch, tmp_path):
    now = [0.0]
    writes = []
    original_atomic_json = runtime.atomic_json

    def counted_atomic_json(path, value):
        original_atomic_json(path, value)
        writes.append(dict(value))

    monkeypatch.setattr(runtime, 'atomic_json', counted_atomic_json)

    path = tmp_path / 'state.json'
    state = RunState(path, clock=lambda:now[0])
    for job_id in range(5000):
        now[0] = job_id * 0.001
        state.save_if_due(
            2.0,
            completed_ops=job_id + 1,
            elapsed_seconds=now[0],
            window_expected=float(job_id + 1),
        )

    pre_flush = RunState(path, clock=lambda:now[0]).data
    assert 0 < pre_flush['completed_ops'] < state.data['completed_ops']
    assert [row['completed_ops'] for row in writes] == sorted(row['completed_ops'] for row in writes)

    state.flush()
    reloaded = RunState(path, clock=lambda:now[0]).data
    assert reloaded['completed_ops'] == 5000
    assert reloaded['window_expected'] == 5000.0
    # initial write + two cadence checkpoints + pre/final/reload writes from this
    # test's explicit restart checks; thousands of updates must not become writes.
    assert len(writes) == 6

    events = []
    sink = RotatingJsonlSink(tmp_path / 'routine.jsonl', max_bytes=320, backups=2)
    telemetry = RoutineTelemetry(lambda event, **fields: events.append((event, fields)),
                                 seconds=1.0, clock=lambda:now[0], routine_sink=sink)
    for job_id in range(5000):
        now[0] = job_id * 0.001
        telemetry.log('completed', job_id=job_id, shares=job_id % 2, blocks=0)
    telemetry.flush()

    assert len(events) <= 12
    assert sum(1 for event, _ in events if event == 'routine_telemetry') <= 6
    retained = list(tmp_path.glob('routine.jsonl*'))
    assert len(retained) <= 3
    assert all(path.stat().st_size > 0 for path in retained)


def test_crashed_gpu_owner_can_restart(tmp_path):
    path = tmp_path / 'gpu.lock'
    code = ('import os; from pathlib import Path; from pmk_miner.runtime import gpu_lock; '
            f'ctx=gpu_lock(lambda *a:None,Path({str(path)!r}),inherited=False); '
            'ctx.__enter__(); os._exit(0)')
    subprocess.run([sys.executable, '-c', code], check=True,
                   env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
    assert path.exists()
    with gpu_lock(lambda *a:None, path, inherited=False):
        assert json.loads((path / 'pmk-owner.json').read_text())['pid'] == os.getpid()
    assert not path.exists()


def test_inherited_gpu_lock_must_exist(tmp_path):
    with pytest.raises(RuntimeError, match='absent'):
        with gpu_lock(lambda *a:None, tmp_path / 'absent', inherited=True):
            pass


def test_foreign_lab_lock_refused(tmp_path):
    lock = tmp_path / '.lab-window.lock'
    lock.mkdir()
    with pytest.raises(RuntimeError, match='another session'):
        LabSession(lock=lock).check()
    (lock / 'pmk-owner.json').write_text(json.dumps({'token': 'other', 'pause_confirmed': True}))
    with pytest.raises(RuntimeError, match='another session'):
        LabSession(lock=lock, token='ours').check()


@pytest.mark.parametrize('outcome', ['resumed', 'fallback-restarted'])
def test_lab_resume_outcome_recorded(tmp_path, outcome):
    lock = tmp_path / 'lock'
    lock.write_text(json.dumps({'token': 'ours', 'pause_confirmed': True}))
    hook = tmp_path / 'resume'
    hook.write_text('#!/bin/sh\nprintf \'{"outcome":"' + outcome + '","confirmed":true}\\n\'\n')
    hook.chmod(0o700)
    report = tmp_path / 'report.json'
    session = LabSession(lock=lock, token='ours', resume_hook=hook, report=report)
    session.check()
    session.finish()
    assert json.loads(report.read_text())['outcome'] == outcome


def test_unconfirmed_resume_recorded_and_fails(tmp_path):
    lock = tmp_path / 'lock'
    lock.write_text(json.dumps({'token': 'ours', 'pause_confirmed': True}))
    hook = tmp_path / 'resume'
    hook.write_text('#!/bin/sh\necho credential-bearing-error\nexit 1\n')
    hook.chmod(0o700)
    report = tmp_path / 'report.json'
    session = LabSession(lock=lock, token='ours', resume_hook=hook, report=report)
    session.check()
    with pytest.raises(RuntimeError, match='unconfirmed'):
        session.finish()
    assert json.loads(report.read_text())['confirmed'] is False
    assert 'credential' not in report.read_text()


def test_lab_pause_and_resume_contract_required(tmp_path):
    lock = tmp_path / 'lock'
    lock.write_text(json.dumps({'token': 'ours', 'pause_confirmed': False}))
    with pytest.raises(RuntimeError, match='pause'):
        LabSession(lock=lock, token='ours').check()
    lock.write_text(json.dumps({'token': 'ours', 'pause_confirmed': True}))
    with pytest.raises(RuntimeError, match='resume hook'):
        LabSession(lock=lock, token='ours').check()


def test_periodic_probe_runs_again_and_failure_stays_due():
    from pmk_miner.runtime import ProbeSchedule
    from types import SimpleNamespace
    now = [0]
    schedule = ProbeSchedule(10, clock=lambda:now[0])
    calls = []
    native = SimpleNamespace(refresh_probe=lambda:calls.append('probe'))
    assert not schedule.due
    now[0] = 10
    assert schedule.due
    schedule.refresh(native)
    assert calls == ['probe'] and not schedule.due
    now[0] = 20
    def failed(): raise RuntimeError('probe mismatch')
    native.refresh_probe = failed
    with pytest.raises(RuntimeError, match='mismatch'):
        schedule.refresh(native)
    assert schedule.due


@pytest.mark.parametrize('seconds', [0, -1, 21601])
def test_probe_interval_cannot_disable_checks(seconds):
    from pmk_miner.runtime import ProbeSchedule
    with pytest.raises(ValueError): ProbeSchedule(seconds)
