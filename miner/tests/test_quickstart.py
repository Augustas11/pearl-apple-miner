"""Quick-start boundaries without GPU or network access."""
import importlib.util
from pathlib import Path
import stat

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pmk_mine', ROOT / 'scripts/pmk_mine.py')
quickstart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quickstart)
admission_spec = importlib.util.spec_from_file_location('pmk_quickstart', ROOT / 'scripts/pmk_quickstart.py')
admission = importlib.util.module_from_spec(admission_spec)
admission_spec.loader.exec_module(admission)


def test_private_wallet_replaces_symlink_without_changing_target(tmp_path):
    target = tmp_path / 'original'
    target.write_text('untouched')
    wallet = tmp_path / 'wallet'
    wallet.symlink_to(target)
    quickstart.write_private(wallet, 'prl1dummywallet\n')
    assert target.read_text() == 'untouched'
    assert not wallet.is_symlink()
    assert stat.S_IMODE(wallet.stat().st_mode) == 0o600
    assert wallet.read_text() == 'prl1dummywallet\n'


def test_status_throttles_routine_events_but_keeps_safety_events(capsys):
    now = [0]
    events = []
    status = quickstart.Status(lambda event, **fields: events.append((event, fields)),
                              clock=lambda: now[0])
    fields = dict(jobs_per_second=2, tops=1.1, accepted=3, rejected=1,
                  expected_share_seconds=120)
    status('routine_telemetry', **fields)
    now[0] = 59
    status('routine_telemetry', **fields)
    status('alert', severity='P0')
    now[0] = 60
    status('routine_telemetry', **fields)
    output = capsys.readouterr().out
    assert output.count('jobs/s') == 2
    assert '1.10 TOPS' in output
    assert 'accepted=3 rejected=1' in output
    assert 'time/share 2.0 min' in output
    assert events == [('alert', {'severity': 'P0'})]


def test_default_worker_is_sanitized(monkeypatch):
    monkeypatch.setattr(quickstart.subprocess, 'check_output', lambda *a, **kw: 'my Mac.example!\n')
    assert quickstart.worker_name() == 'my-Mac-example'


def test_base_mac_job_fits_existing_memory_budget():
    small = quickstart.choose_shape(8 * 1024**3, 80)
    assert small.m == small.n == 4096
    assert small.validate(8 * 1024**3) <= 2 * 1024**3
    large = quickstart.choose_shape(16 * 1024**3, 80)
    assert large.m == large.n == 8192
    assert quickstart.choose_shape(32 * 1024**3, 10).m == 4096
    assert quickstart.choose_shape(32 * 1024**3).m == 4096


@pytest.mark.parametrize('last_probe,valid_hours,passed,expected', [
    (100_000, 6, True, True),
    (100_001, 6, True, False),
    (78_399, 24, True, False),
    (100_000, 0, True, False),
    (100_000, 6, False, False),
    (float('nan'), 6, True, False),
])
def test_admission_never_extends_six_hour_window(last_probe, valid_hours, passed, expected):
    assert admission._record_fresh(dict(last_probe_unix=last_probe,
        valid_hours=valid_hours, g3_passed=passed), 100_000) is expected
