"""Crash-safe local ownership and the external lab wrapper contract."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, separators=(',', ':'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class RunState:
    def __init__(self, path, *, clock=time.monotonic):
        self.path = Path(path)
        self.clock = clock
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            'run_id': uuid.uuid4().hex, 'completed_ops': 0, 'elapsed_seconds': 0,
            'window_expected': 0, 'window_shares': 0, 'daily': 0,
        }
        self._last_save = 0.0
        self.save()

    def save(self, **updates):
        self.data.update(updates)
        atomic_json(self.path, self.data)
        self._last_save = self.clock()

    def save_if_due(self, seconds, **updates):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('checkpoint interval must be positive')
        self.data.update(updates)
        if self.clock() - self._last_save >= seconds:
            self.save()
            return True
        return False

    def flush(self, **updates):
        self.save(**updates)


class RotatingJsonlSink:
    """Small append-only JSONL sink with bounded local retention."""
    def __init__(self, path, *, max_bytes=4 * 1024 * 1024, backups=4):
        if max_bytes <= 0 or backups < 0:
            raise ValueError('routine telemetry retention must be positive')
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.backups = int(backups)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def write(self, event, **fields):
        self._rotate_if_needed()
        with self.path.open('a', encoding='utf-8') as stream:
            os.chmod(self.path, 0o600)
            stream.write(json.dumps({'time': time.time(), 'event': event, **fields},
                                    separators=(',', ':'), sort_keys=True) + '\n')

    def _rotate_if_needed(self):
        if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
            return
        if self.backups == 0:
            self.path.unlink(missing_ok=True)
            return
        oldest = self.path.with_name(f'{self.path.name}.{self.backups}')
        oldest.unlink(missing_ok=True)
        for index in range(self.backups - 1, 0, -1):
            source = self.path.with_name(f'{self.path.name}.{index}')
            if source.exists():
                source.replace(self.path.with_name(f'{self.path.name}.{index + 1}'))
        self.path.replace(self.path.with_name(f'{self.path.name}.1'))


class RoutineTelemetry:
    """Bound routine per-job events while keeping alerts and submissions immediate."""
    ROUTINE_EVENTS = frozenset({'gpu_dispatch', 'completed', 'python_overhead'})

    def __init__(self, sink, *, seconds=5.0, clock=time.monotonic, routine_sink=None):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('telemetry interval must be positive')
        self.sink = sink
        self.routine_sink = routine_sink
        self.seconds = seconds
        self.clock = clock
        self._window_start = clock()
        self._counts = {}
        self._last = {}
        self._sampled = set()

    def log(self, event, **fields):
        if event not in self.ROUTINE_EVENTS:
            self.flush()
            self.sink(event, **fields)
            return
        now = self.clock()
        if now - self._window_start >= self.seconds:
            self.flush(reset_window=True)
            now = self.clock()
        self._counts[event] = self._counts.get(event, 0) + 1
        self._last[event] = {k: v for k, v in fields.items() if k in {'job_id', 'wall_seconds', 'gpu_seconds', 'shares', 'blocks', 'inflight'}}
        if event not in self._sampled:
            self._sampled.add(event)
            self.sink(event, **fields)

    def flush(self, *, reset_window=False):
        if not self._counts:
            if reset_window:
                self._window_start = self.clock()
                self._sampled.clear()
            return
        fields = {
            'counts': dict(self._counts),
            'last': dict(self._last),
            'window_seconds': max(0.0, self.clock() - self._window_start),
        }
        self.sink('routine_telemetry', **fields)
        if self.routine_sink is not None:
            self.routine_sink.write('routine_telemetry', **fields)
        self._counts.clear()
        self._last.clear()
        if reset_window:
            self._window_start = self.clock()
            self._sampled.clear()


def _alive(pid):
    if type(pid) is not int or pid <= 0:
        return True  # Unknown ownership cannot be reclaimed.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def gpu_lock(log, path=Path('/tmp/pmm-gpu-bench.lock'), *, inherited=None):
    """Retain interoperability with the directory lock used by GPU harnesses.

    A separate OS lock serializes PMK owners/recovery; only a PMK directory with
    a recorded dead PID is reclaimable. Unknown/other harness locks stay intact.
    """
    path = Path(path)
    inherited = os.environ.get('PMK_GPU_LOCK_HELD') == '1' if inherited is None else inherited
    if inherited:
        if not path.is_dir():
            raise RuntimeError('inherited GPU lock is absent')
        yield
        return
    mutex = path.with_name(path.name + '.pmk-owner-lock')
    with mutex.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        marker = path / 'pmk-owner.json'
        while True:
            try:
                path.mkdir()
                atomic_json(marker, {'owner': 'pmk', 'pid': os.getpid()})
                break
            except FileExistsError:
                try:
                    owner = json.loads(marker.read_text())
                    if owner.get('owner') == 'pmk' and not _alive(owner.get('pid')):
                        # Do not delete anything an external owner placed here.
                        if set(path.iterdir()) == {marker}:
                            marker.unlink()
                            path.rmdir()
                            continue
                except (OSError, ValueError):
                    pass
                log('gpu_lock_wait')
                time.sleep(1)
        try:
            yield
        finally:
            marker.unlink()
            path.rmdir()


class LabSession:
    """Hooks for an optional outer wrapper that pauses other GPU workloads; this never pauses/restarts them itself."""
    def __init__(self, *, token=None, resume_hook=None, report=None, lock=None):
        self.token = token
        self.resume_hook = resume_hook
        self.report = Path(report) if report else None
        self.lock = Path(lock) if lock else Path.home() / '.lab-window.lock'
        self.owned = False

    def check(self):
        if not self.lock.exists():
            if self.token:
                raise RuntimeError('lab ownership token supplied without a lab lock')
            return
        source = self.lock / 'pmk-owner.json' if self.lock.is_dir() else self.lock
        try:
            owner = json.loads(source.read_text())
        except (OSError, ValueError):
            raise RuntimeError('lab lock is owned by another session') from None
        if not self.token or owner.get('token') != self.token:
            raise RuntimeError('lab lock is owned by another session')
        if owner.get('pause_confirmed') is not True:
            raise RuntimeError('lab provider pause is not confirmed')
        if not self.resume_hook or not self.report:
            raise RuntimeError('owned lab window requires resume hook and outcome report')
        self.owned = True

    def finish(self):
        if not self.owned:
            return None
        result = {'outcome': 'failed', 'confirmed': False, 'time': time.time()}
        try:
            completed = subprocess.run([str(self.resume_hook)], capture_output=True,
                                       timeout=600, check=False)
            report = json.loads(completed.stdout)
            if (completed.returncode == 0 and report.get('confirmed') is True
                    and report.get('outcome') in {'resumed', 'fallback-restarted'}):
                result.update(outcome=report['outcome'], confirmed=True)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        # Never persist arbitrary hook stdout/stderr (which can contain secrets).
        atomic_json(self.report, result)
        if not result['confirmed']:
            raise RuntimeError('provider resume unconfirmed; see resume outcome report')
        return result


class ProbeSchedule:
    """Quiescent periodic known-answer checks; a failed check stays due."""
    def __init__(self, seconds=6*3600, clock=time.monotonic):
        if not 0 < seconds <= 6*3600:
            raise ValueError('probe interval must be positive and at most six hours')
        self.seconds, self.clock = seconds, clock
        self.deadline = clock() + seconds

    @property
    def due(self):
        return self.clock() >= self.deadline

    def refresh(self, native):
        native.refresh_probe()
        self.deadline = self.clock() + self.seconds
