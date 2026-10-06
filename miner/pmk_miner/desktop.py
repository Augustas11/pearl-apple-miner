"""Session activity, power policy, and measured GPU duty cycling."""
from __future__ import annotations

import asyncio
import ctypes as C
import math
import signal
from pathlib import Path
import time
from contextlib import contextmanager


class DesktopNative:
    """Power services do not create a Metal device or dispatch GPU work."""
    def __init__(self, library=None):
        path = library or Path(__file__).resolve().parents[2] / 'libpmk/.build/release/libpmk.dylib'
        self.lib = C.CDLL(str(path))
        self.lib.pmk_activity_begin.argtypes = [C.c_char_p]
        self.lib.pmk_activity_begin.restype = C.c_void_p
        self.lib.pmk_activity_end.argtypes = [C.c_void_p]
        self.lib.pmk_activity_end.restype = None
        self.lib.pmk_power_source.argtypes = []
        self.lib.pmk_power_source.restype = C.c_int32
        self.lib.pmk_thermal_state.argtypes = []
        self.lib.pmk_thermal_state.restype = C.c_int32

    @contextmanager
    def activity(self, reason='pmk mining session'):
        handle = self.lib.pmk_activity_begin(reason.encode('utf-8'))
        if not handle:
            raise RuntimeError('could not begin pmk mining activity')
        try:
            yield
        finally:
            self.lib.pmk_activity_end(handle)

    def power_source(self):
        return {0: 'unknown', 1: 'ac', 2: 'battery', 3: 'desktop'}.get(self.lib.pmk_power_source(), 'unknown')

    def thermal_state(self):
        return {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical', 4: 'unknown'}.get(
            self.lib.pmk_thermal_state(), 'unknown')


class DesktopControls:
    """Gate new dispatches; never cancel a job because power changes.

    Reduced intensity serializes GPU bursts across slots so another lane cannot
    fill the intended idle gap. CPU preparation and proof work remain concurrent.
    Busy time is the union of Metal timestamp intervals, not a requested setting.
    """
    def __init__(self, power_source, *, on_battery='pause', intensity=100,
                 log=lambda *a, **k: None, clock=time.monotonic, poll_seconds=10):
        if on_battery not in ('pause', 'run'):
            raise ValueError('on-battery must be pause or run')
        if not math.isfinite(intensity) or not 10 <= intensity <= 100:
            raise ValueError('intensity must be in [10, 100]')
        self.power_source = power_source
        self.on_battery = on_battery
        self.intensity = intensity
        self.log, self.clock = log, clock
        self.poll_seconds = poll_seconds
        self._next_poll = float('-inf')
        self._source = None
        self._paused = False
        self._gate = asyncio.Lock()
        self._ready_at = 0.0
        self._started = clock()
        self._busy = 0.0
        self._intervals = []

    def refresh(self):
        now = self.clock()
        if now < self._next_poll:
            return
        try:
            source = self.power_source()
        except Exception as exc:
            source = 'unknown'
            self.log('power_source_unavailable', error_type=type(exc).__name__)
        self._next_poll = now + self.poll_seconds
        paused = self.on_battery == 'pause' and source not in ('ac', 'desktop')
        if source != self._source or paused != self._paused:
            self.log('power_state', source=source, policy=self.on_battery,
                     paused=paused, intensity=self.intensity)
        self._source, self._paused = source, paused

    async def wait_ready(self, stopped):
        while not stopped():
            self.refresh()
            if not self._paused:
                return True
            await asyncio.sleep(.1)
        return False

    async def before_dispatch(self, stopped):
        if self.intensity < 100:
            await self._gate.acquire()
        try:
            while await self.wait_ready(stopped):
                delay = self._ready_at - self.clock()
                if delay <= 0:
                    return True
                await asyncio.sleep(min(.1, delay))
            self.abort_dispatch()
            return False
        except BaseException:
            self.abort_dispatch()
            raise

    def abort_dispatch(self):
        if self.intensity < 100 and self._gate.locked():
            self._gate.release()

    def after_gpu(self, start, end):
        seconds = max(0.0, end - start)
        # Completed command buffers can be observed out of order across slots.
        if seconds and math.isfinite(start) and math.isfinite(end):
            intervals = sorted(self._intervals + [(start, end)])
            merged = []
            for left, right in intervals:
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            # Only the newest interval can overlap a later completion on this
            # serial Metal queue; fold older intervals into the running total.
            self._busy += sum(right-left for left, right in merged[:-1])
            self._intervals = merged[-1:]
        if self.intensity < 100:
            # Metal timestamps share the monotonic clock. Host completion and
            # result handling have already idled the GPU; count that interval
            # toward the duty-cycle gap instead of delaying it a second time.
            self._ready_at = end + seconds * (100 / self.intensity - 1)
        self.abort_dispatch()

    @property
    def busy_seconds(self):
        return self._busy + sum(end-start for start, end in self._intervals)

    def snapshot(self):
        elapsed = max(0, self.clock() - self._started)
        fraction = min(1.0, self.busy_seconds / elapsed) if elapsed else 0.0
        return dict(gpu_busy_fraction=fraction, gpu_busy_pct=100*fraction,
                    gpu_busy_seconds=self.busy_seconds, intensity=self.intensity,
                    power_source=self._source, paused=self._paused)

    async def monitor(self, stop):
        while not stop.is_set():
            self.refresh()
            self.log('power_telemetry', **self.snapshot())
            try:
                await asyncio.wait_for(stop.wait(), self.poll_seconds)
            except asyncio.TimeoutError:
                pass


@contextmanager
def mining_session(*, on_battery, intensity, log):
    # Cover synchronous admission/startup before the async miner installs its
    # graceful drain handlers. Unwinding also terminates any G3 helper child.
    previous = {}
    def interrupt(signum, _frame):
        raise SystemExit(128 + signum)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        native = DesktopNative()
        with native.activity():
            yield DesktopControls(native.power_source, on_battery=on_battery,
                                  intensity=intensity, log=log)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
