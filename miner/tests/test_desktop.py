import argparse
import asyncio
import signal
from functools import wraps
from types import SimpleNamespace

import pytest

def async_test(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


from pmk_miner.cli import add_desktop_flags
from pmk_miner.desktop import DesktopControls, DesktopNative


def test_activity_released_on_every_python_exit():
    events = []
    native = DesktopNative.__new__(DesktopNative)
    native.lib = SimpleNamespace(pmk_activity_begin=lambda reason: events.append(reason) or 123,
                                 pmk_activity_end=lambda handle: events.append(handle))
    for error in (None, RuntimeError, KeyboardInterrupt, asyncio.CancelledError):
        try:
            with native.activity():
                assert events[-1] == b'pmk mining session'
                if error:
                    raise error()
        except BaseException as exc:
            assert isinstance(exc, error)
        assert events[-1] == 123
    assert len(events) == 8


def test_power_source_enum_mapping():
    native = DesktopNative.__new__(DesktopNative)
    for value, expected in [(0,'unknown'),(1,'ac'),(2,'battery'),(3,'desktop'),(-1,'unknown')]:
        native.lib = SimpleNamespace(pmk_power_source=lambda: value)
        assert native.power_source() == expected


@async_test
async def test_battery_pauses_new_dispatch_and_ac_resumes_without_cancel():
    state = ['battery']; events=[]
    control = DesktopControls(lambda: state[0], poll_seconds=.01,
                              log=lambda event, **kw: events.append((event,kw)))
    pending = asyncio.create_task(control.before_dispatch(lambda: False))
    await asyncio.sleep(.03)
    assert not pending.done()
    state[0] = 'ac'
    assert await asyncio.wait_for(pending, 1)
    control.after_gpu(10,11)
    assert [row[1]['paused'] for row in events] == [True, False]
    assert control.busy_seconds == 1


@async_test
@pytest.mark.parametrize('source,policy', [('desktop','pause'),('ac','pause'),('battery','run')])
async def test_allowed_power_sources(source, policy):
    control = DesktopControls(lambda: source, on_battery=policy)
    assert await control.before_dispatch(lambda: False)
    control.after_gpu(1,2)


@async_test
async def test_stop_while_paused_releases_dispatch_gate():
    control = DesktopControls(lambda:'battery', intensity=50)
    stopped = [False]
    task = asyncio.create_task(control.before_dispatch(lambda: stopped[0]))
    await asyncio.sleep(.01)
    stopped[0] = True
    assert await asyncio.wait_for(task, 1) is False
    assert not control._gate.locked()


@async_test
async def test_intensity_coordinates_slots_and_idles_after_completed_gpu():
    now=[0.0]
    control = DesktopControls(lambda:'ac', intensity=50, clock=lambda:now[0])
    assert await control.before_dispatch(lambda: False)
    waiting=asyncio.create_task(control.before_dispatch(lambda:False))
    await asyncio.sleep(.01)
    assert not waiting.done()
    now[0]=.1
    control.after_gpu(0,.1)
    assert control._ready_at == pytest.approx(.2)
    await asyncio.sleep(.01)
    assert not waiting.done()
    now[0]=.2
    assert await asyncio.wait_for(waiting,1)
    control.after_gpu(.2,.3)
    now[0]=.4
    assert control.snapshot()['gpu_busy_fraction'] == pytest.approx(.5)


def test_actual_busy_time_merges_overlapping_timestamps():
    now=[0.0]
    control=DesktopControls(lambda:'ac', clock=lambda:now[0])
    control.after_gpu(1,3)
    control.after_gpu(2,4)
    now[0]=6
    assert control.busy_seconds == 3
    assert control.snapshot()['gpu_busy_pct'] == 50


@async_test
async def test_completion_observation_delay_counts_toward_gpu_idle_gap():
    now=[0.0]
    control=DesktopControls(lambda:'ac', intensity=50, clock=lambda:now[0])
    assert await control.before_dispatch(lambda:False)
    # GPU finished at .1; the host did not observe completion until .25.
    # The requested .1-second idle gap has therefore already elapsed.
    now[0]=.25
    control.after_gpu(0,.1)
    assert control._ready_at == pytest.approx(.2)
    assert await asyncio.wait_for(control.before_dispatch(lambda:False),.1)
    control.abort_dispatch()


@async_test
async def test_cancelled_gate_wait_does_not_release_another_lane():
    control=DesktopControls(lambda:'ac', intensity=50)
    assert await control.before_dispatch(lambda:False)
    waiter=asyncio.create_task(control.before_dispatch(lambda:False))
    await asyncio.sleep(.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert control._gate.locked()
    control.abort_dispatch()


@pytest.mark.parametrize('flag,low,high', [('--pool-silence-timeout',30,1800),
    ('--intensity',10,100),('--benchmark',10,600),('--difficulty',1,2**64)])
def test_numeric_cli_ranges_and_nonfinite(flag,low,high,capsys):
    parser=argparse.ArgumentParser()
    add_desktop_flags(parser)
    for invalid in (str(low-1),str(high+10000),'nan','inf','-inf','nonsense'):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([flag,invalid])
        assert exc.value.code == 2
        assert flag in capsys.readouterr().err
    for valid in (str(low),str(high)):
        assert getattr(parser.parse_args([flag,valid]),flag[2:].replace('-','_')) == float(valid)


def test_cli_defaults():
    parser=argparse.ArgumentParser();add_desktop_flags(parser)
    args=parser.parse_args([])
    assert (args.on_battery,args.intensity,args.pool_silence_timeout,args.difficulty)==('pause',100,180,2**21)
    assert args.benchmark is None
    assert parser.parse_args(['--benchmark']).benchmark == 60


@async_test
async def test_power_read_error_fails_closed_and_is_retried():
    state=['failure'];now=[0.0];events=[]
    def read():
        if state[0]=='failure': raise OSError('unavailable')
        return state[0]
    desktop=DesktopControls(read, clock=lambda:now[0], log=lambda *a,**k:events.append((a,k)))
    desktop.refresh()
    assert desktop._paused and desktop._source=='unknown'
    pending=asyncio.create_task(desktop.before_dispatch(lambda:False))
    await asyncio.sleep(.01)
    assert not pending.done()
    state[0]='ac';now[0]=10.0
    assert await asyncio.wait_for(pending,1)
    assert any(args[0]=='power_source_unavailable' for args,fields in events)
    desktop.abort_dispatch()


@pytest.mark.parametrize('signum',[signal.SIGINT,signal.SIGTERM])
def test_startup_signal_unwinds_activity_and_restores_handlers(monkeypatch,signum):
    import os
    import pmk_miner.desktop as module
    events=[]
    native=DesktopNative.__new__(DesktopNative)
    native.lib=SimpleNamespace(pmk_activity_begin=lambda reason:123,
        pmk_activity_end=lambda handle:events.append(handle))
    monkeypatch.setattr(module,'DesktopNative',lambda:native)
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    with pytest.raises(SystemExit) as caught:
        with module.mining_session(on_battery='pause',intensity=100,log=lambda *a,**k:None):
            os.kill(os.getpid(),signum)
    assert caught.value.code==128+signum
    assert events==[123]
    assert all(signal.getsignal(sig)==handler for sig,handler in previous.items())
