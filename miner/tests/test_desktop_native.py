"""Real macOS assertions: no Metal device or GPU work is needed."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]


def assertion_lines(reason):
    output = subprocess.check_output(['pmset','-g','assertions'], text=True)
    return [line.strip() for line in output.splitlines() if reason in line]


def wait_assertion(reason, present):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        lines=assertion_lines(reason)
        if bool(lines)==present:
            return lines
        time.sleep(.05)
    raise AssertionError(f'pmk activity assertion present={bool(lines)}, expected={present}')


@pytest.mark.parametrize('exit_kind',['normal','exception','sigint','sigterm'])
def test_real_activity_assertion_released_after_exit(exit_kind):
    reason='pmk-b11-test-'+uuid.uuid4().hex
    code='''
import signal, sys
from pmk_miner.desktop import DesktopNative

def stop(signum, frame):
    raise SystemExit(0)

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, stop)
with DesktopNative().activity(sys.argv[1]):
    print('active', flush=True)
    command=sys.stdin.readline().strip()
    if command == 'exception':
        raise RuntimeError('intentional activity lifetime check')
print('ended', flush=True)
sys.stdin.readline()
'''
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'miner')
    process=subprocess.Popen([sys.executable,'-c',code,reason],stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    try:
        assert process.stdout.readline().strip()=='active'
        lines=wait_assertion(reason,True)
        assert any('PreventUserIdleSystemSleep' in line for line in lines), lines
        assert not any('PreventUserIdleDisplaySleep' in line for line in lines), lines
        if exit_kind in ('sigint','sigterm'):
            process.send_signal(signal.SIGINT if exit_kind=='sigint' else signal.SIGTERM)
        else:
            process.stdin.write(exit_kind+'\n');process.stdin.flush()
        if exit_kind=='normal':
            assert process.stdout.readline().strip()=='ended'
            # Verify end releases the assertion while the same process is alive.
            wait_assertion(reason,False)
            assert process.poll() is None
            process.stdin.write('quit\n');process.stdin.flush()
        process.wait(timeout=10)
        assert process.returncode == (1 if exit_kind=='exception' else 0)
        wait_assertion(reason,False)
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=5)
        process.stdin.close();process.stdout.close();process.stderr.close()
