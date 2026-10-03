#!/usr/bin/env python3
"""Exercise mine.sh with real GPU proofs, a loopback pool, and SIGINT."""
import asyncio
import json
import os
from pathlib import Path
import signal
import time

from pmk_pool_mock_e2e import MockPool, DEFAULT_WALLET

ROOT = Path(__file__).resolve().parents[1]


async def main():
    if not os.environ.get('PMK_HOME'):
        raise SystemExit('Set PMK_HOME to a disposable test directory and run install.sh there first.')
    # Exercise the real refresh path without waiting six hours. Only test state is aged.
    admission = Path(os.environ['PMK_HOME']).expanduser() / 'g3-admission.json'
    record = json.loads(admission.read_text())
    for device in record['devices']:
        device['last_probe_unix'] = time.time() - 6 * 3600 + 55
    admission.write_text(json.dumps(record))
    print('[quickstart-e2e] test admission expires in 55s; expecting an early G3 refresh', flush=True)
    pool = MockPool(difficulty=10_000, rotate_seconds=15, block_difficulty=1_000_000_000)
    await pool.start()
    process = None
    events = []
    statuses = []
    refreshed = []
    try:
        process = await asyncio.create_subprocess_exec(
            str(ROOT / 'scripts/mine.sh'), '--wallet', DEFAULT_WALLET,
            '--pool', pool.url(), cwd=ROOT, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

        async def read_output():
            async for raw in process.stdout:
                line = raw.decode(errors='replace').rstrip()
                print(line, flush=True)
                if line.startswith('{'):
                    events.append(json.loads(line))
                if line.startswith('[pmk]') and 'jobs/s' in line:
                    statuses.append(line)
                if line.startswith('G3 admission passed:'):
                    refreshed.append(line)

        reader = asyncio.create_task(read_output())
        # Includes waiting for another owner's GPU lock and random share arrivals.
        async with asyncio.timeout(1800):
            # Keep running for a second status line, even if three shares arrive early.
            while pool.stats.accepted < 3 or len(statuses) < 2:
                if process.returncode is not None:
                    raise AssertionError(f'mine.sh exited early: {process.returncode}')
                await asyncio.sleep(.25)
            print(f'[quickstart-e2e] mock accepted={pool.stats.accepted}; sending SIGINT', flush=True)
            os.kill(process.pid, signal.SIGINT)
            rc = await asyncio.wait_for(process.wait(), 60)
            await reader
        summary = next(row for row in reversed(events) if row.get('event') == 'pool_summary')
        assert rc == 0, rc
        assert summary['accepted'] >= 3 and summary['gate_failures'] == 0
        assert summary['rejected'] == 0 and not summary['failed'], summary
        assert pool.stats.invalid == 0 and pool.stats.wrong_job_id == 0
        assert refreshed, 'near-expiry admission was not refreshed'
        try:
            owner = json.loads(Path('/tmp/pmm-gpu-bench.lock/pmk-owner.json').read_text())
            assert owner.get('pid') != process.pid, 'miner left its GPU lock behind'
        except FileNotFoundError:
            pass  # Another waiter may acquire the lock immediately after this miner exits.
        print(f"PASS: accepted={summary['accepted']}, rejected=0, SIGINT exit=0, "
              f"status_lines={len(statuses)}, G3 refreshed, GPU lock released", flush=True)
    finally:
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 60)
            except TimeoutError:
                process.kill()
                await process.wait()
        await pool.stop()


if __name__ == '__main__':
    asyncio.run(main())
