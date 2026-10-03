#!/usr/bin/env python3
"""Beta CLI: private wallet files, admission, and existing pool telemetry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'miner'))


def worker_name():
    name = subprocess.check_output(['hostname', '-s'], text=True).strip()
    return re.sub(r'[^A-Za-z0-9_-]', '-', name).strip('-')[:64] or 'mac'


def write_private(path, content):
    # Replace atomically, without following an existing wallet symlink.
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def gpu_core_count():
    try:
        devices = json.loads(subprocess.check_output(
            ['system_profiler', 'SPDisplaysDataType', '-json'], text=True, timeout=15))
        return max(int(device.get('sppci_cores', 0))
                   for device in devices.get('SPDisplaysDataType', []))
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def choose_shape(ram, gpu_cores=0):
    from pmk_miner.pipeline import Shape
    # Leave command-buffer headroom on base/fanless chips, even with plenty of RAM.
    dimensions = (8192, 4096, 2048) if gpu_cores >= 32 else (4096, 2048)
    for dimension in dimensions:
        shape = Shape(m=dimension, n=dimension)
        try:
            shape.validate(ram)
            return shape
        except ValueError:
            continue
    raise ValueError('not enough memory for a mining job')


class Status:
    """Render the miner's existing telemetry; retain structured safety events."""
    def __init__(self, sink, clock=time.monotonic):
        self.sink = sink
        self.clock = clock
        self.last = float('-inf')

    def __call__(self, event, **fields):
        if event == 'routine_telemetry':
            now = self.clock()
            if now - self.last >= 60:
                self.last = now
                seconds = fields.get('expected_share_seconds')
                estimate = f'{seconds / 60:.1f} min' if seconds else 'waiting for work'
                print(f"[pmk] {fields['jobs_per_second']:.2f} jobs/s | {fields['tops']:.2f} TOPS | "
                      f"shares accepted={fields['accepted']} rejected={fields['rejected']} | "
                      f"expected time/share {estimate}", flush=True)
        elif event == 'pool_summary':
            self.sink(event, **fields)
            print(f"[pmk] stopped: accepted={fields['accepted']} "
                  f"rejected={fields['rejected'] + fields['stale']}", flush=True)
        elif event not in {'gpu_dispatch', 'completed', 'python_overhead'}:
            self.sink(event, **fields)


def main():
    parser = argparse.ArgumentParser(prog='scripts/mine.sh')
    parser.add_argument('--wallet', required=True, help='your Pearl wallet (prl1...)')
    parser.add_argument('--worker', help='worker name (default: sanitized short hostname)')
    parser.add_argument('--pool', default='stratum+tcp://sg.pearl.herominers.com:1200')
    args = parser.parse_args()
    if not re.fullmatch(r'prl1[a-z0-9]{6,252}', args.wallet):
        parser.error('wallet must be a single prl1... address; never enter a private key')
    worker = args.worker if args.worker is not None else worker_name()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', worker):
        parser.error('worker must use 1–64 letters, numbers, underscores or hyphens')
    from pmk_miner.pool import PoolClient
    # Validate the URL without opening a connection, before writing wallet files.
    try:
        PoolClient(args.pool, args.wallet, worker)
    except ValueError:
        parser.error('pool must be a valid stratum+tcp://host:port or stratum+ssl://host:port URL')
    shape = choose_shape(int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])), gpu_core_count())
    state = Path(os.environ.get('PMK_HOME', Path.home() / '.pmk')).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    write_private(state / 'wallet', args.wallet + '\n')
    write_private(state / 'wallet-allowlist', args.wallet + '\n')
    config = state / 'pool.toml'
    write_private(config, f'm = {shape.m}\nn = {shape.n}\nk = {shape.k}\nslots = {shape.slots}\n'
                  '[run]\nstate_dir = ' + json.dumps(str(state / 'pool')) +
                  '\ntelemetry_interval_seconds = 5\n')
    os.environ['PMK_G3_ADMISSION_FILE'] = str(state / 'g3-admission.json')
    # A beta run owns its GPU lock and never consumes a long-run harness token.
    os.environ.pop('PMK_GPU_LOCK_HELD', None)
    for key in tuple(os.environ):
        if key.startswith(('B4_LAB_', 'PMK_LAB_', 'PMK_B4_LAB_')):
            os.environ.pop(key)
    from pmk_quickstart import ensure_admission
    from pmk_miner import __main__ as miner
    def refresh_admission(**kwargs):
        path = ensure_admission(inherited=True, **kwargs)
        records = json.loads(path.read_text())['devices']
        return min(record['last_probe_unix'] + min(record.get('valid_hours', 6), 6) * 3600
                   for record in records if record.get('g3_passed') is True)
    sys.argv = ['pmk_miner', '--mode', 'pool', '--pool-url', args.pool,
                '--wallet-file', str(state / 'wallet'), '--wallet-allowlist',
                str(state / 'wallet-allowlist'), '--worker', worker, '--config', str(config)]
    print(f'[pmk] job size {shape.m}x{shape.n}x{shape.k}; checking GPU admission; '
          'Ctrl-C stops mining cleanly.', flush=True)
    return miner.main(standalone=True, pool_log=Status(miner.log),
                      admission_refresh=refresh_admission)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
