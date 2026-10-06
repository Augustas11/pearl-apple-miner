# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual CLI submission/restart orchestration without GPU work."""
import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

import pmk_miner.__main__ as cli
from pmk_miner.native import NativeError
from pmk_miner.monitor import bits_to_target
from pmk_miner.transport import GatewayJob, SubmissionLedger, SubmissionOutcome, TransportError
from pmk_miner.v4_admission import V4_G3_ADMISSION_ENV, V4_VENDOR_PEARL_FP8


def write_v4_admission(tmp_path, monkeypatch):
    now=time.time()
    record={"schema":"pmk-v4-admission-v1","gpu_name":"Unit GPU","device_class":"Apple7-9",
        "os_build":"unit-os","cache_key":"unit-cache","kernel":"E","metal_language":"3.1",
        "library_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vendor_pearl_fp8":V4_VENDOR_PEARL_FP8,"v4_g3_passed":True,"exact_cells":100_000_000,
        "last_probe_unix":now-1,"valid_until_unix":now+3600}
    path=tmp_path / "v4-admission.json"
    path.write_text(json.dumps({"devices":[record]}), encoding="utf-8")
    monkeypatch.setenv(V4_G3_ADMISSION_ENV,str(path))
    return path


def test_stop_cancels_long_lived_submission_trackers():
    async def exercise():
        cancelled=asyncio.Event()

        async def tracker():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task=asyncio.create_task(tracker())
        await asyncio.sleep(0)
        await cli.cancel_pending_tasks({task})
        assert task.cancelled()
        assert cancelled.is_set()

    asyncio.run(exercise())


def setup_miner(monkeypatch, tmp_path, outcomes, *, lose_ack=False):
    monkeypatch.setenv('PMK_DEVICE_CLASS', 'Apple9')
    header = (1).to_bytes(4, 'little') + bytes(64) + (1).to_bytes(4, 'little') + (0x177fd82e).to_bytes(4, 'little')
    job = GatewayJob(header, bits_to_target(0x177fd82e), 3)
    log = tmp_path / 'gateway.log'
    log.touch()
    config = {'m': 128, 'n': 128, 'payout': {'script': '5120' + '11'*32, 'hrp': 'rprl'},
              'gateway': {'log_file': str(log)},
              'run': {'state_dir': str(tmp_path / 'state'), 'max_accepted': 1, 'template_poll_seconds': .001}}
    calls = {'submitted': [], 'dispatched': 0, 'native': 0}
    class Gateway:
        def __init__(self, *_): pass
        async def get_job(self): return job
        async def submit(self, source, proof):
            calls['submitted'].append((source.submission_id, proof))
            if lose_ack: raise TransportError('ack lost')
    class Node:
        def __init__(self, *_): pass
        async def get_best_block_hash(self): return 'block'
        async def get_block(self, *_):
            return {'version': job.version, 'previousblockhash': job.prev_hash,
                    'bits': job.bits_hex, 'merkleroot': job.merkle_root, 'time': job.timestamp,
                    'rawtx': [{'vout': [{'scriptPubKey': {'hex': config['payout']['script']}}]}]}
    class Native:
        probe_key = 'unit-test'
        def __init__(self): calls['native'] += 1
        def close(self): pass
    class Pipeline:
        def __init__(self, *_): self.stopped = False
        def cancel(self): self.stopped = True
        def set_template(self, source): self.source = source
        async def run(self, index, target, bits, submit, current):
            calls['dispatched'] += 1
            await submit(self.source, 'YQ==' if index == 0 else 'Yg==')
            await asyncio.sleep(.005)
            return SimpleNamespace(cancelled=self.stopped, share_count=0)
    pending = iter(outcomes)
    class Tracker:
        def __init__(self, *_args, **_kw): pass
        async def track(self, source):
            await asyncio.sleep(.01)
            return next(pending)
    monkeypatch.setattr(cli, 'GatewayClient', Gateway)
    monkeypatch.setattr(cli, 'NodeRpcClient', Node)
    monkeypatch.setattr(cli, 'Native', Native)
    monkeypatch.setattr(cli, 'Pipeline', Pipeline)
    monkeypatch.setattr(cli, 'SubmissionTracker', Tracker)
    monkeypatch.setattr(cli.NodeRpcConfig, 'from_sources', lambda **kw: SimpleNamespace(mining_address='test'))
    monkeypatch.setattr(cli, 'validate_payout_startup', lambda *a, **kw: None)
    monkeypatch.setattr(GatewayJob, 'authorize_coinbase', lambda *a, **kw: None)
    monkeypatch.setattr(cli.Shape, 'validate', lambda *a, **kw: 1)
    monkeypatch.setattr(cli.Shape, 'memory_estimate', lambda *a, **kw: 1)
    monkeypatch.setattr(cli, 'log', lambda *a, **kw: None)
    return job, config, calls, SimpleNamespace(
        gateway='localhost:1', config=tmp_path/'config', kernel='sg')


def test_restart_confirms_outstanding_before_dispatch_and_never_resubmits(monkeypatch, tmp_path):
    job, config, calls, args = setup_miner(monkeypatch, tmp_path, [SubmissionOutcome.ACCEPTED])
    ledger = SubmissionLedger(tmp_path/'state/submissions.jsonl')
    entry = ledger.prepare(job, 'YQ==')
    asyncio.run(cli.mine(args, config))
    assert calls['submitted'] == []
    assert calls['dispatched'] == 0
    assert SubmissionLedger(ledger.path).entries()[entry.submission_id].outcome == SubmissionOutcome.ACCEPTED


def test_restart_reconstructs_outstanding_v4_submission_without_downgrade(monkeypatch, tmp_path):
    write_v4_admission(tmp_path,monkeypatch)
    job, config, calls, args = setup_miner(monkeypatch, tmp_path, [SubmissionOutcome.ACCEPTED])
    ancestors=(b"a"*108,b"b"*108)
    v4_job=GatewayJob(job.header,bits_to_target(0x177fd82e),4,ancestor_headers=ancestors)
    ledger=SubmissionLedger(tmp_path/'state/submissions.jsonl')
    entry=ledger.prepare(v4_job,'YQ==')
    tracked=[]

    class Tracker:
        def __init__(self, *_args, **_kw): pass
        async def track(self, source):
            tracked.append(source)
            return SubmissionOutcome.ACCEPTED

    monkeypatch.setattr(cli, 'SubmissionTracker', Tracker)
    asyncio.run(cli.mine(args,config))
    assert calls['submitted'] == []
    assert calls['dispatched'] == 0
    assert tracked[0].cert_version == 4
    assert tracked[0].ancestor_headers == ancestors
    assert SubmissionLedger(ledger.path).entries()[entry.submission_id].outcome == SubmissionOutcome.ACCEPTED


def test_unknown_submission_stops_and_remains_halted_after_restart(monkeypatch, tmp_path):
    _, config, calls, args = setup_miner(monkeypatch, tmp_path, [SubmissionOutcome.UNKNOWN_SUBMISSION], lose_ack=True)
    with pytest.raises(cli.FatalDeviceError, match='failed closed'):
        asyncio.run(cli.mine(args, config))
    assert len(calls['submitted']) == 1
    with pytest.raises(cli.FatalDeviceError, match='durable safety halt'):
        asyncio.run(cli.mine(args, config))
    assert len(calls['submitted']) == 1
    assert calls['native'] == 1


def test_consensus_invalid_stops_and_persists_halt(monkeypatch, tmp_path):
    _, config, calls, args = setup_miner(monkeypatch, tmp_path, [SubmissionOutcome.CONSENSUS_INVALID])
    with pytest.raises(cli.FatalDeviceError, match='failed closed'):
        asyncio.run(cli.mine(args, config))
    assert len(calls['submitted']) == 1
    assert SubmissionLedger(tmp_path/'state/submissions.jsonl').fail_closed_entries()


def test_same_template_bounded_backup_used_after_proving_error(monkeypatch, tmp_path):
    _, config, calls, args = setup_miner(monkeypatch, tmp_path, [SubmissionOutcome.PROVING_ERROR, SubmissionOutcome.ACCEPTED])
    asyncio.run(cli.mine(args, config))
    assert len(calls['submitted']) == 2
    assert len({proof for _, proof in calls['submitted']}) == 2
    assert len({sid for sid, _ in calls['submitted']}) == 2


@pytest.mark.parametrize('outcome', [SubmissionOutcome.STALE, SubmissionOutcome.DUPLICATE])
def test_terminal_template_is_not_reenabled_by_restart(monkeypatch, tmp_path, outcome):
    _, config, calls, args = setup_miner(monkeypatch, tmp_path, [outcome])
    async def bounded_run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cli.mine(args, config), .08)
    asyncio.run(bounded_run())
    assert len(calls['submitted']) == 1
    before = calls['dispatched']
    asyncio.run(bounded_run())
    assert len(calls['submitted']) == 1
    assert calls['dispatched'] == before


def test_native_error_message_is_redacted_and_bounded(monkeypatch, tmp_path):
    from pmk_miner.transport import NodeRpcConfig
    exc = NativeError('pmk_init', -103, 'startup failed password=supersecret bare-secret ' + 'x' * 400)
    fields = cli._native_error_fields(
        exc,
        NodeRpcConfig('http://127.0.0.1:1', 'alice', 'bare-secret'),
    )
    assert fields['native_function'] == 'pmk_init'
    assert fields['native_code'] == -103
    assert 'supersecret' not in fields['error_message']
    assert 'bare-secret' not in fields['error_message']
    assert len(fields['error_message']) <= 256


def _run_cli_main(monkeypatch, config):
    import sys
    monkeypatch.setattr(sys, 'argv', ['pmk', '--mode', 'solo', '--gateway', '127.0.0.1:1', '--config', str(config)])
    return cli.main()
