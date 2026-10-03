"""Pool policy and payout guards without network or GPU work."""
from pathlib import Path

import pytest

from pmk_miner.pool_runtime import load_wallet, rejection_alarm


def test_wallet_requires_operator_allowlist(tmp_path):
    wallet = tmp_path / 'wallet'
    allowed = tmp_path / 'allowed'
    wallet.write_text('prl1operatorwallet\n')
    allowed.write_text('# approved by operator\nprl1someoneelse\n')
    with pytest.raises(ValueError, match='absent'):
        load_wallet(wallet, allowed)
    allowed.write_text('# approved\nprl1operatorwallet\n')
    assert load_wallet(wallet, allowed) == 'prl1operatorwallet'
    with pytest.raises(ValueError, match='required'):
        load_wallet(wallet, None)


@pytest.mark.parametrize('value', ['', 'prl1a b', 'prl1\nsecond', 'é', 'x' * 257])
def test_wallet_rejects_malformed(tmp_path, value):
    path = tmp_path / 'wallet'
    path.write_text(value)
    with pytest.raises(ValueError):
        load_wallet(path, path)


@pytest.mark.parametrize('outcome', ['invalid', 'low-difficulty'])
def test_local_gate_rejection_policy(outcome):
    assert rejection_alarm(outcome, 0) == 'pool rejected SG config; try next pool'
    assert rejection_alarm(outcome, 1).startswith('P0:')


@pytest.mark.parametrize('outcome', ['accepted', 'stale', 'duplicate', 'timeout', 'transport'])
def test_other_outcomes_do_not_claim_config_incompatibility(outcome):
    assert rejection_alarm(outcome, 1) is None


def test_pool_cli_never_loads_node_credentials_or_logs_arbitrary_errors(monkeypatch, tmp_path, capsys):
    import sys
    import pmk_miner.__main__ as cli
    config = tmp_path / 'pool.toml'
    config.write_text('[gateway]\nenv_file="/must/not/be/read"\n')
    monkeypatch.setattr(sys, 'argv', ['pmk', '--mode', 'pool', '--config', str(config)])
    node_calls = []
    def forbidden(**_kwargs):
        node_calls.append(True)
        raise AssertionError('node credentials were read')
    monkeypatch.setattr(cli.NodeRpcConfig, 'from_sources', forbidden)
    called = []
    def check(_self):
        called.append(True)
        raise RuntimeError('wallet=prl1private-value secret=private-credential')
    monkeypatch.setattr(cli.LabSession, 'check', check)
    assert cli.main() == 1
    assert called
    assert not node_calls
    output = capsys.readouterr().out
    assert 'RuntimeError' in output
    assert 'private-value' not in output
    assert 'private-credential' not in output


@pytest.mark.parametrize('gate', ['device', 'verifier'])
def test_fatal_gate_is_counted_in_pool_summary(monkeypatch, tmp_path, gate):
    import asyncio
    from types import SimpleNamespace
    from pmk_miner import pool_runtime as runtime
    from pmk_miner.pipeline import FatalDeviceError

    wallet = tmp_path / 'wallet'
    wallet.write_text('prl1syntheticwallet\n')
    args = SimpleNamespace(wallet_file=wallet, wallet_allowlist=wallet,
                           pool_url='stratum+tcp://127.0.0.1:1', worker='test')
    error = FatalDeviceError('injected startup gate', gate=gate)
    def fail_native():
        raise error
    async def idle_client(_self, stop):
        await stop.wait()
    monkeypatch.setattr(runtime, 'Native', fail_native)
    monkeypatch.setattr(runtime.PoolClient, 'run', idle_client)
    events = []
    with pytest.raises(FatalDeviceError) as caught:
        asyncio.run(runtime.mine_pool(args,
            {'run': {'state_dir': str(tmp_path / 'state')}},
            lambda event, **fields: events.append((event, fields)),
            lambda _shape: (1, 1)))
    assert caught.value is error
    summary = next(fields for event, fields in events if event == 'pool_summary')
    assert summary['gate_failures'] == 1
    assert summary['failed'] is True
    assert summary['completed_ops'] == 0
