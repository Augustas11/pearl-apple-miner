"""Pure verdict regressions: never certify compatibility without acceptance."""
import argparse

import pytest

from pool_window import classify_summary, sanitize_pool_url


@pytest.mark.parametrize(
    'outcome,status,exit_code',
    [('accepted', 'PASS', 0), ('timeout', 'INCONCLUSIVE', 3),
     ('transport', 'INCONCLUSIVE', 3), ('stale', 'FAIL', 2),
     ('invalid', 'FAIL', 2), ('duplicate', 'FAIL', 2),
     ('low_difficulty', 'FAIL', 2)],
)
def test_t2_first_share_verdict(outcome, status, exit_code):
    event = {'submitted': 1, outcome: 1, 'poisson_ok': True}
    result = classify_summary(event, argparse.Namespace(max_accepted=1, max_submitted=1), returncode=0)
    assert result['status'] == status
    assert result['pass'] is (status == 'PASS')
    assert result['exit_code'] == exit_code


@pytest.mark.parametrize('event,rc,error,timed_out,status', [
    ({'submitted': 0}, 0, None, False, 'INCONCLUSIVE'),
    (None, -15, 'pool miner exceeded window deadline', True, 'INCONCLUSIVE'),
    (None, 0, None, False, 'FAIL'),
    ({'timeout': 1}, -15, 'interrupted', False, 'FAIL'),
    ({'gate_failures': 1}, 0, None, False, 'FAIL'),
    ({'failed': True}, 0, None, False, 'FAIL'),
    ({'invalid': 1}, -15, 'pool miner exceeded window deadline', True, 'FAIL'),
    ({'accepted': 1, 'invalid': 1}, 0, None, False, 'FAIL'),
    ({'accepted': 1, 'timeout': 1}, 0, None, False, 'FAIL'),
    ({'accepted': 1, 'poisson_ok': False}, 0, None, False, 'FAIL'),
])
def test_t2_missing_verdict_and_failure_precedence(event, rc, error, timed_out, status):
    result = classify_summary(event, argparse.Namespace(max_accepted=1, max_submitted=1),
                              returncode=rc, error=error, timed_out=timed_out)
    assert result['status'] == status
    assert result['pass'] is False


@pytest.mark.parametrize('accepted', [0, 19, 20])
def test_t3_requires_acceptance_target(accepted):
    result = classify_summary({'accepted': accepted, 'submitted': 20},
                              argparse.Namespace(max_accepted=20, max_submitted=20), returncode=0)
    assert result['pass'] is (accepted == 20)


@pytest.mark.parametrize('scheme', ['stratum+tcp', 'stratum+ssl', 'stratum+tls', 'tls'])
def test_supported_url_schemes(scheme):
    assert sanitize_pool_url(f'{scheme}://localhost:1200') == f'{scheme}://localhost:1200'
