from __future__ import annotations

import pytest

from pmk_miner.monitor import (
    MonitorAlarm,
    ShareMonitor,
    bits_to_target,
    choose_share_nbits,
    expected_blocks_per_day,
    expected_shares,
    poisson_interval,
    script_pubkey_from_p2tr_address,
    target_to_bits_floor,
    validate_coinbase_payout,
    validate_payout_startup,
)


REGTEST_ADDR = "rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"


def test_compact_round_trip_floor():
    for bits in (0x1D00FFFF, 0x1E010000, 0x1A07FFF8):
        target = bits_to_target(bits)
        compact = target_to_bits_floor(target)
        assert bits_to_target(compact) <= target
        assert compact == bits


def test_choose_share_nbits_targets_about_one_share_per_minute():
    ops = 1_000_000_000_000
    share_bits = choose_share_nbits(ops, target_shares_per_minute=1.0)
    expected = expected_shares(int(ops * 60), share_bits)
    assert 0.99 <= expected <= 1.01


def test_expected_blocks_per_day_formula():
    target = bits_to_target(0x1E010000)
    expected = expected_blocks_per_day(2_000_000, target)
    assert expected == pytest.approx(2_000_000 * 86400 * target / (1 << 257))


def test_poisson_exact_interval_known_window():
    assert poisson_interval(0) == (0, 0)
    low, high = poisson_interval(10.0, alpha=0.001)
    assert low == 2
    assert high == 22


def test_poisson_interval_stable_for_g6_daily_window():
    low, high = poisson_interval(1440.0, alpha=0.001)
    assert low == 1317
    assert high == 1566


def test_share_monitor_alarms_on_lost_find_window():
    share_bits = target_to_bits_floor(1 << 240)
    target = bits_to_target(share_bits)
    completed_ops = int(10 * (1 << 257) / target)
    monitor = ShareMonitor(share_bits)
    window = monitor.record(completed_ops=completed_ops, shares=0)
    assert window.expected == pytest.approx(expected_shares(completed_ops, share_bits))
    assert window.alarm is True


def test_share_monitor_enforces_fixed_window_max():
    share_bits = target_to_bits_floor(1 << 240)
    monitor = ShareMonitor(share_bits, max_completed_ops=10)
    monitor.record(completed_ops=10, shares=1)
    with pytest.raises(MonitorAlarm, match="fixed window"):
        monitor.record(completed_ops=1, shares=0)


def test_payout_startup_validates_regtest_hrp_and_script():
    script_hex = validate_payout_startup(REGTEST_ADDR, "rprl")
    assert script_hex.startswith("5120")
    assert len(bytes.fromhex(script_hex)) == 34
    assert validate_payout_startup(REGTEST_ADDR, "rprl", approved_script_hex=script_hex) == script_hex
    with pytest.raises(MonitorAlarm, match="HRP"):
        validate_payout_startup(REGTEST_ADDR, "prl")


def test_coinbase_payout_check_accepts_approved_script():
    approved = script_pubkey_from_p2tr_address(REGTEST_ADDR, "rprl").hex()
    block = {
        "tx": [
            {
                "vout": [
                    {
                        "value": 50,
                        "scriptPubKey": {"hex": approved, "address": REGTEST_ADDR},
                    }
                ]
            }
        ]
    }
    assert validate_coinbase_payout(block, approved_script_hex=approved, expected_hrp="rprl")


def test_coinbase_payout_check_uses_pearl_getblock_rawtx_schema():
    approved = script_pubkey_from_p2tr_address(REGTEST_ADDR, "rprl").hex()
    block = {
        "hash": "00" * 32,
        "tx": ["coinbase-txid-only"],
        "rawtx": [
            {
                "txid": "coinbase-txid-only",
                "vout": [
                    {
                        "n": 0,
                        "value": 50,
                        "scriptPubKey": {"hex": approved, "address": REGTEST_ADDR},
                    }
                ],
            }
        ],
    }
    assert validate_coinbase_payout(block, approved_script_hex=approved, expected_hrp="rprl")


def test_coinbase_payout_check_rejects_wrong_script():
    approved = script_pubkey_from_p2tr_address(REGTEST_ADDR, "rprl").hex()
    block = {"tx": [{"vout": [{"scriptPubKey": {"hex": "6a"}}]}]}
    with pytest.raises(MonitorAlarm, match="approved script"):
        validate_coinbase_payout(block, approved_script_hex=approved)
