"""T0 consensus/kernel bound parity at mainnet bits, using real native helpers."""
import pearl_mining as pm
import pytest

from pmk_miner.native import words
from pmk_miner.pool import validate_pool_notify
from pmk_miner.scheme import V3_SCHEME, compact_to_target

VECTORS = [(2**21, 0x1A07FFF8), (2_000_000, 0x1A086373),
           (50_000, 0x1B014F8A), (10_000, 0x1B068DB2)]


@pytest.mark.parametrize('difficulty,share_bits', VECTORS)
def test_mainnet_notify_bounds_match_consensus_and_kernel_encoding(difficulty, share_bits):
    header = pm.IncompleteBlockHeader(0x20000000, bytes.fromhex('11'*32),
        bytes.fromhex('22'*32), 1790986448, 0x177FD82E)
    target = (0xFFFF << 208) // difficulty
    job = validate_pool_notify(dict(job_id=f'00000000_{difficulty}',
        header=bytes(header.to_bytes()).hex(), target=f'{target:064x}',
        height=122415, cert_version=3), session_id=1)
    cfg = pm.MiningConfiguration(4096, 128, pm.MMAType.Int7xInt7ToInt32,
        pm.PeriodicPattern.from_list([0,8,64,72]),
        pm.PeriodicPattern.from_list([0,1,8,9,32,33,40,41]), None)
    encoded = bytes(cfg.to_bytes())
    assert job.bits == 0x177FD82E and job.share_nbits == share_bits
    block = V3_SCHEME.target_bound(job.block_target, encoded)
    share = V3_SCHEME.nbits_bound(job.share_nbits, encoded, job.target)
    assert block == int(pm.extract_difficulty_bound(job.bits, cfg))
    assert share == int(pm.extract_difficulty_bound(share_bits, cfg))
    assert 0 < block <= share < 2**256
    assert compact_to_target(share_bits) <= target
    for bound in (block, share):
        assert int.from_bytes(bytes(words(bound)), 'little') == bound
    # Compact rounding can only tighten a share bound, never ease it.
    assert share <= V3_SCHEME.target_bound(target, encoded)
