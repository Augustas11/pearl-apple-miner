#!/usr/bin/env python3
"""Independent verification of every PMKVEC01 field using the existing NumPy oracle."""
import pathlib
import struct
import sys

import numpy as np
from blake3 import blake3
import pearl_mining as pm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'bench' / 'f1_k3'))
import oracle


def main():
    encoded = sys.stdin.buffer.read()
    assert encoded[:8] == b'PMKVEC01'
    count, = struct.unpack_from('<I', encoded, 8)
    fields = {}
    at = 12
    for _ in range(count):
        n, = struct.unpack_from('<H', encoded, at)
        at += 2
        name = encoded[at:at+n].decode('ascii')
        at += n
        n, = struct.unpack_from('<Q', encoded, at)
        at += 8
        assert name not in fields
        fields[name] = encoded[at:at+n]
        assert len(fields[name]) == n
        at += n
    assert at == len(encoded)
    m, n, k, r = struct.unpack('<4I', fields['dimensions'])
    header = pm.IncompleteBlockHeader.from_bytes(fields['header'])
    cfg = pm.MiningConfiguration.from_bytes(fields['config'])
    assert bytes(header.to_bytes()) == fields['header']
    assert bytes(cfg.to_bytes()) == fields['config']
    assert cfg.common_dim == k and cfg.rank == r
    a = np.frombuffer(fields['padded_a'], dtype=np.int8)[:m*k].reshape(m, k)
    bt = np.frombuffer(fields['padded_bt'], dtype=np.int8)[:n*k].reshape(n, k)
    # Inputs and headers are pinned, independently of Rust's export.
    assert fields['header'] == bytes(range(76))
    assert np.array_equal(a.ravel(), (np.arange(m*k) * 17 + 3) % 129 - 64)
    assert np.array_equal(bt.ravel(), (np.arange(n*k) * 29 + 11) % 129 - 64)
    expected = {
        'dimensions': struct.pack('<4I', m, n, k, r),
        'header': bytes(header.to_bytes()), 'config': bytes(cfg.to_bytes()),
        'padded_a': oracle.pad_to_chunk(a.tobytes()),
        'padded_bt': oracle.pad_to_chunk(bt.tobytes()),
        'salt_key_a': oracle.SEED_SALT_A, 'salt_key_b': oracle.SEED_SALT_B,
    }
    jk = oracle.job_key(fields['header'], fields['config'])
    ra = blake3(expected['padded_a'], key=jk).digest()
    rb = blake3(expected['padded_bt'], key=jk).digest()
    sa = oracle.bind_root(ra, m, oracle.SEED_SALT_A)
    sb = oracle.bind_root(rb, n, oracle.SEED_SALT_B)
    bs, ass = oracle.seeds(jk, ra, rb, m, n)
    expected.update(job_key=jk, raw_root_a=ra, raw_root_b=rb,
                    salt_input_a=ra + struct.pack('<I', m) + bytes(28),
                    salt_input_b=rb + struct.pack('<I', n) + bytes(28),
                    salted_root_a=sa, salted_root_b=sb,
                    seed_input_a=bs + sa, seed_input_b=jk + sb,
                    a_noise_seed=ass, b_noise_seed=bs)
    al = oracle.uniform_rows(oracle.LABEL_A, ass, list(range(m)), r)
    ar = oracle.perm_matrix(oracle.LABEL_A, ass, k, r)
    bl = oracle.perm_matrix(oracle.LABEL_B, bs, k, r)
    br = oracle.uniform_rows(oracle.LABEL_B, bs, list(range(n)), r)
    na = al[:, ar[:, 0]] - al[:, ar[:, 1]]
    nb = br[:, bl[:, 0]] - br[:, bl[:, 1]]
    ap = a.astype(np.int64) + na
    btp = bt.astype(np.int64) + nb
    for name, value in [('e_al', al), ('e_br_t', br), ('noise_a', na), ('noise_bt', nb),
                        ('noised_a', ap), ('noised_bt', btp)]:
        expected[name] = value.astype(np.int8).tobytes()
    expected['e_ar_t'] = ar.astype('<u4').tobytes()
    expected['e_bl'] = bl.astype('<u4').tobytes()
    assert set(fields) == set(expected)
    for name, value in expected.items():
        assert fields[name] == value, f'{name}: intermediate mismatch'
    print(blake3(encoded).hexdigest())
    rs, cs, jp = oracle.transcripts(ap, btp.T, cfg.rows_pattern.to_list(), cfg.cols_pattern.to_list())
    for ri, ci in [(0, 0), (1, 1), (len(rs)-1, len(cs)-1)]:
        transcript = jp[ri, ci]
        print(int(rs[ri, 0]), int(cs[ci, 0]), *map(int, transcript), oracle.jackpot_hash(transcript, ass).hex())


if __name__ == '__main__':
    main()
