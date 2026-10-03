"""Cross-check of the K3-SG pattern + oracle + GPU kernel against Pearl's own miner and verifier (pearl_mining 0.3.1).

Same method as bench/f1_k3/crosscheck.py, with the SG MiningConfiguration (rows_pattern [0,8,16,24],
cols_pattern [0,1,8,9,16,17,24,25], r=128, Int7xInt7ToInt32): signals pinned with signal_range=(c,c) so the oracle
reproduces the whole job, predicts Pearl's winning tile BEFORE mine(), then checks opened rows/cols, raw roots,
verify_plain_proof_for_cert_version(3) and nbits_override just below/above the hash; finally the same noised job runs
through the production K3-SG kernel (`k3sg run`) with Pearl's bound.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

import numpy as np
import pearl_mining as pm

import sg_oracle
from sg_oracle import harness, oracle

M, N, K, R = 128, 128, 2048, 128
BIN = os.environ.get("K3SG_BIN", "/tmp/k3sg")
WORK = os.environ.get("K3SG_WORK", "/tmp/k3sg_jobs")


def make_header(nbits: int):
    return pm.IncompleteBlockHeader(version=0, prev_block=bytes(range(32)), merkle_root=b"0123456789abcdef" * 2,
                                    timestamp=0x66666666, nbits=nbits)


def make_config():
    rp = pm.PeriodicPattern.from_list(sg_oracle.ROWS_PATTERN)
    cp = pm.PeriodicPattern.from_list(sg_oracle.COLS_PATTERN)
    assert rp.to_list() == sg_oracle.ROWS_PATTERN and cp.to_list() == sg_oracle.COLS_PATTERN
    assert pm.PeriodicPattern.from_bytes(rp.to_bytes()).to_list() == sg_oracle.ROWS_PATTERN
    assert pm.PeriodicPattern.from_bytes(cp.to_bytes()).to_list() == sg_oracle.COLS_PATTERN
    assert [tuple(x) for x in rp.shape] == [tuple(x) for x in oracle.pattern_shape(sg_oracle.ROWS_PATTERN)], (rp.shape,)
    assert [tuple(x) for x in cp.shape] == [tuple(x) for x in oracle.pattern_shape(sg_oracle.COLS_PATTERN)], (cp.shape,)
    return pm.MiningConfiguration(common_dim=K, rank=R, mma_type=pm.MMAType.Int7xInt7ToInt32, rows_pattern=rp, cols_pattern=cp)


def _mine_child(c: int, nbits: int, q) -> None:
    try:
        proof = pm.mine(M, N, K, make_header(nbits), make_config(), signal_range=(c, c), cert_version=pm.CERT_VERSION_ZK_V3)
        q.put(("ok", proof.to_base64()))
    except Exception as e:  # noqa: BLE001 - report verbatim
        q.put(("err", repr(e)))


def mine_with_timeout(c: int, nbits: int, timeout: float = 120.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_mine_child, args=(c, nbits, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return ("timeout", None)
    return q.get()


def oracle_job(c: int, nbits: int):
    header, config = make_header(nbits), make_config()
    print(f"  config: hash_tile_h={config.hash_tile_h} hash_tile_w={config.hash_tile_w} rows.shape={config.rows_pattern.shape} "
          f"cols.shape={config.cols_pattern.shape} bytes={bytes(config.to_bytes()).hex()}")
    jk = oracle.job_key(bytes(header.to_bytes()), bytes(config.to_bytes()))
    A = np.full((M, K), c, dtype=np.int64)
    Bt = np.full((N, K), c, dtype=np.int64)
    a_bytes, bt_bytes = A.astype(np.int8).tobytes(), Bt.astype(np.int8).tobytes()
    assert oracle.pad_to_chunk(a_bytes) == bytes(pm.pad_to_chunk_boundary(a_bytes))
    raw_a = oracle.blake3(oracle.pad_to_chunk(a_bytes), key=jk).digest()
    raw_b = oracle.blake3(oracle.pad_to_chunk(bt_bytes), key=jk).digest()
    b_seed, a_seed = oracle.seeds(jk, raw_a, raw_b, M, N)
    na, nb = oracle.noise(K, R, b_seed, a_seed, M, N)
    Ap = A + na
    Bp = (Bt + nb).T
    assert np.abs(na).max() <= 63 and np.abs(nb).max() <= 63
    tiles = sg_oracle.all_tiles(Ap, Bp, a_seed)
    bound = oracle.extract_difficulty_bound(nbits, sg_oracle.H, sg_oracle.W, K)
    assert config.hash_tile_h == sg_oracle.H and config.hash_tile_w == sg_oracle.W
    return header, raw_a, raw_b, a_seed, Ap, Bp, tiles, bound


def nbits_around(hv: int, f: int) -> tuple[int, int]:
    for e in range(3, 33):
        unit = (1 << (8 * (e - 3))) * f
        q = hv // unit
        if 0x8000 <= q < 0x7FFFFF:
            return (e << 24) | q, (e << 24) | (q + 1)
    raise ValueError("no bracketing nbits")


def main() -> int:
    fails = 0
    nbits = 0x1E100000   # difficulty 2^236 x (h*w*k = 32*2048 = 2^16) = bound 2^252 -> per-tile p ~ 1/16
    print(f"pearl_mining (.venv 0.3.1); config m={M} n={N} k={K} r={R} rows={sg_oracle.ROWS_PATTERN} cols={sg_oracle.COLS_PATTERN}; "
          f"header nbits=0x{nbits:08x}; cert_version={pm.CERT_VERSION_ZK_V3}; kernel {BIN}")
    for c in (0, 1, -64, 64):
        print(f"\nsignal c={c}:")
        header, raw_a, raw_b, a_seed, Ap, Bp, tiles, bound = oracle_job(c, nbits)
        win = [i for i, t in enumerate(tiles) if t["hv"] <= bound]
        print(f"  a_noise_seed={a_seed.hex()} bound=0x{bound:064x}")
        print(f"  oracle: {len(win)}/{len(tiles)} tiles win; first (Pearl order) = tile #{win[0] if win else None}")
        if not win:
            print("  skipped: no winning tile (mine() would loop forever)")
            continue
        t = tiles[win[0]]
        print(f"  predicted: t_rows={t['t_rows']} t_cols={t['t_cols']} rows={t['rows']} cols={t['cols']}")
        print(f"  predicted jackpot hash (LE) = {t['hash'].hex()}")
        status, payload = mine_with_timeout(c, nbits)
        if status != "ok":
            print(f"  FAIL pearl_mining.mine: {status} {payload}")
            fails += 1
            continue
        proof = pm.PlainProof.from_base64(payload)
        checks = {
            "opened A rows == predicted rows": list(proof.a.row_indices) == t["rows"],
            "opened B^T rows == predicted cols": list(proof.bt.row_indices) == t["cols"],
            "raw A root == oracle": bytes(proof.a.root) == raw_a,
            "raw B^T root == oracle": bytes(proof.bt.root) == raw_b,
            "m, n, k, noise_rank": (proof.m, proof.n, proof.k, proof.noise_rank) == (M, N, K, R),
        }
        ok, msg = pm.verify_plain_proof_for_cert_version(3, header, proof)
        checks[f"verify_plain_proof_for_cert_version(3) accepts [{msg}]"] = bool(ok)
        lo, hi = nbits_around(t["hv"], sg_oracle.H * sg_oracle.W * K)
        for nb in (lo, hi):
            expect = t["hv"] <= oracle.extract_difficulty_bound(nb, sg_oracle.H, sg_oracle.W, K)
            ok2, msg2 = pm.verify_plain_proof_for_cert_version(3, header, proof, nbits_override=nb)
            checks[f"nbits_override=0x{nb:08x}: oracle says {'accept' if expect else 'reject'}, verifier {'accepts' if ok2 else 'rejects'} [{msg2[:70]}]"] = bool(ok2) == expect
        for k_, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'} {k_}")
            fails += 0 if v else 1
        d = f"{WORK}/pearl_c{c}"
        cases = [{"name": "pearl_bound", "bound_block": bound, "bound_share": oracle.U256_MAX, "cap_block": 128, "cap_share": 128}]
        harness.write_job(d, Ap, Bp, a_seed, cases)
        sg_oracle.write_tiles(d, tiles)
        harness.K3_BIN = BIN
        out = harness.run_gpu(d)
        okg, summ = harness.check_case(d, cases[0], tiles)
        cb, cs, blk, shr = harness.read_out(d, cases[0])
        gpu_has = (t["t_rows"], t["t_cols"]) in {(int(s[0]), int(s[1])) for s in blk[:cb]}
        print(f"  {'PASS' if okg else 'FAIL'} GPU K3-SG on this job vs oracle (python check): {summ}")
        print(f"  {'PASS' if 'JOB' in out and 'PASS' in out.splitlines()[-1] else 'FAIL'} GPU K3-SG swift-side check: {out.splitlines()[-1].strip()}")
        print(f"  {'PASS' if gpu_has else 'FAIL'} GPU block array contains Pearl's winning tile ({t['t_rows']},{t['t_cols']})")
        fails += (0 if okg else 1) + (0 if gpu_has else 1) + (0 if 'PASS' in out.splitlines()[-1] else 1)
    print(f"\nCROSS-CHECK vs pearl_mining: {'PASS' if fails == 0 else f'FAIL ({fails} failures)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
