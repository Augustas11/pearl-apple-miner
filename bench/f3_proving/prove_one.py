"""F3: time Pearl ZK certificate generation at the v1 production pattern for one (k, m, n).

Run one process per measurement (clean peak RSS, cold circuit cache):
  RAYON_NUM_THREADS=N .venv/bin/python bench/f3_proving/prove_one.py --k 4096 [--m 128 --n 64] [--reps 2]
Prints one JSON line.
"""
import argparse, json, resource, sys, time, os
import pearl_mining as pm

ROWS = [0, 8, 64, 72]
COLS = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
RANK = 128


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


def header(nbits, salt):
    return pm.IncompleteBlockHeader(version=0, prev_block=salt.to_bytes(32, "little"),
                                    merkle_root=b"0123456789abcdef" * 2, timestamp=0x66666666, nbits=nbits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nbits", type=lambda s: int(s, 0), default=0x1E010000)
    ap.add_argument("--reps", type=int, default=2, help="proofs generated in this process (1st = cold)")
    a = ap.parse_args()
    cfg = pm.MiningConfiguration(common_dim=a.k, rank=RANK, mma_type=pm.MMAType.Int7xInt7ToInt32,
                                 rows_pattern=pm.PeriodicPattern.from_list(ROWS),
                                 cols_pattern=pm.PeriodicPattern.from_list(COLS))
    out = dict(k=a.k, m=a.m, n=a.n, threads_env=os.environ.get("RAYON_NUM_THREADS"), reps=[])
    for rep in range(a.reps):
        h = header(a.nbits, 7000 + rep)
        t0 = time.perf_counter()
        plain = pm.mine(a.m, a.n, a.k, h, cfg, cert_version=3)
        t_mine = time.perf_counter() - t0
        ok, msg = pm.verify_plain_proof_for_cert_version(3, h, plain)
        assert ok, msg
        rss0 = rss_mb()
        t0 = time.perf_counter()
        zk = pm.generate_proof_for_cert_version(3, h, plain)
        t_prove = time.perf_counter() - t0
        rss1 = rss_mb()
        t0 = time.perf_counter()
        vok, vmsg = pm.verify_proof_for_cert_version(3, h, zk)
        t_ver = time.perf_counter() - t0
        out["reps"].append(dict(mine_s=t_mine, plain_ok=ok, prove_s=t_prove, verify_s=t_ver, zk_ok=vok, zk_msg=vmsg,
                                peak_rss_mb=rss1, rss_before_prove_mb=rss0,
                                proof_bytes=len(zk.proof_data), public_bytes=len(zk.public_data)))
    print(json.dumps(out))


main()
