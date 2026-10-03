"""F1 correctness: production K3 kernel (k3 run) vs CPU oracle, bit-exact.

(a) every transcript + hash of full jobs (256x256x4096, 128x64x65536, + extras) via share bound = U256 max
(b) boundary vectors: bound = hash-1 / hash / hash+1 (and +-2^(32i)) for min / median / max-hash tiles
(c) non-saturated deterministic wins and losses (run twice)
(d) slot overflow: bound = max with production capacities (block 4, share 64)
(e) simultaneous block + share finds land in the right arrays
Operands: uniform in [-127,127] (noised range) and a +-127 extreme-magnitude job.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import harness
import oracle

U = oracle.U256_MAX
WORK = os.environ.get("K3_WORK", "/tmp/k3_f1_jobs")


def job_cases(tiles: list[dict], boundary: bool) -> list[dict]:
    T = len(tiles)
    hs = sorted(t["hv"] for t in tiles)
    cases = [
        {"name": "a_all", "bound_block": U // 32, "bound_share": U, "cap_block": T, "cap_share": T},
        {"name": "c_det_run1", "bound_block": U // 200, "bound_share": U // 16, "cap_block": T, "cap_share": T},
        {"name": "c_det_run2", "bound_block": U // 200, "bound_share": U // 16, "cap_block": T, "cap_share": T},
        {"name": "d_overflow_prodcaps", "bound_block": U, "bound_share": U, "cap_block": 4, "cap_share": 64},
        {"name": "d_overflow_share_only", "bound_block": 0, "bound_share": U, "cap_block": 4, "cap_share": 64},
        {"name": "e_simul_bigcaps", "bound_block": U // 40, "bound_share": U // 6, "cap_block": T, "cap_share": T},
        {"name": "e_simul_prodcaps", "bound_block": hs[2], "bound_share": hs[40], "cap_block": 4, "cap_share": 64},
        {"name": "e_simul_exact_fit", "bound_block": hs[3], "bound_share": hs[63], "cap_block": 4, "cap_share": 64},
    ]
    if boundary:
        picks = {"min": hs[0], "median": hs[T // 2], "max": hs[-1]}
        for label, h in picks.items():
            for dname, delta in (("m1", -1), ("eq", 0), ("p1", 1)):
                b = h + delta
                if 0 <= b <= U:
                    cases.append({"name": f"b_{label}_{dname}", "bound_block": b, "bound_share": b, "cap_block": T, "cap_share": T})
        h = picks["median"]
        for i in range(1, 8):
            for sgn, dn in ((-1, "m"), (1, "p")):
                b = h + sgn * (1 << (32 * i))
                if 0 <= b <= U:
                    # block bound one step tighter than share bound: tile classified share-only vs both
                    cases.append({"name": f"b_median_{dn}2^{32 * i}", "bound_block": b - 1 if b > 0 else 0, "bound_share": b,
                                  "cap_block": T, "cap_share": T})
    return cases


def main() -> int:
    rng = np.random.default_rng(20261002)
    jobs = [  # (name, m, n, k, operand kind, boundary vectors)
        ("j1_256x256x4096", 256, 256, 4096, "uniform", True),
        ("j2_128x64x65536", 128, 64, 65536, "uniform", True),
        ("j3_256x256x2048", 256, 256, 2048, "uniform", False),
        ("j4_384x192x4096", 384, 192, 4096, "uniform", False),
        ("j5_128x64x65536_pm127", 128, 64, 65536, "pm127", True),
    ]
    total_cases = fails = slots_checked = 0
    for name, m, n, k, kind, boundary in jobs:
        if kind == "uniform":
            Ap = rng.integers(-127, 128, (m, k), dtype=np.int64)
            Bp = rng.integers(-127, 128, (k, n), dtype=np.int64)
        else:
            Ap = rng.choice(np.array([-127, 127]), (m, k))
            Bp = rng.choice(np.array([-127, 127]), (k, n))
        key = rng.bytes(32)
        t0 = time.time()
        tiles = oracle.all_tiles(Ap, Bp, key)
        t_or = time.time() - t0
        assert len(tiles) == m * n // 64
        assert len({(t["t_rows"], t["t_cols"]) for t in tiles}) == len(tiles)
        assert len({t["hv"] for t in tiles}) == len(tiles), "hash tie (would make boundary tests ambiguous)"
        d = f"{WORK}/{name}"
        cases = job_cases(tiles, boundary)
        harness.write_job(d, Ap, Bp, key, cases)
        gpu_log = harness.run_gpu(d)
        print(f"job {name}: m={m} n={n} k={k} operands={kind} tiles={len(tiles)} key={key.hex()[:16]}... (oracle {t_or:.1f} s)")
        for line in gpu_log.strip().splitlines()[1:]:
            print("   gpu" + line)
        for c in cases:
            ok, summ = harness.check_case(d, c, tiles)
            cb, cs, _, _ = harness.read_out(d, c)
            slots_checked += min(cb, c["cap_block"]) + min(cs, c["cap_share"])
            total_cases += 1
            fails += 0 if ok else 1
            print(f"   {'PASS' if ok else 'FAIL'} {c['name']:<26} {summ}")
        # (a) explicit: the a_all share array must hold every tile exactly once
        cb, cs, blk, shr = harness.read_out(d, cases[0])
        got = {(int(s[0]), int(s[1])) for s in shr[:cs]}
        full = cs == len(tiles) and got == {(t["t_rows"], t["t_cols"]) for t in tiles}
        print(f"   (a) all {len(tiles)} transcripts+hashes present and bit-exact: {'YES' if full else 'NO'}")
        fails += 0 if full else 1
        # (c) deterministic non-saturated: both runs found the same non-trivial sets
        r1 = harness.read_out(d, cases[1]); r2 = harness.read_out(d, cases[2])
        s1 = {(int(s[0]), int(s[1])) for s in r1[3][:r1[1]]}; s2 = {(int(s[0]), int(s[1])) for s in r2[3][:r2[1]]}
        det = s1 == s2 and 0 < len(s1) < len(tiles) and r1[0] == r2[0]
        print(f"   (c) wins={len(s1)} losses={len(tiles) - len(s1)} (share), block wins={r1[0]}; run1 == run2: {'YES' if det else 'NO'}")
        fails += 0 if det else 1
        if boundary:
            # (b) explicit classification for the min-hash tile
            res = {c["name"]: harness.read_out(d, c) for c in cases if c["name"].startswith("b_min_")}
            bmin = (res["b_min_m1"][1], res["b_min_eq"][1], res["b_min_p1"][1])
            okb = bmin == (0, 1, 1)
            print(f"   (b) min-hash tile share finds at bound hash-1/hash/hash+1 = {bmin} (expect (0, 1, 1)): {'YES' if okb else 'NO'}")
            fails += 0 if okb else 1
    print(f"\nTOTAL: {total_cases} cases, {slots_checked} slots compared bit-exact (transcript[16] + hash[8] + t_rows/t_cols), "
          f"{fails} failures")
    print("F1 CORRECTNESS: " + ("BIT-EXACT (all checks passed)" if fails == 0 else "FAILED"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
