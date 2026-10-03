"""K3-SG correctness: production kernel (`k3sg run`, variant 3) vs the CPU oracle, bit-exact, for every tile config.

Jobs (operands full-range uniform in [-127,127] unless noted):
  256x256x4096 (+ boundary vectors), 128x128x65536 (+ boundary vectors), 256x256x2048, 384x256x4096,
  128x128x65536 with +-127 extreme magnitudes (max |chunk sum| = 128*127^2 = 2,064,512 < 2^24) (+ boundary vectors).
Cases per job (bench/f1_k3/correct.py job_cases): (a) all transcripts+hashes, (b) bound = hash-1/hash/hash+1 and
+-2^(32i), (c) deterministic wins/losses run twice, (d) slot overflow at capacities 4/64, (e) simultaneous block+share.
Every case is checked twice: in Python (bench/f1_k3/harness.check_case) and in Swift against tiles.bin.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

import sg_oracle
from sg_oracle import harness

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "f1_k3"))
from correct import job_cases  # noqa: E402

BIN = os.environ.get("K3SG_BIN", "/tmp/k3sg")
WORK = os.environ.get("K3SG_WORK", "/tmp/k3sg_jobs")
CFGS = os.environ.get("K3SG_CFGS", "64x64x16x2x2x1,64x64x16x2x2x0,64x64x16x2x2x2,64x64x32x2x2x1,64x64x8x2x2x2,128x64x16x4x2x1,128x64x16x4x2x2,64x128x16x2x4x1,128x128x16x4x4x1").split(",")


def main() -> int:
    rng = np.random.default_rng(20261003)
    jobs = [
        ("j1_256x256x4096", 256, 256, 4096, "uniform", True),
        ("j2_128x128x65536", 128, 128, 65536, "uniform", True),
        ("j3_256x256x2048", 256, 256, 2048, "uniform", False),
        ("j4_384x256x4096", 384, 256, 4096, "uniform", False),
        ("j5_128x128x65536_pm127", 128, 128, 65536, "pm127", True),
    ]
    print(f"K3-SG correctness: bin={BIN} cfgs={CFGS}")
    print(f"pattern rows={sg_oracle.ROWS_PATTERN} cols={sg_oracle.COLS_PATTERN}")
    total = fails = slots = 0
    for name, m, n, k, kind, boundary in jobs:
        if kind == "uniform":
            Ap = rng.integers(-127, 128, (m, k), dtype=np.int64)
            Bp = rng.integers(-127, 128, (k, n), dtype=np.int64)
        else:
            Ap = rng.choice(np.array([-127, 127]), (m, k))
            Bp = rng.choice(np.array([-127, 127]), (k, n))
        key = rng.bytes(32)
        t0 = time.time()
        tiles = sg_oracle.all_tiles(Ap, Bp, key)
        t_or = time.time() - t0
        assert len(tiles) == m * n // 32
        assert len({(t["t_rows"], t["t_cols"]) for t in tiles}) == len(tiles)
        assert len({t["hv"] for t in tiles}) == len(tiles), "hash tie"
        d = f"{WORK}/{name}"
        cases = job_cases(tiles, boundary)
        harness.write_job(d, Ap, Bp, key, cases)
        sg_oracle.write_tiles(d, tiles)
        p = subprocess.run([BIN, "run", d, "cfgs=" + ",".join(CFGS)], capture_output=True, text=True)
        print(f"job {name}: m={m} n={n} k={k} operands={kind} tiles={len(tiles)} key={key.hex()[:16]}... (oracle {t_or:.1f} s)")
        swift_fail_lines = [l for l in p.stdout.splitlines() if " FAIL " in l or l.strip().startswith("JOB") and "FAIL" in l]
        swift_ok = p.returncode == 0 and not swift_fail_lines
        npass = sum(1 for l in p.stdout.splitlines() if "  PASS " in l)
        print(f"   swift-side check: {npass} case-runs PASS, rc={p.returncode} -> {'PASS' if swift_ok else 'FAIL'}")
        if not swift_ok:
            print(p.stdout[-4000:], p.stderr[-2000:])
            fails += 1
        for ci, cfg in enumerate(CFGS):
            suffix = "" if ci == 0 else f"__{cfg}"
            cfails = 0
            for c in cases:
                cc = dict(c, name=c["name"] + suffix)
                ok, summ = harness.check_case(d, cc, tiles)
                cb, cs, _, _ = harness.read_out(d, cc)
                slots += min(cb, c["cap_block"]) + min(cs, c["cap_share"])
                total += 1
                if not ok:
                    cfails += 1
                    print(f"   FAIL cfg {cfg} {c['name']}: {summ}")
                elif ci == 0:
                    print(f"   PASS {c['name']:<26} {summ}")
            # (a) every tile exactly once; (c) deterministic; (b) min-tile classification
            cb, cs, blk, shr = harness.read_out(d, dict(cases[0], name=cases[0]["name"] + suffix))
            full = cs == len(tiles) and {(int(s[0]), int(s[1])) for s in shr[:cs]} == {(t["t_rows"], t["t_cols"]) for t in tiles}
            r1 = harness.read_out(d, dict(cases[1], name=cases[1]["name"] + suffix))
            r2 = harness.read_out(d, dict(cases[2], name=cases[2]["name"] + suffix))
            s1 = {(int(s[0]), int(s[1])) for s in r1[3][:r1[1]]}
            s2 = {(int(s[0]), int(s[1])) for s in r2[3][:r2[1]]}
            det = s1 == s2 and 0 < len(s1) < len(tiles) and r1[0] == r2[0]
            okb = True
            if boundary:
                res = {c["name"]: harness.read_out(d, dict(c, name=c["name"] + suffix)) for c in cases if c["name"].startswith("b_min_")}
                okb = (res["b_min_m1"][1], res["b_min_eq"][1], res["b_min_p1"][1]) == (0, 1, 1)
            cfails += (0 if full else 1) + (0 if det else 1) + (0 if okb else 1)
            print(f"   cfg {cfg:<16} cases {len(cases)}: {'PASS' if cfails == 0 else f'FAIL ({cfails})'}"
                  f"  (a) all {len(tiles)} tiles once: {'YES' if full else 'NO'}  (c) wins={len(s1)} run1==run2: {'YES' if det else 'NO'}"
                  + (f"  (b) min tile hash-1/hash/hash+1 = (0,1,1): {'YES' if okb else 'NO'}" if boundary else ""))
            fails += cfails
    print(f"\nTOTAL: {total} case-runs ({len(CFGS)} cfgs), {slots} slots compared bit-exact (Python), {fails} failures")
    print("K3-SG CORRECTNESS: " + ("BIT-EXACT (all checks passed)" if fails == 0 else "FAILED"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
