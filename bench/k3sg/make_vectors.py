"""Pre-generate K3-SG oracle vectors for the Studio window (bench/k3sg/studio/vectors/<job>/).

Each job dir: job.json + A.bin + B.bin (harness.write_job format) + tiles.bin (all oracle tiles, 26 LE u32 each) so the
Studio needs only the k3sg binary (`k3sg run DIR cfgs=...` checks every case bit-exact in Swift; no Python/blake3 there).
Jobs:
  v1_256x256x4096        uniform [-127,127], cases (a)-(e) + boundary vectors hash-1/hash/hash+1, +-2^(32i)
  v2_256x256x4096_pm127  +-127 extreme magnitudes (max |chunk sum| = 2,064,512 < 2^24), same cases
  v3_128x128x65536       uniform, same cases (k = 2^16 = max legal k, 512 rank chunks, 32 slot wraps)
  v4_pearl_c64           the pearl_mining cross-check job (signal c=64, real job key / salted v3 seeds / noise,
                         Pearl's bound from nbits 0x1E100000): block array must hold exactly Pearl's winning tiles
"""
from __future__ import annotations

import os
import sys

import numpy as np

import crosscheck_sg
import sg_oracle
from sg_oracle import harness, oracle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "f1_k3"))
from correct import job_cases  # noqa: E402

OUT = os.environ.get("K3SG_VECTORS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio", "vectors"))


def emit(name: str, Ap, Bp, key: bytes, cases: list[dict], tiles: list[dict]) -> None:
    d = f"{OUT}/{name}"
    harness.write_job(d, Ap, Bp, key, cases)
    sg_oracle.write_tiles(d, tiles)
    print(f"{name}: m={Ap.shape[0]} n={Bp.shape[1]} k={Ap.shape[1]} tiles={len(tiles)} cases={len(cases)} key={key.hex()[:16]}...")


def main() -> int:
    rng = np.random.default_rng(20261004)
    for name, m, n, k, kind in (("v1_256x256x4096", 256, 256, 4096, "uniform"),
                                ("v2_256x256x4096_pm127", 256, 256, 4096, "pm127"),
                                ("v3_128x128x65536", 128, 128, 65536, "uniform")):
        if kind == "uniform":
            Ap = rng.integers(-127, 128, (m, k), dtype=np.int64)
            Bp = rng.integers(-127, 128, (k, n), dtype=np.int64)
        else:
            Ap = rng.choice(np.array([-127, 127]), (m, k))
            Bp = rng.choice(np.array([-127, 127]), (k, n))
        key = rng.bytes(32)
        tiles = sg_oracle.all_tiles(Ap, Bp, key)
        assert len({t["hv"] for t in tiles}) == len(tiles), "hash tie"
        emit(name, Ap, Bp, key, job_cases(tiles, True), tiles)
    _, _, _, a_seed, Ap, Bp, tiles, bound = crosscheck_sg.oracle_job(64, 0x1E100000)
    win = [(t["t_rows"], t["t_cols"]) for t in tiles if t["hv"] <= bound]
    print(f"  pearl job: Pearl's first winning tile (threads_partition order) = {win[0]}, {len(win)} winners")
    emit("v4_pearl_c64", Ap, Bp, a_seed,
         [{"name": "pearl_bound", "bound_block": bound, "bound_share": oracle.U256_MAX, "cap_block": 128, "cap_share": len(tiles)}], tiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
