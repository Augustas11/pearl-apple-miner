"""K3-SG oracle glue: bench/f1_k3/oracle.py (CPU int64 + blake3 port of zk-pow jackpot) with the K3-SG pattern.

Pattern = the per-lane element set of a 32x32 fp32 simdgroup_matrix tile (verified by `k3sg probe`):
rows_pattern [0,8,16,24] (h=4), cols_pattern [0,1,8,9,16,17,24,25] (w=8), h*w = 32, periods 32/32.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "f1_k3"))
import harness  # noqa: E402  (bench/f1_k3: job-dir I/O + slot comparison, pattern-agnostic)
import oracle  # noqa: E402

ROWS_PATTERN = [0, 8, 16, 24]
COLS_PATTERN = [0, 1, 8, 9, 16, 17, 24, 25]
H, W = len(ROWS_PATTERN), len(COLS_PATTERN)


def all_tiles(Ap: np.ndarray, Bp: np.ndarray, key32: bytes) -> list[dict]:
    """oracle.all_tiles with the SG pattern (Pearl tile order: row set major, col set minor)."""
    rs, cs, jp = oracle.transcripts(Ap, Bp, ROWS_PATTERN, COLS_PATTERN)
    out = []
    for i in range(len(rs)):
        for j in range(len(cs)):
            h = oracle.jackpot_hash(jp[i, j], key32)
            out.append({"t_rows": int(rs[i, 0]), "t_cols": int(cs[j, 0]), "rows": rs[i].tolist(), "cols": cs[j].tolist(),
                        "jp": [int(v) for v in jp[i, j]], "hash": h, "hv": oracle.u256_le(h)})
    return out


def write_tiles(d: str, tiles: list[dict]) -> None:
    """tiles.bin for the Swift-side check: per tile 26 LE u32 = t_rows, t_cols, jp[16], hash[8]."""
    arr = np.zeros((len(tiles), harness.SLOT_WORDS), dtype="<u4")
    for i, t in enumerate(tiles):
        arr[i, 0], arr[i, 1] = t["t_rows"], t["t_cols"]
        arr[i, 2:18] = t["jp"]
        arr[i, 18:26] = np.frombuffer(t["hash"], dtype="<u4")
    arr.tofile(f"{d}/tiles.bin")
