"""Job-dir I/O between the CPU oracle and the K3 GPU host (k3 run JOBDIR)."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import oracle

SLOT_WORDS = 26
GUARD_SLOTS = 8
CANARY = 0xA5A5A5A5
K3_BIN = os.environ.get("K3_BIN", "/tmp/k3")


def write_job(d: str, Ap: np.ndarray, Bp: np.ndarray, key32: bytes, cases: list[dict]) -> None:
    """Ap: m x k, Bp: k x n (row-major int8 in [-127,127]); cases: name, bound_block, bound_share (ints), caps."""
    os.makedirs(d, exist_ok=True)
    assert Ap.min() >= -127 and Ap.max() <= 127 and Bp.min() >= -127 and Bp.max() <= 127
    m, k = Ap.shape
    n = Bp.shape[1]
    np.ascontiguousarray(Ap, dtype=np.int8).tofile(f"{d}/A.bin")
    np.ascontiguousarray(Bp, dtype=np.int8).tofile(f"{d}/B.bin")
    js = {"m": m, "n": n, "k": k, "key": list(np.frombuffer(key32, dtype="<u4").tolist()),
          "cases": [{"name": c["name"], "cap_block": c["cap_block"], "cap_share": c["cap_share"],
                     "bound_block": oracle.words_le(c["bound_block"]), "bound_share": oracle.words_le(c["bound_share"])}
                    for c in cases]}
    with open(f"{d}/job.json", "w") as f:
        json.dump(js, f)


def run_gpu(d: str) -> str:
    p = subprocess.run([K3_BIN, "run", d], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"k3 run failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    return p.stdout


def read_out(d: str, case: dict):
    raw = np.fromfile(f"{d}/out_{case['name']}.bin", dtype="<u4")
    nb = (case["cap_block"] + GUARD_SLOTS) * SLOT_WORDS
    ns = (case["cap_share"] + GUARD_SLOTS) * SLOT_WORDS
    assert raw.size == 2 + nb + ns, (raw.size, 2 + nb + ns)
    ctr = raw[:2]
    blk = raw[2:2 + nb].reshape(-1, SLOT_WORDS)
    shr = raw[2 + nb:].reshape(-1, SLOT_WORDS)
    return int(ctr[0]), int(ctr[1]), blk, shr


def check_case(d: str, case: dict, tiles: list[dict]) -> tuple[bool, str]:
    """Compare one GPU case against the oracle tiles. Every written slot must equal the oracle tile at (t_rows, t_cols)
    bit for bit (transcript + hash); the slot set must equal the oracle find set when it fits; counters must equal the
    oracle find counts; slots beyond capacity (guards) and unwritten slots must hold the canary."""
    by_xy = {(t["t_rows"], t["t_cols"]): t for t in tiles}
    exp_b = {(t["t_rows"], t["t_cols"]) for t in tiles if t["hv"] <= case["bound_block"]}
    exp_s = {(t["t_rows"], t["t_cols"]) for t in tiles if t["hv"] <= case["bound_share"]}
    cb, cs, blk, shr = read_out(d, case)
    errs: list[str] = []
    for label, ctr, cap, arr, exp in (("block", cb, case["cap_block"], blk, exp_b), ("share", cs, case["cap_share"], shr, exp_s)):
        if ctr != len(exp):
            errs.append(f"{label} counter {ctr} != oracle {len(exp)}")
        written = min(ctr, cap)
        seen = set()
        for i in range(written):
            s = arr[i]
            xy = (int(s[0]), int(s[1]))
            t = by_xy.get(xy)
            if t is None:
                errs.append(f"{label} slot {i}: (t_rows,t_cols)={xy} is not a Pearl tile origin")
                continue
            if xy in seen:
                errs.append(f"{label} slot {i}: duplicate tile {xy}")
            seen.add(xy)
            if [int(v) for v in s[2:18]] != t["jp"]:
                errs.append(f"{label} slot {i} tile {xy}: transcript mismatch")
            if np.asarray(s[18:26], dtype="<u4").tobytes() != t["hash"]:
                errs.append(f"{label} slot {i} tile {xy}: hash mismatch")
            if xy not in exp:
                errs.append(f"{label} slot {i} tile {xy}: not a find per oracle (hash > bound)")
        if ctr <= cap and seen != exp:
            errs.append(f"{label} slot set != oracle find set ({len(seen)} vs {len(exp)})")
        if not np.all(arr[written:] == CANARY):
            errs.append(f"{label}: write outside the {written} valid slots (capacity {cap}, guard {GUARD_SLOTS})")
    summary = f"block ctr={cb} (oracle {len(exp_b)}, cap {case['cap_block']}), share ctr={cs} (oracle {len(exp_s)}, cap {case['cap_share']})"
    return (not errs), summary + ("" if not errs else " ERRORS: " + "; ".join(errs[:8]) + (f" (+{len(errs) - 8} more)" if len(errs) > 8 else ""))
