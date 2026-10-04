#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Generate the bundled v4 startup known-answer fixture using Pearl's core."""
from __future__ import annotations

import base64
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pmk_v4_paths import bundle_root, verify_pearl_pin

ROOT = bundle_root(__file__)
sys.path.insert(0, str(ROOT / "miner"))
from pmk_miner.native import V4GpuJobDesc, V4TileResult  # noqa: E402

PIN = "f696760b259500ecb608469ea3953aeabbe78948"
OUT = ROOT / "libpmk/resources/v4_probe"
LIB = ROOT / "pmkcore/v4/target/release/libpmkcore_v4.dylib"
ORACLE = ROOT / "pmkcore/v4/target/release/pmkcore-v4-oracle"
P, U32, U64 = C.c_void_p, C.c_uint32, C.c_uint64


def main():
    verify_pearl_pin(ROOT, PIN)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "bench/v4_emulation/vectors/b9_g3/const_256x256_k4096_s102/metadata.json").read_text())
    header = C.create_string_buffer(base64.b64decode(meta["proposed_header_b64"]))
    ancestor = C.create_string_buffer(base64.b64decode(meta["ancestor_header_b64"]))
    core = C.CDLL(str(LIB))
    signatures = {
        "pmkcore_v4_init": [U32],
        "pmkcore_v4_job_create_grid_b200": [P, P, P, U64, U32, U32, U32, C.POINTER(P)],
        "pmkcore_v4_prepare_oracle_noised": [P],
        "pmkcore_v4_gpu_descriptor": [P, C.POINTER(V4GpuJobDesc)],
        "pmkcore_v4_tile_cpu_oracle": [P, U32, U32, P, P, C.POINTER(V4TileResult)],
        "pmkcore_v4_build_plain_proof": [P, U32, U32, P, U64, C.POINTER(U64)],
        "pmkcore_v4_verify_plain_proof": [P, P, U64, P, C.POINTER(C.c_uint8)],
    }
    for name, args in signatures.items():
        getattr(core, name).argtypes = args
        getattr(core, name).restype = C.c_int32
    core.pmkcore_v4_job_free.argtypes = [P]
    core.pmkcore_v4_job_free.restype = None

    def check(rc):
        if rc != 0:
            raise RuntimeError(f"core v4 returned {rc}")

    check(core.pmkcore_v4_init(2))
    job = P()
    m = n = 64
    k = 4096
    check(core.pmkcore_v4_job_create_grid_b200(header, ancestor, None, 0, m, n, k, C.byref(job)))
    try:
        check(core.pmkcore_v4_prepare_oracle_noised(job))
        desc = V4GpuJobDesc()
        check(core.pmkcore_v4_gpu_descriptor(job, C.byref(desc)))
        for field, name, count in (
            ("a_values", "a_clean.bin", m*k), ("bt_values", "b_clean.bin", n*k),
            ("a_noised", "a.bin", m*k), ("bt_noised", "b.bin", n*k),
            ("a_noise_e", "a_noise_e.bin", m*32), ("bt_noise_e", "b_noise_e.bin", n*32),
            ("a_noise_f", "a_noise_f.bin", k*32), ("bt_noise_f", "b_noise_f.bin", k*32),
            ("a_alpha", "a_alpha.bf16", m*2), ("bt_alpha", "b_alpha.bf16", n*2),
            ("a_beta", "a_beta.bf16", m*2), ("bt_beta", "b_beta.bf16", n*2),
        ):
            (OUT / name).write_bytes(C.string_at(getattr(desc, field), count))
        bound = C.create_string_buffer(b"\xff"*32)
        tiles = []
        for row in range(0, m, 16):
            for col in range(0, n, 16):
                tile = V4TileResult()
                check(core.pmkcore_v4_tile_cpu_oracle(job, row, col, bound, bound, C.byref(tile)))
                assert tile.policy_pass == 1
                tiles.append({"row": row, "col": col, "message": bytes(tile.message).hex(), "hash": bytes(tile.hash).hex()})
        block_limit = sorted(int.from_bytes(bytes.fromhex(t["hash"]), "little") for t in tiles)[7]
        # A block candidate can also satisfy the independent share predicate.
        slots = []
        for tile in tiles:
            if int.from_bytes(bytes.fromhex(tile["hash"]), "little") <= block_limit:
                slots.append(dict(tile, kind="block"))
            slots.append(dict(tile, kind="share"))
        needed = U64()
        rc = core.pmkcore_v4_build_plain_proof(job, 0, 0, None, 0, C.byref(needed))
        check(rc)
        assert needed.value > 0
        proof = C.create_string_buffer(needed.value)
        check(core.pmkcore_v4_build_plain_proof(job, 0, 0, proof, len(proof), C.byref(needed)))
        accepted = C.c_uint8()
        check(core.pmkcore_v4_verify_plain_proof(header, proof, needed.value, None, C.byref(accepted)))
        assert accepted.value == 1
        (OUT / "plain_proof.bin").write_bytes(proof.raw[:needed.value])
        subprocess.run([str(ORACLE), "ref", str(OUT), str(m), str(n), str(k)], env=dict(os.environ, RAYON_NUM_THREADS="2"), check=True)
        manifest = {
            "schema": "pmk-v4-startup-probe-v1", "upstream_pin": PIN,
            "m": m, "n": n, "k": k, "rank": 32,
            "proposed_header": header.raw[:76].hex(), "ancestor_header": ancestor.raw[:108].hex(),
            "jackpot_key": bytes(desc.jackpot_key).hex(),
            "block_bound": block_limit.to_bytes(32, "little").hex(), "share_bound": (b"\xff"*32).hex(),
            "blocks_expected": 8, "shares_expected": 16,
            "tiles": slots, "plain_proofs_accepted": 1,
            "core_sha256": hashlib.sha256(LIB.read_bytes()).hexdigest(),
            "oracle_sha256": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"},
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"startup fixture: {m*n} exact C cells, {len(tiles)} upstream policy/fold/hash tiles, one accepted plain proof; {OUT}")
    finally:
        core.pmkcore_v4_job_free(job)


if __name__ == "__main__":
    main()
