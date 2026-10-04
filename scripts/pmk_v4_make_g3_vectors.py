#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Generate resumable B9 G3 references with pmkcore's pinned Pearl B200 oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

from pmk_v4_paths import bundle_root, verify_pearl_pin

ROOT = bundle_root(__file__)
PIN = "f696760b259500ecb608469ea3953aeabbe78948"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def adversarial(directory: Path, family: str, m: int, n: int, k: int, seed: int):
    """Test inputs only; all arithmetic/reference outputs still come from Pearl."""
    for side, rows in (("a", m), ("b", n)):
        rng = random.Random(f"B9/{family}/{seed}/{side}")
        data = bytearray(rows * k)
        for base in range(0, len(data), 32):
            mode = rng.randrange(5) if family == "adv_edge" else 0
            for t in range(32):
                if mode == 1:
                    choices = (0x00, 0x80, 0x00, 0x80, 0x01, 0x81)
                elif mode == 2:
                    choices = (0x7e, 0xfe, 0x01, 0x81, 0x08, 0x88)
                elif mode == 3:
                    choices = (0x01, 0x02, 0x07, 0x81, 0x82, 0x87)
                elif mode == 4:
                    choices = (0x78, 0xf8, 0x7e, 0xfe)
                else:
                    choices = None
                if choices:
                    code = rng.choice(choices)
                else:
                    code = rng.randrange(256)
                    while code & 0x7f == 0x7f:
                        code = rng.randrange(256)
                data[base + t] = code
        (directory / f"{side}.bin").write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=ROOT / "pmkcore/v4/target/release/pmkcore-v4-oracle")
    parser.add_argument("--output", type=Path, default=ROOT / "bench/v4_emulation/vectors/b9_g3")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--small", action="store_true", help="diagnostic subset; cannot earn G3 admission")
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("--threads must be 1..8")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    verify_pearl_pin(ROOT, PIN)
    oracle_hash = sha(args.oracle)
    cases = [("const", 256, 256, k, i) for i, k in enumerate((1024, 4096, 16384), 101)]
    cases += [("adv_uniform", 256, 256, 4096, 201), ("adv_edge", 256, 256, 4096, 202)]
    if not args.small:
        cases += [("const", 2048, 2048, 4096, 301)]
        cases += [("const", 4096, 4096, 4096, i) for i in range(401, 407)]
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
    manifest = {"upstream_pin": PIN, "oracle_executable": str(args.oracle), "oracle_sha256": oracle_hash, "cases": []}
    for family, m, n, k, seed in cases:
        name = f"{family}_{m}x{n}_k{k}_s{seed}"
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        stamp = directory / "provenance.json"
        expected = {"upstream_pin": PIN, "oracle_sha256": oracle_hash, "family": family, "m": m, "n": n, "k": k, "seed": seed}
        previous = json.loads(stamp.read_text()) if stamp.exists() else {}
        valid = all(previous.get(key) == value for key, value in expected.items())
        if valid:
            valid = all(
                (directory / f"{part}.bin").exists()
                and sha(directory / f"{part}.bin") == previous.get(f"{part}_sha256")
                for part in ("a", "b", "c_b200")
            )
        if not valid:
            start = time.monotonic()
            print(f"GENERATE {name}", flush=True)
            if family == "const":
                seed_hex = hashlib.sha256(f"B9/{seed}".encode()).hexdigest()
                subprocess.run([str(args.oracle), "gen", str(m), str(n), str(k), seed_hex, str(directory)], env=environment, check=True)
            else:
                adversarial(directory, family, m, n, k, seed)
            subprocess.run([str(args.oracle), "ref", str(directory), str(m), str(n), str(k)], env=environment, check=True)
            previous = dict(expected, directory=str(directory), generation_seconds=time.monotonic() - start)
            for part in ("a", "b", "c_b200"):
                previous[f"{part}_sha256"] = sha(directory / f"{part}.bin")
            stamp.write_text(json.dumps(previous, indent=2) + "\n")
        print(json.dumps({"reference": name, "cells": m * n, "seconds": previous.get("generation_seconds"), "cached": valid}), flush=True)
        manifest["cases"].append(previous)
        # A resumable partial manifest is deliberately unable to pass the G3 matrix.
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if sha(args.oracle) != oracle_hash:
        raise RuntimeError("oracle executable changed while references were generated")
    print(f"Reference manifest: {args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
