#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    manifest = json.loads(source.read_text(encoding="utf-8"))
    cases = []
    for case in manifest["cases"]:
        name = Path(case["directory"]).name
        cases.append({
            "name": name,
            "upstream_pin": case["upstream_pin"],
            "oracle_sha256": case["oracle_sha256"],
            "family": case["family"],
            "m": case["m"],
            "n": case["n"],
            "k": case["k"],
            "seed": case["seed"],
            "a_sha256": case["a_sha256"],
            "b_sha256": case["b_sha256"],
            "c_b200_sha256": case["c_b200_sha256"],
        })
    oracle_sha256 = manifest["oracle_sha256"]
    if len(sys.argv) > 3:
        oracle_sha256 = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
    slim = {
        "schema": "pmk-v4-g3-slim-v1",
        "upstream_pin": manifest["upstream_pin"],
        "oracle_sha256": oracle_sha256,
        "cases": cases,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(slim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
