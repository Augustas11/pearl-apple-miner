#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import urllib.error
import urllib.request


def label() -> str:
    try:
        return subprocess.check_output(["scutil", "--get", "ComputerName"], text=True, timeout=3).strip()
    except Exception:
        return platform.node() or "Mac"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    body = {
        "version": "0.2.0-beta.17",
        "state": args.state,
        "uptime_s": 0,
        "label": label(),
        "last_error": "installer correctness check failed" if args.state == "error" else None,
    }
    request = urllib.request.Request(
        str(config["api_base"]).rstrip("/") + "/api/hb",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {config['miner_token']}",
            "X-Install-Id": str(config["install_id"]),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except (OSError, urllib.error.URLError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
