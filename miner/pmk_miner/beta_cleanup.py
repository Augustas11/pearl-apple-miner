# SPDX-License-Identifier: Apache-2.0
"""Detached uninstall finisher; loaded before it deletes the release tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

LABEL = "tech.malibu.pearl"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request_path = args.request
    config = json.loads(request_path.read_text(encoding="utf-8"))
    home = Path(config["home"])
    app = home / "Library/Application Support/MalibuPearl"
    logs = home / "Library/Logs/MalibuPearl"
    plist = home / "Library/LaunchAgents/tech.malibu.pearl.plist"
    cli = home / ".local/bin/pearl-miner"
    time.sleep(0.5)
    plist.unlink(missing_ok=True)
    if cli.is_symlink():
        cli.unlink(missing_ok=True)
    shutil.rmtree(app, ignore_errors=True)
    shutil.rmtree(logs, ignore_errors=True)
    request_path.unlink(missing_ok=True)
    # Bootout is deliberately last: launchd may terminate every process in the
    # job's coalition, including this detached finisher.
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
