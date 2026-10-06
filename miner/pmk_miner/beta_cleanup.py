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
    body = json.dumps({"version": config["version"], "state": "uninstalled"}, separators=(",", ":"))
    def curl_quote(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")
    request_path.write_text("\n".join((
        f'url = "{curl_quote(str(config["api_base"]).rstrip("/") + "/api/hb")}"',
        'request = "POST"',
        'header = "Content-Type: application/json"',
        f'header = "Authorization: Bearer {curl_quote(config["miner_token"])}"',
        f'header = "X-Install-Id: {curl_quote(config["install_id"])}"',
        f'data = "{curl_quote(body)}"',
        'silent', 'show-error', 'fail', 'max-time = 10',
    )) + "\n", encoding="utf-8")
    os.chmod(request_path, 0o600)
    time.sleep(0.5)
    plist.unlink(missing_ok=True)
    if cli.is_symlink():
        cli.unlink(missing_ok=True)
    shutil.rmtree(app, ignore_errors=True)
    shutil.rmtree(logs, ignore_errors=True)
    for delay in (0, 1, 2, 4, 8, 16):
        if delay:
            time.sleep(delay)
        result = subprocess.run(["/usr/bin/curl", "--config", str(request_path)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                check=False)
        if result.returncode == 0:
            break
    request_path.unlink(missing_ok=True)
    # Bootout is deliberately last: launchd may terminate every process in the
    # job's coalition, including this detached finisher.
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
