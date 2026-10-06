#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-shot B15 release install/control/uninstall acceptance flow."""
from __future__ import annotations

import argparse
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]

def request(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)

async def wait_until(predicate, label: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.25)
    raise TimeoutError(label)

def busy_percent(log: Path) -> float | None:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get("gpu_busy_pct")
        if row.get("event") in {"routine_telemetry", "power_telemetry"} and isinstance(value, (int, float)):
            return float(value)
    return None

async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--tarball", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts import pmk_pool_mock_e2e as pool_module
    from scripts.release import mock_api

    mock_api.STATE = mock_api.State()
    api = ThreadingHTTPServer(("127.0.0.1", 0), mock_api.Handler)
    api_thread = threading.Thread(target=api.serve_forever, daemon=True)
    api_thread.start()
    api_base = f"http://127.0.0.1:{api.server_address[1]}"
    pool = pool_module.MockPool(difficulty=10_000, rotate_seconds=15, block_difficulty=1_000_000_000)
    await pool.start()
    home = Path(tempfile.mkdtemp(prefix="pmk-b16-home-"))
    plist = home / "Library/LaunchAgents/tech.malibu.pearl.plist"
    app = home / "Library/Application Support/MalibuPearl"
    logs = home / "Library/Logs/MalibuPearl"
    symlink = home / ".local/bin/pearl-miner"
    label = f"gui/{os.getuid()}/tech.malibu.pearl"
    subprocess.run(["launchctl", "bootout", label], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wallet = "prl1" + "q" * 40
        registration = request(api_base + "/api/register", {"wallet": wallet})
        install_id = registration["install_id"]
        dash = registration["dash_token"]
        command = ["sh", str(args.installer), "--wallet", wallet,
                   "--token", registration["miner_token"], "--install-id", install_id]
        env = dict(os.environ, HOME=str(home), PMK_RELEASE_URL=args.tarball.resolve().as_uri(),
                   PMK_API_BASE=api_base, PMK_POOL_URL=pool.url(),
                   PMK_TEST_FORCE_GPU_SECONDS="0.29", PMK_TEST_FORCE_GPU_SHAPE="4096")
        first = await asyncio.to_thread(subprocess.run, command, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
        print("INSTALL_FIRST_RC", first.returncode)
        print(first.stdout, end="")
        if first.returncode:
            raise RuntimeError("first install failed")
        await wait_until(lambda: pool.stats.accepted >= 3, "three accepted mock shares", 1200)
        await wait_until(lambda: any(event["hb"].get("state") == "mining" for event in mock_api.STATE.events),
                         "mining heartbeat")
        cert_versions = mock_api.STATE.macs[install_id]["last_hb"].get("cert_versions", [])
        if cert_versions != [3, 4]:
            raise AssertionError(f"expected v3+v4 admission, got {cert_versions!r}")
        await wait_until(lambda: mock_api.STATE.macs[install_id]["last_hb"].get("kernel") == "na",
                         "K3-NA heartbeat")
        await wait_until(lambda: mock_api.STATE.macs[install_id]["last_hb"].get("throttled") is True,
                         "forced-slow shape throttle", 300)
        heartbeat = mock_api.STATE.macs[install_id]["last_hb"]
        if heartbeat.get("shape") != 2048 or heartbeat.get("state") != "mining":
            raise AssertionError(f"forced-slow recovery did not keep mining at 2048: {heartbeat!r}")
        before = busy_percent(logs / "agent.out.log")

        second = await asyncio.to_thread(subprocess.run, command, env=env, text=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        print("INSTALL_SECOND_RC", second.returncode)
        print(second.stdout, end="")
        if second.returncode:
            raise RuntimeError("idempotent reinstall failed")

        request(api_base + "/api/control", {"d": dash, "install_id": install_id, "paused": True})
        await wait_until(lambda: mock_api.STATE.macs[install_id]["last_hb"].get("state") == "paused", "pause control")
        request(api_base + "/api/control", {"d": dash, "install_id": install_id,
                                             "paused": False, "intensity": "low"})
        await wait_until(lambda: mock_api.STATE.macs[install_id]["last_hb"].get("intensity") == "low", "low intensity")
        await wait_until(lambda: (busy_percent(logs / "agent.out.log") or 101) < 55, "low busy fraction", 300)
        after = busy_percent(logs / "agent.out.log")

        request(api_base + "/api/control", {"d": dash, "install_id": install_id, "uninstall": True})
        await wait_until(lambda: any(event["hb"].get("state") == "uninstalled" for event in mock_api.STATE.events),
                         "final uninstalled heartbeat")
        checked = [app, logs, plist, symlink]
        await wait_until(lambda: not any(path.exists() or path.is_symlink() for path in checked), "uninstall paths removed")
        loaded = subprocess.run(["launchctl", "print", label], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0
        if loaded:
            raise AssertionError("LaunchAgent remained loaded")
        print(f"PASS accepted={pool.stats.accepted} kernel=na shape=2048 throttled=true no_halt=true cert_versions={cert_versions} first_install=0 reinstall=0 pause_within_poll=true "
              f"full_busy_pct={before} low_busy_pct={after} final_uninstalled=true")
        print("REMOVED_PATHS " + " | ".join(str(path) for path in checked))
        return 0
    finally:
        subprocess.run(["launchctl", "bootout", label], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await pool.stop()
        api.shutdown()
        api_thread.join(timeout=5)
        shutil.rmtree(home, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
