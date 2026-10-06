#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-shot B17 release install/local-control/uninstall acceptance flow."""
from __future__ import annotations

import argparse
import asyncio
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]


async def wait_until(predicate, label: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
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


def local_request(base: str, path: str, *, body: dict | None = None,
                  headers: dict[str, str] | None = None,
                  host: str | None = None) -> tuple[int, dict[str, str], bytes]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request_headers = dict(headers or {})
    request_headers.setdefault("Accept", "application/json")
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    if host is not None:
        request_headers["Host"] = host
    req = urllib.request.Request(base + path, data=data,
                                 method="GET" if data is None else "POST", headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def local_json(base: str, path: str, **kwargs) -> dict:
    status, _headers, body = local_request(base, path, **kwargs)
    if status != 200:
        raise ValueError(f"local request returned HTTP {status}")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("local request returned non-object JSON")
    return value


def status_document(base: str) -> dict:
    return local_json(base, "api/status")


def status_payload(base: str) -> dict:
    value = status_document(base).get("status", {})
    return value if isinstance(value, dict) else {}


def port_is_closed(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return False
    except OSError:
        return True


class PageInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.network_resources: list[str] = []
        self._script: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if tag == "script":
            if values.get("src"):
                self.network_resources.append(values["src"])
            self._script = []
        elif tag in {"img", "iframe", "audio", "video", "source"} and values.get("src"):
            self.network_resources.append(values["src"])
        elif tag == "link" and values.get("href"):
            self.network_resources.append(values["href"])

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None


def check_page(html: str) -> None:
    inspector = PageInspector()
    inspector.feed(html)
    if not inspector.scripts or not any("api/status" in script and "api/control" in script for script in inspector.scripts):
        raise AssertionError("local page did not serve its control JavaScript")
    if "Malibu Pearl <span>&middot; This Mac</span>" not in html or "Open on HeroMiners" not in html:
        raise AssertionError("local page content is incomplete")
    urls = inspector.network_resources + re.findall(r"url\([\"']?(https?://[^\"')]+)", html)
    for script in inspector.scripts:
        urls.extend(re.findall(r"fetch\([\"'](https?://[^\"']+)", script))
        urls.extend(re.findall(r"https?://[^\"']+", script))
    allowed = {"fonts.googleapis.com", "fonts.gstatic.com", "pearl.herominers.com"}
    unexpected = sorted({urlparse(url).hostname for url in urls if urlparse(url).hostname not in allowed})
    if unexpected:
        raise AssertionError(f"unexpected page resource hosts: {unexpected}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--tarball", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts import pmk_pool_mock_e2e as pool_module

    pool = pool_module.MockPool(difficulty=10_000, rotate_seconds=15, block_difficulty=1_000_000_000)
    await pool.start()
    home = Path(tempfile.mkdtemp(prefix="pmk-b17-home-"))
    plist = home / "Library/LaunchAgents/tech.malibu.pearl.plist"
    app = home / "Library/Application Support/MalibuPearl"
    logs = home / "Library/Logs/MalibuPearl"
    config = app / "config.json"
    state = app / "state.json"
    symlink = home / ".local/bin/pearl-miner"
    fallback_cli = app / "current/bin/pearl-miner"
    label = f"gui/{os.getuid()}/tech.malibu.pearl"
    subprocess.run(["launchctl", "bootout", label], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port: int | None = None
    try:
        wallet = "prl1" + "q" * 40
        command = ["sh", str(args.installer), "--wallet", wallet]
        env = dict(os.environ, HOME=str(home), PMK_RELEASE_URL=args.tarball.resolve().as_uri(),
                   PMK_POOL_URL=pool.url(), PMK_TEST_FORCE_GPU_SECONDS="0.29",
                   PMK_TEST_FORCE_GPU_SHAPE="4096")
        installed = await asyncio.to_thread(
            subprocess.run, command, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=3600)
        print("INSTALL_RC", installed.returncode)
        print(installed.stdout, end="")
        if installed.returncode:
            raise RuntimeError("install failed")
        for expected in (
            "Done. Your Mac is mining Pearl.",
            "Your control page just opened in your browser. Bookmark it.",
            "To open it again later, run: pearl-miner open",
        ):
            if expected not in installed.stdout:
                raise AssertionError(f"installer omitted: {expected}")
        fallback_line = f"Pearl miner command: {fallback_cli}"
        if installed.stdout.count(fallback_line) != 1:
            raise AssertionError("fallback CLI path was not printed exactly once")

        cfg = json.loads(config.read_text(encoding="utf-8"))
        initial = json.loads(state.read_text(encoding="utf-8"))
        secret = cfg["local_secret"]
        port = initial["local_port"]
        if not re.fullmatch(r"[0-9a-f]{64}", secret) or not 47811 <= port <= 47820:
            raise AssertionError("invalid local endpoint configuration")
        if stat.S_IMODE(config.stat().st_mode) != 0o600 or stat.S_IMODE(state.stat().st_mode) != 0o600:
            raise AssertionError("local config/state permissions are not 0600")
        base = f"http://127.0.0.1:{port}/{secret}/"

        first_status = status_payload(base)
        if first_status.get("state") != "checking":
            raise AssertionError(f"expected Starting up/checking first, got {first_status.get('state')!r}")
        page_status, page_headers, page_body = local_request(base, "")
        if page_status != 200 or "Access-Control-Allow-Origin" in page_headers:
            raise AssertionError("local page response headers are unsafe")
        check_page(page_body.decode("utf-8"))

        secret_path = f"/{secret}/"
        wrong_host = local_request(f"http://127.0.0.1:{port}", secret_path, host="evil.example")[0]
        no_secret = local_request(f"http://127.0.0.1:{port}", "/api/status")[0]
        no_header = local_request(base, "api/control", body={"paused": True})[0]
        foreign_origin = local_request(
            base, "api/control", body={"paused": True},
            headers={"X-Pearl-Local": secret, "Origin": "https://evil.example"})[0]
        if (wrong_host, no_secret, no_header, foreign_origin) != (403, 404, 403, 403):
            raise AssertionError("local HTTP security gates failed")

        await wait_until(lambda: pool.stats.accepted >= 3, "three accepted mock shares", 1200)
        await wait_until(lambda: status_payload(base).get("state") == "mining" and
                         float(status_payload(base).get("tops") or 0) > 0,
                         "Mining with measured TOPS", 1200)
        await wait_until(lambda: status_payload(base).get("kernel") == "na", "K3-NA status")
        await wait_until(lambda: status_payload(base).get("throttled") is True and
                         status_payload(base).get("shape") == 2048,
                         "forced-slow shape step-down", 300)
        mining = status_payload(base)
        if not mining.get("v4_ready") or mining.get("cert_versions") != [3, 4]:
            raise AssertionError("v4 admission is not ready")
        await wait_until(lambda: (busy_percent(logs / "agent.out.log") or 0) > 0,
                         "full busy fraction", 300)
        before = busy_percent(logs / "agent.out.log")

        started = time.monotonic()
        local_json(base, "api/control", body={"paused": True},
                   headers={"X-Pearl-Local": secret,
                            "Origin": f"http://127.0.0.1:{port}"})
        await wait_until(lambda: status_payload(base).get("state") == "paused", "pause control", 3)
        pause_s = time.monotonic() - started
        paused = status_payload(base)
        if paused.get("tops") is not None:
            raise AssertionError("paused status retained stale TOPS")

        local_json(base, "api/control", body={"paused": False},
                   headers={"X-Pearl-Local": secret,
                            "Origin": f"http://localhost:{port}"})
        await wait_until(lambda: status_payload(base).get("state") == "mining" and
                         float(status_payload(base).get("tops") or 0) > 0,
                         "resume control", 300)
        local_json(base, "api/control", body={"intensity": "low"},
                   headers={"X-Pearl-Local": secret})
        await wait_until(lambda: status_document(base).get("controls", {}).get("intensity") == "low",
                         "low intensity control", 3)
        await wait_until(lambda: busy_percent(logs / "agent.out.log") is not None and
                         busy_percent(logs / "agent.out.log") < (before or 101),
                         "low busy fraction", 300)
        after = busy_percent(logs / "agent.out.log")

        removal = local_json(base, "api/control", body={"uninstall": True},
                             headers={"X-Pearl-Local": secret,
                                      "Origin": f"http://127.0.0.1:{port}"})
        if removal != {"ok": True, "removing": True}:
            raise AssertionError("remove response did not confirm removal")
        checked = [app, logs, plist, symlink]
        await wait_until(lambda: not any(path.exists() or path.is_symlink() for path in checked),
                         "uninstall paths removed", 60)
        await wait_until(lambda: port_is_closed(port), "local control port closed", 10)
        loaded = subprocess.run(["launchctl", "print", label], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0
        if loaded:
            raise AssertionError("LaunchAgent remained loaded")
        processes = subprocess.run(["pgrep", "-f", str(home)], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if processes.returncode == 0 and processes.stdout.strip():
            raise AssertionError("a process from the temporary install remained running")

        print(
            "PASS status_sequence=checking->mining tops_gt_zero=true "
            f"accepted={pool.stats.accepted} kernel=na shape=2048 throttled=true "
            f"v4_ready=true pause_seconds={pause_s:.2f} paused_tops=none resumed=true "
            f"full_busy_pct={before} low_busy_pct={after} removed=true port_closed=true "
            "security=403,404,403,403 page_check=true no_processes=true"
        )
        print("REMOVED_PATHS " + " | ".join(str(path) for path in checked + [fallback_cli]))
        return 0
    finally:
        subprocess.run(["launchctl", "bootout", label], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await pool.stop()
        if port is not None and not port_is_closed(port):
            print("WARNING local control port remained open", file=sys.stderr)
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
