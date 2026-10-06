# SPDX-License-Identifier: Apache-2.0
"""Local user CLI for the Malibu Pearl beta LaunchAgent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .beta_agent import config_path, state_path


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class LocalControlError(RuntimeError):
    pass


def control_url(*, wait_seconds: float = 0) -> tuple[str, str]:
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        config = read_json(config_path(), {})
        state = read_json(state_path(), {})
        secret = config.get("local_secret") if isinstance(config, dict) else None
        port = state.get("local_port") if isinstance(state, dict) else None
        if (isinstance(secret, str) and re.fullmatch(r"[0-9a-f]{64}", secret) and
                isinstance(port, int) and 47811 <= port <= 47820):
            return f"http://127.0.0.1:{port}/{secret}/", secret
        if time.monotonic() >= deadline:
            raise LocalControlError("Pearl's local control page is not ready.")
        time.sleep(0.2)


def request_local(path: str, changes: dict[str, object] | None = None,
                  *, wait_seconds: float = 0) -> dict[str, object]:
    deadline = time.monotonic() + max(0, wait_seconds)
    last_error: Exception | None = None
    while True:
        try:
            base, secret = control_url(wait_seconds=0)
            data = None if changes is None else json.dumps(changes, separators=(",", ":")).encode("utf-8")
            headers = {"Accept": "application/json"}
            method = "GET"
            if data is not None:
                method = "POST"
                headers.update({"Content-Type": "application/json", "X-Pearl-Local": secret})
            req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise LocalControlError("Pearl's local agent returned an invalid response.")
            return value
        except (OSError, ValueError, urllib.error.URLError, LocalControlError) as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise LocalControlError("Pearl's local agent is not responding.") from last_error
        time.sleep(0.2)


def update_control(**changes: object) -> dict[str, object]:
    return request_local("api/control", changes)


def cmd_status(_args) -> int:
    try:
        state = request_local("api/status")
    except LocalControlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    status = state.get("status", {})
    controls = state.get("controls", {})
    if not isinstance(status, dict) or not isinstance(controls, dict):
        print("Pearl's local agent returned an invalid response.", file=sys.stderr)
        return 1
    tops = status.get("tops")
    speed = f"{tops} TOPS" if isinstance(tops, (int, float)) else "—"
    print(f"State: {status.get('state', 'unknown')}")
    print(f"Speed: {speed}")
    print(f"Shares: accepted={status.get('shares_accepted', 0)} rejected={status.get('shares_rejected', 0)}")
    print(f"Intensity: {controls.get('intensity', status.get('intensity', 'full'))}")
    return 0


def cmd_pause(_args) -> int:
    try:
        update_control(paused=True)
    except LocalControlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Pearl miner paused.")
    return 0


def cmd_resume(_args) -> int:
    try:
        update_control(paused=False)
    except LocalControlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Pearl miner resumed.")
    return 0


def cmd_intensity(args) -> int:
    try:
        update_control(intensity=args.level)
    except LocalControlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Pearl miner intensity set to {args.level}.")
    return 0


def cmd_uninstall(_args) -> int:
    try:
        update_control(uninstall=True)
    except LocalControlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Pearl miner uninstall requested.")
    return 0


def cmd_open(args) -> int:
    try:
        request_local("api/status", wait_seconds=args.wait)
        url, _secret = control_url()
    except LocalControlError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result = subprocess.run(["/usr/bin/open", url], check=False)
    if result.returncode:
        print("The control page could not be opened.", file=sys.stderr)
        return result.returncode
    return 0


def cmd_logs(_args) -> int:
    paths = [
        Path.home() / "Library/Logs/MalibuPearl/agent.out.log",
        Path.home() / "Library/Logs/MalibuPearl/agent.err.log",
    ]
    for path in paths:
        if not path.exists():
            continue
        print(f"==> {path.name} <==")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-50:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pearl-miner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("pause").set_defaults(func=cmd_pause)
    sub.add_parser("resume").set_defaults(func=cmd_resume)
    intensity = sub.add_parser("intensity")
    intensity.add_argument("level", choices=("low", "medium", "full"))
    intensity.set_defaults(func=cmd_intensity)
    sub.add_parser("uninstall").set_defaults(func=cmd_uninstall)
    sub.add_parser("logs").set_defaults(func=cmd_logs)
    open_page = sub.add_parser("open")
    open_page.add_argument("--wait", type=float, default=0, help=argparse.SUPPRESS)
    open_page.set_defaults(func=cmd_open)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
