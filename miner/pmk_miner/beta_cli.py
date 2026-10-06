# SPDX-License-Identifier: Apache-2.0
"""Local user CLI for the Malibu Pearl beta LaunchAgent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .beta_agent import control_path, state_path, write_json


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def update_control(**changes) -> dict[str, object]:
    current = read_json(control_path(), {})
    current.update(changes)
    write_json(control_path(), current)
    return current


def cmd_status(_args) -> int:
    state = read_json(state_path(), {})
    hb = state.get("heartbeat", {}) if isinstance(state, dict) else {}
    controls = state.get("controls", {}) if isinstance(state, dict) else {}
    if not hb:
        print("Pearl miner is installed, but no status has been reported yet.")
        return 1
    print(f"State: {hb.get('state', 'unknown')}")
    print(f"Speed: {hb.get('tops', 0)} TOPS")
    print(f"Shares: accepted={hb.get('shares_accepted', 0)} rejected={hb.get('shares_rejected', 0)}")
    print(f"Intensity: {controls.get('intensity', hb.get('intensity', 'full'))}")
    return 0


def cmd_pause(_args) -> int:
    update_control(paused=True)
    print("Pearl miner paused.")
    return 0


def cmd_resume(_args) -> int:
    update_control(paused=False)
    print("Pearl miner resumed.")
    return 0


def cmd_intensity(args) -> int:
    update_control(intensity=args.level)
    print(f"Pearl miner intensity set to {args.level}.")
    return 0


def cmd_uninstall(_args) -> int:
    update_control(uninstall=True)
    print("Pearl miner uninstall requested.")
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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
