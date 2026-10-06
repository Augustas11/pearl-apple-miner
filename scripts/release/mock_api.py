#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""In-memory beta API mock for release installer and agent tests."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import time
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_DESIRED = {
    "paused": False,
    "intensity": "full",
    "only_ac": True,
    "start_at_login": True,
    "uninstall": False,
    "poll_s": 2,
}


class State:
    def __init__(self) -> None:
        self.wallets: dict[str, str] = {}
        self.tokens: dict[str, str] = {}
        self.dash: dict[str, str] = {}
        self.macs: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def register(self, wallet: str, dash_token: str | None = None) -> dict[str, str]:
        install_id = secrets.token_hex(16)
        miner_token = secrets.token_hex(32)
        dash_token = dash_token or secrets.token_hex(32)
        self.wallets[install_id] = wallet
        self.tokens[install_id] = miner_token
        self.dash[install_id] = dash_token
        self.macs[install_id] = {
            "install_id": install_id,
            "label": None,
            "last_seen": None,
            "desired": dict(DEFAULT_DESIRED),
            "last_hb": {},
        }
        return {
            "install_id": install_id,
            "miner_token": miner_token,
            "dash_token": dash_token,
            "dashboard_url": f"http://127.0.0.1/d/{dash_token}",
        }

    def authorize(self, install_id: str | None, token: str | None) -> bool:
        return bool(install_id and token and self.tokens.get(install_id) == token)


STATE = State()


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length))


def write_json(handler: BaseHTTPRequestHandler, status: int, value: Any) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    server_version = "PearlMockAPI/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            dash = parse_qs(parsed.query).get("d", [""])[0]
            macs = []
            wallet = None
            now = time.time()
            for install_id, record in STATE.macs.items():
                if STATE.dash[install_id] != dash:
                    continue
                wallet = STATE.wallets[install_id]
                hb = dict(record["last_hb"])
                last_seen = record["last_seen"]
                macs.append({
                    **hb,
                    "install_id": install_id,
                    "label": hb.get("label") or record.get("label"),
                    "last_seen": last_seen,
                    "online": bool(last_seen and now - last_seen <= 90),
                    "desired": record["desired"],
                })
            write_json(self, 200, {"wallet": wallet, "macs": macs})
            return
        if parsed.path == "/__mock/events":
            write_json(self, 200, {"events": STATE.events})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/register":
            body = read_json(self)
            wallet = str(body.get("wallet", ""))
            if not wallet.startswith("prl1") or not 34 <= len(wallet) <= 94:
                write_json(self, 400, {"error": "invalid wallet"})
                return
            write_json(self, 200, STATE.register(wallet, body.get("dash_token")))
            return
        if parsed.path == "/api/hb":
            install_id = self.headers.get("X-Install-Id")
            auth = self.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if not STATE.authorize(install_id, token):
                write_json(self, 401, {"error": "bad token"})
                return
            body = read_json(self)
            record = STATE.macs[install_id]
            record["last_seen"] = time.time()
            record["label"] = body.get("label", record.get("label"))
            record["last_hb"] = dict(body)
            event = {"time": record["last_seen"], "install_id": install_id, "hb": body}
            STATE.events.append(event)
            write_json(self, 200, record["desired"])
            return
        if parsed.path == "/api/control":
            body = read_json(self)
            dash = body.get("d")
            install_id = str(body.get("install_id", ""))
            if install_id not in STATE.macs or STATE.dash[install_id] != dash:
                write_json(self, 404, {"error": "missing install"})
                return
            desired = STATE.macs[install_id]["desired"]
            for key in ("paused", "only_ac", "start_at_login", "uninstall"):
                if key in body:
                    desired[key] = bool(body[key])
            if body.get("intensity") in {"low", "medium", "full"}:
                desired["intensity"] = body["intensity"]
            if "poll_s" in body:
                desired["poll_s"] = max(1, int(body["poll_s"]))
            write_json(self, 200, desired)
            return
        if parsed.path == "/__mock/control":
            body = read_json(self)
            install_id = str(body.get("install_id", ""))
            if install_id not in STATE.macs:
                write_json(self, 404, {"error": "missing install"})
                return
            desired = STATE.macs[install_id]["desired"]
            desired.update({k: v for k, v in body.items() if k in desired})
            write_json(self, 200, desired)
            return
        self.send_error(404)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    host, port = server.server_address[:2]
    print(json.dumps({"base_url": f"http://{host}:{port}", "port": port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
