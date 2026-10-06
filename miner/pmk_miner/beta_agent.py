# SPDX-License-Identifier: Apache-2.0
"""Local-control beta agent for non-technical macOS installs."""
from __future__ import annotations

import argparse
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import plistlib
import platform
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

VERSION = "0.2.0-beta.18"
LABEL = "tech.malibu.pearl"
CONTROL_PORTS = range(47811, 47821)
REGIONS = ("de", "fr", "es", "fi", "ru", "ca", "us", "us2", "us3", "mx", "br", "kz", "hk", "kr", "in", "sg", "tr", "au")
INTENSITY_PERCENT = {"low": 25, "medium": 50, "full": 100}
VALID_STATES = {"checking", "mining", "paused", "paused_battery", "paused_thermal", "error", "uninstalled"}
SHAPE_LEVELS = (8192, 4096, 2048)
SHAPE_BUDGET_SECONDS = 0.4
SHAPE_DOWN_THRESHOLD_SECONDS = SHAPE_BUDGET_SECONDS * 0.70
SHAPE_UP_THRESHOLD_SECONDS = SHAPE_BUDGET_SECONDS * 0.45
SHAPE_UP_HYSTERESIS_SECONDS = 300.0


def app_root() -> Path:
    return Path.home() / "Library/Application Support/MalibuPearl"


def log_root() -> Path:
    return Path.home() / "Library/Logs/MalibuPearl"


def launch_agent_path() -> Path:
    return Path.home() / "Library/LaunchAgents/tech.malibu.pearl.plist"


def current_root() -> Path:
    return app_root() / "current"


def config_path() -> Path:
    return app_root() / "config.json"


def control_path() -> Path:
    return app_root() / "control.json"


def state_path() -> Path:
    return app_root() / "state.json"


def private_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def write_json(path: Path, value, *, mode: int = 0o600) -> None:
    private_write(path, json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.chmod(path, mode)


def redact_wallet(wallet: str | None) -> str | None:
    if not wallet:
        return wallet
    return wallet if len(wallet) <= 12 else f"{wallet[:6]}...{wallet[-6:]}"


def sanitize_worker(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-_")
    return value[:32] or "Mac"


def computer_name() -> str:
    for command in (["scutil", "--get", "ComputerName"], ["hostname", "-s"]):
        try:
            value = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
            if value:
                return value
        except (OSError, subprocess.SubprocessError):
            pass
    return "Mac"


def choose_pool_region(timeout: float = 1.25) -> str:
    timings: list[tuple[float, str]] = []
    for region in REGIONS:
        host = f"{region}.pearl.herominers.com"
        started = time.monotonic()
        try:
            with socket.create_connection((host, 1200), timeout=timeout):
                timings.append((time.monotonic() - started, region))
        except OSError:
            continue
    return min(timings)[1] if timings else "sg"


def pool_url(region: str) -> str:
    if region.startswith("stratum+"):
        return region
    return f"stratum+tcp://{region}.pearl.herominers.com:1200"


def platform_info() -> dict[str, object]:
    chip = "Apple Silicon"
    gpu_cores = 0
    try:
        hw = subprocess.check_output(["system_profiler", "SPHardwareDataType", "-json"], text=True, timeout=15)
        items = json.loads(hw).get("SPHardwareDataType", [])
        if items:
            chip = items[0].get("chip_type") or items[0].get("machine_model") or chip
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    try:
        displays = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=15)
        gpu_cores = max(int(item.get("sppci_cores", 0)) for item in json.loads(displays).get("SPDisplaysDataType", []))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        gpu_cores = 0
    try:
        macos = subprocess.check_output(["sw_vers", "-productVersion"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        macos = platform.mac_ver()[0]
    return {"chip": chip, "gpu_cores": gpu_cores, "macos": macos}


def power_source() -> str:
    try:
        from .desktop import DesktopNative

        return DesktopNative().power_source()
    except Exception:
        pass
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True, stderr=subprocess.DEVNULL, timeout=5)
        return "battery" if "Battery Power" in out else "ac"
    except (OSError, subprocess.SubprocessError):
        return "desktop"


def thermal_state() -> str:
    try:
        from .desktop import DesktopNative

        return DesktopNative().thermal_state()
    except Exception:
        pass
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True, stderr=subprocess.DEVNULL, timeout=5)
        lowered = out.lower()
        if "critical" in lowered:
            return "critical"
        if "serious" in lowered or "high" in lowered:
            return "serious"
        if "fair" in lowered:
            return "fair"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def set_run_at_load(enabled: bool) -> None:
    plist = launch_agent_path()
    if not plist.exists():
        return
    data = plistlib.loads(plist.read_bytes())
    if bool(data.get("RunAtLoad", False)) == enabled:
        return
    data["RunAtLoad"] = enabled
    tmp = plist.with_suffix(".plist.tmp")
    tmp.write_bytes(plistlib.dumps(data, sort_keys=True))
    os.replace(tmp, plist)


def _initial_kernel(cfg: dict[str, object]) -> str:
    """Report the kernel this Mac will run before the miner's first event arrives."""
    try:
        from .kernel import resolve_v3_kernel
        kernel, _ = resolve_v3_kernel(str(cfg.get("kernel", "auto")))
        return kernel
    except Exception:
        return "sg"


class MinerProcess:
    def __init__(self, cfg: dict[str, object]) -> None:
        self.cfg = cfg
        self.process: subprocess.Popen[str] | None = None
        self.events = collections.deque(maxlen=512)
        self.tops_samples = collections.deque(maxlen=120)
        self.ops_samples = collections.deque(maxlen=120)
        self.accepted = 0
        self.rejected = 0
        self.last_error: str | None = None
        self._reader: threading.Thread | None = None
        self._settings: tuple[str, bool] | None = None
        self.restart_requested = threading.Event()
        self.shape_override: int | None = None
        self.active_shape: int | None = None
        self.initial_shape: int | None = None
        self.kernel = _initial_kernel(cfg)
        self.throttled = False
        self.working = False
        self.last_shape_change = float("-inf")
        self.headroom_since: float | None = None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, *, intensity: str, only_ac: bool) -> None:
        requested = (intensity, only_ac, self.shape_override)
        if self.running() and self._settings == requested:
            return
        if self.running():
            self.stop()
        root = current_root()
        python = root / "bin/python3"
        if not python.exists():
            python = root / "bin/python"
        script = root / "scripts/pmk_mine.py"
        env = os.environ.copy()
        env["PMK_HOME"] = str(app_root() / "pmk-home")
        env["PYTHONPATH"] = os.pathsep.join(part for part in (str(root / "miner"), str(root), env.get("PYTHONPATH", "")) if part)
        env.setdefault("PYTHONUNBUFFERED", "1")
        wallet_file = app_root() / "wallet"
        private_write(wallet_file, str(self.cfg["wallet"]) + "\n")
        args = [str(python), str(script),
                "--wallet-file", str(wallet_file),
                "--worker", str(self.cfg["worker"]),
                "--pool", str(self.cfg["pool_url"]),
                "--on-battery", "pause" if only_ac else "run",
                "--intensity", str(INTENSITY_PERCENT[intensity]),
                "--kernel", str(self.cfg.get("kernel", "auto"))]
        if self.shape_override is not None:
            args.extend(["--shape", str(self.shape_override)])
        log_root().mkdir(parents=True, exist_ok=True)
        stdout = open(log_root() / "agent.out.log", "a", encoding="utf-8")
        self.process = subprocess.Popen(args, cwd=root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            start_new_session=True)
        self._reader = threading.Thread(target=self._read_output, args=(self.process, stdout), daemon=True)
        self._reader.start()
        self._settings = requested
        self.last_error = None
        self.restart_requested.clear()

    def stop(self, timeout: float = 20.0) -> None:
        self.working = False
        self.tops_samples.clear()
        self.ops_samples.clear()
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
        self.process = None
        self._settings = None

    def _read_output(self, proc: subprocess.Popen[str], mirror) -> None:
        assert proc.stdout is not None
        wallet = str(self.cfg.get("wallet", ""))
        for line in proc.stdout:
            safe = line.replace(wallet, redact_wallet(wallet) or "<wallet>")
            mirror.write(safe)
            mirror.flush()
            self._parse_line(safe.strip())
        mirror.close()

    def _parse_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        now = time.time()
        if isinstance(event, dict):
            name = event.get("event")
            self.events.append(event)
            if name == "pool_outcome":
                self.accepted = max(self.accepted, int(event.get("accepted", self.accepted) or 0))
                classification = str(event.get("classification", ""))
                if classification not in ("accepted", "transport", "timeout", ""):
                    self.rejected += 1
            elif name == "routine_telemetry":
                tops = event.get("tops")
                if isinstance(tops, (int, float)):
                    self.tops_samples.append((now, float(tops)))
                completed_ops = event.get("completed_ops")
                if isinstance(completed_ops, int) and completed_ops >= 0:
                    self.ops_samples.append((now, completed_ops))
                    if completed_ops > 0:
                        self.working = True
                p90 = event.get("gpu_p90_seconds")
                if isinstance(p90, (int, float)):
                    self._observe_gpu_p90(float(p90), now)
                shape = event.get("shape")
                if isinstance(shape, dict) and isinstance(shape.get("m"), int):
                    self.active_shape = int(shape["m"])
                if isinstance(event.get("throttled"), bool):
                    self.throttled = bool(event["throttled"])
                if event.get("kernel") in ("sg", "na"):
                    self.kernel = str(event["kernel"])
            elif name == "completed":
                self.working = True
            elif name == "pool_startup":
                shape = event.get("shape")
                if isinstance(shape, dict) and isinstance(shape.get("m"), int):
                    self.active_shape = int(shape["m"])
                    if self.initial_shape is None:
                        self.initial_shape = self.active_shape
                if event.get("effective_kernel") in ("sg", "na"):
                    self.kernel = str(event["effective_kernel"])
            elif name in ("shape_changed", "shape_budget_minimum"):
                shape = event.get("shape")
                if isinstance(shape, dict) and isinstance(shape.get("m"), int):
                    self.active_shape = int(shape["m"])
                if isinstance(event.get("throttled"), bool):
                    self.throttled = bool(event["throttled"])
            elif name == "shape_throttle":
                self._step_down(str(event.get("reason") or "GPU command budget"), now)
            elif name == "fatal":
                if event.get("gate") == "shape_budget":
                    self._step_down(str(event.get("error_message") or "GPU command budget"), now)
                else:
                    self.last_error = str(event.get("error_message") or event.get("error_type") or "miner failed")[:200]
            return
        match = re.search(r"([0-9]+(?:\.[0-9]+)?) TOPS .*accepted=([0-9]+) rejected=([0-9]+)", line)
        if match:
            self.tops_samples.append((now, float(match.group(1))))
            if float(match.group(1)) > 0:
                self.working = True
            self.accepted = max(self.accepted, int(match.group(2)))
            self.rejected = max(self.rejected, int(match.group(3)))

    def _step_down(self, reason: str, now: float) -> bool:
        current = self.active_shape or self.shape_override
        if current not in SHAPE_LEVELS:
            return False
        index = SHAPE_LEVELS.index(current)
        if index >= len(SHAPE_LEVELS) - 1:
            self.last_error = f"minimum shape exceeded GPU budget: {reason}"[:200]
            return False
        self.shape_override = SHAPE_LEVELS[index + 1]
        self.throttled = True
        self.last_shape_change = now
        self.headroom_since = None
        self.last_error = None
        self.restart_requested.set()
        return True

    def _observe_gpu_p90(self, p90: float, now: float) -> None:
        if p90 > SHAPE_DOWN_THRESHOLD_SECONDS:
            self._step_down(f"p90 {p90:.3f}s approaches {SHAPE_BUDGET_SECONDS:.3f}s budget", now)
            return
        if not self.throttled or p90 >= SHAPE_UP_THRESHOLD_SECONDS:
            self.headroom_since = None
            return
        if self.headroom_since is None:
            self.headroom_since = now
            return
        if (now - self.headroom_since < SHAPE_UP_HYSTERESIS_SECONDS or
                now - self.last_shape_change < SHAPE_UP_HYSTERESIS_SECONDS):
            return
        current = self.active_shape or self.shape_override
        ceiling = self.initial_shape
        if current not in SHAPE_LEVELS or ceiling not in SHAPE_LEVELS:
            return
        index = SHAPE_LEVELS.index(current)
        if index == 0:
            return
        candidate = SHAPE_LEVELS[index - 1]
        if candidate > ceiling:
            return
        self.shape_override = candidate
        self.throttled = candidate < ceiling
        self.last_shape_change = now
        self.headroom_since = None
        self.restart_requested.set()

    def tops_60s(self) -> float:
        cutoff = time.time() - 60
        ops = [(ts, value) for ts, value in self.ops_samples if ts >= cutoff]
        if len(ops) >= 2 and ops[-1][0] > ops[0][0]:
            return max(0.0, (ops[-1][1] - ops[0][1]) / (ops[-1][0] - ops[0][0]) / 1e12)
        values = [value for ts, value in self.tops_samples if ts >= cutoff]
        return sum(values) / len(values) if values else 0.0


class ControlHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], agent: "Agent") -> None:
        self.agent = agent
        super().__init__(address, ControlRequestHandler)


class ControlRequestHandler(BaseHTTPRequestHandler):
    """Loopback API with strict host, path-secret, and CSRF checks."""

    server: ControlHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        # Paths contain the local secret, so access logging is intentionally disabled.
        return

    def _host_allowed(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _route(self) -> str | None:
        try:
            path = urllib.parse.urlsplit(self.path).path
        except ValueError:
            return None
        prefix = f"/{self.server.agent.local_secret}/"
        if not path.startswith(prefix):
            return None
        return path[len(prefix):]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        if content_type.startswith("text/html"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src https://fonts.gstatic.com; script-src 'unsafe-inline'; "
                "connect-src 'self' https://pearl.herominers.com; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'",
            )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _checked_route(self) -> str | None:
        if not self._host_allowed():
            self._json(403, {"error": "forbidden"})
            return None
        route = self._route()
        if route is None:
            self._json(404, {"error": "not found"})
        return route

    def do_GET(self) -> None:
        # Bare http://localhost:<port>/ forwards to the control page, so people only need to
        # remember one short address. Host is still checked (DNS rebinding); other sites can
        # navigate here but can't read the redirect or press buttons (POST needs X-Pearl-Local).
        if self._host_allowed() and urllib.parse.urlsplit(self.path).path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", f"/{self.server.agent.local_secret}/")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        route = self._checked_route()
        if route is None:
            return
        if route == "":
            from .control_page import render_control_page

            html = render_control_page(
                wallet=str(self.server.agent.cfg["wallet"]),
                local_secret=self.server.agent.local_secret,
                port=self.server.server_address[1],
                computer_name=str(self.server.agent.cfg.get("label") or "This Mac"),
            ).encode("utf-8")
            self._send(200, html, "text/html; charset=utf-8")
            return
        if route == "api/status":
            self._json(200, self.server.agent.status_document())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        route = self._checked_route()
        if route is None:
            return
        if route != "api/control":
            self._json(404, {"error": "not found"})
            return
        secret = self.headers.get("X-Pearl-Local", "")
        if not secrets.compare_digest(secret, self.server.agent.local_secret):
            self._json(403, {"error": "forbidden"})
            return
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        if origin is not None and origin not in allowed_origins:
            self._json(403, {"error": "forbidden"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 1 or length > 4096:
            self._json(400, {"error": "invalid request"})
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError
            controls = self.server.agent.update_controls(request)
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"error": "invalid control"})
            return
        self._json(200, {"ok": True, "removing": bool(controls.get("uninstall"))})


class Agent:
    def __init__(self, cfg: dict[str, object]) -> None:
        self.cfg = cfg
        self.local_secret = str(cfg["local_secret"])
        self.controls = {"paused": False, "intensity": "full", "only_ac": True,
                         "start_at_login": True, "uninstall": False}
        stored_controls = read_json(control_path(), {})
        if isinstance(stored_controls, dict):
            self._merge_valid_controls(stored_controls)
        write_json(control_path(), self.controls)
        self.miner = MinerProcess(cfg)
        self.started = time.monotonic()
        self.info = platform_info()
        self.state = "checking"
        self.last_error: str | None = None
        self.effective_intensity = str(self.controls["intensity"])
        self.local_port: int | None = None
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._status: dict[str, object] = {}
        self._httpd: ControlHTTPServer | None = None
        self._http_thread: threading.Thread | None = None

    def run(self) -> int:
        try:
            self.start_control_server()
            self.loop()
            return 0
        except Exception as exc:
            self.state = "error"
            self.last_error = str(exc)[:200] or "the local agent stopped"
            if self.local_port is not None:
                self.publish_status()
            return 1
        finally:
            self.stop_control_server()
            self.miner.stop()

    def start_control_server(self) -> None:
        previous = read_json(state_path(), {})
        preferred = previous.get("local_port") if isinstance(previous, dict) else None
        candidates = []
        if isinstance(preferred, int) and preferred in CONTROL_PORTS:
            candidates.append(preferred)
        candidates.extend(port for port in CONTROL_PORTS if port not in candidates)
        last_error: OSError | None = None
        for port in candidates:
            try:
                self._httpd = ControlHTTPServer(("127.0.0.1", port), self)
                self.local_port = port
                break
            except OSError as exc:
                last_error = exc
        if self._httpd is None:
            raise RuntimeError("local control ports 47811 through 47820 are unavailable") from last_error
        self.publish_status()
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="pearl-local-control",
            daemon=True,
        )
        self._http_thread.start()

    def stop_control_server(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
        self._httpd = None
        self._http_thread = None

    def loop(self) -> None:
        while True:
            self.apply_controls()
            self.publish_status()
            with self._lock:
                removing = bool(self.controls["uninstall"])
            if removing:
                self.state = "uninstalled"
                self.publish_status()
                self.uninstall()
                return
            self._wake.wait(1.0)
            self._wake.clear()

    def _merge_valid_controls(self, desired: dict[str, object]) -> None:
        for key in ("paused", "only_ac", "start_at_login", "uninstall"):
            if type(desired.get(key)) is bool:
                self.controls[key] = desired[key]
        if desired.get("intensity") in INTENSITY_PERCENT:
            self.controls["intensity"] = str(desired["intensity"])

    def update_controls(self, desired: dict[str, object]) -> dict[str, object]:
        allowed = {"paused", "intensity", "only_ac", "start_at_login", "uninstall"}
        if not desired or set(desired) - allowed:
            raise ValueError("unknown control")
        for key, value in desired.items():
            if key == "intensity":
                if value not in INTENSITY_PERCENT:
                    raise ValueError("invalid intensity")
            elif type(value) is not bool:
                raise ValueError("invalid toggle")
            elif key == "uninstall" and value is not True:
                raise ValueError("invalid uninstall")
        with self._lock:
            self._merge_valid_controls(desired)
            snapshot = dict(self.controls)
            write_json(control_path(), snapshot)
        self._wake.set()
        return snapshot

    def apply_controls(self) -> None:
        with self._lock:
            controls = dict(self.controls)
        set_run_at_load(bool(controls.get("start_at_login", True)))
        thermal = thermal_state()
        if thermal == "critical":
            self.miner.stop()
            self.state = "paused_thermal"
            return
        if self.miner.restart_requested.is_set():
            self.miner.stop()
            self.miner.restart_requested.clear()
        paused = bool(controls.get("paused", False))
        only_ac = bool(controls.get("only_ac", True))
        intensity = str(controls.get("intensity", "full"))
        if thermal == "serious":
            intensity = "low"
        self.effective_intensity = intensity
        if paused:
            self.miner.stop()
            self.state = "paused"
        else:
            self.miner.start(intensity=intensity, only_ac=only_ac)
            if only_ac and power_source() == "battery":
                self.state = "paused_battery"
            elif self.miner.running():
                self.state = "mining" if self.miner.working else "checking"
                self.last_error = None
            else:
                self.state = "error"
                self.last_error = self.miner.last_error or "miner is not running"

    def payload(self, *, state: str | None = None) -> dict[str, object]:
        current_state = state or self.state
        if current_state not in VALID_STATES:
            current_state = "error"
        certs = [3]
        if Path(str(self.cfg.get("v4_admission", ""))).exists():
            certs.append(4)
        result = {
            "version": str(self.cfg.get("version", VERSION)),
            "label": str(self.cfg.get("label") or computer_name()),
            "state": current_state,
            "tops": round(self.miner.tops_60s(), 3) if current_state == "mining" else None,
            "shares_accepted": self.miner.accepted,
            "shares_rejected": self.miner.rejected,
            "uptime_s": int(time.monotonic() - self.started),
            "power": power_source(),
            "kernel": self.miner.kernel,
            "shape": self.miner.active_shape,
            "throttled": self.miner.throttled,
            "cert_versions": certs,
            "v4_ready": 4 in certs,
            "intensity": self.effective_intensity,
            "last_error": self.last_error or self.miner.last_error,
            **self.info,
        }
        return result

    def publish_status(self) -> None:
        payload = self.payload()
        with self._lock:
            controls = {key: value for key, value in self.controls.items() if key != "uninstall"}
            document = {
                "status": payload,
                "controls": controls,
                "wallet": redact_wallet(str(self.cfg.get("wallet", ""))),
                "local_port": self.local_port,
            }
            self._status = document
        write_json(state_path(), document)

    def status_document(self) -> dict[str, object]:
        with self._lock:
            # Round-tripping gives handlers an immutable snapshot without exposing
            # the config's full wallet or local secret.
            return json.loads(json.dumps(self._status))

    def uninstall(self) -> None:
        self.miner.stop()
        fd, raw = tempfile.mkstemp(prefix="malibu-pearl-uninstall-", suffix=".json")
        request = Path(raw)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"home": str(Path.home())}, stream)
        os.chmod(request, 0o600)
        subprocess.Popen([sys.executable, "-m", "pmk_miner.beta_cleanup", "--request", str(request)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_path())
    args = parser.parse_args(argv)
    cfg = read_json(args.config, {})
    secret = cfg.get("local_secret") if isinstance(cfg, dict) else None
    if (not isinstance(cfg, dict) or not cfg.get("wallet") or not cfg.get("pool_url") or
            not isinstance(secret, str) or not re.fullmatch(r"[0-9a-f]{64}", secret)):
        print("beta agent missing config", file=sys.stderr)
        return 2
    return Agent(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
