# SPDX-License-Identifier: Apache-2.0
"""Website-managed beta agent for non-technical macOS installs."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import plistlib
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

VERSION = "0.2.0"
FAST_REPORT_S = 11  # the server accepts one heartbeat per 10 s
LABEL = "tech.malibu.pearl"
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


class HeartbeatClient:
    def __init__(self, api_base: str, install_id: str, token: str) -> None:
        self.url = api_base.rstrip("/") + "/api/hb"
        self.install_id = install_id
        self.token = token
        self.backoff = 30.0
        self.retry_delay = 30.0

    def post(self, payload: dict[str, object]) -> dict[str, object] | None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.token}",
                     "X-Install-Id": self.install_id})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                self.backoff = 30.0
                self.retry_delay = 30.0
                if response.status != 200:
                    return None
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise PermissionError("heartbeat token rejected") from exc
            self.retry_delay = self.backoff
            self.backoff = min(300.0, self.backoff * 2)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self.retry_delay = self.backoff
            self.backoff = min(300.0, self.backoff * 2)
        return None


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
        env["PMK_API_BASE"] = str(self.cfg.get("api_base", "https://pearl.malibu.tech"))
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


class Agent:
    def __init__(self, cfg: dict[str, object]) -> None:
        self.cfg = cfg
        self.client = HeartbeatClient(str(cfg.get("api_base", "https://pearl.malibu.tech")),
                                      str(cfg["install_id"]), str(cfg["miner_token"]))
        self.controls = {"paused": False, "intensity": "full", "only_ac": True,
                         "start_at_login": True, "uninstall": False, "poll_s": 30}
        self.controls.update(read_json(control_path(), {}))
        self.miner = MinerProcess(cfg)
        self.started = time.monotonic()
        self.info = platform_info()
        self.state = "checking"
        self.last_error: str | None = None
        self.effective_intensity = str(self.controls["intensity"])

    def run(self) -> int:
        try:
            try:
                self.loop()
            except PermissionError as exc:
                self.state = "error"
                self.last_error = str(exc)
                write_json(state_path(), {"heartbeat": self.payload(), "controls": self.controls,
                                          "wallet": redact_wallet(str(self.cfg.get("wallet", "")))})
            return 0
        finally:
            self.miner.stop()

    def loop(self) -> None:
        while True:
            local = read_json(control_path(), {})
            if isinstance(local, dict):
                self.controls.update({key: local[key] for key in self.controls if key in local})
            self.apply_controls()
            payload = self.payload()
            write_json(state_path(), {"heartbeat": payload, "controls": self.controls,
                                      "wallet": redact_wallet(str(self.cfg.get("wallet", "")))})
            desired = self.client.post(payload)
            if desired:
                self.merge_controls(desired)
                if self.controls.get("uninstall"):
                    self.uninstall()
                    return
                self.apply_controls()
            sleep_for = int(self.controls.get("poll_s") or 30)
            # Report a change (pause, resume, intensity) or a start-up as soon as the server allows,
            # so the dashboard confirms it in seconds instead of a full poll later.
            reported = (payload.get("state"), payload.get("intensity"))
            if (self.state, self.effective_intensity) != reported or self.state == "checking":
                sleep_for = min(sleep_for, FAST_REPORT_S)
            if desired is None:
                sleep_for = int(self.client.retry_delay)
            self.miner.restart_requested.wait(max(1, sleep_for))

    def merge_controls(self, desired: dict[str, object]) -> None:
        for key in ("paused", "only_ac", "start_at_login", "uninstall"):
            if isinstance(desired.get(key), bool):
                self.controls[key] = bool(desired[key])
        if desired.get("intensity") in INTENSITY_PERCENT:
            self.controls["intensity"] = str(desired["intensity"])
        if isinstance(desired.get("poll_s"), (int, float)) and float(desired["poll_s"]) >= 1:
            self.controls["poll_s"] = int(desired["poll_s"])
        write_json(control_path(), self.controls)

    def apply_controls(self) -> None:
        set_run_at_load(bool(self.controls.get("start_at_login", True)))
        thermal = thermal_state()
        if thermal == "critical":
            self.miner.stop()
            self.state = "paused_thermal"
            return
        if self.miner.restart_requested.is_set():
            self.miner.stop()
            self.miner.restart_requested.clear()
        paused = bool(self.controls.get("paused", False))
        only_ac = bool(self.controls.get("only_ac", True))
        intensity = str(self.controls.get("intensity", "full"))
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
            "tops": round(self.miner.tops_60s(), 3) if current_state == "mining" else 0.0,
            "shares_accepted": self.miner.accepted,
            "shares_rejected": self.miner.rejected,
            "uptime_s": int(time.monotonic() - self.started),
            "power": power_source(),
            "kernel": self.miner.kernel,
            "shape": self.miner.active_shape,
            "throttled": self.miner.throttled,
            "cert_versions": certs,
            "intensity": self.effective_intensity,
            "last_error": self.last_error or self.miner.last_error,
            **self.info,
        }
        return result

    def uninstall(self) -> None:
        self.miner.stop()
        fd, raw = tempfile.mkstemp(prefix="malibu-pearl-uninstall-", suffix=".json")
        request = Path(raw)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"home": str(Path.home()), "version": str(self.cfg.get("version", VERSION)),
                       "api_base": self.cfg.get("api_base", "https://pearl.malibu.tech"),
                       "install_id": self.cfg["install_id"],
                       "miner_token": self.cfg["miner_token"]}, stream)
        os.chmod(request, 0o600)
        subprocess.Popen([sys.executable, "-m", "pmk_miner.beta_cleanup", "--request", str(request)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_path())
    args = parser.parse_args(argv)
    cfg = read_json(args.config, {})
    if not isinstance(cfg, dict) or not cfg.get("miner_token") or not cfg.get("install_id"):
        print("beta agent missing config", file=sys.stderr)
        return 2
    return Agent(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
