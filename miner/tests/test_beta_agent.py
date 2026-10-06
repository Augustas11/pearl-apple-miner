import json
import os
import plistlib
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

from pmk_miner import beta_agent
from pmk_miner.beta_agent import MinerProcess, sanitize_worker
from pmk_miner import beta_cleanup
from pmk_miner import beta_cli


def _agent(tmp_path, monkeypatch):
    monkeypatch.setattr(beta_agent, "app_root", lambda: tmp_path)
    monkeypatch.setattr(beta_agent, "control_path", lambda: tmp_path / "control.json")
    monkeypatch.setattr(beta_agent, "state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(beta_agent, "platform_info", lambda: {})
    monkeypatch.setattr(beta_agent, "power_source", lambda: "ac")
    monkeypatch.setattr(beta_agent, "thermal_state", lambda: "unknown")
    cfg = {"local_secret": "a" * 64, "wallet": "prl1" + "q" * 40,
           "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1", "label": "Test Mac"}
    return beta_agent.Agent(cfg)


def _request(agent, path, *, method="GET", host=None, headers=None, body=None):
    port = agent.local_port
    request_headers = dict(headers or {})
    request_headers["Host"] = host or f"127.0.0.1:{port}"
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def test_local_control_http_security_and_api(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    agent.start_control_server()
    try:
        root = f"/{agent.local_secret}/"
        status, headers, body = _request(agent, root + "api/status")
        assert status == 200
        assert json.loads(body)["status"]["state"] == "checking"
        assert "Access-Control-Allow-Origin" not in headers

        assert _request(agent, root, host="evil.example")[0] == 403
        assert _request(agent, "/api/status")[0] == 404
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", agent.local_port, timeout=3)
        conn.request("GET", "/", headers={"Host": f"localhost:{agent.local_port}"})
        response = conn.getresponse()
        assert response.status == 302
        assert response.getheader("Location") == root
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", agent.local_port, timeout=3)
        conn.request("GET", "/", headers={"Host": "evil.example"})
        assert conn.getresponse().status == 403
        conn.close()
        assert _request(agent, root + "api/control", method="POST", body={"paused": True})[0] == 403
        assert _request(agent, root + "api/control", method="POST",
                        headers={"X-Pearl-Local": agent.local_secret,
                                 "Origin": "https://evil.example"}, body={"paused": True})[0] == 403

        status, _, body = _request(agent, root + "api/control", method="POST",
                                  headers={"X-Pearl-Local": agent.local_secret,
                                           "Origin": f"http://localhost:{agent.local_port}"},
                                  body={"paused": True, "intensity": "low"})
        assert status == 200
        assert json.loads(body) == {"ok": True, "removing": False}
        assert agent.controls["paused"] is True
        assert agent.controls["intensity"] == "low"
        assert json.loads((tmp_path / "control.json").read_text())["paused"] is True
    finally:
        agent.stop_control_server()


def test_miner_output_parses_telemetry_and_redacts_wallet(tmp_path):
    cfg = {"wallet": "prl1abcdef1234567890", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"}
    miner = MinerProcess(cfg)
    miner._parse_line('{"event":"pool_outcome","classification":"accepted","accepted":3}')
    miner._parse_line('{"event":"pool_outcome","classification":"invalid","accepted":3}')
    miner._parse_line("[pmk] 2.00 jobs/s | 4.50 TOPS | shares accepted=5 rejected=1 | expected time/share 1.0 min")
    assert miner.accepted == 5
    assert miner.rejected == 1
    assert miner.tops_60s() == 4.5


def test_miner_uses_completed_ops_for_rolling_tops(monkeypatch):
    miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    now = [100.0]
    monkeypatch.setattr(beta_agent.time, "time", lambda: now[0])
    miner._parse_line('{"event":"routine_telemetry","tops":99,"completed_ops":1000000000000}')
    now[0] = 110.0
    miner._parse_line('{"event":"routine_telemetry","tops":99,"completed_ops":21000000000000}')
    assert miner.tops_60s() == 2.0


def test_slow_jobs_step_down_without_halting(monkeypatch):
    miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    now = [100.0]
    monkeypatch.setattr(beta_agent.time, "time", lambda: now[0])
    miner._parse_line('{"event":"pool_startup","shape":{"m":8192,"n":8192},"effective_kernel":"na"}')
    miner._parse_line('{"event":"routine_telemetry","gpu_p90_seconds":0.29}')
    assert miner.shape_override == 4096
    assert miner.throttled is True
    assert miner.restart_requested.is_set()
    assert miner.last_error is None
    assert miner.kernel == "na"

    miner.restart_requested.clear()
    miner.active_shape = 4096
    now[0] += 1
    miner._parse_line('{"event":"fatal","gate":"shape_budget","error_message":"predicted command buffer exceeds 400 ms"}')
    assert miner.shape_override == 2048
    assert miner.restart_requested.is_set()
    assert miner.last_error is None


def test_shape_step_up_requires_sustained_five_minute_headroom(monkeypatch):
    miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    miner.initial_shape = 8192
    miner.active_shape = 4096
    miner.shape_override = 4096
    miner.throttled = True
    miner.last_shape_change = 10.0
    miner._observe_gpu_p90(0.10, 20.0)
    miner._observe_gpu_p90(0.10, 319.0)
    assert miner.shape_override == 4096
    miner._observe_gpu_p90(0.10, 320.0)
    assert miner.shape_override == 8192
    assert miner.throttled is False
    assert miner.restart_requested.is_set()


def test_control_merge_is_persisted_and_sanitized(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    agent.update_controls({"paused": True, "intensity": "medium", "only_ac": False,
                           "start_at_login": False})
    stored = json.loads((tmp_path / "control.json").read_text())
    assert stored == {"paused": True, "intensity": "medium", "only_ac": False,
                      "start_at_login": False, "uninstall": False}
    assert sanitize_worker(" Family's MacBook Pro!!! ") == "Family-s-MacBook-Pro"


def test_cli_updates_control_file(tmp_path, monkeypatch, capsys):
    agent = _agent(tmp_path, monkeypatch)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(agent.cfg))
    monkeypatch.setattr(beta_cli, "config_path", lambda: config)
    monkeypatch.setattr(beta_cli, "state_path", lambda: tmp_path / "state.json")
    agent.start_control_server()
    try:
        assert beta_cli.main(["pause"]) == 0
        assert agent.controls["paused"] is True
        assert beta_cli.main(["intensity", "low"]) == 0
        assert agent.controls["intensity"] == "low"
        assert "intensity set to low" in capsys.readouterr().out
    finally:
        agent.stop_control_server()


def test_cli_open_uses_secret_local_url_without_printing_it(tmp_path, monkeypatch, capsys):
    agent = _agent(tmp_path, monkeypatch)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(agent.cfg))
    monkeypatch.setattr(beta_cli, "config_path", lambda: config)
    monkeypatch.setattr(beta_cli, "state_path", lambda: tmp_path / "state.json")
    calls = []
    monkeypatch.setattr(beta_cli.subprocess, "run",
                        lambda argv, **_kwargs: calls.append(argv) or SimpleNamespace(returncode=0))
    agent.start_control_server()
    try:
        assert beta_cli.main(["open"]) == 0
        assert calls == [["/usr/bin/open",
                          f"http://127.0.0.1:{agent.local_port}/{agent.local_secret}/"]]
        assert agent.local_secret not in capsys.readouterr().out
    finally:
        agent.stop_control_server()


def test_start_at_login_updates_plist_without_stopping_agent(tmp_path, monkeypatch):
    plist = tmp_path / "tech.malibu.pearl.plist"
    plist.write_bytes(plistlib.dumps({"Label": beta_agent.LABEL, "RunAtLoad": True}))
    monkeypatch.setattr(beta_agent, "launch_agent_path", lambda: plist)
    beta_agent.set_run_at_load(False)
    assert plistlib.loads(plist.read_bytes())["RunAtLoad"] is False


def test_uninstall_starts_detached_cleanup_without_exposing_credentials(tmp_path, monkeypatch):
    calls = []
    agent = _agent(tmp_path, monkeypatch)
    agent.miner.stop = lambda: calls.append(("stop", None))
    monkeypatch.setattr(beta_agent.tempfile, "mkstemp", lambda **_kw: (os.open(tmp_path / "request.json", os.O_CREAT | os.O_RDWR, 0o600), str(tmp_path / "request.json")))
    monkeypatch.setattr(beta_agent.subprocess, "Popen", lambda argv, **_kw: calls.append(("spawn", argv)) or object())
    agent.uninstall()
    assert calls[0] == ("stop", None)
    assert calls[1][0] == "spawn"
    request = json.loads((tmp_path / "request.json").read_text())
    assert request == {"home": str(beta_agent.Path.home())}


def test_detached_cleanup_removes_local_install_and_boots_out_last(tmp_path, monkeypatch):
    home = tmp_path / "home"
    app = home / "Library/Application Support/MalibuPearl"
    logs = home / "Library/Logs/MalibuPearl"
    plist = home / "Library/LaunchAgents/tech.malibu.pearl.plist"
    cli = home / ".local/bin/pearl-miner"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    plist.parent.mkdir(parents=True)
    plist.write_text("plist")
    cli.parent.mkdir(parents=True)
    cli.symlink_to(app / "current/bin/pearl-miner")
    request = tmp_path / "cleanup.json"
    request.write_text(json.dumps({"home": str(home)}))
    calls = []
    monkeypatch.setattr(beta_cleanup.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(beta_cleanup.subprocess, "run",
                        lambda argv, **_kwargs: calls.append(argv) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(sys, "argv", ["beta_cleanup", "--request", str(request)])
    assert beta_cleanup.main() == 0
    assert not app.exists() and not logs.exists() and not plist.exists() and not cli.is_symlink()
    assert not request.exists()
    assert calls == [["launchctl", "bootout", f"gui/{os.getuid()}/{beta_cleanup.LABEL}"]]


def test_kernel_is_known_before_first_miner_event(monkeypatch):
    monkeypatch.setattr(beta_agent, "_initial_kernel", lambda cfg: "na")
    miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    assert miner.kernel == "na"


def test_initial_kernel_follows_device_class(monkeypatch):
    from pmk_miner import kernel as kernel_mod
    monkeypatch.setattr(kernel_mod, "detect_device_class", lambda: "Apple10")
    assert beta_agent._initial_kernel({"kernel": "auto"}) == "na"
    monkeypatch.setattr(kernel_mod, "detect_device_class", lambda: "Apple9")
    assert beta_agent._initial_kernel({"kernel": "auto"}) == "sg"


def test_not_working_until_gpu_work_completes_and_stop_clears_speed(monkeypatch):
    miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    now = [100.0]
    monkeypatch.setattr(beta_agent.time, "time", lambda: now[0])
    assert miner.working is False
    miner._parse_line('{"event":"pool_startup","effective_kernel":"na"}')
    assert miner.working is False
    miner._parse_line('{"event":"routine_telemetry","completed_ops":1000000000000}')
    now[0] = 110.0
    miner._parse_line('{"event":"routine_telemetry","completed_ops":21000000000000}')
    assert miner.working is True
    assert miner.tops_60s() == 2.0
    miner.stop()
    assert miner.working is False
    assert miner.tops_60s() == 0.0


def test_payload_reports_zero_speed_unless_mining(tmp_path, monkeypatch):
    agent = beta_agent.Agent.__new__(beta_agent.Agent)
    agent.cfg = {"version": "t", "label": "Mac"}
    agent.miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    agent.miner.tops_samples.append((beta_agent.time.time(), 5.0))
    agent.state = "paused"
    agent.started = beta_agent.time.monotonic()
    agent.effective_intensity = "full"
    agent.last_error = None
    agent.info = {}
    monkeypatch.setattr(beta_agent, "power_source", lambda: "ac")
    assert agent.payload()["tops"] is None
    assert agent.payload(state="mining")["tops"] == 5.0


def test_local_status_file_records_port_without_full_wallet_or_secret(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    agent.start_control_server()
    try:
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["local_port"] == agent.local_port
        assert state["wallet"] != agent.cfg["wallet"]
        assert agent.local_secret not in (tmp_path / "state.json").read_text()
        assert state["status"]["tops"] is None
    finally:
        agent.stop_control_server()


def test_installer_accepts_and_ignores_legacy_credentials(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uname = fake_bin / "uname"
    uname.write_text("#!/bin/sh\nprintf '%s\\n' Linux\n")
    uname.chmod(0o755)
    installer = Path(__file__).resolve().parents[2] / "scripts/release/install.sh"
    result = subprocess.run(
        ["sh", str(installer), "--wallet", "prl1" + "q" * 40,
         "--token", "legacy-token", "--install-id", "legacy-id"],
        env=dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert result.returncode == 1
    assert "Usage:" not in result.stdout
    assert "token is not valid" not in result.stdout
    assert "Apple Silicon Macs" in result.stdout
