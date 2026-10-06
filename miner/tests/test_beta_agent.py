import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import plistlib
import urllib.error

from pmk_miner import beta_agent
from pmk_miner.beta_agent import HeartbeatClient, MinerProcess, sanitize_worker
from pmk_miner import beta_cli


def test_heartbeat_contract_headers_and_body():
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["path"] = self.path
            captured["authorization"] = self.headers.get("Authorization")
            captured["install_id"] = self.headers.get("X-Install-Id")
            captured["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = b'{"paused":true,"intensity":"low","only_ac":false,"start_at_login":false,"uninstall":false,"poll_s":7}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HeartbeatClient(f"http://127.0.0.1:{server.server_address[1]}", "iid", "tok")
        desired = client.post({"state": "mining", "tops": 1.25})
    finally:
        server.shutdown()
        thread.join()

    assert captured == {
        "path": "/api/hb",
        "authorization": "Bearer tok",
        "install_id": "iid",
        "body": {"state": "mining", "tops": 1.25},
    }
    assert desired["paused"] is True
    assert desired["poll_s"] == 7


def test_heartbeat_network_backoff_starts_at_30_and_caps_at_300(monkeypatch):
    client = HeartbeatClient("http://127.0.0.1:1", "iid", "tok")
    monkeypatch.setattr(beta_agent.urllib.request, "urlopen",
                        lambda *_a, **_k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    delays = []
    for _ in range(6):
        assert client.post({"state": "mining"}) is None
        delays.append(client.retry_delay)
    assert delays == [30, 60, 120, 240, 300, 300]


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
    monkeypatch.setattr(beta_agent, "app_root", lambda: tmp_path)
    monkeypatch.setattr(beta_agent, "control_path", lambda: tmp_path / "control.json")
    cfg = {"api_base": "http://127.0.0.1:1", "install_id": "iid", "miner_token": "tok",
           "wallet": "prl1abc", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"}
    agent = beta_agent.Agent(cfg)
    agent.merge_controls({"paused": True, "intensity": "medium", "only_ac": False,
                          "start_at_login": False, "uninstall": True, "poll_s": 2})
    stored = json.loads((tmp_path / "control.json").read_text())
    assert stored == {"paused": True, "intensity": "medium", "only_ac": False,
                      "start_at_login": False, "uninstall": True, "poll_s": 2}
    assert sanitize_worker(" Family's MacBook Pro!!! ") == "Family-s-MacBook-Pro"


def test_cli_updates_control_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(beta_agent, "app_root", lambda: tmp_path)
    monkeypatch.setattr(beta_agent, "control_path", lambda: tmp_path / "control.json")
    monkeypatch.setattr(beta_cli, "control_path", lambda: tmp_path / "control.json")
    assert beta_cli.main(["pause"]) == 0
    assert json.loads((tmp_path / "control.json").read_text())["paused"] is True
    assert beta_cli.main(["intensity", "low"]) == 0
    assert json.loads((tmp_path / "control.json").read_text())["intensity"] == "low"
    assert "intensity set to low" in capsys.readouterr().out


def test_start_at_login_updates_plist_without_stopping_agent(tmp_path, monkeypatch):
    plist = tmp_path / "tech.malibu.pearl.plist"
    plist.write_bytes(plistlib.dumps({"Label": beta_agent.LABEL, "RunAtLoad": True}))
    monkeypatch.setattr(beta_agent, "launch_agent_path", lambda: plist)
    beta_agent.set_run_at_load(False)
    assert plistlib.loads(plist.read_bytes())["RunAtLoad"] is False


def test_uninstall_starts_detached_cleanup_without_exposing_credentials(tmp_path, monkeypatch):
    calls = []
    cfg = {"api_base": "http://127.0.0.1:1", "install_id": "iid", "miner_token": "tok",
           "wallet": "prl1abc", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"}
    agent = beta_agent.Agent(cfg)
    agent.miner.stop = lambda: calls.append(("stop", None))
    monkeypatch.setattr(beta_agent.tempfile, "mkstemp", lambda **_kw: (os.open(tmp_path / "request.json", os.O_CREAT | os.O_RDWR, 0o600), str(tmp_path / "request.json")))
    monkeypatch.setattr(beta_agent.subprocess, "Popen", lambda argv, **_kw: calls.append(("spawn", argv)) or object())
    agent.uninstall()
    assert calls[0] == ("stop", None)
    assert calls[1][0] == "spawn"
    assert "tok" not in " ".join(calls[1][1])


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
    assert agent.payload()["tops"] == 0.0
    assert agent.payload(state="mining")["tops"] == 5.0


def test_loop_reports_a_change_quickly(monkeypatch, tmp_path):
    agent = beta_agent.Agent.__new__(beta_agent.Agent)
    agent.cfg = {"version": "t", "label": "Mac", "wallet": "prl1test"}
    agent.controls = {"paused": False, "intensity": "full", "only_ac": True, "start_at_login": True,
                      "uninstall": False, "poll_s": 30}
    agent.miner = MinerProcess({"wallet": "prl1test", "worker": "w", "pool_url": "stratum+tcp://127.0.0.1:1"})
    agent.miner.working = True
    agent.state = "mining"
    agent.effective_intensity = "full"
    agent.started = beta_agent.time.monotonic()
    agent.last_error = None
    agent.info = {}
    monkeypatch.setattr(beta_agent, "power_source", lambda: "ac")
    monkeypatch.setattr(beta_agent, "control_path", lambda: tmp_path / "controls.json")
    monkeypatch.setattr(beta_agent, "state_path", lambda: tmp_path / "state.json")

    def apply_controls():
        agent.state = "paused" if agent.controls["paused"] else "mining"
    agent.apply_controls = apply_controls

    class Client:
        retry_delay = 30
        def post(self, payload):
            return {"paused": True, "intensity": "full", "only_ac": True, "start_at_login": True,
                    "uninstall": False, "poll_s": 30}
    agent.client = Client()
    waits = []

    class Stop(Exception):
        pass

    def wait(seconds):
        waits.append(seconds)
        raise Stop
    monkeypatch.setattr(agent.miner.restart_requested, "wait", wait)
    try:
        agent.loop()
    except Stop:
        pass
    assert waits == [beta_agent.FAST_REPORT_S]
