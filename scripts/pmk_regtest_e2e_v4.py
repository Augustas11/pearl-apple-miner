#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Local cert-v4 regtest harness for a native PlainProofV4 miner command."""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pmk_v4_paths import bundle_root

ROOT = bundle_root(__file__)
BUILD_ROOT = ROOT / "bench/v4_emulation/pearl-build"
PEARL_PIN = os.environ.get("PMK_PEARL_V4_PIN", "f696760b259500ecb608469ea3953aeabbe78948")
PEARLD = Path(os.environ.get("PMK_PEARLD_V4", BUILD_ROOT / "bin/pearld"))
PEARL_SRC = Path(os.environ.get("PMK_PEARL_V4_SRC", BUILD_ROOT / "src" / f"pearl-{PEARL_PIN}"))
GATEWAY_PYTHON = Path(os.environ.get("PMK_GATEWAY_PYTHON_V4", BUILD_ROOT / "gateway-python/bin/python"))
PATCHED_GATEWAY_SRC = Path(os.environ.get("PMK_GATEWAY_V4_PATCHED_SRC", BUILD_ROOT / "patched-gateway/pearl-gateway-v4/src"))
MINER_PYTHON = Path(os.environ.get("PMK_V4_MINER_PYTHON", ROOT / ".venv/bin/python"))
EVIDENCE_DIR = Path(os.environ.get("PMK_EVIDENCE_DIR", ROOT / "bench/evidence")).resolve()
EVIDENCE_PREFIX = os.environ.get("PMK_EVIDENCE_PREFIX", "b9_v4")
if not EVIDENCE_PREFIX.replace("_", "").isascii() or not EVIDENCE_PREFIX.replace("_", "").isalnum():
    raise ValueError("invalid evidence prefix")
MINING_ADDR = "rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"


@dataclass(slots=True)
class Proc:
    name: str
    process: subprocess.Popen[str]
    log_path: Path
    sentinel_write_fd: int


class Rpc:
    def __init__(self, url: str, user: str, password: str) -> None:
        self.url = url
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.authorization = f"Basic {token}"
        self.request_id = 0

    def call(self, method: str, params: list[Any] | None = None, *, timeout: float = 15) -> Any:
        self.request_id += 1
        payload = json.dumps({"jsonrpc": "1.0", "id": self.request_id, "method": method, "params": params or []}).encode()
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"content-type": "text/plain", "authorization": self.authorization},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
        if not isinstance(body, dict) or body.get("error") is not None:
            raise RuntimeError(f"RPC {method} rejected: {body.get('error') if isinstance(body, dict) else body}")
        return body["result"]


def log(message: str) -> None:
    print(f"[pmk-v4-e2e] {message}", flush=True)


def json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def completed_job_count(events: list[dict[str, Any]]) -> int:
    telemetry_counts = [
        int(event.get("counts", {}).get("completed", 0) or 0)
        for event in events
        if event.get("event") == "routine_telemetry" and isinstance(event.get("counts"), dict)
    ]
    return sum(telemetry_counts) if telemetry_counts else sum(
        1 for event in events if event.get("event") == "completed"
    )


def wait_for_miner_stopped_exit(
    miner_proc: Proc,
    *,
    transition_timeout: float = 30.0,
    exit_timeout: float = 5.0,
) -> dict[str, float]:
    """Wait for the flushed stop boundary, then enforce the exit budget."""
    transition_deadline = time.monotonic() + transition_timeout
    stopped_time: float | None = None
    while time.monotonic() < transition_deadline:
        stopped_times = [
            float(event.get("time", 0.0))
            for event in json_events(miner_proc.log_path)
            if event.get("event") == "stopped"
        ]
        if stopped_times:
            stopped_time = max(stopped_times)
            break
        if miner_proc.process.poll() is not None:
            break
        time.sleep(0.05)
    tail = miner_proc.log_path.read_text(errors="replace").splitlines()[-80:] if miner_proc.log_path.exists() else []
    if stopped_time is None:
        raise RuntimeError("v4 miner did not log stopped before exit; tail:\n" + "\n".join(tail))
    wait_started = time.time()
    exit_budget = max(0.0, exit_timeout - (wait_started - stopped_time))
    try:
        miner_proc.process.wait(timeout=exit_budget)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"v4 miner did not exit within {exit_timeout:g}s after stopped; tail:\n" + "\n".join(tail)) from exc
    exit_time = time.time()
    stopped_to_exit = max(0.0, exit_time - stopped_time)
    if stopped_to_exit > exit_timeout:
        raise RuntimeError(f"v4 miner exit exceeded {exit_timeout:g}s after stopped: {stopped_to_exit:.3f}s")
    return {
        "wait_started": wait_started,
        "exit_time": exit_time,
        "stopped_time": stopped_time,
        "stopped_to_exit_seconds": stopped_to_exit,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def start_proc(name: str, args: list[str], log_path: Path, *, env: dict[str, str]) -> Proc:
    secure_dir(log_path.parent)
    log_path.touch(mode=0o600, exist_ok=True)
    log_path.chmod(0o600)
    fh = log_path.open("w", encoding="utf-8")
    sentinel_read_fd, sentinel_write_fd = os.pipe()
    guard = ROOT / "scripts" / "studio_b4" / "process_guard.py"
    if not guard.is_file():
        os.close(sentinel_read_fd)
        os.close(sentinel_write_fd)
        fh.close()
        raise RuntimeError(f"missing child process guard: {guard}")
    command = [sys.executable, str(guard), str(sentinel_read_fd), "--", *args]
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            pass_fds=(sentinel_read_fd,),
        )
    except BaseException:
        os.close(sentinel_read_fd)
        os.close(sentinel_write_fd)
        fh.close()
        raise
    os.close(sentinel_read_fd)
    fh.close()
    log(f"started {name} pid={process.pid} log={log_path}")
    return Proc(name, process, log_path, sentinel_write_fd)


def redact_text(text: str, secrets_to_hide: list[str]) -> str:
    for secret in secrets_to_hide:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def assert_alive(procs: list[Proc], *, allow_success: set[str] | None = None, secrets_to_hide: list[str] | None = None) -> None:
    allow_success = allow_success or set()
    secrets_to_hide = secrets_to_hide or []
    for proc in procs:
        rc = proc.process.poll()
        if rc is None or (proc.name in allow_success and rc == 0):
            continue
        tail = ""
        if proc.log_path.exists():
            tail = "\n".join(proc.log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{proc.name} exited with {rc}; tail:\n{redact_text(tail, secrets_to_hide)}")


def signal_proc_group(proc: Proc, sig: signal.Signals) -> None:
    if proc.process.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
        return
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    try:
        proc.process.send_signal(sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_all(procs: list[Proc]) -> None:
    for proc in reversed(procs):
        signal_proc_group(proc, signal.SIGTERM)
    deadline = time.time() + 8
    for proc in reversed(procs):
        while proc.process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
    lingering: list[str] = []
    for proc in reversed(procs):
        signal_proc_group(proc, signal.SIGKILL)
        try:
            proc.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            lingering.append(f"{proc.name}:{proc.process.pid}")
        finally:
            try:
                os.close(proc.sentinel_write_fd)
            except OSError:
                pass
    if lingering:
        raise RuntimeError(f"regtest child processes remained alive after cleanup: {', '.join(lingering)}")


def wait_for_rpc(rpc: Rpc, deadline: float) -> int:
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return int(rpc.call("getblockcount", timeout=5))
        except (OSError, urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"pearld RPC did not become ready: {last_error}")


def make_run_root(explicit: Path | None) -> Path:
    parent = explicit or Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    parent = parent.expanduser().resolve(strict=False)
    if parent == ROOT or ROOT in parent.parents:
        raise ValueError("regtest run parent must be outside the repository")
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="pmk-v4-regtest-", dir=parent))
    run.chmod(0o700)
    return run


def gateway_env(base: dict[str, str], *, rpc_url: str, rpc_user: str, rpc_pass: str, gw_port: int, tap_log: Path) -> dict[str, str]:
    if not PATCHED_GATEWAY_SRC.exists():
        raise SystemExit(
            f"missing patched V4 gateway source {PATCHED_GATEWAY_SRC}; "
            "run scripts/pmk_build_gateway_python_v4.sh first"
        )
    paths = [
        str(PATCHED_GATEWAY_SRC),
        str(PEARL_SRC / "miner/miner-base/src"),
        str(PEARL_SRC / "miner/miner-utils/src"),
        str(ROOT / "miner"),
        base.get("PYTHONPATH", ""),
    ]
    env = dict(base)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(p for p in paths if p),
            "PEARLD_RPC_URL": rpc_url,
            "PEARLD_RPC_USER": rpc_user,
            "PEARLD_RPC_PASSWORD": rpc_pass,
            "PEARLD_MINING_ADDRESS": MINING_ADDR,
            "MINER_RPC_TRANSPORT": "tcp",
            "MINER_RPC_HOST": "127.0.0.1",
            "MINER_RPC_PORT": str(gw_port),
            "RTAP_LOG": str(tap_log),
            "RTAP_CORRUPT_FIRST": "1",
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "2"),
        }
    )
    return env


CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BECH32M_CONST = 0x2BC830A3


def bech32_polymod(values: list[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            if (top >> i) & 1:
                chk ^= generator[i]
    return chk


def bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def convertbits(data: list[int], frombits: int, tobits: int) -> bytes:
    acc = 0
    bits = 0
    ret: list[int] = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or value >> frombits:
            raise ValueError("invalid bech32 payload")
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if bits and ((acc << (tobits - bits)) & maxv):
        raise ValueError("non-zero bech32 padding")
    return bytes(ret)


def assert_v4_python(env: dict[str, str], log_path: Path) -> None:
    code = "import pearl_mining as p; assert int(p.CERT_VERSION_PLAIN_FP8)==4; print('pearl_mining_v4_ok')"
    proc = start_proc("v4-python-check", [str(GATEWAY_PYTHON), "-c", code], log_path, env=env)
    try:
        try:
            rc = proc.process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("v4 Python identity check timed out") from exc
        if rc != 0:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"v4 Python identity check failed rc={rc}: {tail}")
    finally:
        stop_all([proc])


def taproot_script_for_address(address: str) -> str:
    lower = address.lower()
    if lower != address:
        raise ValueError("mixed-case bech32 address")
    sep = lower.rfind("1")
    if sep < 1:
        raise ValueError("invalid bech32 address")
    hrp = lower[:sep]
    data = [CHARSET.index(c) for c in lower[sep + 1 :]]
    if bech32_polymod(bech32_hrp_expand(hrp) + data) != BECH32M_CONST:
        raise ValueError("invalid bech32m checksum")
    payload = data[:-6]
    version = payload[0]
    program = convertbits(payload[1:], 5, 8)
    if hrp != "rprl" or version != 1 or len(program) != 32:
        raise ValueError(f"expected rprl taproot address, got hrp={hrp!r} version={version}")
    return "5120" + program.hex()


def coinbase_pays_script(rpc: Rpc, height: int, expected_script: str) -> bool:
    block_hash = rpc.call("getblockhash", [height])
    block = rpc.call("getblock", [block_hash, 2])
    txs = block.get("rawtx") or block.get("tx") or block.get("transactions") or []
    if not txs or not isinstance(txs[0], dict):
        return False
    for vout in txs[0].get("vout") or []:
        if str((vout.get("scriptPubKey") or {}).get("hex", "")).lower() == expected_script.lower():
            return True
    return False


def make_miner_config(
    path: Path,
    *,
    rpc_url: str,
    rpc_user: str,
    rpc_pass: str,
    gateway_log: Path,
    target_blocks: int,
    m: int,
    n: int,
    k: int,
    slots: int,
) -> None:
    admission = os.environ.get("PMK_V4_G3_ADMISSION_FILE")
    v4_section = f'\n[v4]\nadmission_file = "{admission}"\n' if admission else ""
    write_text(
        path,
        f"""m = {m}
n = {n}
k = {k}
slots = {slots}
{v4_section}

[gateway]
log_file = "{gateway_log}"

[node_rpc]
rpc_url = "{rpc_url}"
rpc_user = "{rpc_user}"
rpc_password = "{rpc_pass}"
mining_address = "{MINING_ADDR}"

[payout]
hrp = "rprl"
script = "{taproot_script_for_address(MINING_ADDR)}"

[run]
max_accepted = {target_blocks}
state_dir = "{path.parent / 'state'}"
template_poll_seconds = 0.25
checkpoint_seconds = 2.0
telemetry_interval_seconds = 5.0
probe_interval_seconds = 21600
""",
    )


def default_miner_cmd(config: Path, gw_port: int) -> list[str]:
    return [
        str(MINER_PYTHON),
        "-m",
        "pmk_miner",
        "--mode",
        "solo",
        "--gateway",
        f"127.0.0.1:{gw_port}",
        "--config",
        str(config),
    ]


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-new-blocks", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--keep-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--m", type=int, default=int(os.environ.get("PMK_V4_REGTEST_M", "128")))
    parser.add_argument("--n", type=int, default=int(os.environ.get("PMK_V4_REGTEST_N", "128")))
    parser.add_argument("--k", type=int, default=int(os.environ.get("PMK_V4_REGTEST_K", "4096")))
    parser.add_argument("--slots", type=int, default=int(os.environ.get("PMK_V4_REGTEST_SLOTS", "2")))
    args = parser.parse_args()

    if not PEARL_SRC.exists():
        raise SystemExit(f"missing derived Pearl source {PEARL_SRC}; run scripts/pmk_build_pearld_v4.sh first")
    stamp = PEARL_SRC / ".pmk-v4-pin"
    if not stamp.exists():
        raise SystemExit(f"missing derived Pearl source stamp {stamp}; run scripts/pmk_build_pearld_v4.sh first")
    if stamp.read_text(encoding="utf-8").strip() != PEARL_PIN:
        raise SystemExit(f"derived Pearl source pin mismatch: {stamp.read_text(encoding='utf-8').strip()} != {PEARL_PIN}")
    if not PEARLD.exists():
        raise SystemExit(f"missing {PEARLD}; run scripts/pmk_build_pearld_v4.sh first")
    if not GATEWAY_PYTHON.exists():
        raise SystemExit(
            f"missing {GATEWAY_PYTHON}; run scripts/pmk_build_gateway_python_v4.sh first"
        )
    if not PATCHED_GATEWAY_SRC.exists():
        raise SystemExit(
            f"missing patched V4 gateway source {PATCHED_GATEWAY_SRC}; "
            "run scripts/pmk_build_gateway_python_v4.sh first"
        )
    if os.environ.get("PMK_GPU_LOCK_HELD") != "1" and not args.preflight_only:
        raise SystemExit("scripts/pmk_regtest_e2e_v4.sh must hold the GPU lock before running")

    run = make_run_root(args.run_root)
    secure_dir(run / "logs")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    rpc_port = int(os.environ.get("PMK_REGTEST_RPC_PORT", free_port()))
    p2p_port = int(os.environ.get("PMK_REGTEST_P2P_PORT", free_port()))
    gw_port = int(os.environ.get("PMK_REGTEST_GATEWAY_PORT", free_port()))
    unused_connect_port = int(os.environ.get("PMK_REGTEST_UNUSED_CONNECT_PORT", free_port()))
    rpc_user = f"reg-{secrets.token_hex(4)}"
    rpc_pass = secrets.token_hex(16)
    rpc_url = f"http://127.0.0.1:{rpc_port}"
    expected_script = taproot_script_for_address(MINING_ADDR)
    secrets_to_hide = [rpc_user, rpc_pass, MINING_ADDR]

    env = os.environ.copy()
    gw_env = gateway_env(env, rpc_url=rpc_url, rpc_user=rpc_user, rpc_pass=rpc_pass, gw_port=gw_port, tap_log=run / "logs/tap.log")
    assert_v4_python(gw_env, run / "logs/v4-python-check.log")
    config_file = run / "pmk_regtest_v4.toml"
    gateway_log = run / "logs/gateway.log"
    make_miner_config(
        config_file,
        rpc_url=rpc_url,
        rpc_user=rpc_user,
        rpc_pass=rpc_pass,
        gateway_log=gateway_log,
        target_blocks=args.target_new_blocks,
        m=args.m,
        n=args.n,
        k=args.k,
        slots=args.slots,
    )
    miner_cmd_text = os.environ.get("PMK_V4_MINER_CMD")
    miner_cmd = shlex.split(miner_cmd_text) if miner_cmd_text else default_miner_cmd(config_file, gw_port)
    if args.preflight_only:
        log(f"PREFLIGHT gateway_python={GATEWAY_PYTHON} miner_cmd={shlex.join(miner_cmd)} config={config_file}")
        write_text(EVIDENCE_DIR / f"{EVIDENCE_PREFIX}_regtest_e2e_preflight.json", json.dumps({
            "gateway_python": str(GATEWAY_PYTHON),
            "patched_gateway_src": str(PATCHED_GATEWAY_SRC),
            "miner_python": str(MINER_PYTHON),
            "miner_cmd": miner_cmd,
            "config_file": str(config_file),
            "shape": {"m": args.m, "n": args.n, "k": args.k, "slots": args.slots},
            "run_root": str(run),
        }, indent=2, sort_keys=True) + "\n")
        if not args.keep_run:
            shutil.rmtree(run, ignore_errors=True)
        return 0

    procs: list[Proc] = []
    summary: dict[str, Any] = {
        "target_new_blocks": args.target_new_blocks,
        "run_root": str(run),
        "expected_coinbase_script": expected_script,
        "miner_cmd": miner_cmd,
        "normal_miner_entry": miner_cmd[:3] == [str(MINER_PYTHON), "-m", "pmk_miner"],
        "shape": {"m": args.m, "n": args.n, "k": args.k, "slots": args.slots},
    }
    try:
        pearld_args = [
            str(PEARLD),
            "--regtest",
            f"--datadir={run / 'pearld'}",
            f"--logdir={run / 'logs/pearld'}",
            f"--rpcuser={rpc_user}",
            f"--rpcpass={rpc_pass}",
            f"--rpclisten=127.0.0.1:{rpc_port}",
            f"--listen=127.0.0.1:{p2p_port}",
            "--nodnsseed",
            f"--connect=127.0.0.1:{unused_connect_port}",
            "--notls",
            "--addrindex",
            "--txindex",
            f"--miningaddr={MINING_ADDR}",
            "--debuglevel=info",
        ]
        procs.append(start_proc("pearld", pearld_args, run / "logs/pearld.out", env=env))
        rpc = Rpc(rpc_url, rpc_user, rpc_pass)
        start_height = wait_for_rpc(rpc, time.time() + 60)
        template = rpc.call("getblocktemplate", [{"rules": ["segwit"]}])
        if int(template.get("requiredcertversion", -1)) != 4:
            raise RuntimeError(f"expected requiredcertversion=4, got {template.get('requiredcertversion')}")
        if not template.get("ancestorheaders"):
            raise RuntimeError("v4 getblocktemplate omitted ancestorheaders")
        summary.update({"start_height": start_height, "template_bits": template.get("bits"), "requiredcertversion": template.get("requiredcertversion")})
        log(f"pearld ready height={start_height} bits={template.get('bits')} cert=4")

        procs.append(start_proc("gateway", [str(GATEWAY_PYTHON), str(ROOT / "scripts/pmk_regtest_gateway_tap_v4.py")], gateway_log, env=gw_env))
        time.sleep(2)
        assert_alive(procs, secrets_to_hide=secrets_to_hide)
        miner_env = dict(gw_env)
        miner_env.update({
            "PMK_REGTEST_GATEWAY": f"127.0.0.1:{gw_port}",
            "PMK_REGTEST_TARGET_BLOCKS": str(args.target_new_blocks),
        })
        miner_proc = start_proc("miner", miner_cmd, run / "logs/miner.log", env=miner_env)
        procs.append(miner_proc)

        goal = start_height + args.target_new_blocks
        deadline = time.time() + args.timeout_seconds
        final_height = start_height
        while time.time() < deadline:
            time.sleep(0.5)
            final_height = int(rpc.call("getblockcount", timeout=8))
            log(f"height={final_height} goal={goal}")
            if final_height >= goal:
                break
            assert_alive(procs, allow_success={"miner"}, secrets_to_hide=secrets_to_hide)
        if final_height < goal:
            raise RuntimeError(f"timed out at height {final_height}; goal was {goal}")

        exit_metrics = wait_for_miner_stopped_exit(miner_proc)
        miner_wait_started = exit_metrics["wait_started"]
        miner_exit_time = exit_metrics["exit_time"]
        if miner_proc.process.returncode != 0:
            raise RuntimeError(f"v4 miner exited with {miner_proc.process.returncode}")
        miner_events = json_events(miner_proc.log_path)
        completed_jobs = completed_job_count(miner_events)

        paid_heights = [height for height in range(start_height + 1, final_height + 1) if coinbase_pays_script(rpc, height, expected_script)]
        tap_text = (run / "logs/tap.log").read_text(errors="replace") if (run / "logs/tap.log").exists() else ""
        corrupt_ok = (
            "NEGATIVE corrupt_proof verdict=rejected" in tap_text
            and "NEGATIVE corrupt_public_data verdict=rejected" in tap_text
            and "NEGATIVE corrupt_proof verdict=error:" not in tap_text
            and "NEGATIVE corrupt_public_data verdict=error:" not in tap_text
        )
        summary.update(
            {
                "final_height": final_height,
                "accepted_blocks": final_height - start_height,
                "coinbase_checked_heights": paid_heights,
                "coinbase_ok": len(paid_heights) == final_height - start_height,
                "corrupted_v4_certificate_rejection": corrupt_ok,
                "tap_log": str(run / "logs/tap.log"),
                "gateway_log": str(run / "logs/gateway.log"),
                "miner_log": str(run / "logs/miner.log"),
                "miner_wait_seconds": max(0.0, miner_exit_time - miner_wait_started),
                "miner_stopped_to_exit_seconds": exit_metrics["stopped_to_exit_seconds"],
                "completed_jobs": completed_jobs,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        summary["accepted_blocks_per_second"] = summary["accepted_blocks"] / max(
            summary["elapsed_seconds"], 1e-9
        )
        summary["completed_jobs_per_second"] = completed_jobs / max(
            summary["elapsed_seconds"], 1e-9
        )
        if not summary["normal_miner_entry"]:
            raise RuntimeError("v4 regtest did not use the normal pmk_miner entry point")
        if not summary["coinbase_ok"]:
            raise RuntimeError(f"coinbase payout mismatch; paid heights={paid_heights}")
        if not corrupt_ok:
            raise RuntimeError("tap log did not show both corrupted v4 certificate rejections")
        write_text(EVIDENCE_DIR / f"{EVIDENCE_PREFIX}_regtest_e2e_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        log(
            f"RESULT accepted_blocks={summary['accepted_blocks']} coinbase_ok={summary['coinbase_ok']} "
            f"corrupted_v4_certificate_rejection={summary['corrupted_v4_certificate_rejection']}"
        )
        return 0
    finally:
        write_text(EVIDENCE_DIR / f"{EVIDENCE_PREFIX}_regtest_e2e_last.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        stop_all(procs)
        if not args.keep_run:
            shutil.rmtree(run, ignore_errors=True)
        else:
            log(f"kept run root {run}")


if __name__ == "__main__":
    raise SystemExit(main())
