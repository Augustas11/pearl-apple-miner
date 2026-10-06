#!/usr/bin/env python3
"""Local regtest e2e harness for pmk_miner.

The harness is intentionally local-only: pearld binds to loopback, the gateway
binds to loopback, and pearld is started with dns seeding disabled plus an inert
loopback-only outbound peer.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PEARLD = ROOT / "vendor/pearl/bin/pearld"
PYTHON = ROOT / ".venv/bin/python"
EVIDENCE_DIR = ROOT / "bench/evidence"
EVIDENCE_PREFIX = os.environ.get("PMK_EVIDENCE_PREFIX", "b3")
if not EVIDENCE_PREFIX.replace("_", "").isascii() or not EVIDENCE_PREFIX.replace("_", "").isalnum():
    raise ValueError("invalid evidence prefix")
MINING_ADDR = "rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"


@dataclass(slots=True)
class Proc:
    name: str
    process: subprocess.Popen[str]
    log_path: Path


class Rpc:
    def __init__(self, url: str, user: str, password: str) -> None:
        self.url = url
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.authorization = f"Basic {token}"
        self.request_id = 0

    def call(self, method: str, params: list[Any] | None = None, *, timeout: float = 15) -> Any:
        self.request_id += 1
        payload = json.dumps(
            {"jsonrpc": "1.0", "id": self.request_id, "method": method, "params": params or []}
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"content-type": "text/plain", "authorization": self.authorization},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            if not isinstance(body, dict) or body.get("error") is not None:
                raise RuntimeError("RPC envelope rejected")
            return body["result"]
        except Exception:
            raise RuntimeError("regtest RPC request failed") from None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def log(message: str) -> None:
    print(f"[pmk-e2e] {message}", flush=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def redact(value: str) -> str:
    return "<redacted>" if value else "<unset>"


def redact_text(text: str, secrets_to_hide: list[str] | tuple[str, ...]) -> str:
    redacted = text
    for secret in secrets_to_hide:
        if secret:
            redacted = redacted.replace(secret, redact(secret))
    return redacted


def secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def read_text_secure(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def start_proc(name: str, args: list[str], log_path: Path, *, env: dict[str, str]) -> Proc:
    secure_dir(log_path.parent)
    log_path.touch(mode=0o600, exist_ok=True)
    log_path.chmod(0o600)
    fh = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    fh.close()
    log(f"started {name} pid={process.pid} log={display_path(log_path)}")
    return Proc(name, process, log_path)


def signal_proc_group(proc: Proc, sig: signal.Signals) -> None:
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
        # The group may still contain gateway workers after its direct
        # launcher exits, so group cleanup must not depend on poll().
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
            signal_proc_group(proc, signal.SIGKILL)
            try:
                proc.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                lingering.append(f"{proc.name}:{proc.process.pid}")
    if lingering:
        raise RuntimeError(f"regtest child processes remained alive after cleanup: {', '.join(lingering)}")


def assert_alive(procs: list[Proc], *, allow_success: set[str] | None = None, secrets_to_hide: list[str] | tuple[str, ...] = ()) -> None:
    allow_success = allow_success or set()
    for proc in procs:
        rc = proc.process.poll()
        if rc is not None:
            if proc.name in allow_success and rc == 0:
                continue
            tail = ""
            if proc.log_path.exists():
                tail = "\n".join(proc.log_path.read_text(errors="replace").splitlines()[-80:])
            tail = redact_text(tail, secrets_to_hide)
            raise RuntimeError(f"{proc.name} exited with {rc}; tail:\n{tail}")


def wait_for_exit(proc: Proc, *, timeout_seconds: float, secrets_to_hide: list[str] | tuple[str, ...] = ()) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        rc = proc.process.poll()
        if rc is None:
            time.sleep(0.25)
            continue
        if rc == 0:
            return True
        tail = ""
        if proc.log_path.exists():
            tail = "\n".join(proc.log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"{proc.name} exited with {rc}; tail:\n{redact_text(tail, secrets_to_hide)}"
        )
    return False


def wait_for_rpc(rpc: Rpc, deadline: float) -> int:
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return int(rpc.call("getblockcount", timeout=5))
        except (OSError, urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"pearld RPC did not become ready: {last_error}")


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


def coinbase_outputs(block: dict[str, Any]) -> list[dict[str, Any]]:
    txs = block.get("rawtx") or block.get("tx") or block.get("transactions") or []
    if not txs:
        return []
    coinbase = txs[0]
    if not isinstance(coinbase, dict):
        return []
    vout = coinbase.get("vout") or []
    return vout if isinstance(vout, list) else []


def coinbase_pays_script(rpc: Rpc, height: int, expected_script: str) -> bool:
    block_hash = rpc.call("getblockhash", [height])
    block = rpc.call("getblock", [block_hash, 2])
    for vout in coinbase_outputs(block):
        script = vout.get("scriptPubKey", {})
        if str(script.get("hex", "")).lower() == expected_script.lower():
            return True
    return False


def summarize_blocks(rpc: Rpc, start_height: int, final_height: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for height in range(start_height + 1, final_height + 1):
        block_hash = rpc.call("getblockhash", [height])
        block = rpc.call("getblock", [block_hash, 1])
        blocks.append(
            {
                "height": height,
                "hash": block.get("hash", block_hash),
                "bits": block.get("bits"),
                "time": block.get("time"),
                "previousblockhash": block.get("previousblockhash"),
            }
        )
    return blocks


def parse_miner_summary(log_path: Path) -> dict[str, Any]:
    payout_verified = 0
    stopped_accepted: int | None = None
    overhead_events: list[dict[str, Any]] = []
    if not log_path.exists():
        return {
            "payout_verified": payout_verified,
            "stopped_accepted": stopped_accepted,
            "python_overhead_events": overhead_events,
        }
    for line in log_path.read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        name = event.get("event")
        if name == "payout_verified":
            payout_verified = max(payout_verified, int(event.get("accepted", 0)))
        elif name == "stopped":
            stopped_accepted = int(event.get("accepted", 0))
        elif name == "python_overhead":
            overhead_events.append(
                {
                    "job_id": event.get("job_id"),
                    "python_overhead_pct": event.get("python_overhead_pct"),
                    "wall_seconds": event.get("wall_seconds"),
                }
            )
    return {
        "payout_verified": payout_verified,
        "stopped_accepted": stopped_accepted,
        "python_overhead_events": overhead_events[-10:],
    }


def build_env(
    base: dict[str, str],
    env_file: Path,
    rpc_url: str,
    rpc_user: str,
    rpc_pass: str,
    *,
    gw_port: int,
    tap_log: Path,
) -> dict[str, str]:
    gateway_env = dict(base)
    gateway_env.update(
        {
            "PEARLD_RPC_URL": rpc_url,
            "PEARLD_RPC_USER": rpc_user,
            "PEARLD_RPC_PASSWORD": rpc_pass,
            "PEARLD_MINING_ADDRESS": MINING_ADDR,
            "MINER_RPC_TRANSPORT": "tcp",
            "MINER_RPC_HOST": "127.0.0.1",
            "MINER_RPC_PORT": str(gw_port),
            "RTAP_LOG": str(tap_log),
            "RTAP_CORRUPT_FIRST": "1",
        }
    )
    write_text(
        env_file,
        "\n".join(
            [
                f"PEARLD_RPC_URL={rpc_url}",
                f"PEARLD_RPC_USER={rpc_user}",
                f"PEARLD_RPC_PASSWORD={rpc_pass}",
                f"PEARLD_MINING_ADDRESS={MINING_ADDR}",
                "",
            ]
        ),
    )
    return gateway_env


def assert_no_log_leaks(paths: list[Path], secrets_to_hide: list[str] | tuple[str, ...]) -> None:
    leaked: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        for secret in secrets_to_hide:
            if secret and secret in text:
                leaked.append(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
                break
    if leaked:
        raise RuntimeError(f"raw credential or payout material leaked in logs: {', '.join(leaked)}")


def scrub_runtime_secrets(root: Path, secrets_to_hide: list[str] | tuple[str, ...]) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        redacted = redact_text(text, secrets_to_hide)
        if redacted != text:
            path.write_text(redacted, encoding="utf-8")
            path.chmod(0o600)


def _is_under(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def make_run_root(explicit: Path | None) -> Path:
    parent = Path(explicit).expanduser() if explicit is not None else Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    resolved_parent = parent.resolve(strict=False)
    resolved_root = ROOT.resolve(strict=True)
    if _is_under(resolved_parent, resolved_root):
        raise ValueError("regtest run parent must be outside the repository")
    if parent.exists() and not parent.is_dir():
        raise ValueError("regtest run parent must be a directory")
    parent.mkdir(parents=True, exist_ok=True)
    child = Path(tempfile.mkdtemp(prefix="pmk-regtest-", dir=resolved_parent))
    child.chmod(0o700)
    return child


def cleanup_run_root(run: Path, *, keep_run: bool, secrets_to_hide: list[str] | tuple[str, ...]) -> None:
    if keep_run:
        scrub_runtime_secrets(run, secrets_to_hide)
        return
    shutil.rmtree(run, ignore_errors=True)


def preflight_gateway_patch(copy_parent: Path) -> None:
    from pmk_miner.gateway_launcher import DEFAULT_PATCH, DEFAULT_SOURCE, patch_gateway_copy

    patched = patch_gateway_copy(DEFAULT_SOURCE, DEFAULT_PATCH, copy_parent)
    try:
        client = read_text_secure(patched.root_dir / "src/pearl_gateway/pearl_client.py")
        submission = read_text_secure(patched.root_dir / "src/pearl_gateway/submission_service.py")
        server = read_text_secure(patched.root_dir / "src/pearl_gateway/miner_rpc/server.py")
        sentinels = {
            "_redact_value(config.rpc_password)": client,
            "ProcessPoolExecutor": submission,
            "run_in_executor": submission,
            "asyncio.BoundedSemaphore": server,
            "proving_queue_full": server,
        }
        missing = [name for name, text in sentinels.items() if name not in text]
        forbidden = [
            name
            for name, text in {
                "rpc_password log": client,
                "plain_proof exception log": submission,
            }.items()
            if (
                name == "rpc_password log"
                and "rpc_password: {config.rpc_password}" in text
            )
            or (name == "plain_proof exception log" and "{plain_proof=}" in text)
        ]
        if missing or forbidden:
            raise RuntimeError(
                "patched gateway sentinel check failed: "
                f"missing={missing or []} forbidden={forbidden or []}"
            )
    finally:
        shutil.rmtree(patched.root_dir.parent, ignore_errors=True)


def make_config(
    path: Path,
    *,
    rpc_url: str,
    rpc_user: str,
    rpc_pass: str,
    gw_log: Path,
    env_file: Path,
    target: int,
    m: int = 128, n: int = 128, k: int = 4096, slots: int = 2,
    kernel: str = "auto",
) -> None:
    write_text(
        path,
        f"""m = {m}
n = {n}
k = {k}
slots = {slots}

[gateway]
env_file = "{env_file}"
log_file = "{gw_log}"

[node_rpc]
rpc_url = "{rpc_url}"
rpc_user = "{rpc_user}"
rpc_password = "{rpc_pass}"
mining_address = "{MINING_ADDR}"

[payout]
hrp = "rprl"
script = "{taproot_script_for_address(MINING_ADDR)}"

[run]
max_accepted = {target}
stop_after_cert_rejection = true
state_dir = "{path.parent / "state"}"
kernel = "{kernel}"
""",
    )


def gateway_cmd(run: Path) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "pmk_miner.gateway_launcher",
        "--source",
        str(ROOT / "vendor/pearl/miner/pearl-gateway"),
        "--copy-parent",
        str(run / "gateway-copy"),
        "--tap-script",
        str(ROOT / "scripts/pmk_regtest_gateway_tap.py"),
        "--",
        "start",
        "--debug",
    ]


def miner_cmd(config: Path, gw_port: int, kernel: str) -> list[str]:
    cmd = [
        str(PYTHON),
        "-m",
        "pmk_miner",
        "--mode",
        "solo",
        "--gateway",
        f"127.0.0.1:{gw_port}",
        "--config",
        str(config),
    ]
    if kernel != "auto":
        cmd.extend(["--kernel", kernel])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-new-blocks", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    for name, default in (("m", 128), ("n", 128), ("k", 4096), ("slots", 2)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--keep-run", action="store_true", help="preserve sanitized runtime files for debugging")
    parser.add_argument("--kernel", choices=("auto", "sg", "na"), default=os.environ.get("PMK_KERNEL", "auto"),
                        help="miner kernel override for cert-v3 runs")
    args = parser.parse_args()

    if not PEARLD.exists():
        raise SystemExit(f"missing {PEARLD}; run scripts/build_pearld.sh first")
    if not PYTHON.exists():
        raise SystemExit(f"missing {PYTHON}; run scripts/setup_env.sh first")
    if os.environ.get("PMK_GPU_LOCK_HELD") != "1":
        raise SystemExit("scripts/pmk_regtest_e2e.sh must hold the GPU lock before running")

    run = make_run_root(args.run_root)
    secure_dir(run)
    secure_dir(run / "logs")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    rpc_port = int(os.environ.get("PMK_REGTEST_RPC_PORT", free_port()))
    p2p_port = int(os.environ.get("PMK_REGTEST_P2P_PORT", free_port()))
    gw_port = int(os.environ.get("PMK_REGTEST_GATEWAY_PORT", free_port()))
    unused_connect_port = int(os.environ.get("PMK_REGTEST_UNUSED_CONNECT_PORT", free_port()))
    rpc_user = f"reg-{secrets.token_hex(4)}"
    rpc_pass = secrets.token_hex(16)
    rpc_url = f"http://127.0.0.1:{rpc_port}"
    env_file = run / "gateway.env"
    config_file = run / "pmk_regtest.toml"
    pearld_log = run / "logs/pearld.out"
    gateway_log = run / "logs/gateway.log"
    tap_log = run / "logs/tap.log"
    miner_log = run / "logs/miner.log"
    expected_script = taproot_script_for_address(MINING_ADDR)
    secrets_to_hide = [rpc_user, rpc_pass, MINING_ADDR]
    preflight_gateway_patch(run / "gateway-preflight")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'miner'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    gateway_env = build_env(
        env,
        env_file,
        rpc_url,
        rpc_user,
        rpc_pass,
        gw_port=gw_port,
        tap_log=tap_log,
    )
    make_config(
        config_file,
        rpc_url=rpc_url,
        rpc_user=rpc_user,
        rpc_pass=rpc_pass,
        gw_log=gateway_log,
        env_file=env_file,
        target=args.target_new_blocks,
        m=args.m, n=args.n, k=args.k, slots=args.slots,
        kernel=args.kernel,
    )

    procs: list[Proc] = []
    summary: dict[str, Any] = {
        "target_new_blocks": args.target_new_blocks,
        "run_root": str(run),
        "mining_address": redact(MINING_ADDR),
        "expected_coinbase_script": expected_script,
        "rpc_user": redact(rpc_user),
        "rpc_password": redact(rpc_pass),
        "kernel": args.kernel,
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
        procs.append(start_proc("pearld", pearld_args, pearld_log, env=env))
        rpc = Rpc(rpc_url, rpc_user, rpc_pass)
        start_height = wait_for_rpc(rpc, time.time() + 60)
        template = rpc.call("getblocktemplate", [{"rules": ["segwit"]}])
        if int(template.get("requiredcertversion", -1)) != 3:
            raise RuntimeError(f"expected requiredcertversion=3, got {template.get('requiredcertversion')}")
        summary["start_height"] = start_height
        summary["template_bits"] = template.get("bits")
        log(f"pearld ready height={start_height} bits={template.get('bits')} cert=3")

        procs.append(start_proc("gateway", gateway_cmd(run), gateway_log, env=gateway_env))
        time.sleep(2)
        assert_alive(procs, secrets_to_hide=secrets_to_hide)
        miner_proc = start_proc("miner", miner_cmd(config_file, gw_port, args.kernel), miner_log, env=env)
        procs.append(miner_proc)

        goal = start_height + args.target_new_blocks
        deadline = time.time() + args.timeout_seconds
        final_height = start_height
        while time.time() < deadline:
            time.sleep(5)
            final_height = int(rpc.call("getblockcount", timeout=8))
            log(f"height={final_height} goal={goal}")
            if final_height >= goal:
                break
            assert_alive(procs, allow_success={"miner"}, secrets_to_hide=secrets_to_hide)
        if final_height < goal:
            raise RuntimeError(f"timed out at height {final_height}; goal was {goal}")

        miner_graceful_stop = wait_for_exit(
            miner_proc,
            timeout_seconds=30,
            secrets_to_hide=secrets_to_hide,
        )
        miner_summary = parse_miner_summary(miner_log)
        if not miner_graceful_stop:
            raise RuntimeError("miner did not exit cleanly within 30s after chain goal")
        if miner_summary["payout_verified"] < args.target_new_blocks:
            raise RuntimeError(
                "miner did not verify all payout confirmations before exit: "
                f"{miner_summary['payout_verified']} < {args.target_new_blocks}"
            )
        if miner_summary["stopped_accepted"] is None or miner_summary["stopped_accepted"] < args.target_new_blocks:
            raise RuntimeError(
                "miner stopped log did not confirm target accepted count: "
                f"{miner_summary['stopped_accepted']} < {args.target_new_blocks}"
            )

        # A proof already handed off when the goal was reached may land during
        # graceful drain. Include every resulting block in the payout check.
        final_height = int(rpc.call("getblockcount", timeout=8))
        blocks = summarize_blocks(rpc, start_height, final_height)
        paid_heights = [height for height in range(start_height + 1, final_height + 1) if coinbase_pays_script(rpc, height, expected_script)]
        tap_text = tap_log.read_text(errors="replace") if tap_log.exists() else ""
        gateway_text = gateway_log.read_text(errors="replace") if gateway_log.exists() else ""
        corrupt_ok = (
            "NEGATIVE corrupt_proof verdict=rejected" in tap_text
            and "NEGATIVE corrupt_public_data verdict=rejected" in tap_text
        )
        summary.update(
            {
                "final_height": final_height,
                "accepted_blocks": final_height - start_height,
                "blocks": blocks,
                "coinbase_checked_heights": paid_heights,
                "coinbase_ok": len(paid_heights) == final_height - start_height,
                "corrupted_certificate_rejection": corrupt_ok,
                "gateway_log_rejection_seen": "Block rejected" in gateway_text or "rejected:" in gateway_text,
                "miner_graceful_stop": miner_graceful_stop,
                "miner_payout_verified": miner_summary["payout_verified"],
                "miner_stopped_accepted": miner_summary["stopped_accepted"],
                "python_overhead_events": miner_summary["python_overhead_events"],
                "tap_log": str(tap_log),
                "miner_log": str(miner_log),
                "gateway_log": str(gateway_log),
            }
        )
        assert_no_log_leaks([pearld_log, gateway_log, tap_log, miner_log], secrets_to_hide)
        if not summary["coinbase_ok"]:
            raise RuntimeError(f"coinbase payout mismatch; paid heights={paid_heights}")
        if not corrupt_ok:
            raise RuntimeError("tap log did not show both corrupted certificate rejections")

        write_text(EVIDENCE_DIR / f"{EVIDENCE_PREFIX}_regtest_e2e_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        log(
            "RESULT accepted_blocks="
            f"{summary['accepted_blocks']} coinbase_ok={summary['coinbase_ok']} "
            f"corrupted_certificate_rejection={summary['corrupted_certificate_rejection']} "
            f"miner_graceful_stop={summary['miner_graceful_stop']}"
        )
        return 0
    finally:
        write_text(EVIDENCE_DIR / f"{EVIDENCE_PREFIX}_regtest_e2e_last.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        stop_all(procs)
        if args.keep_run:
            cleanup_run_root(run, keep_run=True, secrets_to_hide=secrets_to_hide)
            log(f"kept sanitized run root {display_path(run)}")
        else:
            cleanup_run_root(run, keep_run=False, secrets_to_hide=secrets_to_hide)


if __name__ == "__main__":
    raise SystemExit(main())
