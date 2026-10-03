import asyncio
import ast
import base64
import importlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MINER = ROOT / "miner"
if str(MINER) not in sys.path:
    sys.path.insert(0, str(MINER))

from pmk_miner.gateway_launcher import (  # noqa: E402
    DEFAULT_PATCH,
    DEFAULT_SOURCE,
    build_gateway_command,
    patch_gateway_copy,
)


@dataclass
class FakeHeader:
    header_bytes: bytes = b"header-for-runtime-test"
    timestamp: int = 123456

    def serialize_without_proof_commitment(self):
        return self.header_bytes


@dataclass
class FakeTemplate:
    header: FakeHeader
    height: int = 7
    required_cert_version: int = 3

    def get_raw_transactions(self):
        return [b"coinbase"]


class FakePlainProof:
    def to_base64(self):
        return "proof64"


class FailingPlainProof:
    def to_base64(self):
        raise RuntimeError("secret-proof-material")


class ImmediateExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class FakeWorkCache:
    def __init__(self, template):
        self.current_template = template


class DelayedSubmissionService:
    def __init__(self, delay=0.05, fail=False):
        self.delay = delay
        self.fail = fail
        self.calls = 0

    async def submit_plain_proof(self, plain_proof, template, identity=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("synthetic submit failure")
        return {"status": "accepted"}

    def close(self):
        pass


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))

    def debug(self, message):
        self.messages.append(str(message))

    def trace(self, message):
        self.messages.append(str(message))

    def exception(self, message):
        self.messages.append(str(message))

    def text(self):
        return "\n".join(self.messages)


@pytest.fixture()
def patched_gateway(tmp_path):
    return patch_gateway_copy(DEFAULT_SOURCE, DEFAULT_PATCH, tmp_path)


def _read(patched, relative):
    return (patched.root_dir / relative).read_text()


def _import_from_patched(patched, module_name):
    for name in list(sys.modules):
        if name == "pearl_gateway" or name.startswith("pearl_gateway."):
            del sys.modules[name]
    sys.path.insert(0, str(patched.src_dir))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(patched.src_dir))


def _submit_request(header_bytes=b"header-for-runtime-test", request_id=1):
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "submitPlainProof",
            "params": {
                "plain_proof": base64.b64encode(b"proof").decode(),
                "mining_job": {
                    "incomplete_header_bytes": base64.b64encode(header_bytes).decode(),
                    "target": 1,
                    "cert_version": 3,
                    "submission_id": "0123456789abcdef0123456789abcdef",
                },
            },
            "id": request_id,
        }
    )


def test_patch_application_removes_credential_and_wallet_leaks(patched_gateway):
    client = _read(patched_gateway, "src/pearl_gateway/pearl_client.py")
    submission = _read(patched_gateway, "src/pearl_gateway/submission_service.py")
    server = _read(patched_gateway, "src/pearl_gateway/miner_rpc/server.py")

    assert "rpc_password: {config.rpc_password}" not in client
    assert "Using mining address: {self.mining_address}" not in client
    assert "_redact_value(config.rpc_password)" in client
    assert "_redact_value(self.mining_address)" in client
    assert "{plain_proof=}" not in server
    assert "{plain_proof=}, {template=}" not in submission
    assert "request params: {params}" not in server
    assert "{params=}" not in client


def test_patch_moves_proving_to_bounded_process_worker(patched_gateway):
    submission = _read(patched_gateway, "src/pearl_gateway/submission_service.py")
    server = _read(patched_gateway, "src/pearl_gateway/miner_rpc/server.py")

    assert "ProcessPoolExecutor" in submission
    assert "run_in_executor" in submission
    assert "plain_proof.to_base64()" in submission
    assert "PlainProof.from_base64(plain_proof_base64)" in submission
    assert "IncompleteBlockHeader.from_bytes(header_bytes)" in submission
    assert "PMK_GATEWAY_PROVING_TEST" not in submission
    assert "PMK_GATEWAY_PROVING_WORKERS" in submission
    assert "asyncio.BoundedSemaphore" in server
    assert "PMK_GATEWAY_PROVING_QUEUE_SIZE" in server
    assert "proving_queue_full" in server


def test_real_plainproof_and_header_primitive_roundtrip(patched_gateway):
    import pearl_mining as pm

    submission_mod = _import_from_patched(patched_gateway, "pearl_gateway.submission_service")
    leaf = b"\x00" * 1024
    merkle = pm.MerkleProof([leaf], [0], b"\x11" * 32, [], 1)
    matrix = pm.MatrixMerkleProof(merkle, [0])
    proof = pm.PlainProof(1, 1, 1, 128, matrix, matrix, None)
    proof_base64 = proof.to_base64()
    assert pm.PlainProof.from_base64(proof_base64).to_base64() == proof_base64

    header = pm.IncompleteBlockHeader(1, b"\x22" * 32, b"\x33" * 32, 123456, 0x1E010000)
    header_bytes = bytes(header.to_bytes())
    worker_template = submission_mod._WorkerBlockTemplate(header_bytes, [b"coinbase"], 3)

    assert worker_template.header.serialize_without_proof_commitment() == header_bytes
    assert worker_template.get_raw_transactions() == [b"coinbase"]
    assert int(worker_template.required_cert_version) == 3


def test_patch_applies_inside_miner_runtime_subtree():
    copy_parent = ROOT / "miner" / ".pmk_regtest" / "test-copy"
    try:
        patched = patch_gateway_copy(DEFAULT_SOURCE, DEFAULT_PATCH, copy_parent)

        assert "ProcessPoolExecutor" in _read(patched, "src/pearl_gateway/submission_service.py")
        assert "rpc_password: {config.rpc_password}" not in _read(
            patched, "src/pearl_gateway/pearl_client.py"
        )
    finally:
        shutil.rmtree(copy_parent, ignore_errors=True)


def test_submit_admission_returns_before_background_task_when_queue_has_room(patched_gateway):
    server_tree = ast.parse(_read(patched_gateway, "src/pearl_gateway/miner_rpc/server.py"))
    process_request = next(
        node
        for node in ast.walk(server_tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process_request"
    )
    source_segment = ast.get_source_segment(
        _read(patched_gateway, "src/pearl_gateway/miner_rpc/server.py"), process_request
    )

    assert "await self._try_admit_proving_job()" in source_segment
    assert "asyncio.create_task(" in source_segment
    assert '"status": "submitted"' in source_segment
    assert '"submission_id": identity["submission_id"]' in source_segment
    assert 'claimed_submission_id = _validate_submission_id' in source_segment
    assert source_segment.index("await self._try_admit_proving_job()") < source_segment.index("MiningJob.from_dict")
    assert source_segment.index("asyncio.create_task(") < source_segment.index(
        '"status": "submitted"'
    )


def test_submission_outcomes_are_structured_and_correlated(patched_gateway):
    submission = _read(patched_gateway, "src/pearl_gateway/submission_service.py")
    server = _read(patched_gateway, "src/pearl_gateway/miner_rpc/server.py")

    for event in (
        "plain_proof_received",
        "block_accepted",
        "block_rejected",
        "block_submission_error",
    ):
        assert event in submission
    for field in ("header_hash", "header_prefix", "template_height", "cert_version"):
        assert field in submission
    assert "block_submission_result" in server
    assert "submit_plain_proof_admitted" in server
    assert "stage=stage" in submission
    assert "phase=stage" in submission
    assert "classification=classification" in submission
    assert "error_type=_error_type(e)" in submission
    assert "logger.exception" not in submission


def test_submission_error_event_classifies_proving_without_error_text(
    patched_gateway, monkeypatch
):
    async def run_case():
        class FakePearlClient:
            async def submit_block(self, block_hex):
                return "accepted"

        capture = CaptureLogger()
        submission_mod.logger = capture
        service = submission_mod.SubmissionService(FakePearlClient(), debug_mode=False)
        service.proving_executor = ImmediateExecutor()

        result = await service.submit_plain_proof(FailingPlainProof(), FakeTemplate(FakeHeader()))
        service.close()
        text = capture.text()

        assert result == {"status": "error: proving-error"}
        assert '"event":"block_submission_error"' in text
        assert '"stage":"proving"' in text
        assert '"phase":"proving"' in text
        assert '"classification":"proving-error"' in text
        assert '"error_type":"RuntimeError"' in text
        assert "secret-proof-material" not in text

    submission_mod = _import_from_patched(patched_gateway, "pearl_gateway.submission_service")
    monkeypatch.setattr(submission_mod, "check_cert_version_eligible", lambda cert, proof: None)
    asyncio.run(run_case())


def test_submission_error_event_classifies_generic_node_exception_as_transport(
    patched_gateway, monkeypatch
):
    async def run_case():
        class FakePearlClient:
            async def submit_block(self, block_hex):
                raise ValueError("secret-node-rpc-reject")

        capture = CaptureLogger()
        submission_mod.logger = capture
        service = submission_mod.SubmissionService(FakePearlClient(), debug_mode=False)
        service.proving_executor = ImmediateExecutor()

        result = await service.submit_plain_proof(FakePlainProof(), FakeTemplate(FakeHeader()))
        service.close()
        text = capture.text()

        assert result == {"status": "error: transport"}
        assert '"event":"block_submission_error"' in text
        assert '"stage":"node"' in text
        assert '"phase":"node"' in text
        assert '"classification":"transport"' in text
        assert '"error_type":"ValueError"' in text
        assert "secret-node-rpc-reject" not in text

    submission_mod = _import_from_patched(patched_gateway, "pearl_gateway.submission_service")
    monkeypatch.setattr(submission_mod, "check_cert_version_eligible", lambda cert, proof: None)
    monkeypatch.setattr(submission_mod, "_generate_block_hex_in_worker", lambda *args: "00")
    asyncio.run(run_case())


def test_submission_explicit_node_rejection_classifies_consensus_invalid(
    patched_gateway, monkeypatch
):
    async def run_case():
        class FakePearlClient:
            async def submit_block(self, block_hex):
                return "rejected: bad-cb"

        capture = CaptureLogger()
        submission_mod.logger = capture
        service = submission_mod.SubmissionService(FakePearlClient(), debug_mode=False)
        service.proving_executor = ImmediateExecutor()

        result = await service.submit_plain_proof(FakePlainProof(), FakeTemplate(FakeHeader()))
        service.close()
        text = capture.text()

        assert result == {"status": "rejected: bad-cb"}
        assert '"event":"block_rejected"' in text
        assert '"classification":"consensus-invalid"' in text

    submission_mod = _import_from_patched(patched_gateway, "pearl_gateway.submission_service")
    monkeypatch.setattr(submission_mod, "check_cert_version_eligible", lambda cert, proof: None)
    monkeypatch.setattr(submission_mod, "_generate_block_hex_in_worker", lambda *args: "00")
    asyncio.run(run_case())


def test_bounded_admission_rejects_full_queue_and_releases_after_success(
    patched_gateway, monkeypatch
):
    async def run_case():
        server_mod.logger = CaptureLogger()
        template = FakeTemplate(FakeHeader())
        service = DelayedSubmissionService(delay=0.08)
        rpc = server_mod.MinerRpcServer(
            FakeWorkCache(template),
            service,
            SimpleNamespace(transport="tcp", port=0, host="127.0.0.1"),
        )
        client = SimpleNamespace(client_id=99)

        first = await rpc._process_request(_submit_request(request_id=1), client)
        second = await rpc._process_request(_submit_request(request_id=2), client)
        assert first["result"]["status"] == "submitted"
        assert first["result"]["submission_id"] == "0123456789abcdef0123456789abcdef"
        assert second["error"]["code"] == -32010
        assert '"event":"proving_queue_full"' in server_mod.logger.text()
        assert '"classification":"transport"' in server_mod.logger.text()

        await asyncio.sleep(0.12)
        third = await rpc._process_request(_submit_request(request_id=3), client)
        assert third["result"]["status"] == "submitted"
        assert third["result"]["submission_id"] == "0123456789abcdef0123456789abcdef"
        await asyncio.sleep(0.12)
        assert service.calls == 2

    monkeypatch.setenv("PMK_GATEWAY_PROVING_QUEUE_SIZE", "1")
    server_mod = _import_from_patched(patched_gateway, "pearl_gateway.miner_rpc.server")
    monkeypatch.setattr(server_mod.PlainProof, "from_base64", staticmethod(lambda _: "proof"))
    asyncio.run(run_case())


def test_bounded_admission_releases_after_background_exception(patched_gateway, monkeypatch):
    async def run_case():
        server_mod.logger = CaptureLogger()
        template = FakeTemplate(FakeHeader())
        rpc = server_mod.MinerRpcServer(
            FakeWorkCache(template),
            DelayedSubmissionService(delay=0.01, fail=True),
            SimpleNamespace(transport="tcp", port=0, host="127.0.0.1"),
        )
        client = SimpleNamespace(client_id=100)

        first = await rpc._process_request(_submit_request(request_id=1), client)
        await asyncio.sleep(0.05)
        second = await rpc._process_request(_submit_request(request_id=2), client)
        assert first["result"]["status"] == "submitted"
        assert first["result"]["submission_id"] == "0123456789abcdef0123456789abcdef"
        assert second["result"]["status"] == "submitted"
        assert second["result"]["submission_id"] == "0123456789abcdef0123456789abcdef"

    monkeypatch.setenv("PMK_GATEWAY_PROVING_QUEUE_SIZE", "1")
    server_mod = _import_from_patched(patched_gateway, "pearl_gateway.miner_rpc.server")
    monkeypatch.setattr(server_mod.PlainProof, "from_base64", staticmethod(lambda _: "proof"))
    asyncio.run(run_case())


def test_process_worker_does_not_block_event_loop(patched_gateway, monkeypatch, tmp_path):
    async def run_case():
        class FakePearlClient:
            async def submit_block(self, block_hex):
                return "accepted"

        service = submission_mod.SubmissionService(FakePearlClient(), debug_mode=False)
        template = helper.FakeTemplate(helper.FakeHeader())
        task = asyncio.create_task(service.submit_plain_proof(FakePlainProof(), template))
        ticks = 0
        start = time.monotonic()
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.02)
        result = await task
        service.close()

        assert result == {"status": "accepted"}
        assert time.monotonic() - start >= 0.18
        assert ticks >= 3

    helper_path = tmp_path / "pmk_gateway_worker_helper.py"
    helper_path.write_text(
        textwrap.dedent(
            """
            import time
            from dataclasses import dataclass


            @dataclass
            class FakeHeader:
                header_bytes: bytes = b"header-for-runtime-test"
                timestamp: int = 123456

                def serialize_without_proof_commitment(self):
                    return self.header_bytes


            @dataclass
            class FakeTemplate:
                header: FakeHeader
                height: int = 7
                required_cert_version: int = 3

                def get_raw_transactions(self):
                    return [b"coinbase"]

            def delayed_block_hex(
                plain_proof_base64,
                header_bytes,
                raw_transactions,
                cert_version,
                debug_mode,
            ):
                assert plain_proof_base64 == "proof64"
                assert header_bytes == b"header-for-runtime-test"
                assert raw_transactions == [b"coinbase"]
                assert cert_version == 3
                time.sleep(0.20)
                return "00"
            """
        )
    )
    existing_pythonpath = os.environ.get("PYTHONPATH")
    worker_pythonpath = os.pathsep.join([str(tmp_path), str(patched_gateway.src_dir)])
    if existing_pythonpath:
        worker_pythonpath = worker_pythonpath + os.pathsep + existing_pythonpath
    monkeypatch.setenv("PYTHONPATH", worker_pythonpath)
    sys.path.insert(0, str(tmp_path))
    helper = importlib.import_module("pmk_gateway_worker_helper")
    submission_mod = _import_from_patched(patched_gateway, "pearl_gateway.submission_service")
    monkeypatch.setattr(submission_mod, "check_cert_version_eligible", lambda cert, proof: None)
    monkeypatch.setattr(submission_mod, "_generate_block_hex_in_worker", helper.delayed_block_hex)
    submission_mod.logger = CaptureLogger()
    sys.path.insert(0, str(patched_gateway.src_dir))
    try:
        asyncio.run(run_case())
    finally:
        sys.path.remove(str(patched_gateway.src_dir))
        sys.path.remove(str(tmp_path))


def test_credentials_are_redacted_from_client_logs(patched_gateway):
    client_mod = _import_from_patched(patched_gateway, "pearl_gateway.pearl_client")
    capture = CaptureLogger()
    client_mod.logger = capture

    client_mod.PearlNodeClient(
        SimpleNamespace(
            rpc_url="http://urluser:supersecret@127.0.0.1:44107",
            rpc_user="rawuser",
            rpc_password="supersecret",
            mining_address="rprl1operatorsecretaddress",
        )
    )

    text = capture.text()
    assert "supersecret" not in text
    assert "rawuser" not in text
    assert "rprl1operatorsecretaddress" not in text
    assert "urluser:supersecret@" not in text


def test_rpc_error_paths_redact_credentials_and_drop_raw_exception_chains(patched_gateway, monkeypatch):
    client_mod = _import_from_patched(patched_gateway, "pearl_gateway.pearl_client")
    capture = CaptureLogger()
    client_mod.logger = capture

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", no_sleep)

    config = SimpleNamespace(
        rpc_url="http://urluser:supersecret@127.0.0.1:44107",
        rpc_user="rawuser",
        rpc_password="supersecret",
        mining_address="rprl1operatorsecretaddress",
    )
    probe_client = client_mod.PearlNodeClient(config)
    token_secret = probe_client._auth_token
    auth_secret = probe_client._auth_headers["Authorization"]
    secrets = ("rawuser", "supersecret", "urluser:supersecret@", token_secret, auth_secret)

    def assert_clean_exception(excinfo):
        rendered = "".join(traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb))
        for secret in secrets:
            assert secret not in str(excinfo.value)
            assert secret not in rendered
        assert excinfo.value.__cause__ is None

    class ConfiguredClient(client_mod.PearlNodeClient):
        pass

    async def expect_transport_error(session=None, create_session_error=None):
        client = ConfiguredClient(config)
        if create_session_error is not None:
            def raising_create_session():
                raise create_session_error
            client._create_session = raising_create_session
        else:
            client.session = session
        with pytest.raises(ConnectionError) as excinfo:
            await client._make_rpc_call("getblocktemplate", [])
        assert_clean_exception(excinfo)

    class PostRaisesSession:
        def post(self, *args, **kwargs):
            raise RuntimeError(f"post rawuser supersecret {token_secret} {auth_secret}")

    class FailingResponse:
        def __init__(self, *, status=200, enter_error=None, json_error=None, exit_error=None, payload=None):
            self.status = status
            self.enter_error = enter_error
            self.json_error = json_error
            self.exit_error = exit_error
            self.payload = payload if payload is not None else {"result": "ok"}

        async def __aenter__(self):
            if self.enter_error is not None:
                raise self.enter_error
            return self

        async def __aexit__(self, exc_type, exc, tb):
            if self.exit_error is not None:
                raise self.exit_error
            return False

        async def json(self):
            if self.json_error is not None:
                raise self.json_error
            return self.payload

    class ResponseSession:
        def __init__(self, response):
            self.response = response

        def post(self, *args, **kwargs):
            return self.response

    async def run_transport_cases():
        await expect_transport_error(
            create_session_error=RuntimeError(f"create rawuser supersecret {token_secret} {auth_secret}")
        )
        await expect_transport_error(PostRaisesSession())
        await expect_transport_error(
            ResponseSession(
                FailingResponse(enter_error=RuntimeError(f"enter rawuser supersecret {token_secret}"))
            )
        )
        await expect_transport_error(
            ResponseSession(
                FailingResponse(json_error=ValueError(f"json rawuser supersecret {token_secret}"))
            )
        )
        await expect_transport_error(
            ResponseSession(
                FailingResponse(exit_error=RuntimeError(f"exit rawuser supersecret {token_secret}"))
            )
        )
        await expect_transport_error(ResponseSession(FailingResponse(status=500)))

    async def run_rpc_error_case():
        client = client_mod.PearlNodeClient(config)
        client.session = ResponseSession(
            FailingResponse(payload={"error": f"bad rawuser supersecret {token_secret}", "result": None})
        )
        with pytest.raises(ValueError) as excinfo:
            await client._make_rpc_call("submitblock", [])
        assert_clean_exception(excinfo)

    asyncio.run(run_transport_cases())
    asyncio.run(run_rpc_error_case())
    text = capture.text()
    for secret in secrets:
        assert secret not in text


def test_generic_rpc_request_error_does_not_return_raw_exception_text(patched_gateway):
    server = _read(patched_gateway, "src/pearl_gateway/miner_rpc/server.py")
    assert 'self._jsonrpc_error(-32000, "Internal server error", request_id=None)' in server
    assert 'self._jsonrpc_error(-32000, "Internal server error", request_id)' in server
    assert 'self._jsonrpc_error(-32000, str(e)' not in server


def test_gateway_config_refuses_nonloopback_tcp_and_symlink_uds(patched_gateway, tmp_path):
    config_mod = _import_from_patched(patched_gateway, "pearl_gateway.config")

    with pytest.raises(ValueError, match="loopback"):
        config_mod.MinerRpcConfig(transport="tcp", host="0.0.0.0", port=8337)

    target = tmp_path / "real.sock"
    symlink = tmp_path / "link.sock"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        config_mod.MinerRpcConfig(transport="uds", socket_path=str(symlink))


def test_launcher_prefers_patched_copy_and_supports_tap(patched_gateway, tmp_path):
    tap = tmp_path / "tap.py"
    tap.write_text("print('tap')\n")

    cmd, env = build_gateway_command(
        patched_gateway,
        ["start", "--debug"],
        tap_script=tap,
        env={"PYTHONPATH": "oldpath"},
    )

    assert cmd == [sys.executable, str(tap.resolve())]
    path_parts = env["PYTHONPATH"].split(os.pathsep)
    assert path_parts[0] == str(tap.parent.resolve())
    assert path_parts[1] == str(patched_gateway.src_dir)
    assert path_parts[-1] == "oldpath"


def test_patched_cli_module_runs_version(patched_gateway):
    cmd, env = build_gateway_command(patched_gateway, ["version"], env={})
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)

    assert "PearlGateway v0.1.0" in result.stdout


@pytest.mark.parametrize('witness', [None, '12'*32])
@pytest.mark.parametrize('extra_transactions', [0, 1, 2, 3])
def test_gateway_coinbase_wire_authorizes_through_miner(patched_gateway, witness, extra_transactions):
    import hashlib
    import pearl_mining as pm
    from pmk_miner.transport import GatewayJob
    from pmk_miner.monitor import validate_payout_startup
    module = _import_from_patched(patched_gateway, 'pearl_gateway.comm.dataclasses')
    address = 'rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2'
    script = validate_payout_startup(address, 'rprl')
    coinbase = module.create_coinbase_transaction(
        height=1, coinbase_value=50_00000000, mining_address=address,
        coinbase_aux={}, default_witness_commitment=witness)
    txids = [coinbase.get_txid()] + [hashlib.sha256(bytes([i])).hexdigest() for i in range(extra_transactions)]
    template = module.BlockTemplate(
        header=module.PearlHeader(incomplete_header=pm.IncompleteBlockHeader(
            1, bytes(32), module.calculate_merkle_root(txids), 1, 0x177fd82e)),
        height=1, raw_transactions=[], coinbase_tx=coinbase,
        coinbase_merkle_branch=module.merkle_branch_for_index(txids, 0),
        required_cert_version=module.CertificateVersion(3))
    wire = module.MiningJob.from_template(template).to_dict()
    job = GatewayJob.from_gateway_dict(wire)
    job.authorize_coinbase(approved_script_hex=script)
    with pytest.raises(ValueError, match='approved'):
        job.authorize_coinbase(approved_script_hex='5120' + '00'*32)
