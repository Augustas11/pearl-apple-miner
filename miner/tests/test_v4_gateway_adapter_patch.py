# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
import asyncio
import base64
import compileall
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pmk_v4_gateway_adapter_check.py"
PATCH = ROOT / "miner" / "gateway_patches" / "0002-b9-v4-safe-async-proving.patch"
SOURCE = ROOT / "vendor" / "pearl-fp8" / "miner" / "pearl-gateway"
V4_PYTHON = ROOT / "bench" / "v4_emulation" / "pearl-build" / "gateway-python" / "bin" / "python"
SUBMISSION_ID = "0123456789abcdef0123456789abcdef"
HEADER = b"h" * 76
OTHER_HEADER = b"s" * 76


def _load_checker():
    spec = importlib.util.spec_from_file_location("pmk_v4_gateway_adapter_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_v4_case(case: str) -> str:
    if not V4_PYTHON.exists():
        pytest.fail(
            f"required V4 gateway Python is missing at {V4_PYTHON}; "
            "run the V4 gateway Python build/setup before adapter validation"
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "miner"), *(p for p in [env.get("PYTHONPATH", "")] if p)]
    )
    result = subprocess.run(
        [str(V4_PYTHON), str(Path(__file__).resolve()), "--v4-case", case],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"V4 adapter subprocess case {case!r} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _drop_gateway_modules():
    for name in list(sys.modules):
        if name == "pearl_gateway" or name.startswith("pearl_gateway."):
            del sys.modules[name]


def _import_from_patched(patched, module_name):
    _drop_gateway_modules()
    sys.path.insert(0, str(patched.src_dir))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(patched.src_dir))


def _require_v4_pm():
    import pearl_mining as pm

    if not hasattr(pm, "CERT_VERSION_PLAIN_FP8"):
        raise RuntimeError(
            "cert-v4 pearl_mining is required; run with the built V4 gateway Python "
            f"at {V4_PYTHON}"
        )
    return pm


def _dummy_plain_proof_v4(pm) -> object:
    leaf = b"\x00" * 1024
    merkle = pm.MerkleProof(
        total_leaves=1,
        leaf_data=[leaf],
        leaf_indices=[0],
        root=b"\x00" * 32,
        siblings=[],
    )
    matrix = pm.MatrixMerkleProof(proof=merkle, row_indices=[0])
    pattern = pm.AxisPattern([(4, pm.DimType.Fold), (4, pm.DimType.Blake)])
    ancestor_header = pm.BlockHeader(
        pm.IncompleteBlockHeader(
            version=0,
            prev_block=b"\x00" * 32,
            merkle_root=b"\x00" * 32,
            timestamp=0,
            nbits=0x1F00FFFF,
        ),
        b"\x00" * 32,
    )
    common = pm.CommonParams(
        2048,
        32,
        pm.Quant.Fp8E4M3Prequant,
        pm.Device.B200,
    )
    a = pm.OperandParams(32, pm.HashId.Blake3Chunk1024, pattern)
    b = pm.OperandParams(32, pm.HashId.Blake3Chunk1024, pattern)
    return pm.PlainProofV4(
        ancestor_header,
        common,
        a,
        b,
        values_a=matrix,
        values_b=matrix,
        scales_a=matrix,
        scales_b=matrix,
        ancestor_chain=[],
    )


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

    def text(self):
        return "\n".join(self.messages)


class FakeHeader:
    def __init__(self, payload=HEADER):
        self.payload = payload
        self.timestamp = 123456789

    def serialize_without_proof_commitment(self):
        return self.payload


class FakeTemplate:
    height = 99

    def __init__(self, cert_version: int, payload=HEADER):
        self.required_cert_version = cert_version
        self.header = FakeHeader(payload)

    def get_raw_transactions(self):
        return [b"coinbase"]


class FakeWorkCache:
    def __init__(self, template):
        self.current_template = template


class RecordingSubmissionService:
    def __init__(self, release: asyncio.Event | None = None):
        self.release = release
        self.calls = []

    async def submit_plain_proof(self, plain_proof, template, identity=None):
        self.calls.append((plain_proof, template, identity))
        if self.release is not None:
            await self.release.wait()
        return {"status": "accepted"}


def _request(mining_job, proof_b64, request_id=1):
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "submitPlainProof",
            "params": {"plain_proof": proof_b64, "mining_job": mining_job.to_dict()},
            "id": request_id,
        }
    )


def _with_patched_gateway(fn):
    checker = _load_checker()
    with tempfile.TemporaryDirectory(prefix="pmk-v4-adapter-test.") as tmp:
        patched = checker.patch_gateway_copy(SOURCE, PATCH, Path(tmp))
        try:
            checker.check_patched_gateway(patched.root_dir)
            return fn(patched)
        finally:
            patched.cleanup()
            _drop_gateway_modules()


def _plain_proof_v4_b64(pm) -> str:
    proof = _dummy_plain_proof_v4(pm)
    encoded = proof.to_base64()
    assert isinstance(pm.PlainProofV4.from_base64(encoded), pm.PlainProofV4)
    return encoded


def _case_compile():
    return _with_patched_gateway(
        lambda patched: compileall.compile_dir(patched.src_dir, quiet=1, force=True)
    )


def _case_roundtrip():
    pm = _require_v4_pm()

    def run(patched):
        data_mod = _import_from_patched(patched, "pearl_gateway.comm.dataclasses")
        ancestor = b"a" * 108
        coinbase = b"coinbase bytes"
        branch = [b"b" * 32, b"c" * 32]
        job = data_mod.MiningJob(
            incomplete_header_bytes=HEADER,
            target=123,
            cert_version=data_mod.CertificateVersion(pm.CERT_VERSION_PLAIN_FP8),
            ancestor_headers=[ancestor],
            coinbase_tx=coinbase,
            coinbase_merkle_branch=branch,
            coinbase_merkle_index=0,
            submission_id=SUBMISSION_ID,
        )
        restored = data_mod.MiningJob.from_dict(job.to_dict())
        assert restored.incomplete_header_bytes == HEADER
        assert int(restored.cert_version) == pm.CERT_VERSION_PLAIN_FP8
        assert restored.ancestor_headers == [ancestor]
        assert restored.coinbase_tx == coinbase
        assert restored.coinbase_merkle_branch == branch
        assert restored.coinbase_merkle_index == 0
        assert restored.submission_id == SUBMISSION_ID

    _with_patched_gateway(run)


def _case_receipt():
    pm = _require_v4_pm()
    proof_b64 = _plain_proof_v4_b64(pm)

    async def run_async(patched):
        os.environ["PMK_GATEWAY_PROVING_QUEUE_SIZE"] = "1"
        server_mod = _import_from_patched(patched, "pearl_gateway.miner_rpc.server")
        data_mod = importlib.import_module("pearl_gateway.comm.dataclasses")
        server_mod.logger = CaptureLogger()
        service = RecordingSubmissionService()
        rpc = server_mod.MinerRpcServer(
            FakeWorkCache(FakeTemplate(pm.CERT_VERSION_PLAIN_FP8)),
            service,
            SimpleNamespace(transport="tcp", port=0, host="127.0.0.1"),
        )
        job = data_mod.MiningJob(
            incomplete_header_bytes=HEADER,
            target=1,
            cert_version=data_mod.CertificateVersion(pm.CERT_VERSION_PLAIN_FP8),
            submission_id=SUBMISSION_ID,
        )
        response = await rpc._process_request(_request(job, proof_b64), SimpleNamespace(client_id=7))
        assert response["result"] == {"status": "submitted", "submission_id": SUBMISSION_ID}
        await asyncio.sleep(0)
        assert isinstance(service.calls[0][0], pm.PlainProofV4)
        assert service.calls[0][2]["submission_id"] == SUBMISSION_ID
        assert not rpc._proving_slots.locked()
        assert "submit_plain_proof_admitted" in server_mod.logger.text()
        assert "block_submission_result" in server_mod.logger.text()

    _with_patched_gateway(lambda patched: asyncio.run(run_async(patched)))


def _case_queue_full():
    pm = _require_v4_pm()
    proof_b64 = _plain_proof_v4_b64(pm)

    async def run_async(patched):
        os.environ["PMK_GATEWAY_PROVING_QUEUE_SIZE"] = "1"
        server_mod = _import_from_patched(patched, "pearl_gateway.miner_rpc.server")
        data_mod = importlib.import_module("pearl_gateway.comm.dataclasses")
        server_mod.logger = CaptureLogger()
        release = asyncio.Event()
        service = RecordingSubmissionService(release)
        rpc = server_mod.MinerRpcServer(
            FakeWorkCache(FakeTemplate(pm.CERT_VERSION_PLAIN_FP8)),
            service,
            SimpleNamespace(transport="tcp", port=0, host="127.0.0.1"),
        )
        job = data_mod.MiningJob(
            incomplete_header_bytes=HEADER,
            target=1,
            cert_version=data_mod.CertificateVersion(pm.CERT_VERSION_PLAIN_FP8),
            submission_id=SUBMISSION_ID,
        )
        first = await rpc._process_request(_request(job, proof_b64, 1), SimpleNamespace(client_id=8))
        assert first["result"]["submission_id"] == SUBMISSION_ID
        while not service.calls:
            await asyncio.sleep(0)
        assert rpc._proving_slots.locked()
        invalid_but_base64 = base64.b64encode(b"not-a-proof").decode()
        queue_full = await rpc._process_request(
            _request(job, invalid_but_base64, 2), SimpleNamespace(client_id=8)
        )
        assert queue_full["error"]["code"] == -32010
        assert queue_full["error"]["message"] == "proving queue full"
        assert len(service.calls) == 1
        assert "plain_proof_decode_error" not in server_mod.logger.text()
        release.set()
        for _ in range(20):
            if not rpc._proving_slots.locked():
                break
            await asyncio.sleep(0.01)
        assert not rpc._proving_slots.locked()
        again = await rpc._process_request(_request(job, proof_b64, 3), SimpleNamespace(client_id=8))
        assert again["result"]["submission_id"] == SUBMISSION_ID

    _with_patched_gateway(lambda patched: asyncio.run(run_async(patched)))


def _case_invalid_id():
    pm = _require_v4_pm()
    proof_b64 = _plain_proof_v4_b64(pm)

    async def run_async(patched):
        os.environ["PMK_GATEWAY_PROVING_QUEUE_SIZE"] = "1"
        server_mod = _import_from_patched(patched, "pearl_gateway.miner_rpc.server")
        data_mod = importlib.import_module("pearl_gateway.comm.dataclasses")
        service = RecordingSubmissionService()
        rpc = server_mod.MinerRpcServer(
            FakeWorkCache(FakeTemplate(pm.CERT_VERSION_PLAIN_FP8)),
            service,
            SimpleNamespace(transport="tcp", port=0, host="127.0.0.1"),
        )
        job = data_mod.MiningJob(
            incomplete_header_bytes=HEADER,
            target=1,
            cert_version=data_mod.CertificateVersion(pm.CERT_VERSION_PLAIN_FP8),
            submission_id="f" * 32,
        )
        payload = json.loads(_request(job, proof_b64))
        payload["params"]["mining_job"]["submission_id"] = "not-lower-hex-32"
        response = await rpc._process_request(json.dumps(payload), SimpleNamespace(client_id=9))
        assert response["error"]["code"] == -32602
        assert "submission_id" in response["error"]["message"]
        assert service.calls == []
        assert not rpc._proving_slots.locked()

    _with_patched_gateway(lambda patched: asyncio.run(run_async(patched)))


def _case_stale():
    pm = _require_v4_pm()
    proof_b64 = _plain_proof_v4_b64(pm)

    async def run_async(patched):
        os.environ["PMK_GATEWAY_PROVING_QUEUE_SIZE"] = "1"
        server_mod = _import_from_patched(patched, "pearl_gateway.miner_rpc.server")
        data_mod = importlib.import_module("pearl_gateway.comm.dataclasses")
        capture = CaptureLogger()
        server_mod.logger = capture
        service = RecordingSubmissionService()
        rpc = server_mod.MinerRpcServer(
            FakeWorkCache(FakeTemplate(pm.CERT_VERSION_PLAIN_FP8, OTHER_HEADER)),
            service,
            SimpleNamespace(transport="tcp", port=0, host="127.0.0.1"),
        )
        job = data_mod.MiningJob(
            incomplete_header_bytes=HEADER,
            target=1,
            cert_version=data_mod.CertificateVersion(pm.CERT_VERSION_PLAIN_FP8),
            submission_id=SUBMISSION_ID,
        )
        await rpc._proving_slots.acquire()
        await rpc.handle_submit_plain_proof(
            pm.PlainProofV4.from_base64(proof_b64),
            job,
            {"submission_id": SUBMISSION_ID},
        )
        assert service.calls == []
        assert not rpc._proving_slots.locked()
        assert '"event":"stale_plain_proof"' in capture.text()
        assert '"classification":"stale"' in capture.text()

    _with_patched_gateway(lambda patched: asyncio.run(run_async(patched)))


V4_CASES = {
    "compile": _case_compile,
    "roundtrip": _case_roundtrip,
    "receipt": _case_receipt,
    "queue_full": _case_queue_full,
    "invalid_id": _case_invalid_id,
    "stale": _case_stale,
}


def test_v4_gateway_adapter_patch_compiles_under_v4_gateway_python():
    assert "case_ok compile" in _run_v4_case("compile")


def test_mining_job_coinbase_ancestor_submission_roundtrip():
    assert "case_ok roundtrip" in _run_v4_case("roundtrip")


def test_receipt_echo_uses_real_plainproofv4_and_releases_slot():
    assert "case_ok receipt" in _run_v4_case("receipt")


def test_queue_full_rejects_before_decode_and_later_releases_slot():
    assert "case_ok queue_full" in _run_v4_case("queue_full")


def test_invalid_submission_id_rejected_before_proof_decode():
    assert "case_ok invalid_id" in _run_v4_case("invalid_id")


def test_stale_submission_is_classified_and_releases_slot():
    assert "case_ok stale" in _run_v4_case("stale")


def test_v4_gateway_adapter_checker_rejects_unpatched_source():
    checker = _load_checker()
    with pytest.raises(checker.AdapterCheckError):
        checker.check_patched_gateway(SOURCE)


def _main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "--v4-case" or argv[2] not in V4_CASES:
        print(f"usage: {argv[0]} --v4-case <{'|'.join(sorted(V4_CASES))}>", file=sys.stderr)
        return 2
    case = argv[2]
    V4_CASES[case]()
    print(f"case_ok {case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
