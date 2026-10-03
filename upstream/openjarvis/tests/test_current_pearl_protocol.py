"""Protocol/API tests for oj_pearl_mps against current Pearl (7039e66f, py-pearl-mining 0.3.1)."""

from __future__ import annotations

import base64

import pearl_mining
import pytest
from oj_pearl_mps._miner_loop_main import (
    _build_mining_config,
    _decode_mining_info,
    _mining_job_params,
)
from oj_pearl_mps._mps_miner_loop_main import (
    _mining_config_for_shape,
    _salted_dims_for,
    _validate_shape,
)
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.miner_rpc.schemas import validate_submit_plain_proof

HEADER = bytes(range(76))


def test_decode_mining_info_reads_gateway_job_including_cert_version():
    job = MiningJob(HEADER, 12345, cert_version=CertificateVersion.ZK_V3)
    assert _decode_mining_info(job.to_dict()) == (HEADER, 12345, 3)


def test_mining_job_params_match_gateway_serialization_and_schema():
    job = MiningJob(HEADER, 2**200, cert_version=CertificateVersion.ZK_V3)
    params = _mining_job_params(HEADER, 2**200, 3)
    assert params == job.to_dict()
    assert MiningJob.from_dict(params) == job
    validate_submit_plain_proof(
        {"plain_proof": base64.b64encode(b"proof").decode(), "mining_job": params}
    )


def test_submit_without_cert_version_is_rejected_by_current_gateway_schema():
    legacy = {"incomplete_header_bytes": base64.b64encode(HEADER).decode(), "target": 1}
    with pytest.raises(Exception, match="cert_version"):
        validate_submit_plain_proof({"plain_proof": "AA==", "mining_job": legacy})


@pytest.mark.parametrize(
    ("cert_version", "expected"),
    [(1, None), (2, None), (3, (128, 256))],
)
def test_salted_dims_follow_cert_version(cert_version, expected):
    assert _salted_dims_for(cert_version, m=128, n=256) == expected


def test_mining_configs_build_on_current_api():
    mps_cfg = _mining_config_for_shape(pearl_mining, k=2048, rank=128)
    cpu_cfg = _build_mining_config(pearl_mining, k=2048, rank=128)
    for cfg in (mps_cfg, cpu_cfg):
        assert cfg.rank == 128
        assert cfg.common_dim == 2048
        assert cfg.moe is None
        assert len(cfg.to_bytes()) > 0


def test_adjust_target_accepts_default_shape_and_rejects_old_rank_64():
    job = MiningJob(HEADER, 2**200, cert_version=CertificateVersion.ZK_V3)
    assert job.adjust_target(_mining_config_for_shape(pearl_mining, k=2048, rank=128)) > 2**200
    with pytest.raises(ValueError, match="below the minimum"):
        job.adjust_target(_mining_config_for_shape(pearl_mining, k=1024, rank=64))


def test_validate_shape():
    _validate_shape(pearl_mining, m=128, n=128, k=2048, rank=128)
    _validate_shape(pearl_mining, m=1024, n=1024, k=8192, rank=128)
    with pytest.raises(ValueError, match="PENALTY_BASE_RANK"):
        _validate_shape(pearl_mining, m=128, n=128, k=1024, rank=64)  # old OJ default
    with pytest.raises(ValueError, match="16\\*rank"):
        _validate_shape(pearl_mining, m=128, n=128, k=1024, rank=128)
