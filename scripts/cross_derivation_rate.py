"""Cross-derivation acceptance rate: mine N shares with one seed derivation, verify under both V3 and V2.

Usage: .venv/bin/python scripts/cross_derivation_rate.py <nbits-hex> <v3|legacy> <N>
Expected: same-derivation pass rate 100%; cross-derivation rate ~= per-transcript hit probability
(2^-5 at nbits 0x1e010000 with rank 128, k 2048, 16x16 hash tile).
"""

import sys

import pearl_mining as pm
import torch
from loguru import logger
from miner_base.block_submission import create_proof
from miner_base.commitment_hash import CommitmentHasher
from miner_base.noise_generation import NoiseGenerator
from miner_base.noisy_gemm import NoisyGemm
from oj_pearl_mps._mps_miner_loop_main import MpsNoisyGemmAdapter, _mining_config_for_shape
from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob

logger.remove()  # silence miner-base INFO logs
nbits = int(sys.argv[1], 16)
salted = sys.argv[2] == "v3"
N = int(sys.argv[3])
m = n = 128
k = 2048
r = 128
dev = torch.device("mps")
cfg = _mining_config_for_shape(pm, k=k, rank=r)
stats = {"v3": 0, "v2": 0, "n": 0}
i = 0
while stats["n"] < N:
    i += 1
    h = pm.IncompleteBlockHeader(
        version=0,
        prev_block=i.to_bytes(32, "little"),
        merkle_root=b"0123456789abcdef" * 2,
        timestamp=0x66666666,
        nbits=nbits,
    )
    hb = h.to_bytes()
    tgt = MiningJob(hb, bits_to_target(nbits), CertificateVersion.ZK_V3).adjust_target(cfg)
    A = torch.randint(-64, 64, (m, k), dtype=torch.int8)
    B = torch.randint(-64, 64, (k, n), dtype=torch.int8)
    ch = CommitmentHasher.commitment_hash(A, B, hb, cfg, salted_dims=(m, n) if salted else None)
    E = NoiseGenerator(noise_rank=r, noise_range=128).generate_noise_metrices(
        key_A=ch.noise_seed_A, key_B=ch.noise_seed_B, A_rows=m, common_dim=k, B_cols=n
    )
    g = MpsNoisyGemmAdapter.build(
        NoisyGemm,
        noise_range=128,
        noise_rank=r,
        hash_tile_h=16,
        hash_tile_w=16,
        matmul_tile_h=r,
        matmul_tile_w=r,
    )
    _, f = g.noisy_gemm(A.to(dev), B.to(dev), *[e.to(dev) for e in E], commitment_hash=ch, pow_target=tgt)
    if not f:
        continue
    p = create_proof(g.get_opened_block_info(), hb)
    stats["n"] += 1
    stats["v3"] += pm.verify_plain_proof_for_cert_version(3, h, p)[0]
    stats["v2"] += pm.verify_plain_proof_for_cert_version(2, h, p)[0]
print(
    f"nbits={nbits:#x} mined_with={'salted(v3)' if salted else 'legacy'} shares={stats['n']} pass_v3={stats['v3']} pass_v2={stats['v2']} rounds={i}"
)
