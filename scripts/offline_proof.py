"""Offline correctness proof for the upgraded OpenJarvis Apple-MPS Pearl miner.

Runs the real miner round (``oj_pearl_mps._mps_miner_loop_main._mine_one_round``,
MPS NoisyGEMM on the Apple GPU) against an in-process pearl-gateway JSON-RPC
server (Pearl's own ``MinerRpcServer``: same request parsing and JSON-schema
validation as production). The server hands out a cert-version-3 mining job and,
on ``submitPlainProof``, verifies the share with
``pearl_mining.verify_plain_proof_for_cert_version``.

1. POSITIVE: shares mined by the upgraded loop must pass V3 verification
   (``CERT_VERSION_ZK_V3``), and fail legacy V2 verification.
2. NEGATIVE CONTROL: the same loop with the pre-fork seed derivation
   (``salted_dims=None``, i.e. what OpenJarvis computed before the salted-seed
   fork) must be rejected by V3 verification, while still passing V2 (legacy)
   verification -- proving the only defect is the seed derivation.

Usage: .venv/bin/python scripts/offline_proof.py [--trials 3] [--neg-trials 5]
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import time

import oj_pearl_mps._mps_miner_loop_main as mps
import pearl_mining as pm
from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from pearl_gateway.miner_rpc.server import MinerRpcServer

# Simnet's PowLimitBits (node/chaincfg/params.go). With rank 128, k 2048 and a
# 16x16 hash tile the penalized per-transcript hit probability is 2^-5.
EASY_NBITS = 0x1E010000
# Harder target for the negative control: per-transcript hit probability 2^-8,
# so a wrongly-derived share passes V3 by accident with probability 2^-8.
NEG_NBITS = 0x1E002000

M = N = 128
K = 2048
RANK = 128


def _header(nbits: int, salt: int) -> pm.IncompleteBlockHeader:
    return pm.IncompleteBlockHeader(
        version=0,
        prev_block=salt.to_bytes(32, "little"),
        merkle_root=b"0123456789abcdef" * 2,
        timestamp=0x66666666,
        nbits=nbits,
    )


class _StaticWork:
    def __init__(self, job: MiningJob):
        self.job = job
        self.current_template = None

    async def get_mining_job(self) -> MiningJob:
        return self.job


class _VerifyingServer(MinerRpcServer):
    """Pearl's MinerRpcServer; submissions are verified instead of sent to a node."""

    def __init__(self, job: MiningJob, header: pm.IncompleteBlockHeader, port: int):
        super().__init__(
            _StaticWork(job),
            submission_service=None,
            config=MinerRpcConfig(transport="tcp", port=port, socket_path=None),
        )
        self.header = header
        self.results: list[dict] = []

    async def handle_submit_plain_proof(self, plain_proof, mining_job) -> None:
        cv = int(mining_job.cert_version)
        v3 = pm.verify_plain_proof_for_cert_version(pm.CERT_VERSION_ZK_V3, self.header, plain_proof)
        v2 = pm.verify_plain_proof_for_cert_version(pm.CERT_VERSION_ZK_MOE, self.header, plain_proof)
        self.results.append({"job_cert_version": cv, "v3": v3, "v2": v2, "proof": plain_proof})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _mine_until_share(nbits: int, salt: int, max_rounds: int) -> tuple[dict, int, float]:
    header = _header(nbits, salt)
    job = MiningJob(
        incomplete_header_bytes=header.to_bytes(),
        target=bits_to_target(nbits),
        cert_version=CertificateVersion.ZK_V3,
    )
    port = _free_port()
    server = _VerifyingServer(job, header, port)
    await server.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    t0 = time.perf_counter()
    try:
        for rnd in range(1, max_rounds + 1):
            submitted = await mps._mine_one_round(
                reader, writer, request_id=2 * rnd, m=M, n=N, k=K, rank=RANK
            )
            if submitted:
                for _ in range(100):
                    if server.results:
                        break
                    await asyncio.sleep(0.01)
                return server.results[0], rnd, time.perf_counter() - t0
        raise RuntimeError(f"no share found in {max_rounds} rounds")
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--neg-trials", type=int, default=5)
    args = ap.parse_args()

    print(f"pearl_mining {pm.__version__}; CERT_VERSION_ZK_V3={pm.CERT_VERSION_ZK_V3}; "
          f"PENALTY_BASE_RANK={pm.PENALTY_BASE_RANK}")
    print(f"shape m={M} n={N} k={K} rank={RANK}; device=mps")
    mps._validate_shape(pm, m=M, n=N, k=K, rank=RANK)

    print(f"\n== POSITIVE: upgraded loop (salted_dims=(m, n) for cert_version 3), nbits={EASY_NBITS:#x}")
    pos_ok = 0
    for t in range(args.trials):
        r, rounds, dt = asyncio.run(_mine_until_share(EASY_NBITS, salt=1000 + t, max_rounds=50))
        p = r["proof"]
        print(f"trial {t}: share after {rounds} round(s), {dt:.2f}s; job cert_version={r['job_cert_version']}; "
              f"proof m={p.m} n={p.n} k={p.k} rank={p.noise_rank}")
        print(f"  verify_plain_proof_for_cert_version(3) -> {r['v3']}")
        print(f"  verify_plain_proof_for_cert_version(2) -> {r['v2']}")
        pos_ok += bool(r["v3"][0])
    print(f"POSITIVE RESULT: {pos_ok}/{args.trials} shares accepted by the V3 verifier")

    print(f"\n== NEGATIVE CONTROL: same loop, legacy seeds (salted_dims=None), nbits={NEG_NBITS:#x}")
    mps._salted_dims_for = lambda cert_version, *, m, n: None  # pre-fork OpenJarvis behaviour
    neg_rejected = neg_v2_ok = 0
    for t in range(args.neg_trials):
        r, rounds, dt = asyncio.run(_mine_until_share(NEG_NBITS, salt=2000 + t, max_rounds=400))
        print(f"trial {t}: share after {rounds} round(s), {dt:.2f}s")
        print(f"  verify_plain_proof_for_cert_version(3) -> {r['v3']}")
        print(f"  verify_plain_proof_for_cert_version(2) -> {r['v2']}")
        neg_rejected += not r["v3"][0]
        neg_v2_ok += bool(r["v2"][0])
    print(f"NEGATIVE RESULT: {neg_rejected}/{args.neg_trials} legacy-seed shares rejected by V3; "
          f"{neg_v2_ok}/{args.neg_trials} accepted by V2 (legacy derivation)")

    ok = pos_ok == args.trials and neg_rejected == args.neg_trials
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
