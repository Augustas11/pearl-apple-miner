"""Experimental Apple-MPS Pearl miner-loop subprocess.

This is the first Apple-GPU path for Pearl mining. It is intentionally
separate from ``_miner_loop_main`` because the CPU path calls Pearl's pure-Rust
``pearl_mining.mine()``, while this path exercises upstream ``miner-base``:

- generate random int7 A/B matrices
- compute Pearl commitment hashes and deterministic noise
- run ``miner_base.NoisyGemm`` on ``torch.device("mps")``
- convert an opened block to ``PlainProof`` and submit it to pearl-gateway

The current implementation still does transcript hashing and proof construction
on CPU because upstream ``miner-base`` uses NumPy/BLAKE3 for those pieces. That
makes this a correctness-first MPS path, not a final high-performance Metal
kernel.

Modified for pearl-metal-miner (see NOTICE / docs/OPENJARVIS_UPGRADE.md):
ported to py-pearl-mining 0.3.1 + miner-base/pearl-gateway at Pearl 7039e66f.
The NoisyGEMM adapter and hot loop are unchanged; the changes are API/protocol
drift only (MiningConfiguration ``moe``, ``cert_version`` in the mining job,
V3 salted noise seeds, rank-penalty minimum rank 128 and k >= 16 * rank).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from ._miner_loop_main import (
    _CONNECT_RETRY_SECONDS,
    _CONNECT_TIMEOUT_SECONDS,
    _FAILURE_BACKOFF_SECONDS,
    _decode_mining_info,
    _make_request,
    _mining_job_params,
    _open_gateway_connection,
    _read_response,
    _send_request,
)

logger = logging.getLogger("oj_pearl_mps.mps_miner_loop")


class MpsNoisyGemmAdapter:
    """Device-safe wrapper around upstream ``miner_base.NoisyGemm``.

    Upstream NoisyGemm is written as a CPU reference and creates some
    intermediate tensors on CPU. This adapter overrides the two places that
    need device-awareness so MPS inputs stay on MPS for matmuls. Transcript
    hashing explicitly copies the small hash tiles back to CPU because the
    upstream ``InnerHasher`` path uses NumPy.
    """

    @staticmethod
    def build(noisy_gemm_cls: type, **kwargs: Any) -> Any:
        import torch
        from miner_base.noisy_gemm import Transcript

        class _MpsNoisyGemm(noisy_gemm_cls):  # type: ignore[misc, valid-type]
            def _accumulate_transcripts(self, transcripts, reduction_count, C_block):  # type: ignore[no-untyped-def]
                return super()._accumulate_transcripts(
                    transcripts,
                    reduction_count,
                    C_block.detach().cpu(),
                )

            def _process_output_tile(self, A, B, k, i, i_max, j, j_max):  # type: ignore[no-untyped-def]
                block_h = i_max - i
                block_w = j_max - j
                hash_tile_h = self.inner_hash.tile_h
                hash_tile_w = self.inner_hash.tile_w
                num_hash_tiles_h = block_h // hash_tile_h
                num_hash_tiles_w = block_w // hash_tile_w
                transcripts = [
                    [self.__class__.Transcript() for _ in range(num_hash_tiles_w)]
                    for _ in range(num_hash_tiles_h)
                ]
                reduction_count = 0
                has_full_tiles = False
                C_block = torch.zeros(
                    (block_h, block_w),
                    dtype=torch.int32,
                    device=A.device,
                )
                for p in range(0, k, self.noise_rank):
                    p_max = min(p + self.noise_rank, k)
                    A_tile = A[i:i_max, p:p_max]
                    B_tile = B[p:p_max, j:j_max]
                    C_tile = torch.matmul(
                        A_tile.to(torch.int32),
                        B_tile.to(torch.int32),
                    )
                    C_block += C_tile
                    is_full_tile = (
                        block_h >= hash_tile_h
                        and block_w >= hash_tile_w
                        and p_max - p == self.noise_rank
                    )
                    if is_full_tile:
                        self._accumulate_transcripts(
                            transcripts=transcripts,
                            reduction_count=reduction_count,
                            C_block=C_block,
                        )
                        reduction_count += 1
                        has_full_tiles = True
                return C_block, has_full_tiles, transcripts

            def _tiled_matmul(self, A, B, pow_key, pow_target):  # type: ignore[no-untyped-def]
                assert A.shape[1] == B.shape[0]
                m, k = A.shape
                n = B.shape[1]
                C = torch.zeros((m, n), dtype=torch.int32, device=A.device)
                hash_tile_h = self.inner_hash.tile_h
                hash_tile_w = self.inner_hash.tile_w
                if (
                    m < hash_tile_h
                    or n < hash_tile_w
                    or self.noise_rank < min(hash_tile_h, hash_tile_w)
                ):
                    raise ValueError(
                        f"{m=}, {n=}, {hash_tile_h=}, {hash_tile_w=}, "
                        f"{self.noise_rank=}, matmul tile should be larger "
                        "than hash tile and noise rank"
                    )
                found_block = False
                for i in range(0, m, self.noise_rank):
                    i_max = min(i + self.noise_rank, m)
                    for j in range(0, n, self.noise_rank):
                        j_max = min(j + self.noise_rank, n)
                        C_block, has_full_tiles, transcripts = (
                            self._process_output_tile(A, B, k, i, i_max, j, j_max)
                        )
                        if not found_block and has_full_tiles:
                            found_block = self._check_tile_transcripts(
                                transcripts, i, j, pow_key, pow_target
                            )
                        C[i:i_max, j:j_max] = C_block
                return C, found_block

        _MpsNoisyGemm.Transcript = Transcript
        return _MpsNoisyGemm(**kwargs)


def _mining_config_for_shape(pearl_mining_module: Any, *, k: int, rank: int) -> Any:
    return pearl_mining_module.MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=pearl_mining_module.MMAType.Int7xInt7ToInt32,
        rows_pattern=pearl_mining_module.PeriodicPattern.from_list(list(range(16))),
        cols_pattern=pearl_mining_module.PeriodicPattern.from_list(list(range(16))),
        moe=None,
    )


def _salted_dims_for(cert_version: int, *, m: int, n: int) -> tuple[int, int] | None:
    """Seed derivation for a job: ``(m, n)`` salts the Merkle roots (cert V3+).

    Mirrors Pearl's reference miner (vllm_miner/gemm_operators.py):
    ``salted_dims=(m, n) if mining_job.cert_version.uses_salted_seeds else None``.
    """
    from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion

    return (m, n) if CertificateVersion(cert_version).uses_salted_seeds else None


def _validate_shape(pearl_mining_module: Any, *, m: int, n: int, k: int, rank: int) -> None:
    """Reject shapes current consensus can never accept, before doing any work.

    Rules from zk-pow/src/api/sanity_checks.rs (public_params_sanity_check and
    the rank-penalty rule; pearl-gateway's MiningJob.adjust_target enforces the
    rank minimum too). The adapter additionally needs m, n >= rank (matmul tile).
    """
    base_rank = int(pearl_mining_module.PENALTY_BASE_RANK)
    problems = []
    if rank < base_rank:
        problems.append(f"rank={rank} < PENALTY_BASE_RANK={base_rank}")
    if rank & (rank - 1) or not 32 <= rank <= 1024:
        problems.append(f"rank={rank} must be a power of two in [32, 1024]")
    if k % 64 or k < 1024 or k < 16 * rank or k > 4 * rank * rank or k > 1 << 16:
        problems.append(f"k={k} must be a multiple of 64 with max(1024, 16*rank) <= k <= min(4*rank^2, 65536)")
    if m < rank or n < rank:
        problems.append(f"m={m}, n={n} must be >= rank={rank} (NoisyGemm matmul tile)")
    if problems:
        raise ValueError("unminable shape: " + "; ".join(problems))


async def _mine_one_round(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    request_id: int,
    m: int,
    n: int,
    k: int,
    rank: int,
) -> bool:
    import pearl_mining
    import torch
    from miner_base.block_submission import create_proof
    from miner_base.commitment_hash import CommitmentHasher
    from miner_base.noise_generation import NoiseGenerator
    from miner_base.noisy_gemm import NoisyGemm
    from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
    from pearl_gateway.comm.dataclasses import MiningJob

    await _send_request(writer, _make_request("getMiningInfo", {}, request_id))
    info_response = await _read_response(reader)
    if "error" in info_response:
        logger.warning("getMiningInfo error: %s", info_response["error"])
        return False

    header_bytes, target, cert_version = _decode_mining_info(info_response["result"])
    mining_config = _mining_config_for_shape(pearl_mining, k=k, rank=rank)
    mining_job = MiningJob(
        incomplete_header_bytes=header_bytes,
        target=target,
        cert_version=CertificateVersion(cert_version),
    )
    adjusted_target = mining_job.adjust_target(mining_config)

    # Matrices stay on CPU for commitment/proof construction and are copied to
    # MPS for NoisyGEMM. Values are int7-compatible: [-64, 63].
    device = torch.device("mps")
    A_cpu = torch.randint(-64, 64, (m, k), dtype=torch.int8)
    B_cpu = torch.randint(-64, 64, (k, n), dtype=torch.int8)
    commitment_hash = CommitmentHasher.commitment_hash(
        A_cpu,
        B_cpu,
        header_bytes,
        mining_config,
        salted_dims=_salted_dims_for(cert_version, m=m, n=n),
    )
    E_AL, E_AR, E_BL, E_BR = NoiseGenerator(
        noise_rank=rank,
        noise_range=128,
    ).generate_noise_metrices(
        key_A=commitment_hash.noise_seed_A,
        key_B=commitment_hash.noise_seed_B,
        A_rows=m,
        common_dim=k,
        B_cols=n,
    )

    gemm = MpsNoisyGemmAdapter.build(
        NoisyGemm,
        noise_range=128,
        noise_rank=rank,
        hash_tile_h=16,
        hash_tile_w=16,
        matmul_tile_h=rank,
        matmul_tile_w=rank,
    )
    _, found = gemm.noisy_gemm(
        A_cpu.to(device),
        B_cpu.to(device),
        E_AL.to(device),
        E_AR.to(device),
        E_BL.to(device),
        E_BR.to(device),
        commitment_hash=commitment_hash,
        pow_target=adjusted_target,
    )
    if not found:
        return False

    opened_block = gemm.get_opened_block_info()
    if opened_block is None:
        raise RuntimeError("NoisyGemm reported found=True without opened block info")
    plain_proof = create_proof(opened_block, header_bytes)
    submit_params = {
        "plain_proof": plain_proof.to_base64(),
        "mining_job": _mining_job_params(header_bytes, target, cert_version),
    }
    await _send_request(
        writer,
        _make_request("submitPlainProof", submit_params, request_id + 1),
    )
    submit_response = await _read_response(reader)
    if "error" in submit_response:
        logger.warning("submitPlainProof rejected: %s", submit_response["error"])
        return False
    # The gateway answers "submitted" and then proves/submits the block
    # asynchronously; node acceptance is only visible in the gateway log/chain.
    logger.info(
        "submitPlainProof result: %s (cert_version=%d)",
        submit_response.get("result"),
        cert_version,
    )
    return True


async def _main_loop(args: argparse.Namespace) -> None:
    import pearl_mining

    _validate_shape(pearl_mining, m=args.m, n=args.n, k=args.k, rank=args.rank)
    reader, writer = await _open_gateway_connection(
        args.gateway_host,
        args.gateway_port,
        timeout_seconds=args.connect_timeout_seconds,
        retry_seconds=args.connect_retry_seconds,
    )
    request_id = 0
    try:
        while True:
            request_id += 2
            try:
                accepted = await _mine_one_round(
                    reader,
                    writer,
                    request_id=request_id,
                    m=args.m,
                    n=args.n,
                    k=args.k,
                    rank=args.rank,
                )
            except Exception:
                logger.exception("MPS mining round failed; retrying after backoff")
                await asyncio.sleep(_FAILURE_BACKOFF_SECONDS)
                continue
            if accepted:
                logger.info("proof handed to gateway")
            else:
                await asyncio.sleep(_FAILURE_BACKOFF_SECONDS)
    finally:
        writer.close()
        await writer.wait_closed()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="oj_pearl_mps._mps_miner_loop_main")
    p.add_argument("--gateway-host", default="127.0.0.1")
    p.add_argument("--gateway-port", type=int, default=8337)
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=_CONNECT_TIMEOUT_SECONDS,
    )
    p.add_argument(
        "--connect-retry-seconds",
        type=float,
        default=_CONNECT_RETRY_SECONDS,
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    try:
        asyncio.run(_main_loop(args))
    except KeyboardInterrupt:
        sys.exit(0)
