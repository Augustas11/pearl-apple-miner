"""Benchmark the upgraded OpenJarvis Apple-MPS Pearl miner (oj_pearl_mps).

Times exactly the pieces of one ``_mine_one_round`` on current Pearl
(py-pearl-mining 0.3.1, cert version 3 / salted seeds):

  prep_hash   CommitmentHasher.commitment_hash(..., salted_dims=(m, n))   [CPU]
  prep_noise  NoiseGenerator.generate_noise_metrices(...)                 [CPU]
  noisy_gemm  MpsNoisyGemmAdapter(NoisyGemm).noisy_gemm(...) on MPS, incl.
              host->MPS copies of A, B and the noise factors (as the loop does)

GOPS = 2*m*n*k / noisy_gemm_time (matmul-equivalent). Warm-up once per shape,
then --reps timed reps (default 3); reports min / median / max.
pow_target=0 (hardest) so no block is opened and every hash tile is checked.
The first rep's C is checked against A@B on CPU (NoisyGEMM must denoise exactly).

Run (after ./scripts/setup_env.sh):  .venv/bin/python bench/oj_upgraded_bench.py
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import time

import pearl_mining
import torch
from miner_base.commitment_hash import CommitmentHasher
from miner_base.noise_generation import NoiseGenerator
from miner_base.noisy_gemm import NoisyGemm
from oj_pearl_mps._mps_miner_loop_main import (
    MpsNoisyGemmAdapter,
    _mining_config_for_shape,
    _salted_dims_for,
)
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion

SHAPES = [(128, 128, 1024, 64), (512, 512, 4096, 128), (1024, 1024, 8192, 128)]


def _chip() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


def _sync() -> None:
    torch.mps.synchronize()


def bench_shape(m: int, n: int, k: int, rank: int, reps: int, header: bytes) -> dict:
    dev = torch.device("mps")
    cfg = _mining_config_for_shape(pearl_mining, k=k, rank=rank)
    salted = _salted_dims_for(int(CertificateVersion.ZK_V3), m=m, n=n)
    noise_gen = NoiseGenerator(noise_rank=rank, noise_range=128)
    hash_t, noise_t, gemm_t = [], [], []
    for rep in range(reps + 1):  # rep 0 = warm-up
        A = torch.randint(-64, 64, (m, k), dtype=torch.int8)
        B = torch.randint(-64, 64, (k, n), dtype=torch.int8)
        t0 = time.perf_counter()
        ch = CommitmentHasher.commitment_hash(A, B, header, cfg, salted_dims=salted)
        t1 = time.perf_counter()
        E = noise_gen.generate_noise_metrices(
            key_A=ch.noise_seed_A, key_B=ch.noise_seed_B, A_rows=m, common_dim=k, B_cols=n
        )
        t2 = time.perf_counter()
        gemm = MpsNoisyGemmAdapter.build(
            NoisyGemm,
            noise_range=128,
            noise_rank=rank,
            hash_tile_h=16,
            hash_tile_w=16,
            matmul_tile_h=rank,
            matmul_tile_w=rank,
        )
        _sync()
        t3 = time.perf_counter()
        C, found = gemm.noisy_gemm(
            A.to(dev), B.to(dev), *(e.to(dev) for e in E), commitment_hash=ch, pow_target=0
        )
        _sync()
        t4 = time.perf_counter()
        if rep == 1:
            ref = torch.matmul(A.to(torch.int32), B.to(torch.int32))
            if not torch.equal(C.cpu(), ref):
                raise AssertionError(f"noisy_gemm result != A@B at shape {(m, n, k, rank)}")
        if rep > 0:
            hash_t.append(t1 - t0)
            noise_t.append(t2 - t1)
            gemm_t.append(t4 - t3)
        print(
            f"  {'warmup' if rep == 0 else f'rep {rep}'}: hash {t1 - t0:.4f}s  noise {t2 - t1:.4f}s  "
            f"noisy_gemm {t4 - t3:.4f}s  found={found}",
            flush=True,
        )
    ops = 2 * m * n * k
    med = statistics.median(gemm_t)
    return {
        "shape": (m, n, k, rank),
        "hash_med": statistics.median(hash_t),
        "noise_med": statistics.median(noise_t),
        "gemm_min": min(gemm_t),
        "gemm_med": med,
        "gemm_max": max(gemm_t),
        "gops_med": ops / med / 1e9,
        "gops_best": ops / min(gemm_t) / 1e9,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.reps < 3:
        raise SystemExit("--reps must be >= 3")
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is not available on this host")

    print(f"host: {_chip()} | macOS {platform.mac_ver()[0]} | python {platform.python_version()}")
    print(f"torch {torch.__version__} | pearl_mining {pearl_mining.__version__} | cert_version 3 (salted seeds)")
    print(f"reps={args.reps} (+1 warm-up per shape); pow_target=0")
    la = os.getloadavg()
    print(f"load average at start (1/5/15 min): {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}; cpus: {os.cpu_count()}")
    header = pearl_mining.IncompleteBlockHeader(
        version=0,
        prev_block=bytes(32),
        merkle_root=b"0123456789abcdef" * 2,
        timestamp=0x66666666,
        nbits=0x1E010000,
    ).to_bytes()

    rows = []
    for m, n, k, rank in SHAPES:
        note = "" if rank >= pearl_mining.PENALTY_BASE_RANK else "  [rank < 128: timing only, not minable on current consensus]"
        print(f"\nshape m={m} n={n} k={k} rank={rank}{note}", flush=True)
        rows.append(bench_shape(m, n, k, rank, args.reps, header))

    la = os.getloadavg()
    print(f"\nload average at end (1/5/15 min): {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}")
    print("\n| m | n | k | rank | commit hash (s) | noise gen (s) | noisy_gemm median (s) | min..max (s) | GOPS median | GOPS best |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m, n, k, rank = r["shape"]
        print(
            f"| {m} | {n} | {k} | {rank} | {r['hash_med']:.4f} | {r['noise_med']:.4f} | {r['gemm_med']:.4f} | "
            f"{r['gemm_min']:.4f}..{r['gemm_max']:.4f} | {r['gops_med']:.3f} | {r['gops_best']:.3f} |"
        )


if __name__ == "__main__":
    main()
