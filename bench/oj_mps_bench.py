# Needs a local copy of OpenJarvis' src/openjarvis/mining/_mps_miner_loop_main.py saved as
# ./oj__mps_miner_loop_main.py (clone https://github.com/open-jarvis/OpenJarvis; Apache-2.0).
# It exec()s lines 40-140 of that file, so it only works against the matching upstream revision.
import time, os, torch, pearl_mining
from typing import Any
src = open("oj__mps_miner_loop_main.py").read().splitlines()
exec("\n".join(src[39:140]))  # OpenJarvis MpsNoisyGemmAdapter, verbatim
from miner_base.commitment_hash import CommitmentHasher
from miner_base.noise_generation import NoiseGenerator
from miner_base.noisy_gemm import NoisyGemm

def cfg(k, rank):
    return pearl_mining.MiningConfiguration(
        common_dim=k, rank=rank, mma_type=pearl_mining.MMAType.Int7xInt7ToInt32,
        rows_pattern=pearl_mining.PeriodicPattern.from_list(list(range(16))),
        cols_pattern=pearl_mining.PeriodicPattern.from_list(list(range(16))),
        moe=None)

header = bytes(76)
dev = torch.device("mps")
for (m, n, k, rank) in [(128,128,1024,64),(128, 128, 1024, 64), (512, 512, 4096, 128), (1024, 1024, 8192, 128)]:
    A = torch.randint(-64, 64, (m, k), dtype=torch.int8); B = torch.randint(-64, 64, (k, n), dtype=torch.int8)
    t0 = time.perf_counter()
    ch = CommitmentHasher.commitment_hash(A, B, header, cfg(k, rank), salted_dims=(m, n))
    E = NoiseGenerator(noise_rank=rank, noise_range=128).generate_noise_metrices(
        key_A=ch.noise_seed_A, key_B=ch.noise_seed_B, A_rows=m, common_dim=k, B_cols=n)
    t1 = time.perf_counter()
    g = MpsNoisyGemmAdapter.build(NoisyGemm, noise_range=128, noise_rank=rank, hash_tile_h=16, hash_tile_w=16,
                                  matmul_tile_h=rank, matmul_tile_w=rank)
    g.noisy_gemm(A.to(dev), B.to(dev), *[e.to(dev) for e in E], commitment_hash=ch, pow_target=0)
    torch.mps.synchronize(); t2 = time.perf_counter()
    ops = 2 * m * n * k
    print(f"OpenJarvis MPS  m={m} n={n} k={k} rank={rank}: prep {t1-t0:.2f}s, noisy_gemm {t2-t1:.2f}s "
          f"-> {ops/(t2-t1)/1e9:.3f} GOPS (matmul-equivalent)")
