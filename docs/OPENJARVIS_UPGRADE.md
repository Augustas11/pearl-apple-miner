# OpenJarvis `apple-mps-pearl` on current Pearl

The OpenJarvis Apple-GPU Pearl miner was written against py-pearl-mining 0.1.0 (May 2026). This document records how it was ported to current Pearl, how the port was proven correct, and how fast it runs.

- Source: OpenJarvis `c4da16e1ca3d21f4cc1905d4200063e564104f0f` (`src/openjarvis/mining/`)
- Target: Pearl `7039e66f3c44f1541cb0e85328061e619ec5904d`, which ships py-pearl-mining 0.3.1, miner-base, and pearl-gateway. The mainnet node under test runs 1.4.6+821c811. The 12 commits between that node and this Pearl commit change no mining or consensus code. Mainnet `getblocktemplate` returns `requiredcertversion: 3`.
- Result: the port mines cert-version-3 shares that the V3 verifier accepts. On simnet, the full pearld, pearl-gateway, and MPS miner stack mined 3 blocks, all accepted. On the M5, the hot loop runs at about 4.5 GOPS at 1024×1024×8192, rank 128.

## Package layout

`upstream/openjarvis/` is a standalone package named `oj-pearl-mps` (import name `oj_pearl_mps`). It does not depend on the OpenJarvis app.

| File | Origin |
|---|---|
| `src/oj_pearl_mps/_mps_miner_loop_main.py` | OpenJarvis `_mps_miner_loop_main.py`, modified |
| `src/oj_pearl_mps/_miner_loop_main.py` | OpenJarvis `_miner_loop_main.py` (shared gateway helpers and the CPU loop), modified |
| `src/oj_pearl_mps/_constants.py` | OpenJarvis `_constants.py`, trimmed and modified |
| `src/oj_pearl_mps/__init__.py`, `tests/test_current_pearl_protocol.py`, `pyproject.toml` | new |
| `LICENSE` | OpenJarvis Apache-2.0 license, copied verbatim |
| `NOTICE` | records the source commit and states that the files were modified |

Some OpenJarvis files were not copied: `apple_mps_pearl.py`, `cpu_pearl.py`, `_pearl_subprocess.py`, `_install.py`, and the others. They are app glue: provider registry, hardware detection, sidecar files, and the subprocess launcher. They import `openjarvis.core.*`, and they do no mining. The miner loop runs directly: `python -m oj_pearl_mps._mps_miner_loop_main --gateway-port 8337`.

**The hot loop is unchanged.** `MpsNoisyGemmAdapter` is byte-identical to upstream. That class covers the NoisyGEMM device adapter, the per-128×128 output-tile loop, the per-rank-chunk `torch.matmul`, and the transcript hashing on CPU. This was checked with `diff` on the class body (output: `ADAPTER IDENTICAL`). The round structure is also the same: get a job, use random int7 A/B, commitment hash, noise, `noisy_gemm`, `create_proof`, then submit. The same applies to the 1-second backoff after a round that finds no share.

## Every change: old → new, why, Pearl reference

Paths in the Pearl column are relative to `vendor/pearl/`.

### 1. `MiningConfiguration` constructor: both loops

- **Old:** `MiningConfiguration(common_dim, rank, mma_type, rows_pattern, cols_pattern, reserved=MiningConfiguration.RESERVED)`
- **New:** `MiningConfiguration(common_dim, rank, mma_type, rows_pattern, cols_pattern, moe=None)`
- **Where:** `_mps_miner_loop_main._mining_config_for_shape` and `_miner_loop_main._build_mining_config`
- **Why:** since 0.2.0 the MoE fork replaced `reserved` with `moe`, and `RESERVED` no longer exists. The old call raises `TypeError`. `None` selects dense mining.
- **Pearl ref:** `py-pearl-mining/tests/test_python_api.py`; `miner/pearl-gateway/src/pearl_gateway/comm/mining_configuration.py` (`PearlMiningConfigurationFactory.create`)

### 2. `getMiningInfo` now carries `cert_version`

- **Old:** `_decode_mining_info(result) -> (header_bytes, target)`
- **New:** `_decode_mining_info(result) -> (header_bytes, target, cert_version)`. It reads `result["cert_version"]`.
- **Why:** the gateway job now carries the template's `requiredcertversion`. That value selects the noise-seed derivation (see item 4), and the miner must send it back when it submits.
- **Pearl ref:** `miner/pearl-gateway/src/pearl_gateway/comm/dataclasses.py` (`MiningJob.to_dict`, `MiningJob.from_template`)

### 3. `MiningJob` requires `cert_version`: MPS loop

- **Old:** `MiningJob(incomplete_header_bytes=..., target=...)`
- **New:** `MiningJob(incomplete_header_bytes=..., target=..., cert_version=CertificateVersion(cert_version))`
- **Why:** `cert_version` is a required dataclass field with no default, so the old call raises `TypeError`. The MPS loop builds this object to call `adjust_target`.
- **Pearl ref:** `miner/pearl-gateway/src/pearl_gateway/comm/dataclasses.py` (`MiningJob`)

### 4. Salted noise seeds (V3, the salted-seed hard fork): MPS loop

- **Old:** `CommitmentHasher.commitment_hash(A, B, header_bytes, mining_config)`
- **New:** `CommitmentHasher.commitment_hash(A, B, header_bytes, mining_config, salted_dims=_salted_dims_for(cert_version, m=m, n=n))`
- **How:** the new helper `_salted_dims_for` returns `(m, n)` when `CertificateVersion(cert_version).uses_salted_seeds` is true (cert 3 or later). Otherwise it returns `None`. This matches Pearl's reference miner.
- **Why:** `salted_dims` is now a required keyword-only argument, so the old call raises `TypeError`. From the fork height on, `(m, n)` must be bound into the Merkle roots before the seed chain runs. A miner that derives seeds the old way produces shares that the V3 verifier rejects. The fork heights are mainnet 99000, testnet 38648, testnet2 83109, and simnet/regtest 1.
- **Pearl ref:**
  - `docs/salted-seed-fork-upgrade-guide.md`
  - `miner/miner-base/src/miner_base/commitment_hash.py` (`bind_root_a` / `bind_root_b`)
  - `miner/vllm-miner/src/vllm_miner/gemm_operators.py:149`: `salted_dims=(m, n) if mining_job.cert_version.uses_salted_seeds else None`

### 5. `submitPlainProof` requires `mining_job.cert_version`: both loops

- **Old:** `"mining_job": {"incomplete_header_bytes": ..., "target": ...}`
- **New:** the same object plus `"cert_version": <int>`, built by the new helper `_miner_loop_main._mining_job_params`. Its output equals `MiningJob.to_dict()`, and a unit test checks that.
- **Why:** the gateway's JSON schema now lists `cert_version` as required. An old submission gets `-32602 Invalid params`, and a unit test checks that against the gateway's own validator.
- **Pearl ref:** `miner/pearl-gateway/src/pearl_gateway/miner_rpc/schemas.py` (`SUBMIT_PLAIN_PROOF_SCHEMA`)

### 6. Rank-penalty minimum: default rank 64 → 128 (MPS) and 32 → 128 (CPU)

- **Old:** MPS `--rank` defaulted to 64. CPU `--rank` defaulted to 32, as did `CPU_PEARL_DEFAULT_RANK`.
- **New:** 128 everywhere.
- **Why:** `MiningJob.adjust_target` raises `noise rank N is below the minimum 128` for any rank below `PENALTY_BASE_RANK` (128). With the old default, every MPS round would raise before mining started. Consensus enforces the same rule after the rank-penalty soft fork.
- **Pearl ref:**
  - `miner/pearl-gateway/src/pearl_gateway/comm/dataclasses.py` (`adjust_target`)
  - `zk-pow/src/api/sanity_checks.rs:13` (`PENALTY_BASE_RANK = 128`) and `check_rank_penalty`
  - `node/chaincfg/params.go` (`RankPenaltyForkHeight`: mainnet 96251)

### 7. Common-dimension bounds: default k 1024 → 2048, plus a fail-fast shape check

- **Old:** `--k` defaulted to 1024 (and `CPU_PEARL_DEFAULT_K` was 1024).
- **New:** `--k` defaults to 2048. The MPS loop also gained `_validate_shape(...)`, called once at startup.
- **Why:** `public_params_sanity_check` requires `k >= 16·r`, which means k ≥ 2048 at r = 128. It also requires `k >= 1024`, `k % 64 == 0`, `k <= 4r²`, `k <= 65536`, and r to be a power of two in [32, 1024]. A share with k = 1024 and r = 128 fails verification. `_validate_shape` rejects such shapes before any GPU work, and also enforces the adapter's own `m, n >= rank`.
- **Pearl ref:** `zk-pow/src/api/sanity_checks.rs` (`public_params_sanity_check`)

### 8. `pearl_mining.mine` needs `cert_version`: CPU loop only

- **Old:** `mine(m, n, k, header, cfg, signal_range=None, wrong_jackpot_hash=False)`
- **New:** the same call plus `cert_version=cert_version`
- **Why:** this keyword-only argument now selects the seed derivation. Leaving it out raises `TypeError`.
- **Pearl ref:** `py-pearl-mining/src/lib.rs:262` (`#[pyo3(signature = (..., *, cert_version))]`)

### 9. `submitPlainProof` success is not acceptance: log wording only

- **Old:** logged `share accepted` whenever the gateway replied without an error.
- **New:** logs `submitPlainProof result: submitted (cert_version=3)` and `proof handed to gateway`. The return value and control flow are unchanged.
- **Why:** the gateway replies `"submitted"` right away. It then builds the ZK proof and submits the block in a background task. Whether the node accepted the block shows up only in the gateway log (`Block accepted by node!`) and on the chain. The gateway also silently drops proofs for stale headers and duplicates.
- **Pearl ref:** `miner/pearl-gateway/src/pearl_gateway/miner_rpc/server.py` (`_process_request`, `handle_submit_plain_proof`); `submission_service.py`

### 10. Standalone packaging, with no behaviour change

- **Imports and names:**
  - Module path `openjarvis.mining.*` became `oj_pearl_mps.*`.
  - The logger names and `argparse` `prog` strings were renamed to match.
  - `_constants.py` was trimmed to the mining constants. The original imports `openjarvis.core.paths`.
- **Docstrings:** they now mention 0.3.1. The `PlainProof.to_base64()` note was re-verified on 0.3.1.
- **Removed:** the unused `import base64` in the MPS loop.

### Checked and unchanged

- `create_proof(opened_block, header_bytes)`
- `NoiseGenerator(noise_rank, noise_range).generate_noise_metrices(...)`
- `NoisyGemm` internals: the adapter's three overrides still match upstream `_accumulate_transcripts`, `_process_output_tile`, and `_tiled_matmul`, apart from device placement
- `PlainProof.to_base64`, `IncompleteBlockHeader.from_bytes`
- `getMiningInfo` takes empty params
- The JSON-RPC 2.0 envelope, sent as line-delimited TCP on 127.0.0.1

The app glue left out of the package has drift of its own. It would need the same changes if it were brought back:

- `apple_mps_pearl.py` defaults: `rank` 64 and `k=CPU_PEARL_DEFAULT_K` (1024).
- `_pearl_subprocess.py` sets `METRICS_BIND`. Current pearl-gateway source does not read it and runs no metrics server.

## Environment

```
./scripts/setup_env.sh     # .venv (Python 3.12), maturin release build of py-pearl-mining 0.3.1,
                           # miner-utils / pearl-gateway / miner-base, torch==2.11.0, oj-pearl-mps (editable)
./scripts/build_pearld.sh  # pearld + prlctl from vendor/pearl (Taskfile build:blockchain steps without go-task)
```

`setup_env.sh` checks that `vendor/pearl` is at `7039e66f`, and clones Pearl if the directory is missing. pearl-gemm is not installed because it is CUDA-only. The script ends with an import check. Output on this machine:

```
pearl_mining 0.3.1
torch 2.11.0 mps available: True
CERT_VERSION_ZK_V3 = 3
OK: env ready at $PMK_ROOT/.venv
```

The toolchain here: go 1.26.4, rustc 1.98.0, uv 0.12.5. go-task is not installed, so `build_pearld.sh` runs the same Taskfile steps directly.

## Proof

### Unit tests

```
.venv/bin/python -m pytest -q upstream/openjarvis/tests
9 passed
```

The tests check the following against Pearl's own code:

- The job decode matches `MiningJob.to_dict()`.
- The submit params pass the gateway's `validate_submit_plain_proof`, and a submission without `cert_version` is rejected.
- `salted_dims` follows the cert version.
- Both config builders work on the 0.3.1 API.
- `adjust_target` accepts rank 128 and rejects the old rank 64.
- The shape validator behaves as specified.

### Offline: V3 verifier, with a negative control

`.venv/bin/python scripts/offline_proof.py`. Evidence: `bench/evidence/offline_proof_m5.txt`.

The script runs the real `_mine_one_round` on MPS against Pearl's own `MinerRpcServer`, which applies the production request parsing and schema validation. The server runs in-process on 127.0.0.1 and hands out a cert-version-3 job. On each submit it decodes the `PlainProof` and runs `verify_plain_proof_for_cert_version` with V3 and with V2.

The shape is m = n = 128, k = 2048, rank 128.

```
== POSITIVE: upgraded loop (salted_dims=(m, n) for cert_version 3), nbits=0x1e010000
trial 0: share after 2 round(s), 0.91s; job cert_version=3; proof m=128 n=128 k=2048 rank=128
  verify_plain_proof_for_cert_version(3) -> (True, 'Mining solution verified successfully')
  verify_plain_proof_for_cert_version(2) -> (False, 'Jackpot condition not satisfied: hash does not meet difficulty target')
...
POSITIVE RESULT: 3/3 shares accepted by the V3 verifier

== NEGATIVE CONTROL: same loop, legacy seeds (salted_dims=None), nbits=0x1e002000
trial 0: share after 12 round(s), 2.09s
  verify_plain_proof_for_cert_version(3) -> (False, 'Jackpot condition not satisfied: hash does not meet difficulty target')
  verify_plain_proof_for_cert_version(2) -> (True, 'Mining solution verified successfully')
...
NEGATIVE RESULT: 5/5 legacy-seed shares rejected by V3; 5/5 accepted by V2 (legacy derivation)

OVERALL: PASS
```

The negative control runs the same loop with the pre-fork derivation (`salted_dims=None`), which is what unpatched OpenJarvis computes. The V3 verifier rejects those shares. The same shares pass V2, which proves they are otherwise well-formed and that the seed derivation is the only defect.

**Why the rejection is probabilistic rather than structural.** The shares carry raw Merkle roots, and the wire format is unchanged. The V3 verifier binds `(m, n)` into those roots and derives different noise seeds. The different seeds produce different noise, a different jackpot message, and a different jackpot-hash key (the A noise seed). The recomputed jackpot hash is therefore a fresh pseudo-random value. A wrongly derived share passes only when that value happens to fall under the bound, so the chance of passing equals the per-transcript hit probability.

- The negative control uses nbits `0x1e002000`. Its bound is 2^229 · 256 · 2048 = 2^248, so accidental acceptance has probability 2^-8 per share.
- At the easy positive target (`0x1e010000`) that probability is 2^-5.
- At mainnet bits `0x177fd82e` it is about 2^-54 at this shape. On mainnet, a legacy-derived share is rejected in practice.

The rate was measured directly with `scripts/cross_derivation_rate.py`. Evidence: `bench/evidence/cross_derivation_m5.txt`.

```
nbits=0x1e010000 mined_with=salted(v3) shares=200 pass_v3=200 pass_v2=6  rounds=227
nbits=0x1e010000 mined_with=legacy     shares=200 pass_v3=11  pass_v2=200 rounds=229
```

Same-derivation shares pass 200/200. Cross-derivation shares pass 6/200 (3.0%) and 11/200 (5.5%), close to the expected 2^-5 (3.1%). This is consistent with a random hash under the target and with no structural acceptance.

### End to end: simnet pearld, pearl-gateway, MPS miner

`./scripts/simnet_e2e.sh 3 1500`. Evidence: `bench/evidence/simnet_e2e_m5.txt`. Raw logs are kept under `run/simnet/logs/`, which is gitignored.

The run used these settings:

- **pearld:** built from `vendor/pearl` with flags from `.github/workflows/integration_tests_ci.yml`: `--simnet --notls --addrindex --txindex --miningaddr=<CI simnet addr>`.
- **Network binding:** RPC on `127.0.0.1:44207` and P2P on `127.0.0.1:18655`, with `--nodnsseed`.
- **Credentials:** random throwaway RPC credentials, generated per run.
- **Data:** the data directory is under `run/simnet/`.
- **Gateway:** pearl-gateway runs with `MINER_RPC_TRANSPORT=tcp` on `127.0.0.1:18337` and `--debug`, so it re-verifies each ZK proof.
- **Miner:** the upgraded MPS loop at m = n = 128, k = 2048, rank 128.
- **Cleanup:** an exit trap stops all three processes. Afterwards, no pearld, gateway, or miner process remained.

**Cert version on simnet.** Simnet sets `MoEForkHeight: 1` and `SaltedSeedForkHeight: 1` in `node/chaincfg/params.go`, so every block from height 1 requires cert version 3. The node confirmed this:

```
[e2e] getblocktemplate: height=1 bits=1e010000 requiredcertversion=3
[e2e] block height=1 hash=ed50065315e8c026827f2311524bbc2e46e4377f51174c432be9755a7f2b1f01 bits=1e010000 time=1790911781
[e2e] block height=2 hash=4941515f971e2a6ce96a9ccd9323e393125265acbfa2b651fabef5ef27aca3cb bits=1e010000 time=1790911818
[e2e] block height=3 hash=d380aaa5ec06def5d58c252e54b3f45128616fd200171ed2fc5cc00675ae9a5d bits=1e010000 time=1790911845
2026-10-02 11:30:17 - INFO - Block accepted by node! - submission_service.py:62
2026-10-02 11:30:44 - INFO - Block accepted by node! - submission_service.py:62
2026-10-02 11:31:12 - INFO - Block accepted by node! - submission_service.py:62
2026-10-02 11:30:17.546 [INF] RPCS: Accepted block ed500653...1f01 via submitblock      (pearld.log)
2026-10-02 11:30:44.727 [INF] RPCS: Accepted block 4941515f...a3cb via submitblock
2026-10-02 11:31:12.180 [INF] RPCS: Accepted block d380aaa5...9a5d via submitblock
[e2e] start_height=0 final_height=3
[e2e] RESULT: PASS
```

The miner submitted 8 plain proofs:

- Three became blocks.
- Four arrived for the height-1 template while its block was already being proven. The gateway answered them with `already_submitted`.
- One was a proof for height 4. It reached the gateway at 11:31:13, after the goal was met, and was still being proven when the script stopped all processes.

The gateway proves each block on CPU, which took about 25–30 s per block on the M5. That time includes the `--debug` re-verification.

## Benchmark

`bench/oj_upgraded_bench.py` times one round's pieces at the shapes below:

- **prep_hash:** `commitment_hash` with `salted_dims=(m, n)`, on CPU.
- **prep_noise:** noise generation, on CPU.
- **noisy_gemm:** the OpenJarvis MPS adapter's `noisy_gemm`, including the host→MPS copies, with `torch.mps.synchronize()` around it.

Each shape gets 1 warm-up plus 3 timed reps, at `pow_target=0`. The first timed rep's C is checked against A@B on CPU, and every shape passed. GOPS = 2mnk / noisy_gemm time. Raw output: `bench/evidence/oj_upgraded_m5.txt`.

Run on Apple M5 (10-core), macOS 26.5, torch 2.11.0. The load average was 8.9 at the start; the machine was shared with unrelated jobs.

| m | n | k | rank | commit hash (s) | noise gen (s) | noisy_gemm median (s) | min..max (s) | GOPS median | GOPS best |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 1024 | 64* | 0.0009 | 0.0222 | 0.0386 | 0.0355..0.0390 | 0.869 | 0.945 |
| 512 | 512 | 4096 | 128 | 0.0055 | 0.1029 | 0.5335 | 0.5263..0.5945 | 4.026 | 4.081 |
| 1024 | 1024 | 8192 | 128 | 0.0214 | 0.2164 | 3.7919 | 3.7496..6.2930 | 4.531 | 4.582 |

\* Rank 64 is below `PENALTY_BASE_RANK`. This shape is timed for comparison with the old OpenJarvis defaults only, and current consensus cannot mine it.

Two earlier runs that day were discarded. Unrelated jobs had the load average at 22–26, and the 1024 shape measured 2.2 and 3.5 GOPS median. The loop runs Python and CPU-side hashing for every 128×128×128 step, so CPU contention slows it directly.

For reference, the prior 0.1.0-patched measurement was 4.2 GOPS, and a raw Metal 4 int8 matmul reaches about 8,800 GOPS at the 1024 shape (`README.md`). The port therefore runs about 1,900× below the raw matmul rate at that shape.

To run on another Mac (for example, the M3 Ultra), use:

```
git clone <this repo> pearl-metal-miner && cd pearl-metal-miner
./scripts/setup_env.sh                       # clones vendor/pearl @ 7039e66f if missing
.venv/bin/python bench/oj_upgraded_bench.py | tee bench/evidence/oj_upgraded_<host>.txt
```

## Remaining limitations

- **Speed.** The hot loop is OpenJarvis's correctness-first design: a Python tile loop, `torch.matmul` per rank chunk, and a GPU→CPU copy plus a NumPy/BLAKE3 inner hash after every chunk. Per-round time stays in seconds.
- **The OpenJarvis loop also sleeps 1 s after every round that finds no share.**
  - At mainnet bits `0x177fd82e` (about 2^183 target), the hit probability per transcript is 2^-54 at 128×128×2048, which takes about 2.8·10^14 rounds per block. At 1024×1024×8192 it is 2^-52, which takes about 1.1·10^12 rounds.
  - At roughly 1.2 s and 5 s per round respectively, the expected time to one solo block is on the order of 10^7 and 10^5 years.
  - Proofs are correct, but mainnet solo mining with this design is not viable.
- **No pool or share-target support.** OpenJarvis's loop targets the block target through pearl-gateway, as written upstream.
- **Mining is pure proof of work.** The matrices are random int7 (`torch.randint(-64, 64)`), not model weights.
- **Platform.** Only dense (non-MoE) mining is supported, on MPS only. The CPU loop (`_miner_loop_main`) was ported and passes unit tests, but it was not run end to end here.
- **E2E coverage.** The end-to-end proof covers simnet only, not testnet or mainnet. Simnet exercises the same cert-v3 salted-seed rules as mainnet after height 99000. Simnet does not enable the rank-penalty fork, but the gateway's `adjust_target` enforces rank ≥ 128 regardless.
- **Upstream gateway logs the pearld RPC password at INFO.** This is in `pearl_client.py:30`. Here the password was throwaway and local only.

## Regtest (certificate-verifying) e2e

**Simnet acceptance does not verify certificates.** `NetBehaviorFlags` gives simnet `BFNoPoWCheck`
(`vendor/pearl/node/blockchain/process.go`), which skips `zkpow.VerifyCertificate` (`validate.go:331`)
and the rank-penalty rule (`validate.go:549`). The earlier simnet "blocks accepted" runs only prove the
plumbing. Regtest has the same forks at height 1 (MoE, RankPenalty, SaltedSeed; required cert v3) but
verifies certificates.

```
./scripts/build_pearld.sh            # once
./scripts/regtest_e2e.sh 3 1500 20   # blocks, timeout s, rank-64 phase s; evidence: bench/evidence/regtest_e2e_m5.txt
```

Regtest specifics (from the Pearl source): `pearld --regtest`, same Bech32 HRP `rprl` as simnet (so the
same mining address works), no `generate` needed (template at height 1 is mineable from genesis),
`PowLimitBits 0x1e010000`. `scripts/regtest_gateway_tap.py` wraps the gateway to log submitblock verdicts
and run the negative control.

Key results (M5, rank 128, k 2048): `getblocktemplate ... requiredcertversion=3`; heights 1, 2, 3 each
`Accepted block ... via submitblock` in pearld.log and `Block accepted by node!` in the gateway; find->accept
(miner hands proof -> node accepts) 16.7-33.3 s per block, nearly all of it gateway ZK proving (node verify+accept
is about 15-40 ms). Negative control: the first genuine block was preceded by two corrupted copies, both
rejected by pearld:

- flipped proof byte: `rejected: certificate verification failed: proof rejected: Proof Invalid`
- flipped public-data byte: `rejected: certificate verification failed: proof commitment mismatch: ...`

Rank-64 / rank-penalty rule: not exercised. The upgraded miner refuses to run it
(`unminable shape: rank=64 < PENALTY_BASE_RANK=128`), so reaching `ErrRankPenalty` would require forging a proof.
