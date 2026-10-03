# pmk

## Quick start

Requirements: an Apple Silicon Mac (M1–M4; M5 is experimental), macOS 14+, Xcode Command Line Tools, Git, [rustup](https://rustup.rs/), and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Augustas11/pearl-apple-miner.git
cd pearl-apple-miner
scripts/install.sh
```

When installation passes its GPU correctness check, start mining with your Pearl wallet:

```sh
scripts/mine.sh --wallet '<your-prl-wallet>'
```

Payouts: check your wallet on https://pearl.herominers.com/. Direct stats URL (check; unverified): `https://pearl.herominers.com/api/stats_address?address=<wallet>`.

Expect full, sustained GPU use. Press Ctrl-C once to stop cleanly. An 80-GPU-core M3 Ultra measured about 16 TOPS: roughly one share per 18 minutes and 0.2 PRL/day at the measured difficulty. For other M1–M4 chips, a rough estimate is 18 min × (80 / GPU cores) per share and 0.2 × (GPU cores / 80) PRL/day; real speed and rewards vary.

Power cost: at full GPU load a Mac draws more power than the PRL it earns is worth at today's price (about $0.20/day of PRL on an M3 Ultra). Mine because you want to back the network or expect PRL to rise, not for profit today.

No wallet yet? Get the official Pearl Wallet from https://github.com/pearl-research-labs/pearl/releases.

Beta: open an issue or DM [@_aug11_](https://x.com/_aug11_) on X.

A Pearl (PRL) proof-of-useful-work miner for Apple Silicon, written for Metal. It mines the cert v3 scheme only.

pmk builds a Pearl job, runs the noisy int8 GEMM and hash search on the GPU, and turns a hit into a PlainProof
that Pearl's own verifier accepts. The GEMM is exact: int8 operands, int32 sums, checked bit for bit against a CPU oracle.

| Component | What it is |
|---|---|
| `pmkcore/` | Rust: job builder, CPU oracle, PlainProof construction. C ABI. |
| `libpmk/` | Swift + Metal: K1 `noise_gen`, K2 `noise_apply`, K3-SG (simdgroup int-exact GEMM). C ABI. |
| `miner/` | Python `pmk_miner`: solo mode (via a Pearl gateway) and pool mode (HeroMiners / Kryptex object-dialect stratum). |
| `scripts/studio_b4/` | Long-run window harness: offline bundle, G3 admission, regtest and pool windows. |
| `bench/`, `tools/`, `docs/` | Kernel prototypes and benchmarks, a Pearl-to-MLX converter, spec and design notes. |
| `upstream/openjarvis/` | Upgraded OpenJarvis Apple-MPS miner loop (Apache-2.0), used as a baseline. |

## Status

Experimental. Results so far, all from this code:

- Regtest blocks accepted at the production shape 8192x8192x4096 on an M3 Ultra, with corrupted certificates rejected.
- Sustained 16.08 TOPS on an M3 Ultra over 600 s, with a GPU idle gap of 0.06%.
- Mainnet pool shares accepted at HeroMiners (October 2026).
- A three-lane code, security and architecture audit finished with 0 critical, high or medium findings.

It mines PRL cert v3 only. The cert v4 / FP8 fork will need a new path:
https://github.com/pearl-research-labs/pips/issues/14

Summaries of the runs are in `bench/evidence/`. Raw logs are not published.

## Requirements

- Apple Silicon. K3-SG targets M1 to M4 (Apple7 to Apple9 GPUs). Tested on an M3 Ultra and, for development, an M5.
- macOS 14 or newer, Xcode with Swift 5.9+ (`libpmk/Package.swift`).
- Rust (stable, 2021 edition; tested with 1.98) and `git`.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Build

From the repo root:

```sh
# 1. Pearl sources (pmkcore depends on vendor/pearl at a pinned commit; vendor/ is gitignored)
scripts/fetch_vendor.sh            # add --fp8 only for bench/v4_emulation

# 2. Python env: Pearl packages, py-pearl-mining (maturin), torch
scripts/setup_env.sh               # creates .venv

# 3. Native pieces
(cd pmkcore && cargo build --release)
(cd libpmk && swift build -c release)

# 4. Miner dependencies (hash-pinned)
uv pip install --python .venv/bin/python --require-hashes -r miner/requirements.lock
```

```sh
# 5. Test vectors (generated, not stored in git: ~21 MB). pmkcore/libpmk tests and the G3 bundle need them.
.venv/bin/python bench/k3sg/make_vectors.py
```

Regtest runs and two harness tests need a local `pearld`: `scripts/build_pearld.sh` (needs Go; about 5 minutes).

Pool mode also needs a py-pearl-mining build with the bound-helper patch:
`scripts/pmk_build_pool_binding.sh --install` (see `miner/README.md`).

What comes from where: `vendor/pearl` (Pearl Research Labs, pinned in `scripts/fetch_vendor.sh`) is the only external
checkout. OpenJarvis is already vendored in `upstream/openjarvis`, so it is not fetched.
`bench/oj_mps_bench.py` additionally needs a copy of one OpenJarvis file (see its header).

## G3 admission (M1 to M4)

On Apple7 to Apple9 GPUs libpmk refuses to mine until a G3 admission record exists for the exact GPU, OS build and
kernel build. The window harness runs the full G3 vector suite and the fail-closed probe, and writes the record:

```sh
scripts/studio_b4/build_bundle.sh      # builds dist/studio_b4/
dist/studio_b4/b4_window.sh            # full window; add --quick for a short self-test
```

Details are in `scripts/studio_b4/README.md`. The record is read from `PMK_G3_ADMISSION_FILE`.
Run it on an otherwise idle machine.

## Run in pool mode

The quick-start wrapper defaults to Singapore and a sanitized short hostname as the worker name:

```sh
scripts/mine.sh --wallet '<your-prl-wallet>' --worker my-mac --pool stratum+tcp://de.pearl.herominers.com:1200
```

It keeps owner-only wallet/allowlist files, admission, and run state in `~/.pmk/` (`PMK_HOME` can override
that directory). It chooses a smaller job on 8 GB Macs or GPUs with fewer than 32 cores and prints jobs/s, TOPS, accepted/rejected shares,
and estimated time per share about once a minute. Rates are averages since startup; share arrivals are random.
It refreshes expired G3 admission at startup and at the six-hour probe, without a lab token or resume hook.
Rerun `scripts/install.sh` after updating the checkout. Builds can take a while; no virtualenv activation is needed.

For manual configuration:

```sh
PYTHONPATH=miner .venv/bin/python -m pmk_miner \
  --mode pool --pool-url stratum+tcp://<region>.pearl.herominers.com:1200 \
  --wallet-file /path/to/wallet.txt \
  --wallet-allowlist /path/to/wallet-allowlist.txt \
  --worker <worker-name> --config /path/to/pool.toml
```

The wallet file holds one wallet. The allowlist holds the wallets you approve; pmk refuses any other. Wallets are
masked in logs. See `miner/README.md` for the config file.

HeroMiners port 1200 regions listed in the pool KB are: `de`, `fr`, `es`, `fi`, `ru`, `ca`, `us`, `us2`, `us3`,
`mx`, `br`, `kz`, `hk`, `kr`, `in`, `sg`, `tr`, and `au`. Replace `<region>` above with the nearest one.

## Solo and regtest

Solo mode talks to a local Pearl gateway and node. Setup, config and the regtest end-to-end run
(`scripts/pmk_regtest_e2e.sh`) are in `miner/README.md`.

## Tests

```sh
(cd pmkcore && cargo test --release)                      # needs .venv with pearl_mining
(cd libpmk && swift test -c release)                      # uses the GPU
PYTHONPATH=miner .venv/bin/python -m pytest miner/tests   # fast subset: -k "not regtest"
.venv/bin/python -m pytest scripts/studio_b4              # harness tests
```

GPU tests take `/tmp/pmm-gpu-bench.lock`; do not run them next to a benchmark.

## Warnings

- Mining puts the GPU under heavy sustained load. Watch temperatures. Fanless Macs throttle hard.
- Earnings are small. One M3 Ultra is on the order of 0.2 PRL per day at the pool rate seen in October 2026.
- This is experimental software with no warranty. Do not run it on a machine you cannot afford to stress.
- Keep wallet files, node credentials and gateway env files out of the repo.

## License

Apache-2.0, see `LICENSE`. Third-party code and notices are in `THIRD_PARTY_NOTICES.md`
(OpenJarvis, Pearl Research Labs, BLAKE3).

## Credits and beta testers

Built by Augustas (Malibu, https://malibu.tech). To beta test, open an issue or DM @_aug11_ on X.
