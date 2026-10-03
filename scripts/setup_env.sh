#!/usr/bin/env bash
# Reproducible Python env for the upgraded OpenJarvis Apple-MPS Pearl miner.
#
#   ./scripts/setup_env.sh
#
# Creates .venv (Python 3.12) and installs:
#   - py-pearl-mining (built from vendor/pearl/py-pearl-mining with maturin, release)
#   - miner-utils, pearl-gateway, miner-base (vendor/pearl/miner/*)
#   - torch==2.11.0 (MPS backend on macOS arm64)
# pearl-gemm is NOT installed: it is CUDA-only.
#
# Needs: uv, a Rust toolchain (cargo), git. Clones Pearl into vendor/pearl at
# PEARL_REV if it is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEARL_REV="${PEARL_REV:-7039e66f3c44f1541cb0e85328061e619ec5904d}"
PEARL_URL="${PEARL_URL:-https://github.com/pearl-research-labs/pearl}"
PEARL="$ROOT/vendor/pearl"
VENV="$ROOT/.venv"

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)"; exit 1; }
command -v cargo >/dev/null || { echo "cargo not found (install Rust: https://rustup.rs)"; exit 1; }

if [ ! -d "$PEARL/.git" ]; then
  mkdir -p "$ROOT/vendor"
  git clone "$PEARL_URL" "$PEARL"
fi
actual_rev="$(git -C "$PEARL" rev-parse HEAD)"
if [ "$actual_rev" != "$PEARL_REV" ]; then
  echo "vendor/pearl is at $actual_rev, expected $PEARL_REV"
  echo "run: git -C vendor/pearl fetch && git -C vendor/pearl checkout $PEARL_REV  (or set PEARL_REV)"
  exit 1
fi

[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
export VIRTUAL_ENV="$VENV"
PY="$VENV/bin/python"

uv pip install --python "$PY" "maturin>=1.7,<2" "torch==2.11.0"

# py-pearl-mining: release build of the Rust extension into the venv.
(cd "$PEARL/py-pearl-mining" && "$VENV/bin/maturin" develop --release)

# Pure-Python Pearl miner packages; --no-deps for the local ones so uv does not
# try to resolve py-pearl-mining / miner-utils from PyPI, then their third-party deps.
uv pip install --python "$PY" --no-deps \
  "$PEARL/miner/miner-utils" "$PEARL/miner/pearl-gateway" "$PEARL/miner/miner-base"
uv pip install --python "$PY" \
  "loguru>=0.7.0" "aiohttp>=3.10.0" "bitcoin-utils>=0.7.0" "blake3>=1.0.7" \
  "fastjsonschema>=2.16.0" "numpy>=1.20.0" "prometheus-client>=0.23.1" "pybase64>=1.4.0" \
  "pydantic>=2.12.5" "pydantic-settings>=2.12.0" "pyyaml>=6.0" "pytest>=8.3"

# The upgraded OpenJarvis miner package (upstream/openjarvis), editable.
uv pip install --python "$PY" --no-deps -e "$ROOT/upstream/openjarvis"

"$PY" - <<'PY'
import pearl_mining, torch, miner_base, pearl_gateway, oj_pearl_mps
print("pearl_mining", pearl_mining.__version__)
print("torch", torch.__version__, "mps available:", torch.backends.mps.is_available())
print("CERT_VERSION_ZK_V3 =", pearl_mining.CERT_VERSION_ZK_V3)
PY
echo "OK: env ready at $VENV"
