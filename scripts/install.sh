#!/usr/bin/env bash
# Build and certify the Pearl Metal miner on an Apple Silicon Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
QUICKSTART="$ROOT/dist/quickstart"

fail() {
  echo "install: $*" >&2
  exit 1
}

[ "$(uname -s)" = "Darwin" ] || fail "macOS is required"
[ "$(uname -m)" = "arm64" ] || fail "Apple Silicon is required"

MACOS_VERSION="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS_VERSION%%.*}"
case "$MACOS_MAJOR" in
  ''|*[!0-9]*) fail "could not read the macOS version" ;;
esac
[ "$MACOS_MAJOR" -ge 14 ] || fail "macOS 14 or newer is required (found $MACOS_VERSION)"

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are missing. Install them with:" >&2
  echo "  xcode-select --install" >&2
  exit 1
fi
if ! xcrun --find swiftc >/dev/null 2>&1; then
  echo "The selected Xcode Command Line Tools do not include Swift. Install/update them with:" >&2
  echo "  xcode-select --install" >&2
  exit 1
fi
if ! command -v rustup >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
  echo "Rust via rustup is missing. Install it with:" >&2
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is missing. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "==> Fetching pinned Pearl sources"
"$ROOT/scripts/fetch_vendor.sh"

echo "==> Building pmkcore (release)"
(cd "$ROOT/pmkcore" && cargo build --release --locked)

echo "==> Building libpmk (release)"
(cd "$ROOT/libpmk" && swift build -c release)

echo "==> Creating the Python 3.12 environment"
if [ ! -x "$PYTHON" ] || ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1; then
  uv venv --python 3.12 --clear "$VENV"
fi

mkdir -p "$QUICKSTART"
RUNTIME_LOCK="$QUICKSTART/runtime-requirements.lock"
TORCH_LOCK="$QUICKSTART/torch-requirement.lock"
"$PYTHON" - "$ROOT/miner/requirements.lock" "$RUNTIME_LOCK" "$TORCH_LOCK" <<'PYCODE'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
lines = source.splitlines(keepends=True)
try:
    start = next(index for index, line in enumerate(lines) if line.startswith("torch=="))
except StopIteration:
    raise SystemExit("torch is absent from miner/requirements.lock")
end = start + 1
while end < len(lines) and lines[end].lstrip().startswith("--hash="):
    end += 1
if end == start + 1:
    raise SystemExit("torch has no hashes in miner/requirements.lock")
Path(sys.argv[2]).write_text("".join(lines[:start] + lines[end:]), encoding="utf-8")
Path(sys.argv[3]).write_text("".join(lines[start:end]), encoding="utf-8")
PYCODE

echo "==> Installing hash-locked Python dependencies"
uv pip install --python "$PYTHON" --no-deps --only-binary=:all: --require-hashes \
  -r "$ROOT/scripts/studio_b4/build-requirements.lock"
uv pip install --python "$PYTHON" --no-deps --only-binary=:all: --require-hashes \
  --index-url https://download.pytorch.org/whl/cpu -r "$TORCH_LOCK"
uv pip install --python "$PYTHON" --no-deps --only-binary=:all: --require-hashes \
  -r "$RUNTIME_LOCK"
uv pip install --python "$PYTHON" --no-deps --only-binary=:all: --require-hashes \
  -r "$ROOT/scripts/studio_b4/test-requirements.lock"

echo "==> Building the patched Pearl pool binding"
"$ROOT/scripts/pmk_build_pool_binding.sh" --python "$PYTHON" --install

echo "==> Installing the vendored Pearl runtime packages"
uv pip install --python "$PYTHON" --no-build-isolation --no-deps \
  "$ROOT/vendor/pearl/miner/miner-utils" \
  "$ROOT/vendor/pearl/miner/miner-base" \
  "$ROOT/vendor/pearl/miner/pearl-gateway"
uv pip check --python "$PYTHON"

echo "==> Generating G3 correctness vectors"
VECTORS="$ROOT/bench/k3sg/studio/vectors"
VECTORS_READY=1
for name in v1_256x256x4096 v2_256x256x4096_pm127 v3_128x128x65536 v4_pearl_c64; do
  for file in job.json A.bin B.bin tiles.bin; do
    [ -s "$VECTORS/$name/$file" ] || VECTORS_READY=0
  done
done
if [ "$VECTORS_READY" -eq 0 ]; then
  PYTHONPATH="$ROOT/bench/k3sg:$ROOT/bench/f1_k3${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$ROOT/bench/k3sg/make_vectors.py"
fi

echo "==> Building the G3 admission helper"
mkdir -p "$QUICKSTART/.build-work/libpmk" "$QUICKSTART/vectors/g3" "$QUICKSTART/bin"
rsync -a --delete --exclude .build "$ROOT/libpmk/" "$QUICKSTART/.build-work/libpmk/"
rsync -a --delete "$VECTORS/" "$QUICKSTART/vectors/g3/"
"$PYTHON" "$ROOT/scripts/studio_b4/build_g3_helper.py" "$ROOT" "$QUICKSTART"

echo "==> Running the full G3 correctness admission"
PYTHONPATH="$ROOT/miner${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$ROOT/scripts/pmk_quickstart.py" admission --force

echo
echo "Install complete. Start mining with:"
echo "  scripts/mine.sh --wallet <prl1...>"
