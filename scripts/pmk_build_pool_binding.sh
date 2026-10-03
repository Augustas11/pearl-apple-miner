#!/bin/bash
# Build py-pearl-mining with pmk's pool-bound helper patch in an isolated Pearl copy.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PEARL_ROOT=""
PYTHON="$ROOT/.venv/bin/python"
OUT_DIR="$ROOT/dist/pool-binding-wheels"
INSTALL=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pearl-root)
      PEARL_ROOT="$2"; shift 2 ;;
    --python)
      PYTHON="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --install)
      INSTALL=1; shift ;;
    *)
      echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PEARL_ROOT" ]; then
  PEARL_ROOT="$ROOT/dist/pool-binding-work/vendor/pearl"
  mkdir -p "$(dirname "$PEARL_ROOT")"
  rsync -a --delete \
    --exclude target --exclude .git --exclude __pycache__ --exclude .pytest_cache \
    "$ROOT/vendor/pearl/" "$PEARL_ROOT/"
fi

PEARL_ROOT=$(cd "$PEARL_ROOT" && pwd)
OUT_DIR=$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)
case "$PEARL_ROOT" in
  "$ROOT"/dist/*) ;;
  *)
    echo "refusing to patch non-isolated Pearl root: $PEARL_ROOT (must be under $ROOT/dist)" >&2
    exit 1
    ;;
esac

PATCH="$ROOT/miner/native_patches/0001-expose-pool-bound-helpers.patch"
TEST_FILE="$PEARL_ROOT/py-pearl-mining/src/lib.rs"
if ! grep -q "fn extract_difficulty_bound<'py>" "$TEST_FILE"; then
  patch -d "$PEARL_ROOT" -p1 < "$PATCH"
fi

mkdir -p "$OUT_DIR"
if ! "$PYTHON" -m maturin --version >/dev/null 2>&1; then
  echo "maturin is not installed for $PYTHON; install the pinned build requirements first" >&2
  exit 1
fi
(cd "$PEARL_ROOT/py-pearl-mining" && "$PYTHON" -m maturin build --release --locked --interpreter "$PYTHON" --out "$OUT_DIR")

WHEEL=$(ls -t "$OUT_DIR"/py_pearl_mining-*.whl | head -1)
if [ "$INSTALL" -eq 1 ]; then
  uv pip install --python "$PYTHON" --force-reinstall --no-deps "$WHEEL"
fi
echo "$WHEEL"
