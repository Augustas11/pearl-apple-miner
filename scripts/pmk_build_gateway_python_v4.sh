#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
# Build an isolated v4 Python gateway environment without mutating .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_ROOT="${PMK_PEARL_V4_BUILD_ROOT:-$ROOT/bench/v4_emulation/pearl-build}"
PIN="${PMK_PEARL_V4_PIN:-f696760b259500ecb608469ea3953aeabbe78948}"
SRC_ROOT="${PMK_PEARL_V4_SRC:-$BUILD_ROOT/src/pearl-$PIN}"
VENV="${PMK_GATEWAY_V4_VENV:-$BUILD_ROOT/gateway-python}"
BASE_PYTHON="${PMK_GATEWAY_V4_BASE_PYTHON:-$ROOT/.venv/bin/python}"
WHEEL_DIR="$BUILD_ROOT/wheels"
TARGET_DIR="$BUILD_ROOT/cache/py-pearl-mining-target"
PATCHED_GATEWAY_ROOT="$BUILD_ROOT/patched-gateway/pearl-gateway-v4"
PATCHED_GATEWAY_TMP_PARENT="$BUILD_ROOT/patched-gateway/tmp"

if [ -z "$SRC_ROOT" ] || [ ! -d "$SRC_ROOT/py-pearl-mining" ]; then
  echo "missing derived Pearl v4 source for pin $PIN; run scripts/pmk_build_pearld_v4.sh first" >&2
  exit 1
fi
if [ ! -f "$SRC_ROOT/.pmk-v4-pin" ]; then
  echo "missing derived Pearl source stamp: $SRC_ROOT/.pmk-v4-pin" >&2
  exit 1
fi
if [ "$(cat "$SRC_ROOT/.pmk-v4-pin")" != "$PIN" ]; then
  echo "derived Pearl source pin mismatch: $(cat "$SRC_ROOT/.pmk-v4-pin") != $PIN" >&2
  exit 1
fi
if [ ! -x "$BASE_PYTHON" ]; then
  echo "missing base Python: $BASE_PYTHON" >&2
  exit 1
fi
if ! "$BASE_PYTHON" -m maturin --version >/dev/null 2>&1; then
  echo "maturin is unavailable in $BASE_PYTHON" >&2
  exit 1
fi

mkdir -p "$WHEEL_DIR" "$TARGET_DIR" "$BUILD_ROOT/cache/cargo-home"
if [ ! -x "$VENV/bin/python" ]; then
  "$BASE_PYTHON" -m venv "$VENV"
fi

OLD_SITE="$("$BASE_PYTHON" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
NEW_SITE="$("$VENV/bin/python" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [ -n "$OLD_SITE" ] && [ -d "$OLD_SITE" ] && [ -n "$NEW_SITE" ]; then
  mkdir -p "$NEW_SITE"
  printf '%s\n' "$OLD_SITE" > "$NEW_SITE/pmk_v3_dependency_fallback.pth"
fi

export CARGO_HOME="$BUILD_ROOT/cache/cargo-home"
export CARGO_BUILD_JOBS="${PMK_V4_CARGO_JOBS:-2}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export CARGO_TARGET_DIR="$TARGET_DIR"

(cd "$SRC_ROOT/py-pearl-mining" && "$BASE_PYTHON" -m maturin build --release --locked --interpreter "$VENV/bin/python" --out "$WHEEL_DIR")
WHEEL="$(ls -t "$WHEEL_DIR"/py_pearl_mining-*.whl | head -1)"
uv pip install --python "$VENV/bin/python" --force-reinstall --no-deps "$WHEEL"

"$VENV/bin/python" - <<'PY'
import pearl_mining as p
assert int(p.CERT_VERSION_PLAIN_FP8) == 4
print("pearl_mining_v4_ready", p.__version__, p.CERT_VERSION_PLAIN_FP8)
PY

python3 - <<PY
import shutil
from pathlib import Path
for path in (Path("$PATCHED_GATEWAY_TMP_PARENT"), Path("$PATCHED_GATEWAY_ROOT")):
    shutil.rmtree(path, ignore_errors=True)
PY
mkdir -p "$PATCHED_GATEWAY_TMP_PARENT" "$(dirname "$PATCHED_GATEWAY_ROOT")"
PATCHED_GATEWAY_COPY="$($BASE_PYTHON "$ROOT/scripts/pmk_v4_gateway_adapter_check.py" --source "$SRC_ROOT/miner/pearl-gateway" --copy-parent "$PATCHED_GATEWAY_TMP_PARENT" --keep | awk '/^v4_gateway_adapter_ok root=/ {sub(/^v4_gateway_adapter_ok root=/, ""); print}')"
if [ -z "$PATCHED_GATEWAY_COPY" ] || [ ! -d "$PATCHED_GATEWAY_COPY/src" ]; then
  echo "failed to create patched V4 gateway copy" >&2
  exit 1
fi
mv "$PATCHED_GATEWAY_COPY" "$PATCHED_GATEWAY_ROOT"
python3 - <<PY
import shutil
from pathlib import Path
shutil.rmtree(Path("$PATCHED_GATEWAY_TMP_PARENT"), ignore_errors=True)
PY
PMK_GATEWAY_V4_PATCHED_SRC="$PATCHED_GATEWAY_ROOT/src" "$BASE_PYTHON" "$ROOT/scripts/pmk_v4_gateway_adapter_check.py" --source "$SRC_ROOT/miner/pearl-gateway" >/dev/null
printf 'patched_gateway_v4_ready %s\n' "$PATCHED_GATEWAY_ROOT/src"

echo "$VENV/bin/python"
