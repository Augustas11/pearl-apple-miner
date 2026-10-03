#!/usr/bin/env bash
# pmk_miner local regtest integration harness.
#
# Usage:
#   scripts/pmk_regtest_e2e.sh [TARGET_NEW_BLOCKS=3] [TIMEOUT_SECONDS=1500]
#
# This wrapper owns the GPU lock. The Python harness inherits PMK_GPU_LOCK_HELD=1
# so the miner process does not try to take the same lock again.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_NEW_BLOCKS="${1:-${PMK_REGTEST_TARGET_BLOCKS:-3}}"
TIMEOUT_SECONDS="${2:-${PMK_REGTEST_TIMEOUT_SECONDS:-1500}}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
LOCK="${PMK_GPU_LOCK_DIR:-/tmp/pmm-gpu-bench.lock}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
EVIDENCE_DIR="$ROOT/bench/evidence"
EVIDENCE_PREFIX="${PMK_EVIDENCE_PREFIX:-b3}"
[[ "$EVIDENCE_PREFIX" =~ ^[A-Za-z0-9_]+$ ]] || { echo "invalid evidence prefix" >&2; exit 2; }
EVIDENCE_TXT="$EVIDENCE_DIR/${EVIDENCE_PREFIX}_regtest_e2e.txt"
HAVE_LOCK=0

cleanup() {
  local status=$?
  if [ "$HAVE_LOCK" = 1 ]; then
    rmdir "$LOCK" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$EVIDENCE_DIR"

echo "[pmk-e2e] waiting for GPU lock $LOCK"
while ! mkdir "$LOCK" 2>/dev/null; do
  echo "[pmk-e2e] GPU lock busy; retrying in 15s"
  sleep 15
done
HAVE_LOCK=1
export PMK_GPU_LOCK_HELD=1

echo "[pmk-e2e] GPU lock acquired"
echo "[pmk-e2e] writing evidence to $EVIDENCE_TXT"

PYTHONPATH="$ROOT/miner${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$ROOT/scripts/pmk_regtest_e2e.py" \
    --target-new-blocks "$TARGET_NEW_BLOCKS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    "$@" 2>&1 | tee "$EVIDENCE_TXT"
