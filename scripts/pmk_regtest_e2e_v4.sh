#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
# Local cert-v4 regtest harness wrapper for the native PMK miner.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_NEW_BLOCKS="${1:-${PMK_REGTEST_TARGET_BLOCKS:-3}}"
TIMEOUT_SECONDS="${2:-${PMK_REGTEST_TIMEOUT_SECONDS:-1500}}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi

LOCK="${PMK_GPU_LOCK_DIR:-/tmp/pmm-gpu-bench.lock}"
export PMK_GATEWAY_PYTHON_V4="${PMK_GATEWAY_PYTHON_V4:-$ROOT/bench/v4_emulation/pearl-build/gateway-python/bin/python}"
PY="${PYTHON:-$PMK_GATEWAY_PYTHON_V4}"
EVIDENCE_DIR="$ROOT/bench/evidence"
EVIDENCE_PREFIX="${PMK_EVIDENCE_PREFIX:-b9_v4}"
EVIDENCE_TXT="$EVIDENCE_DIR/${EVIDENCE_PREFIX}_regtest_e2e.txt"
for arg in "$@"; do
  if [ "$arg" = "--preflight-only" ]; then
    EVIDENCE_TXT="$EVIDENCE_DIR/${EVIDENCE_PREFIX}_regtest_e2e_preflight.txt"
    break
  fi
done

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
NEEDS_GPU_LOCK=1
for arg in "$@"; do
  if [ "$arg" = "--preflight-only" ]; then
    NEEDS_GPU_LOCK=0
    break
  fi
done

if [ "$NEEDS_GPU_LOCK" = 1 ]; then
  echo "[pmk-v4-e2e] waiting for GPU lock $LOCK"
  while ! mkdir "$LOCK" 2>/dev/null; do
    echo "[pmk-v4-e2e] GPU lock busy; retrying in 15s"
    sleep 15
  done
  HAVE_LOCK=1
  export PMK_GPU_LOCK_HELD=1
  echo "[pmk-v4-e2e] GPU lock acquired"
else
  echo "[pmk-v4-e2e] preflight-only; skipping GPU lock"
fi
echo "[pmk-v4-e2e] writing evidence to $EVIDENCE_TXT"

PYTHONPATH="$ROOT/miner${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$ROOT/scripts/pmk_regtest_e2e_v4.py" \
    --target-new-blocks "$TARGET_NEW_BLOCKS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    "$@" 2>&1 | tee "$EVIDENCE_TXT"
