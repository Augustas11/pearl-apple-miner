#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
seconds=60
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  seconds="$1"
  shift
fi
v4=0
forward=()
for argument in "$@"; do
  if [ "$argument" = "--v4" ]; then
    v4=1
  else
    forward+=("$argument")
  fi
done
LOG_DIR="${HOME}/.pmk"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/benchmark.log"
if [ "$v4" -eq 1 ]; then
  if [ "${#forward[@]}" -ne 0 ]; then
    echo "benchmark: --v4 does not accept miner benchmark options" >&2
    exit 2
  fi
  LOCK="${PMK_GPU_LOCK_DIR:-/tmp/pmm-gpu-bench.lock}"
  owned=0
  cleanup() {
    if [ "$owned" -eq 1 ]; then
      rmdir "$LOCK" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  if [ "${PMK_GPU_LOCK_HELD:-0}" != 1 ]; then
    echo "Waiting for GPU lock $LOCK..."
    until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
    owned=1
    export PMK_GPU_LOCK_HELD=1
  fi
  export PMK_V4_G3_ADMISSION_FILE="${PMK_HOME:-$HOME/.pmk}/v4-g3-admission.json"
  V4_LOG="$LOG_DIR/benchmark-v4.jsonl"
  echo "Running paired v3/v4 benchmark (full log: $V4_LOG)..."
  "$ROOT/.venv/bin/python" "$ROOT/scripts/pmk_v4_paired_bench.py" \
    --run-gpu --quick --warmup 1 --measured 2 --output "$V4_LOG" >/dev/null
  "$ROOT/.venv/bin/python" - "$V4_LOG" <<'PYCODE'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
summary = next(row for row in reversed(rows) if row.get("event") == "summary")
print(
    f"v3 {summary['v3']['tops_eq_median']:.2f} TOPS | "
    f"v4 {summary['v4']['tops_eq_median']:.2f} TOPS-eq | "
    f"v4/v3 {summary['v4_to_v3_tops_ratio']:.2f}x"
)
PYCODE
  exit 0
fi
echo "Benchmarking for ${seconds}s (full log: $LOG)..."
"$ROOT/scripts/mine.sh" --benchmark "$seconds" "${forward[@]}" 2>&1 | tee "$LOG" | grep -v --line-buffered '^{'
exit "${PIPESTATUS[0]}"
