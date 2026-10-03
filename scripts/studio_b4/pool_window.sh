#!/usr/bin/env bash
# Pool-mode long-run window entrypoint.
#
# An optional outer wrapper owns pause/resume of other GPU workloads and GPU lock setup. This
# script only runs pmk in pool mode and records the miner's summary event.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/scripts/studio_b4/pool_window.py" ]; then
  ROOT="$SCRIPT_DIR"
elif [ "$(basename "$SCRIPT_DIR")" = "studio_b4" ] && [ -f "$SCRIPT_DIR/pool_window.py" ]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
  echo "cannot locate bundle root from $SCRIPT_DIR" >&2
  exit 1
fi

TMLX="${TMLX:-$HOME/pmk-b4-work}"
LOG_DIR="${POOL_LOG_ROOT:-$TMLX/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/pool-$STAMP.log"

exec > >(tee -a "$LOG") 2>&1
echo "pool_window log=$LOG root=$ROOT start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

PY_BOOT="${PYTHON:-${PY_BOOT:-${PMK_PYTHON:-python3}}}"
if [ ! -x "$PY_BOOT" ]; then
  PY_BOOT="${PY_BOOT_FALLBACK:-python3}"
fi

export POOL_WINDOW_LOG="$LOG"
export POOL_WINDOW_ROOT="$ROOT"
exec "$PY_BOOT" "$ROOT/scripts/studio_b4/pool_window.py" "$@"
