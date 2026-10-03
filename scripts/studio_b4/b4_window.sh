#!/usr/bin/env bash
# B4 long-run window entrypoint.
#
# Run from the bundle root after copying it to the target Mac, e.g.:
#   cd ~/pmk-b4 && ./b4_window.sh
#
# This script is intentionally local-only. If you pause other GPU workloads
# through your own wrapper, that wrapper owns lock/pause state; this script
# never touches those surfaces.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/scripts/studio_b4/b4_window.py" ]; then
  ROOT="$SCRIPT_DIR"
elif [ "$(basename "$SCRIPT_DIR")" = "studio_b4" ] && [ -f "$SCRIPT_DIR/b4_window.py" ]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
  echo "cannot locate bundle root from $SCRIPT_DIR" >&2
  exit 1
fi

TMLX="${TMLX:-$HOME/pmk-b4-work}"
LOG_DIR="${B4_LOG_ROOT:-$TMLX/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/b4-$STAMP.log"

exec > >(tee -a "$LOG") 2>&1
echo "b4_window log=$LOG root=$ROOT start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

PY_BOOT="${PYTHON:-${PY_BOOT:-${PMK_PYTHON:-python3}}}"
if [ ! -x "$PY_BOOT" ]; then
  PY_BOOT="${PY_BOOT_FALLBACK:-python3}"
fi

export B4_WINDOW_LOG="$LOG"
export B4_WINDOW_ROOT="$ROOT"
exec "$PY_BOOT" "$ROOT/scripts/studio_b4/b4_window.py" "$@"
