#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
seconds=60
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  seconds="$1"
  shift
fi
LOG_DIR="${HOME}/.pmk"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/benchmark.log"
echo "Benchmarking for ${seconds}s (full log: $LOG)..."
"$ROOT/scripts/mine.sh" --benchmark "$seconds" "$@" 2>&1 | tee "$LOG" | grep -v --line-buffered '^{'
exit "${PIPESTATUS[0]}"
