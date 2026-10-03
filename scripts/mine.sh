#!/usr/bin/env bash
# Foreground entry point; exec preserves Ctrl-C and the miner's exit status.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo 'Usage: scripts/mine.sh --wallet <prl1...> [--worker <name>] [--pool <url>]'
  echo 'Default pool: stratum+tcp://sg.pearl.herominers.com:1200'
  echo 'Default worker: sanitized hostname -s. Ctrl-C stops mining.'
  exit 0
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo 'Run scripts/install.sh first.' >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/pmk_mine.py" "$@"
