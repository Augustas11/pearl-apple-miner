#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the Pearl reference sources that pmk builds against into vendor/ (gitignored).
#
#   scripts/fetch_vendor.sh            # vendor/pearl at the pinned commit (needed by pmkcore)
#   scripts/fetch_vendor.sh --fp8      # also vendor/pearl-fp8 (needed by cert-v4)
#
# Pins can be overridden with PEARL_REV / PEARL_FP8_REV / PEARL_URL.
# OpenJarvis is NOT needed: the upgraded miner loop is already in upstream/openjarvis.
# bench/oj_mps_bench.py additionally needs a copy of OpenJarvis' _mps_miner_loop_main.py
# (see its header); fetch https://github.com/open-jarvis/OpenJarvis yourself if you want that benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEARL_URL="${PEARL_URL:-https://github.com/pearl-research-labs/pearl}"
PEARL_REV="${PEARL_REV:-7039e66f3c44f1541cb0e85328061e619ec5904d}"
PEARL_FP8_REV="${PEARL_FP8_REV:-f696760b259500ecb608469ea3953aeabbe78948}"

fetch() { # dir rev
  local dir="$ROOT/vendor/$1" rev="$2"
  mkdir -p "$ROOT/vendor"
  if [ ! -d "$dir/.git" ]; then
    git clone "$PEARL_URL" "$dir"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$rev" ]; then
    git -C "$dir" fetch origin "$rev" 2>/dev/null || git -C "$dir" fetch origin
    git -C "$dir" checkout --detach "$rev"
  fi
  echo "vendor/$1 at $(git -C "$dir" rev-parse HEAD)"
}

fetch pearl "$PEARL_REV"
if [ "${1:-}" = "--fp8" ]; then
  fetch pearl-fp8 "$PEARL_FP8_REV"
fi
