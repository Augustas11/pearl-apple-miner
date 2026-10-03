#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
while ! mkdir /tmp/pmm-gpu-bench.lock 2>/dev/null; do
  echo 'GPU correctness lock occupied; retrying in 15 seconds.'
  sleep 15
done
trap 'rmdir /tmp/pmm-gpu-bench.lock' EXIT
mkdir -p libpmk/tests/evidence
{
  sw_vers
  sysctl -n hw.model
  echo 'B2 correctness only; production 64x64x16x2x2x2; no performance claims'
  swift test --package-path libpmk -c release
  .venv/bin/python -u libpmk/tests/integration.py
} 2>&1 | tee libpmk/tests/evidence/b2_libpmk_tests.txt
