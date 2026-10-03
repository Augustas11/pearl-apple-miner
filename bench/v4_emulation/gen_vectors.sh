#!/usr/bin/env bash
# Regenerates the test vectors (deterministic seeds) and Pearl-oracle outputs.
# Usage: ./gen_vectors.sh            (large set: 3 policy families 2048x2048x4096 + 2 adversarial 1024x1024x4096)
#        ./gen_vectors.sh small      (256x256x4096 for all 5 families)
set -euo pipefail
cd "$(dirname "$0")"
O=oracle/target/release/v4-oracle
(cd oracle && cargo build --release -q)
K=4096
if [[ "${1:-}" == small ]]; then
  for f in const uniform gauss adv_uniform adv_edge; do
    $O gen $f 256 256 $K 1 vectors/${f}_256
    $O ref vectors/${f}_256 256 256 $K
  done
  exit 0
fi
for f in const uniform gauss; do
  $O gen $f 2048 2048 $K 11 vectors/${f}_2048
  $O ref vectors/${f}_2048 2048 2048 $K
done
for f in adv_uniform adv_edge; do
  $O gen $f 1024 1024 $K 12 vectors/${f}_1024
  $O ref vectors/${f}_1024 1024 1024 $K
done
