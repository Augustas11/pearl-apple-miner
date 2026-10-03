#!/usr/bin/env bash
# Extra-large correctness set (~1e8 output cells with the 2048/1024 set from gen_vectors.sh).
set -euo pipefail
cd "$(dirname "$0")"
O=oracle/target/release/v4-oracle
(cd oracle && cargo build --release -q)
K=4096
for spec in const:21 uniform:22 gauss:23 const:24 gauss:25; do
  f=${spec%%:*}; s=${spec##*:}
  $O gen $f 4096 4096 $K $s vectors/${f}_4096_s$s
  $O ref vectors/${f}_4096_s$s 4096 4096 $K
done
for spec in adv_uniform:26 adv_edge:27; do
  f=${spec%%:*}; s=${spec##*:}
  $O gen $f 2048 2048 $K $s vectors/${f}_2048_s$s
  $O ref vectors/${f}_2048_s$s 2048 2048 $K
done
