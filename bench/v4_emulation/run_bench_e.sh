#!/usr/bin/env bash
# Kernel E (fp32 simdgroup grid-exact, rewritten 2x2-block version) and its no-check GEMM reference,
# same protocol as run_bench.sh. Output: ../evidence/v4_emulation_m5_bench_e.txt
set -uo pipefail
cd "$(dirname "$0")/metal"
swiftc -O v4emu.swift -o v4emu
OUT=../../evidence/v4_emulation_m5_bench_e.txt
export V4EMU_COOL=${V4EMU_COOL:-8} V4EMU_ROUNDS=1
{
  echo "# kernel E (EB=2 blocks/SIMD-group, EW=1) on $(sysctl -n hw.model), $(date)"
  for round in 1 2; do
    echo "## round $round"
    for spec in const_2048:2048 const_256:256 const_4096_s21:4096 uniform_2048:2048 gauss_2048:2048; do
      d=../vectors/${spec%%:*}; s=${spec##*:}
      R=5; [[ $d == *const* ]] || R=3
      echo "### kernel=e dir=$(basename $d) shape=${s}x${s}x4096"
      ./v4emu bench $d $s $s 4096 $R e 2>&1 | grep -E "BENCH|baseline|fail|error"
      echo "### kernel=e cfg=[V4EMU_DEFINES=E_NO_CHECK] dir=$(basename $d) shape=${s}x${s}x4096"
      V4EMU_DEFINES=E_NO_CHECK ./v4emu bench $d $s $s 4096 5 e 2>&1 | grep -E "BENCH|baseline|fail|error"
    done
  done
} | tee $OUT
