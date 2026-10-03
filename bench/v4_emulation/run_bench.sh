#!/usr/bin/env bash
# Throughput of the bit-exact kernels, each batch bracketed by the plain int8 matmul2d baseline
# (bench/int8bench.swift kernel, 4096^3, tile 128x64) inside one /tmp/pmm-gpu-bench.lock window.
# Two rounds, kernels alternated. Output: ../evidence/v4_emulation_m5_bench.txt
set -uo pipefail
cd "$(dirname "$0")/metal"
swiftc -O v4emu.swift -o v4emu
OUT=../../evidence/v4_emulation_m5_bench.txt
REPS=${REPS:-5}
export V4EMU_COOL=${V4EMU_COOL:-8} V4EMU_ROUNDS=1
run() {  # env-config kernel dir size reps
  local cfg=$1 k=$2 d=$3 s=$4 r=$5
  echo "### kernel=$k cfg=[$cfg] dir=$(basename $d) shape=${s}x${s}x4096"
  env $cfg ./v4emu bench $d $s $s 4096 $r $k 2>&1 | grep -E "BENCH|baseline|fail|error"
}
{
  echo "# v4 emulation throughput on $(sysctl -n hw.model) ($(sysctl -n machdep.cpu.brand_string)), macOS $(sw_vers -productVersion), $(date)"
  echo "# TOPS-eq = 2*M*N*K / time. Each batch: 2 warm-up + $REPS timed reps, baseline int8 4096^3 x3 before and x3 after (same lock window)."
  for round in 1 2; do
    echo "## round $round"
    for fam in const uniform gauss; do
      for s in 2048 256; do
        d=../vectors/${fam}_${s}; [[ $s == 2048 ]] || d=../vectors/${fam}_256
        run "V4EMU_X=1" a $d $s $REPS
        run "V4EMU_TM=32 V4EMU_TN=32" b $d $s $REPS
        run "V4EMU_TM=32 V4EMU_TN=32" b3 $d $s $REPS
        R=$REPS; [[ $fam == const ]] || R=3
        run "V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1" c2 $d $s $R
        run "V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1" d $d $s $R
        run "V4EMU_X=1" e $d $s $R
        run "V4EMU_DEFINES=E_NO_CHECK" e $d $s $REPS
      done
    done
    for k in "V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1:d" "V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1:c2" "V4EMU_X=1:e" "V4EMU_DEFINES=E_NO_CHECK:e" "V4EMU_X=1:a" "V4EMU_TM=32 V4EMU_TN=32:b3"; do
      run "${k%%:*}" ${k##*:} ../vectors/const_4096_s21 4096 $REPS
    done
  done
} | tee $OUT
