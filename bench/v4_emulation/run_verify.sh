#!/usr/bin/env bash
# Bit-exactness of every kernel variant vs Pearl's oracle on all generated vector sets.
# No GPU lock needed (correctness only). Output: ../evidence/v4_emulation_m5_verify.txt
set -uo pipefail
cd "$(dirname "$0")/metal"
swiftc -O v4emu.swift -o v4emu
OUT=../../evidence/v4_emulation_m5_verify.txt
{
  echo "# v4 emulation bit-exactness vs Pearl B200::matmul_fp8 (vendor/pearl-fp8 2569546), $(date)"
  echo "# host: $(sysctl -n hw.model) $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion)"
  for d in ../vectors/*/; do
    d=${d%/}; [[ -f $d/c_b200.bin ]] || continue
    [[ $(basename $d) == kdep_* ]] && continue  # k != 4096 sets: see v4_emulation_m5_kdep.txt
    n=$(basename $d); sz=${n##*_}; sz=${sz%%_*}
    case $n in *_256*) S=256;; *_1024*) S=1024;; *_2048*) S=2048;; *_4096*) S=4096;; esac
    echo "## $n (${S}x${S}x4096)"
    ./v4emu verify $d $S $S 4096 a 2>&1 | grep -E "VERIFY|got|fail"
    V4EMU_TM=32 V4EMU_TN=32 ./v4emu verify $d $S $S 4096 b,b3 2>&1 | grep -E "VERIFY|got|fail"
    V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1 ./v4emu verify $d $S $S 4096 c2 2>&1 | grep -E "VERIFY|got|fail"
    V4EMU_TM=64 V4EMU_TN=32 V4EMU_WG=1 ./v4emu verify $d $S $S 4096 d 2>&1 | grep -E "VERIFY|got|fail"
    ./v4emu verify $d $S $S 4096 e 2>&1 | grep -E "VERIFY|got|fail"
  done
} | tee $OUT
echo "cells verified per kernel:"; grep VERIFY $OUT | awk '{print $2}' | sort | uniq -c
