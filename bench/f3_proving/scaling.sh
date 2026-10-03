#!/bin/bash
# Core scaling (k=4096 and k=16384) + re-run of large k that were disturbed by foreign load in the first sweep.
cd "$(dirname "$0")/../.."
LOG=bench/evidence/f3_proving_m5.txt
R=.venv/bin/python; S=bench/f3_proving/run_f3.py
for k in 4096 16384; do for t in 1 2 4 6 10; do $R $S --log $LOG --k $k --threads $t --reps 2 --max-load-wait 300; done; done
for k in 32768 65536; do $R $S --log $LOG --k $k --threads all --reps 1 --max-load-wait 600; done
