#!/bin/bash
# Second pass in a quiet window (loadavg < 3): clean all-core numbers with 3 reps, 1-thread k=4096, real-shape m=n=4096.
cd "$(dirname "$0")/../.."
LOG=bench/evidence/f3_proving_m5.txt
R=.venv/bin/python; S=bench/f3_proving/run_f3.py
echo '# --- pass 2 (quiet window) ---' >> $LOG
for k in 2048 4096 8192 16384; do $R $S --log $LOG --k $k --threads all --reps 3 --max-load-wait 300; done
$R $S --log $LOG --k 2048 --m 4096 --n 4096 --threads all --reps 2
$R $S --log $LOG --k 4096 --threads 1 --reps 2 --max-load-wait 300
