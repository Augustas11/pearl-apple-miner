#!/bin/bash
# Full F3 sweep. Usage: bench/f3_proving/grid.sh
cd "$(dirname "$0")/../.."
LOG=bench/evidence/f3_proving_m5.txt
for k in 2048 4096 8192 16384 32768 65536; do
  .venv/bin/python bench/f3_proving/run_f3.py --log $LOG --k $k --threads all --reps 2 --max-load-wait ${LW:-1800}
done
