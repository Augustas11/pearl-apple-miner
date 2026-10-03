#!/usr/bin/env python3
"""Summarise bench/evidence/v4_emulation_m5_bench.txt and _verify.txt into markdown tables."""
import re
import sys
from collections import defaultdict
from pathlib import Path

ev = Path(__file__).resolve().parents[1] / "evidence"
bench = (ev / "v4_emulation_m5_bench.txt").read_text()
verify = (ev / "v4_emulation_m5_verify.txt").read_text() if (ev / "v4_emulation_m5_verify.txt").exists() else ""

# --- throughput ---------------------------------------------------------------------------
rows = defaultdict(list)  # (kernel, family, shape) -> [(med_tops, min_tops, ratio, base, fb)]
base_line = None
for line in bench.splitlines():
    if "baseline int8" in line:
        base_line = line
        continue
    m = re.search(r"BENCH r\d (.+?) (\S+) (\d+x\d+x\d+): median [\d.]+ ms \(([\d.]+) TOPS-eq\), min [\d.]+ ms \(([\d.]+) TOPS-eq\), reps \d+(.*)", line)
    if not m or base_line is None:
        continue
    kernel, d, shape, med, mn, tail = m.groups()
    b = re.search(r"([\d.]+) TOPS \| loadavg before \[([\d. ]+)\] after \[([\d. ]+)\].*= ([\d.]+) -> normalized", base_line)
    fam = re.sub(r"_(256|1024|2048|4096)(_s\d+)?$", "", d)
    fb = re.search(r"fallback(?:-windows)? ([\d.]+)%", tail)
    rows[(kernel, fam, shape)].append(
        (float(med), float(mn), float(b.group(4)), float(b.group(1)), fb.group(1) if fb else "-", b.group(2).split()[0], b.group(3).split()[0])
    )
    base_line = None

print("| Kernel | Family | Shape | TOPS-eq median (r1 / r2) | TOPS-eq min (best) | ratio to int8 baseline (median) | normalized x19 | baseline TOPS (r1 / r2) | loadavg 1m | fallback % |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (k, f, s), v in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
    meds = " / ".join(f"{x[0]:.3f}" for x in v)
    best = max(x[1] for x in v)
    ratios = sorted(x[2] for x in v)
    r = ratios[len(ratios) // 2]
    bases = " / ".join(f"{x[3]:.1f}" for x in v)
    la = " / ".join(x[5] for x in v)
    print(f"| {k} | {f} | {s} | {meds} | {best:.3f} | {r:.4f} | {r * 19:.2f} | {bases} | {la} | {v[0][4]} |")

# --- correctness --------------------------------------------------------------------------
if verify:
    tot = defaultdict(lambda: [0, 0, 0])  # kernel -> cells, mismatches, runs
    per = []
    for line in verify.splitlines():
        m = re.search(r"VERIFY (.+?) (\S+): (\d+) cells, (\d+) mismatches -> (\S+) \| (.*?) \|", line)
        if not m:
            continue
        k, d, cells, bad, verdict, extra = m.groups()
        kk = re.sub(r"\(.*", "", k)
        tot[kk][0] += int(cells)
        tot[kk][1] += int(bad)
        tot[kk][2] += 1
        fb = re.search(r"= ([\d.]+)%", extra)
        per.append((kk, d, int(cells), int(bad), verdict, fb.group(1) if fb else "-"))
    print()
    print("| Kernel | cells verified (all sets) | mismatches | exact |")
    print("|---|---|---|---|")
    for k, (c, b, n) in sorted(tot.items()):
        print(f"| {k} | {c:,} ({n} sets) | {b} | {'Y' if b == 0 else 'N'} |")
    print()
    print("| Kernel | Set | cells | mismatches | fallback % |")
    print("|---|---|---|---|---|")
    for kk, d, c, b, v, fb in per:
        print(f"| {kk} | {d} | {c:,} | {b} | {fb} |")
