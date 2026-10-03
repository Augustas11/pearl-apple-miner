#!/usr/bin/env python3
"""Summarize a window's per-step outputs (<log>.d/*.out) against kb §4.3 pass criteria.

Prints one JSON line of numbers and one line per criterion; exit 1 if any
criterion that has data fails (criteria without data print "n/a").
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def read(d: Path, name: str) -> str:
    p = d / f"{name}.out"
    return p.read_text(errors="replace") if p.exists() else ""


def ppl(d, tag):
    m = re.search(r"Perplexity: ([0-9.]+) ± ([0-9.]+)", read(d, f"ppl-{tag}"))
    return (float(m.group(1)), float(m.group(2))) if m else None


def bench(d, tag):
    m = re.findall(r"Averages: prompt_tps=([0-9.]+), generation_tps=([0-9.]+), peak_memory=([0-9.]+)",
                   read(d, f"bench-{tag}"))
    return dict(zip(("prompt_tps", "generation_tps", "peak_gb"), map(float, m[-1]))) if m else None


def convert(d, tag):
    m = re.findall(r"done (\{.*\})", read(d, f"convert-{tag}"))
    return json.loads(m[-1]) if m else None


def parity(d):
    p = d / "parity.json"
    return json.loads(p.read_text()) if p.exists() else None


def mmlu(d, tag):
    best = None
    for f in sorted((d / f"mmlu-{tag}").glob("*.json")) if (d / f"mmlu-{tag}").exists() else []:
        r = json.loads(f.read_text()).get("mmlu")
        if r:
            best = {k: v for k, v in r.items() if k.startswith("acc")}
    return best


def main(d: Path) -> int:
    nums = {
        "convert": {t: convert(d, t) for t in ("exact", "std8")},
        "ppl": {t: ppl(d, t) for t in ("exact", "std8", "mc8", "mcbf16")},
        "bench": {t: bench(d, t) for t in ("exact", "mc8", "mc4")},
        "mmlu": {t: mmlu(d, t) for t in ("exact", "mcbf16")},
    }
    par = parity(d)
    if par:
        nums["parity"] = par["pairs"]
        nums["kernel_check"] = par["kernel_check"]
    print("SUMMARY " + json.dumps(nums))

    crit = []
    cx = nums["convert"]["exact"]
    if cx:
        crit.append(("exact int7 layers verified dequant == int7*s (fp32)",
                     cx["int7_verified_bit_exact"] > 0, f"{cx['int7_verified_bit_exact']} layers"))
    if par and par["kernel_check"]:
        crit.append(("exact GPU kernel weights == bf16(int7*s)", all(par["kernel_check"].values()),
                     f"{sum(par['kernel_check'].values())}/{len(par['kernel_check'])} layers"))
    p = nums["ppl"]
    if p["exact"] and p["mc8"]:
        r = p["exact"][0] / p["mc8"][0]
        crit.append(("ppl exact <= +1% vs mlx-community 8bit", r <= 1.01, f"ratio {r:.4f}"))
    if p["exact"] and p["mcbf16"]:
        r = p["exact"][0] / p["mcbf16"][0]
        crit.append(("ppl exact <= +3% vs bf16", r <= 1.03, f"ratio {r:.4f}"))
    if par:
        prim = par["pairs"][0]
        crit.append((f"top-1 agreement >= 99% ({prim['mlx']} vs {prim['ref']})", prim["top1"] >= 0.99,
                     f"top1 {prim['top1']:.4f}, mean KL {prim['mean_kl']:.2e}"))
    b = nums["bench"]
    if b["exact"] and b["mc8"]:
        r = b["exact"]["generation_tps"] / b["mc8"]["generation_tps"]
        crit.append(("decode tok/s exact within 10% of mlx-community 8bit", r >= 0.9, f"ratio {r:.3f}"))
    if not crit:
        print("CRITERIA n/a (no data)")
    for name, ok, detail in crit:
        print(f"CRITERION {'PASS' if ok else 'FAIL'}: {name} [{detail}]")
    return 0 if all(ok for _, ok, _ in crit) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
