#!/usr/bin/env python3
"""Summarize native JSONL profiling and perf.py final JSON without hiding failures."""
import argparse
import json
import statistics
from pathlib import Path


def rows(path):
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def report(path):
    data = list(rows(path))
    native = [r for r in data if r.get('event') == 'pmk_job_profile']
    # Exclude the cold compilation/first allocation from steady-state medians.
    steady = native[2:] if len(native) > 4 else native
    result = {'evidence': str(path), 'native_jobs': len(native), 'median_ms': {}}
    med = result['median_ms']
    for key in ('cpu_validate_ms', 'cpu_alloc_ms', 'cpu_encode_ms', 'cpu_complete_ms'):
        if steady:
            med[key] = statistics.median(r[key] for r in steady)
    for label, names in [('K1', {'noise_dense_a', 'noise_dense_b', 'noise_sparse_a', 'noise_sparse_b'}),
                         ('K2', {'noise_apply_a', 'noise_apply_b', 'noise_apply_b_tiled'}), ('K3', {'k3'})]:
        values = [sum(s['gpu_ms'] for s in r['gpu_stages'] if s['name'] in names) for r in steady]
        if values:
            med[label] = statistics.median(values)
    if med.get('K3'):
        result['K1_K2_over_K3'] = (med['K1'] + med['K2']) / med['K3']
        ratios = []
        for row in steady:
            k3 = sum(stage['gpu_ms'] for stage in row['gpu_stages'] if stage['name'] == 'k3')
            noise = sum(stage['gpu_ms'] for stage in row['gpu_stages'] if stage['name'].startswith('noise_'))
            if k3:
                ratios.append(noise / k3)
        result['same_job_K1_K2_over_K3_median'] = statistics.median(ratios) if ratios else None
    final = next((r for r in reversed(data) if 'tops' in r or 'pipeline' in r or 'error' in r), None)
    if final:
        result['benchmark'] = final
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', nargs='+')
    args = parser.parse_args()
    print(json.dumps([report(p) for p in args.evidence], indent=2, sort_keys=True))
