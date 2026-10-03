import json, sys
for l in open(sys.argv[1]):
    if l[0] != '{': continue
    r = json.loads(l)
    if 'reps' not in r: print('FAIL', r.get('threads'), r.get('rc'), r.get('stderr','')[-300:]); continue
    print(f"k={r['k']} m={r['m']} n={r['n']} thr={r['threads']} rss={r.get('time_l_peak_rss_mb')} wall={r['wall_s']} load0={r['load_before'][0]:.1f}",
          [(round(x['prove_s'],1), round(x['verify_s'],3), x['proof_bytes']) for x in r['reps']])
