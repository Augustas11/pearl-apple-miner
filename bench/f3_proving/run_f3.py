"""Machine-sharing-aware runner: lock /tmp/pmm-gpu-bench.lock, wait for loadavg<=8, run prove_one.py as a subprocess,
log loadavg before/after and /usr/bin/time -l peak RSS. Appends JSON lines to the --log file.
  .venv/bin/python bench/f3_proving/run_f3.py --log bench/evidence/f3_proving_m5.txt --k 4096 --threads 1 [--m 128 --n 64 --reps 2]
"""
import argparse, json, os, subprocess, sys, time, re

LOCK = "/tmp/pmm-gpu-bench.lock"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load1():
    return os.getloadavg()[0]


def wait_load(max_wait):
    t0 = time.time()
    while load1() > 8 and time.time() - t0 < max_wait:
        time.sleep(15)
    return load1() > 8  # True = proceeding while loaded (flag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--threads", default="all")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-load-wait", type=int, default=180)
    a = ap.parse_args()
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    if a.threads != "all":
        env["RAYON_NUM_THREADS"] = a.threads
    while True:  # lock
        try:
            os.mkdir(LOCK); break
        except FileExistsError:
            time.sleep(15)
    try:
        flagged = wait_load(a.max_load_wait)
        l0 = os.getloadavg()
        cmd = ["/usr/bin/time", "-l", os.path.join(ROOT, ".venv/bin/python"), os.path.join(HERE, "prove_one.py"),
               "--k", str(a.k), "--m", str(a.m), "--n", str(a.n), "--reps", str(a.reps)]
        t0 = time.time()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
        wall = time.time() - t0
        l1 = os.getloadavg()
    finally:
        os.rmdir(LOCK)
    rec = dict(threads=a.threads, wall_s=round(wall, 2), load_before=l0, load_after=l1, loaded_flag=flagged, rc=p.returncode)
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    if m: rec["time_l_peak_rss_mb"] = round(int(m.group(1)) / 2**20, 1)
    try:
        rec.update(json.loads(p.stdout.strip().splitlines()[-1]))
    except Exception:
        rec["stdout"] = p.stdout[-500:]; rec["stderr"] = p.stderr[-800:]
    line = json.dumps(rec)
    print(line)
    with open(a.log, "a") as f:
        f.write(line + "\n")

main()
