#!/usr/bin/env bash
# K3-SG M3 Ultra window (budget 30 min; expected ~8 min). Run on an otherwise idle machine with other GPU workloads paused;
# this script never touches launchd, other services, locks, /tmp/pmm-gpu-bench.lock or sysctl settings (all k3sg calls pass `nolock`).
# Steps: preflight -> (1) simdgroup_matrix layout probe -> (2) correctness vs pre-generated oracle vectors, all cfgs
# -> (3) cfg sweep at 4096^3 -> (4) paired perf (31 rounds) at 4096^2x4096 and 8192^2x4096 incl. base_f32 (P3 baseline),
# base_i8, fold, int8bench (Metal 4 matmul2d) and MLX fp32 -> (5) 240 s sustained k3 at 8192^2x4096, TOPS per 10 s.
# Log: ~/pmk-work/logs/sg-<UTC>.log (+ per-step outputs in sg-<UTC>.d/).
# Exit: 0 = all pass; 1 = step failure (crash / timeout / compile error / missing output);
#       2 = criterion failure (probe FAIL, any correctness FAIL, or P3 k3/base_f32 median < 0.80 at either shape).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/pmk-work/venv/bin/python}
TMLX=${TMLX:-$HOME/pmk-work}
BUDGET_S=${BUDGET_S:-1800}
CFGS=${CFGS:-64x64x16x2x2x1,64x64x16x2x2x0,64x64x16x2x2x2,64x64x32x2x2x1,64x64x8x2x2x2,128x64x16x4x2x1,128x64x16x4x2x2,64x128x16x2x4x1,128x128x16x4x4x1}
ROUNDS=${ROUNDS:-31}
SUSTAIN_S=${SUSTAIN_S:-240}
START_S=$(date +%s)
FAILED=(); SKIPPED=()
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$TMLX/logs"
LOG="$TMLX/logs/sg-$stamp.log"; OUTD="$TMLX/logs/sg-$stamp.d"; mkdir -p "$OUTD"
exec > >(tee -a "$LOG") 2>&1
echo "window=sg log=$LOG budget=${BUDGET_S}s here=$HERE py=$PY rounds=$ROUNDS sustain=${SUSTAIN_S}s"

elapsed() { echo $(( $(date +%s) - START_S )); }
# step <name> <est_s> <timeout_s> <required 1|0> -- cmd args...   (same contract as the pearl2mlx window helpers)
step() {
  local name=$1 est=$2 to=$3 req=$4; shift 5
  local el; el=$(elapsed)
  if [ "$req" = 0 ] && [ $((el + est)) -gt "$BUDGET_S" ]; then
    echo "=== SKIP $name: elapsed ${el}s + est ${est}s > budget ${BUDGET_S}s"; SKIPPED+=("$name"); return 0
  fi
  echo "=== STEP $name est=${est}s timeout=${to}s elapsed=${el}s at $(date -u +%H:%M:%SZ)"
  echo "+ $*"
  local t0 rc; t0=$(date +%s)
  perl -e 'alarm shift @ARGV; exec @ARGV or die "exec failed: $!\n"' "$to" "$@" 2>&1 | tee "$OUTD/$name.out"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 142 ] && echo "!!! $name: TIMEOUT after ${to}s"
  echo "=== END $name rc=$rc took=$(( $(date +%s) - t0 ))s"
  [ "$rc" = 0 ] || FAILED+=("$name")
  return 0
}

echo "--- preflight (read-only) $(date -u)"
hostname; sw_vers
sysctl -n hw.model machdep.cpu.brand_string hw.memsize hw.ncpu
system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores|Metal" | sed 's/^ */  /'
echo "loadavg $(sysctl -n vm.loadavg)"; uptime
pmset -g therm 2>&1 | sed 's/^/  /'
ps -Ao pid,pcpu,rss,comm -r | head -8
echo "--- package integrity"
MANIFEST_OUT=$(cd "$HERE" && shasum -a 256 -c MANIFEST.sha256 2>&1); MANIFEST_RC=$?
echo "$MANIFEST_OUT" | grep -v ': OK$'; echo "manifest check rc=$MANIFEST_RC"

BIN="$HERE/k3sg"
BIN_OUT=$("$BIN" 2>&1)
if [[ "$BIN_OUT" != *"usage: k3sg"* ]]; then
  echo "!!! prebuilt k3sg does not run here (verbatim):"; echo "$BIN_OUT" | head -5
  if command -v swiftc >/dev/null 2>&1; then
    echo "--- rebuilding from k3sg.swift with $(swiftc --version 2>&1 | head -1)"
    swiftc -O "$HERE/k3sg.swift" -o "$HERE/k3sg.local" 2>&1 | tail -20 && BIN="$HERE/k3sg.local"
  else
    echo "FATAL: no swiftc on this machine and the prebuilt binary does not run"; exit 1
  fi
fi
echo "binary: $BIN ($(shasum -a 256 "$BIN" | cut -c1-16))"

# (1) layout probe: fail closed -> correctness still runs (diagnostic) but the summary says DO NOT MINE
step probe 10 120 1 -- "$BIN" probe
# (2) correctness: every pre-generated job, every cfg, bit-exact in Swift against the oracle tiles
for v in "$HERE"/vectors/*/; do
  j=$(basename "$v")
  step "correct-$j" 60 600 1 -- "$BIN" run "$v" "cfgs=$CFGS"
done
# (3) cfg sweep (k3 only)
step sweep 120 600 1 -- "$BIN" sweep 4096x4096x4096 rounds=5 "cfgs=$CFGS" nolock
BEST=$(grep -E '^BEST ' "$OUTD/sweep.out" 2>/dev/null | awk '{print $2}')
BEST=${BEST:-${CFGS%%,*}}
echo "cfg for perf/sustain: $BEST"
# (4) paired perf
MLXARG=()
if [ -x "$PY" ] && "$PY" -c "import mlx.core" 2>/dev/null; then MLXARG=("mlx=$PY"); else echo "!!! MLX reference unavailable ($PY)"; fi
step perf 300 900 1 -- "$BIN" perf 4096x4096x4096 8192x8192x4096 "rounds=$ROUNDS" "cfg=$BEST" nolock ${MLXARG[@]+"${MLXARG[@]}"}
# (5) sustained
step sustain $((SUSTAIN_S + 30)) $((SUSTAIN_S + 120)) 0 -- "$BIN" sustain "$SUSTAIN_S" 8192x8192x4096 "cfg=$BEST" nolock

echo "--- post $(date -u) loadavg $(sysctl -n vm.loadavg)"; pmset -g therm 2>&1 | sed 's/^/  /'
echo "=================== SUMMARY (elapsed $(elapsed)s) ==================="
grep -h -E '^PROBE:|measured rows_pattern|^  FAIL' "$OUTD/probe.out" 2>/dev/null
grep -h -E 'JOB .*: (PASS|FAIL)|SKIP' "$OUTD"/correct-*.out 2>/dev/null | sed 's|/.*/vectors/||'
grep -h -E 'k3 cfg|^BEST' "$OUTD/sweep.out" 2>/dev/null
grep -h -E '^shape|^  (int8bench|mlx|base_f32|base_i8|fold|k3) |P3|k3/|fold/|base_' "$OUTD/perf.out" 2>/dev/null | grep -v 'raw ms' | cut -c1-140
grep -h -E '^  first' "$OUTD/sustain.out" 2>/dev/null
# criteria vs step failures: a probe/correctness step that exited non-zero but printed its FAIL verdict is a criterion
# failure (exit 2); anything that crashed, timed out or printed no verdict is a step failure (exit 1)
CRIT=(); STEPFAIL=()
[ "$MANIFEST_RC" = 0 ] || STEPFAIL+=("manifest")
grep -q '^PROBE: PASS' "$OUTD/probe.out" 2>/dev/null || CRIT+=("probe")
for f in "$OUTD"/correct-*.out; do
  n=$(basename "$f" .out)
  grep -qE 'JOB .*: PASS' "$f" || CRIT+=("$n")
done
[ "$(grep -c 'P3 gate.*PASS' "$OUTD/perf.out" 2>/dev/null)" = 2 ] || CRIT+=("P3")
for n in "${FAILED[@]:-}"; do
  [ -n "$n" ] || continue
  case "$n" in
    probe) grep -q '^PROBE: FAIL' "$OUTD/probe.out" || STEPFAIL+=("$n") ;;
    correct-*) grep -qE 'JOB .*: FAIL' "$OUTD/$n.out" || STEPFAIL+=("$n") ;;
    *) STEPFAIL+=("$n") ;;
  esac
done
echo "STEP FAILURES: ${STEPFAIL[*]:-none}; CRITERION FAILURES: ${CRIT[*]:-none}; SKIPPED: ${SKIPPED[*]:-none}"
if printf '%s\n' "${CRIT[@]:-}" | grep -qE '^(probe|correct-)' || printf '%s\n' "${STEPFAIL[@]:-}" | grep -qE '^(probe|correct-)'; then
  echo "VERDICT: DO NOT MINE on this device (probe or correctness did not pass)"
else
  echo "VERDICT: probe + correctness PASS on this device (mining allowed with cfg $BEST)"
fi
echo "log: $LOG"
[ ${#STEPFAIL[@]} -eq 0 ] || exit 1
[ ${#CRIT[@]} -eq 0 ] || exit 2
exit 0
