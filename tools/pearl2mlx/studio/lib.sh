# Shared helpers for window scripts (convert, generate, perplexity, benchmark). Sourced, not executed.
# Read-only w.r.t. the machine: never touches launchd, other services, locks or
# sysctl settings. Run window scripts on an otherwise idle machine.
# shellcheck disable=SC2034  # variables are consumed by the sourcing window script
set -uo pipefail

PY=${PY:-$HOME/pearl2mlx-work/venv/bin/python}
VBIN=$(dirname "$PY")
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMLX=${TMLX:-$HOME/pearl2mlx-work}
BUDGET_S=${BUDGET_S:-2400}          # 40 min window budget
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}       # no downloads inside a window
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-$HF_HUB_OFFLINE}
START_S=$(date +%s)
FAILED=()
SKIPPED=()

PEARL_REPO=pearl-ai/Llama-3.1-8B-Instruct-pearl
PEARL_REV=5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0
MC8_REPO=mlx-community/Meta-Llama-3.1-8B-Instruct-8bit;  MC8_REV=142d42800404
MC4_REPO=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit;  MC4_REV=""
MCBF_REPO=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16; MCBF_REV=f8311090f9ee

init_log() {
  local stamp; stamp=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$TMLX/logs"
  LOG="$TMLX/logs/$1-$stamp.log"
  OUTD="$TMLX/logs/$1-$stamp.d"     # per-step outputs, parsed by summary.py
  mkdir -p "$OUTD"
  exec > >(tee -a "$LOG") 2>&1
  echo "window=$1 log=$LOG budget=${BUDGET_S}s tools=$TOOLS py=$PY HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
}

elapsed() { echo $(( $(date +%s) - START_S )); }

# step <name> <est_s> <timeout_s> <required 1|0> -- cmd args...
# Optional steps are skipped when elapsed + est would exceed the budget.
# Output goes to the log and to $OUTD/<name>.out. A non-zero rc marks FAIL.
step() {
  local name=$1 est=$2 to=$3 req=$4; shift 5
  local el; el=$(elapsed)
  if [ "$req" = 0 ] && [ $((el + est)) -gt "$BUDGET_S" ]; then
    echo "=== SKIP $name: elapsed ${el}s + est ${est}s > budget ${BUDGET_S}s"
    SKIPPED+=("$name"); return 0
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

preflight() {
  echo "--- preflight (read-only) $(date -u)"
  hostname; sw_vers 2>/dev/null
  sysctl hw.memsize
  sysctl iogpu.wired_limit_mb
  vm_stat
  df -h "$HOME"
  echo "--- watched / top-RSS processes (rss KB)"
  local pids; pids=$([ -n "${PMK_WATCH_PATTERN:-}" ] && pgrep -fi "$PMK_WATCH_PATTERN" | paste -sd, -)
  [ -n "$pids" ] && ps -o pid,rss,etime,command -p "$pids" | cut -c1-200 || echo "(no watched process; set PMK_WATCH_PATTERN to list one)"
  ps -axm -o pid,rss,comm | head -8
  "$PY" -c 'import sys, mlx.core as mx, mlx_lm; print("python", sys.version.split()[0], "mlx", mx.__version__, "mlx_lm", mlx_lm.__version__)'
  local mod
  for mod in datasets lm_eval; do
    "$PY" -c "import $mod" 2>/dev/null && echo "dep $mod: ok" || echo "dep $mod: MISSING"
  done
}

# missing <name>: record a step that cannot run because its model is not cached
missing() { echo "=== FAIL $1: model not in the HF cache"; FAILED+=("$1(not-cached)"); }

need_free_gb() {
  local kb; kb=$(df -k "$TMLX" | awk 'NR==2{print $4}')
  echo "free on $TMLX: $((kb / 1048576)) GB (need $1)"
  [ $((kb / 1048576)) -ge "$1" ] || { echo "FATAL: not enough disk"; exit 1; }
}

# resolve <var> <repo> <rev>: local HF-cache snapshot path or "" (logged)
resolve() {
  local p
  if p=$("$PY" "$TOOLS/studio/resolve.py" "$2" ${3:+"$3"}); then
    echo "resolved $2${3:+@$3} -> $p"; printf -v "$1" '%s' "$p"
  else
    echo "!!! $2${3:+@$3} not cached"; printf -v "$1" '%s' ""
  fi
}

finish() {
  local el; el=$(elapsed)
  echo "--- $1 finished in ${el}s; failed: ${FAILED[*]:-none}; skipped: ${SKIPPED[*]:-none}"
  "$PY" "$TOOLS/studio/summary.py" "$OUTD"
  local crit=$?
  echo "log: $LOG"
  [ ${#FAILED[@]} -eq 0 ] || exit 1
  [ "$crit" = 0 ] || exit 2
  exit 0
}
