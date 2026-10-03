#!/usr/bin/env bash
# Certificate-VERIFYING end-to-end: pearld (regtest) <- pearl-gateway <- OpenJarvis MPS miner.
#
#   ./scripts/regtest_e2e.sh [TARGET_NEW_BLOCKS=3] [TIMEOUT_SECONDS=1500] [RANK64_SECONDS=300]
#
# Unlike simnet (BFNoPoWCheck: certificate + rank-penalty checks skipped,
# node/blockchain/process.go NetBehaviorFlags), regtest runs zkpow.VerifyCertificate and
# CheckRankPenalty. Phases: (1) positive: mine >=N blocks; the first genuine block is
# preceded by two corrupted copies (flipped proof byte / flipped public-data byte) that
# pearld must reject; (2) rank-64 miner, refused client-side (rank < PENALTY_BASE_RANK).
# Everything binds to 127.0.0.1, throwaway creds, data under run/regtest, all stopped on exit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_NEW_BLOCKS="${1:-3}"
TIMEOUT_SECONDS="${2:-1500}"
RANK64_SECONDS="${3:-20}"
PEARLD="$ROOT/vendor/pearl/bin/pearld"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run/regtest"
LOGS="$RUN/logs"
RPC_PORT=44307
P2P_PORT=18755
GW_PORT=18437
RPC_USER="reg-$(openssl rand -hex 4)"
RPC_PASS="$(openssl rand -hex 16)"
# Regtest and simnet share Bech32 HRP "rprl" (node/chaincfg/params.go), so the simnet address is valid.
MINING_ADDR="rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"
LOCK=/tmp/pmm-gpu-bench.lock
HAVE_LOCK=0

[ -x "$PEARLD" ] || { echo "missing $PEARLD (run scripts/build_pearld.sh)"; exit 1; }
rm -rf "$RUN"; mkdir -p "$RUN/pearld" "$LOGS"

PIDS=()
stop_pids() {
  for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true; done
  PIDS=()
}
stop_workers() {  # gateway + miner only; pearld keeps running for chain queries
  local keep=()
  for pid in "${PIDS[@]:-}"; do
    if [ "$pid" = "${NODE_PID:-}" ]; then keep+=("$pid"); else kill "$pid" 2>/dev/null || true; fi
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do [ "$pid" != "${NODE_PID:-}" ] && kill -9 "$pid" 2>/dev/null || true; done
  PIDS=("${keep[@]}")
}
cleanup() {
  stop_pids
  if [ "$HAVE_LOCK" = 1 ]; then rmdir "$LOCK" 2>/dev/null || true; fi
  echo "[e2e] stopped; gpu lock released"
}
trap cleanup EXIT INT TERM

rpc() { curl -s --max-time 10 --user "$RPC_USER:$RPC_PASS" -H 'content-type: text/plain;' \
  --data-binary "{\"jsonrpc\":\"1.0\",\"id\":\"e2e\",\"method\":\"$1\",\"params\":${2:-[]}}" "http://127.0.0.1:$RPC_PORT/"; }
height() { rpc getblockcount | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["result"])' 2>/dev/null; }

echo "[e2e] waiting for GPU lock $LOCK"
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
HAVE_LOCK=1; echo "[e2e] GPU lock acquired"

echo "[e2e] starting pearld REGTEST (rpc 127.0.0.1:$RPC_PORT)"
"$PEARLD" --regtest --datadir="$RUN/pearld" --logdir="$LOGS/pearld" \
  --rpcuser="$RPC_USER" --rpcpass="$RPC_PASS" --rpclisten="127.0.0.1:$RPC_PORT" \
  --listen="127.0.0.1:$P2P_PORT" --nodnsseed --notls --addrindex --txindex \
  --miningaddr="$MINING_ADDR" --debuglevel=info > "$LOGS/pearld.out" 2>&1 &
PIDS+=($!)
NODE_PID=$!
for _ in $(seq 1 60); do rpc getblockcount | grep -q '"result"' && break; sleep 1; done
START_HEIGHT="$(height)"
echo "[e2e] pearld up; height=$START_HEIGHT"
rpc getblocktemplate '[{"rules":["segwit"]}]' | "$PY" -c '
import json, sys
r = json.load(sys.stdin)
if r.get("error"): print("[e2e] getblocktemplate error:", r["error"]); sys.exit(1)
t = r["result"]
print(f"[e2e] getblocktemplate: height={t["height"]} bits={t["bits"]} requiredcertversion={t.get("requiredcertversion")}")
assert t.get("requiredcertversion") == 3
'

start_gateway() {  # start_gateway CORRUPT_FIRST
  PEARLD_RPC_URL="http://127.0.0.1:$RPC_PORT" PEARLD_RPC_USER="$RPC_USER" PEARLD_RPC_PASSWORD="$RPC_PASS" \
  PEARLD_MINING_ADDRESS="$MINING_ADDR" MINER_RPC_TRANSPORT=tcp MINER_RPC_PORT="$GW_PORT" MINER_RPC_HOST=127.0.0.1 \
  RTAP_LOG="$LOGS/tap.log" RTAP_CORRUPT_FIRST="$1" \
    "$PY" "$ROOT/scripts/regtest_gateway_tap.py" >> "$LOGS/gateway.log" 2>&1 &
  PIDS+=($!)
}
start_miner() {  # start_miner RANK LOGFILE
  "$PY" -m oj_pearl_mps._mps_miner_loop_main --gateway-host 127.0.0.1 --gateway-port "$GW_PORT" \
    --m 128 --n 128 --k 2048 --rank "$1" >> "$2" 2>&1 &
  PIDS+=($!)
}

echo "[e2e] === PHASE 1: positive (rank 128, k 2048) + corrupted-certificate negative control"
start_gateway 1
start_miner 128 "$LOGS/miner128.log"
GOAL=$((START_HEIGHT + TARGET_NEW_BLOCKS))
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))
HEIGHT="$START_HEIGHT"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 5
  HEIGHT="$(height || echo "$HEIGHT")"
  echo "[e2e] $(date +%H:%M:%S) height=$HEIGHT (goal $GOAL)"
  [ "$HEIGHT" -ge "$GOAL" ] && break
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "[e2e] process $pid died"; break 2; }; done
done
stop_workers

echo "[e2e] ---- chain"
for h in $(seq "$((START_HEIGHT + 1))" "$HEIGHT"); do
  hash="$(rpc getblockhash "[$h]" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["result"])')"
  rpc getblock "[\"$hash\", 1]" | "$PY" -c '
import json, sys
b = json.load(sys.stdin)["result"]
print(f"[e2e] block height={b["height"]} hash={b["hash"]} bits={b["bits"]} time={b["time"]}")'
done
echo "[e2e] ---- tap log (epoch-seconds, submitblock verdicts incl. NEGATIVE controls)"
cat "$LOGS/tap.log" || true
echo "[e2e] ---- find->accept latency (miner hands proof -> pearld submitblock accepted)"
"$PY" - "$LOGS/tap.log" <<'PYEOF'
import sys
found = None
for line in open(sys.argv[1]):
    t, _, rest = line.partition(" ")
    t = float(t)
    if rest.startswith("FOUND"):
        found = t
    elif rest.startswith("SUBMIT") and "verdict=accepted" in rest and found is not None:
        print(f"[e2e] {rest.split()[1]}: find->accept {t - found:.2f}s")
        found = None
PYEOF
echo "[e2e] ---- gateway log"
grep -aE "Block submission result|Block accepted|Block rejected|Rejecting proof" "$LOGS/gateway.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -n 40 || true
echo "[e2e] ---- miner log (found->handed timestamps)"
grep -aE "proof handed|submitPlainProof" "$LOGS/miner128.log" | tail -n 20 || true
echo "[e2e] ---- pearld log (rejections)"
grep -aiE "accepted block|reject|certificate|rank penalty" "$LOGS/pearld/regtest/pearld.log" 2>/dev/null | sort -u | tail -n 20 || true
POS=FAIL; [ "$HEIGHT" -ge "$GOAL" ] && POS=PASS
NEG=FAIL; grep -q "NEGATIVE corrupt_proof verdict=rejected" "$LOGS/tap.log" && grep -q "NEGATIVE corrupt_public_data verdict=rejected" "$LOGS/tap.log" && NEG=PASS

echo "[e2e] === PHASE 2: rank-64 miner (rank-penalty rule), ${RANK64_SECONDS}s"
H2="$(height)"
SUBS0="$(grep -c 'SUBMIT n=.* verdict=' "$LOGS/tap.log" || true)"
start_gateway 0
start_miner 64 "$LOGS/miner64.log"
D2=$(( $(date +%s) + RANK64_SECONDS ))
while [ "$(date +%s)" -lt "$D2" ]; do
  sleep 5
  [ "$(grep -c 'SUBMIT n=.* verdict=' "$LOGS/tap.log" || true)" -gt "$SUBS0" ] && break
done
stop_workers
echo "[e2e] ---- rank-64: the upgraded miner refuses to mine it client-side (PENALTY_BASE_RANK=128), so the node's rank-penalty rule cannot be reached without forging a proof"
grep -a "unminable shape" "$LOGS/miner64.log" | tail -n 1 || true
echo "[e2e] ---- rank-64 tap/gateway/miner"
tail -n 6 "$LOGS/tap.log"
sed 's/\x1b\[[0-9;]*m//g' "$LOGS/gateway.log" | grep -aE "Rejecting proof|Block rejected|Block accepted" | tail -n 5 || true
grep -aE "submitPlainProof|rejected" "$LOGS/miner64.log" | tail -n 5 || true
echo "[e2e] height before rank-64=$H2 after=$(height || echo n/a)"
echo "[e2e] start_height=$START_HEIGHT final_height=$HEIGHT"
echo "[e2e] RESULT positive=$POS negative_corrupt_cert=$NEG"
[ "$POS" = PASS ] && [ "$NEG" = PASS ]
