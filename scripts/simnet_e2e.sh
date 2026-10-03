#!/usr/bin/env bash
# End-to-end proof: pearld (simnet) <- pearl-gateway <- upgraded OpenJarvis MPS miner loop.
#
#   ./scripts/simnet_e2e.sh [TARGET_NEW_BLOCKS=2] [TIMEOUT_SECONDS=1200]
#
# Everything binds to 127.0.0.1, uses throwaway RPC credentials, keeps data and
# logs under run/ (gitignored), and stops every process on exit.
# Flags follow vendor/pearl/.github/workflows/integration_tests_ci.yml (simnet,
# --notls, --txindex, --addrindex, CI's simnet --miningaddr).
# Simnet activates MoEForkHeight=1 and SaltedSeedForkHeight=1
# (node/chaincfg/params.go), so every mined block requires cert version 3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_NEW_BLOCKS="${1:-2}"
TIMEOUT_SECONDS="${2:-1200}"
PEARLD="$ROOT/vendor/pearl/bin/pearld"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run/simnet"
LOGS="$RUN/logs"
RPC_PORT=44207
P2P_PORT=18655
GW_PORT=18337
RPC_USER="sim-$(openssl rand -hex 4)"
RPC_PASS="$(openssl rand -hex 16)"
MINING_ADDR="rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"

[ -x "$PEARLD" ] || { echo "missing $PEARLD (run scripts/build_pearld.sh)"; exit 1; }
rm -rf "$RUN"
mkdir -p "$RUN/pearld" "$LOGS"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  echo "[e2e] stopped pids: ${PIDS[*]:-}"
}
trap cleanup EXIT INT TERM

rpc() {  # rpc METHOD [JSON_PARAMS]
  curl -s --max-time 10 --user "$RPC_USER:$RPC_PASS" -H 'content-type: text/plain;' \
    --data-binary "{\"jsonrpc\":\"1.0\",\"id\":\"e2e\",\"method\":\"$1\",\"params\":${2:-[]}}" \
    "http://127.0.0.1:$RPC_PORT/"
}

echo "[e2e] starting pearld simnet (rpc 127.0.0.1:$RPC_PORT, p2p 127.0.0.1:$P2P_PORT)"
"$PEARLD" --simnet --datadir="$RUN/pearld" --logdir="$LOGS/pearld" \
  --rpcuser="$RPC_USER" --rpcpass="$RPC_PASS" --rpclisten="127.0.0.1:$RPC_PORT" \
  --listen="127.0.0.1:$P2P_PORT" --nodnsseed --notls --addrindex --txindex \
  --miningaddr="$MINING_ADDR" --debuglevel=info > "$LOGS/pearld.out" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 60); do
  rpc getblockcount | grep -q '"result"' && break
  sleep 1
done
START_HEIGHT="$(rpc getblockcount | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["result"])')"
echo "[e2e] pearld up; height=$START_HEIGHT"
rpc getblocktemplate '[{"rules":["segwit"]}]' | "$PY" -c '
import json, sys
r = json.load(sys.stdin)
if r.get("error"):
    print("[e2e] getblocktemplate error:", r["error"]); sys.exit(1)
t = r["result"]
print(f"[e2e] getblocktemplate: height={t["height"]} bits={t["bits"]} requiredcertversion={t.get("requiredcertversion")}")
'

echo "[e2e] starting pearl-gateway (miner rpc tcp 127.0.0.1:$GW_PORT)"
PEARLD_RPC_URL="http://127.0.0.1:$RPC_PORT" PEARLD_RPC_USER="$RPC_USER" PEARLD_RPC_PASSWORD="$RPC_PASS" \
PEARLD_MINING_ADDRESS="$MINING_ADDR" MINER_RPC_TRANSPORT=tcp MINER_RPC_PORT="$GW_PORT" \
MINER_RPC_HOST=127.0.0.1 \
  "$ROOT/.venv/bin/pearl-gateway" start --debug > "$LOGS/gateway.log" 2>&1 &
PIDS+=($!)

echo "[e2e] starting upgraded MPS miner loop (m=128 n=128 k=2048 rank=128)"
"$PY" -m oj_pearl_mps._mps_miner_loop_main --gateway-host 127.0.0.1 --gateway-port "$GW_PORT" \
  --m 128 --n 128 --k 2048 --rank 128 > "$LOGS/miner.log" 2>&1 &
PIDS+=($!)

GOAL=$((START_HEIGHT + TARGET_NEW_BLOCKS))
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))
HEIGHT="$START_HEIGHT"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 5
  HEIGHT="$(rpc getblockcount | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["result"])' 2>/dev/null || echo "$HEIGHT")"
  echo "[e2e] $(date +%H:%M:%S) height=$HEIGHT (goal $GOAL)"
  [ "$HEIGHT" -ge "$GOAL" ] && break
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "[e2e] process $pid died"; break 2; }; done
done

echo "[e2e] ---- chain"
for h in $(seq "$((START_HEIGHT + 1))" "$HEIGHT"); do
  hash="$(rpc getblockhash "[$h]" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["result"])')"
  rpc getblock "[\"$hash\", 1]" | "$PY" -c '
import json, sys
b = json.load(sys.stdin)["result"]
print(f"[e2e] block height={b["height"]} hash={b["hash"]} bits={b["bits"]} time={b["time"]}")
'
done
echo "[e2e] ---- gateway log (submission lines)"
grep -E "Block submission result|Block accepted|Block rejected|Rejecting proof|Updating block template|Error" "$LOGS/gateway.log" | tail -n 40 || true
echo "[e2e] ---- miner log (tail)"
grep -E "submitPlainProof|proof handed|rejected|failed|Error" "$LOGS/miner.log" | tail -n 20 || true
echo "[e2e] start_height=$START_HEIGHT final_height=$HEIGHT"
if [ "$HEIGHT" -ge "$GOAL" ]; then echo "[e2e] RESULT: PASS"; else echo "[e2e] RESULT: FAIL"; exit 1; fi
