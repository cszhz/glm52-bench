#!/bin/bash
# Start both nodes and block until the server answers /health_generate.
# Usage: ./start-all.sh [timeout_seconds]      (default 3600)
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

TIMEOUT="${1:-3600}"

echo "== launching rank 1 on $NODE1 =="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE1" \
    "cd $SCRIPT_DIR && ./start.sh 1"

echo "== launching rank 0 on $NODE0 (local) =="
./start.sh 0

echo "== waiting for http://$NODE0:$PORT to come up (timeout ${TIMEOUT}s) =="
deadline=$(( SECONDS + TIMEOUT ))
while [ $SECONDS -lt $deadline ]; do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo
        echo "== server is up after ${SECONDS}s =="
        curl -s "http://127.0.0.1:$PORT/get_server_info" \
            | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ("model_path","context_length","tp_size","dp_size","ep_size","max_running_requests","attention_backend","moe_backend") if k in d}, indent=2))' \
            2>/dev/null || true
        exit 0
    fi
    # Bail out early if a rank died.
    for r in 0 1; do
        pf="$LOG_DIR/rank${r}.pid"
        [ -f "$pf" ] || continue
        if [ "$r" = 0 ]; then
            kill -0 "$(cat "$pf")" 2>/dev/null && continue
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE1" \
                "kill -0 \$(cat $pf) 2>/dev/null" && continue
        fi
        echo >&2
        echo "start-all.sh: rank $r exited; tail of its log:" >&2
        tail -40 "$LOG_DIR/rank${r}.log" >&2 || true
        exit 1
    done
    printf '.'
    sleep 10
done

echo >&2
echo "start-all.sh: timed out after ${TIMEOUT}s" >&2
tail -40 "$LOG_DIR/rank0.log" >&2 || true
exit 1
