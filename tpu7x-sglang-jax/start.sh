#!/bin/bash
# Start the sglang-jax server on THIS node only.
# Usage: ./start.sh [node_rank]     (rank is auto-detected from the local IP)
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

RANK="${1:-$(detect_rank)}"
LOG="$LOG_DIR/rank${RANK}.log"
PIDFILE="$LOG_DIR/rank${RANK}.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "start.sh: rank $RANK already running (pid $(cat "$PIDFILE"))" >&2
    exit 1
fi

# Refuse to start on top of somebody else's TPU job -- the chips are
# exclusive, and a second process just fails with a confusing libtpu error.
if pgrep -f 'sgl_jax[.]launch_server' >/dev/null 2>&1; then
    echo "start.sh: another sgl_jax.launch_server is already on this node's TPU:" >&2
    pgrep -af 'sgl_jax[.]launch_server' >&2
    echo "start.sh: run ./stop.sh (or kill it) first." >&2
    exit 1
fi

mapfile -t ARGS < <(server_args "$RANK" | tr ' ' '\n' | grep -v '^$')

{
    echo "=============================================================="
    echo "rank=$RANK host=$(hostname) started=$(date -Is)"
    echo "repo=$REPO"
    echo "python=$PY"
    echo "cmd: $PY -m sgl_jax.launch_server ${ARGS[*]}"
    echo "=============================================================="
} > "$LOG"

setsid "$PY" -m sgl_jax.launch_server "${ARGS[@]}" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo "start.sh: rank $RANK pid $(cat "$PIDFILE") -> $LOG"
