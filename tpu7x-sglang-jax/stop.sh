#!/bin/bash
# Stop the server on THIS node.
cd "$(dirname "$0")"
source ./env.sh

for pf in "$LOG_DIR"/rank*.pid; do
    [ -f "$pf" ] || continue
    pid="$(cat "$pf")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "stop.sh: SIGTERM $pid ($(basename "$pf"))"
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
done

# launch_server forks a scheduler/detokenizer tree; make sure nothing keeps
# the TPU claimed.
for _ in $(seq 1 15); do
    pgrep -f 'sgl_jax[.]launch_server' >/dev/null 2>&1 || break
    sleep 1
done
if pgrep -f 'sgl_jax[.]launch_server' >/dev/null 2>&1; then
    echo "stop.sh: still alive, SIGKILL"
    pkill -9 -f 'sgl_jax[.]launch_server' || true
    pkill -9 -f 'sglang[:][:]' || true
    sleep 2
fi

if pgrep -f 'sgl_jax[.]launch_server' >/dev/null 2>&1; then
    echo "stop.sh: WARNING processes remain:" >&2
    pgrep -af 'sgl_jax[.]launch_server' >&2
    exit 1
fi
echo "stop.sh: clean on $(hostname)"
