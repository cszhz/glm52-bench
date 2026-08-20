#!/bin/bash
# Ad-hoc input-length sweep at fixed concurrency, for one-off points that are
# not part of the A-E matrix. bench-matrix.sh is the main benchmark.
#
# Usage:
#   ./bench.sh                          # run the default sweep
#   ./bench.sh 1024 1024 32             # single point: isl osl concurrency
#   NUM_PROMPTS=128 ./bench.sh 4096 512 16
#
# Writes $RESULT_DIR/<date>-<tag>/in<isl>_out<osl>_c<conc>.jsonl per point --
# same layout as bench-matrix.sh minus the workload letter -- then aggregates
# with the same summarize.py.
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
TAG="${TAG:-lensweep}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-$RESULT_DIR/$STAMP-$TAG}"
mkdir -p "$OUT_DIR"

if ! curl -sf -m 5 "$BASE_URL/health" >/dev/null; then
    echo "bench.sh: no server at $BASE_URL -- run ./start-all.sh first" >&2
    exit 1
fi

# isl:osl:concurrency
if [ $# -ge 3 ]; then
    POINTS=("$1:$2:$3")
else
    POINTS=(
        "1024:1024:1"
        "1024:1024:8"
        "1024:1024:32"
        "4096:1024:32"
        "16384:1024:32"
        "32768:1024:32"
    )
fi

run_point() {
    local isl="$1" osl="$2" conc="$3"
    # Default to 4 rounds of the concurrency level, min 8 prompts.
    local n="${NUM_PROMPTS:-$(( conc * 4 < 8 ? 8 : conc * 4 ))}"
    local name="in${isl}_out${osl}_c${conc}"
    local json="$OUT_DIR/$name.jsonl"
    local log="$OUT_DIR/$name.log"

    echo "== $name (num_prompts=$n) =="
    curl -sf -m 30 -X POST "$BASE_URL/flush_cache" >/dev/null 2>&1 || true

    "$PY" -m sgl_jax.bench_serving \
        --backend sgl-jax \
        --base-url "$BASE_URL" \
        --model "$MODEL_PATH" \
        --tokenizer "$BENCH_TOKENIZER" \
        --dataset-name random \
        --random-input-len "$isl" \
        --random-output-len "$osl" \
        --random-range-ratio 1.0 \
        --num-prompts "$n" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --warmup-requests 1 \
        --seed "$RANDOM_SEED" \
        --disable-tqdm \
        --output-file "$json" \
        --tag "$name" \
        2>&1 | tee "$log"
}

for p in "${POINTS[@]}"; do
    IFS=: read -r isl osl conc <<< "$p"
    run_point "$isl" "$osl" "$conc" || echo "bench.sh: point $p FAILED (continuing)" >&2
done

echo
echo "== summary -> $OUT_DIR =="
"$PY" ./summarize.py "$OUT_DIR"
