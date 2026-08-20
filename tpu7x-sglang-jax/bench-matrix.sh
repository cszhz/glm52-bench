#!/bin/bash
# Concurrency-sweep matrix, shaped after the harness in
# github.com/cszhz/Qwen-3.5-397B/scripts (bench_jax.sh / bench_all.sh) so the
# results line up cell-for-cell with theirs.
#
# Their harness is `vllm bench serve` against Qwen3.5-397B on vLLM/torch-tpu;
# this one is `sgl_jax.bench_serving` against GLM-5.2-FP8 on sglang-jax. Same
# methodology, different model and stack -- the numbers are NOT directly
# comparable, the shape of the sweep is.
#
# Flag mapping (verified against both codebases):
#   --random-range-ratio 0   (vllm) == --random-range-ratio 1.0 (sglang)
#       sglang: compute_random_lens() -> randint(int(len*ratio), len+1),
#       so 1.0 pins every prompt at exactly random_input_len.
#   --ignore-eos             (vllm) == sglang default
#       bench_serving.py:238 sends "ignore_eos": not args.disable_ignore_eos.
#   num_prompts = clamp(2*concurrency, 16, 128)   -- same formula as theirs.
#   seed 42                                       -- same as theirs.
#
# Usage:
#   ./bench-matrix.sh                 # D, E, A, C at c=1,2,4,8,16,32
#   WORKLOADS="A C" ./bench-matrix.sh
#   CONCURRENCIES="1 8 32" ./bench-matrix.sh
#   OUT_DIR=results/rerun ./bench-matrix.sh
#
# Resumable: a point whose .jsonl already exists is skipped, so an interrupted
# run can just be restarted.
set -uo pipefail

cd "$(dirname "$0")"
source ./env.sh

BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-$RESULT_DIR/$STAMP-matrix}"
BENCH_SEED="${BENCH_SEED:-42}"
mkdir -p "$OUT_DIR"

read -r -a CONCS <<< "${CONCURRENCIES:-1 2 4 8 16 32}"

# name:isl:osl -- labels match the reference harness's workload letters.
# B (1024/8192, decode-heavy) is defined but not in the default set: at ~48 ms
# per output token an 8192-token decode is ~6.5 min per request, which roughly
# doubles the total runtime. Add it with WORKLOADS="... B".
declare -A WL=(
    [D]="1024:1"      # pure prefill
    [E]="8192:1"      # pure prefill, the reference repo's headline metric
    [A]="1024:1024"   # balanced
    [C]="8192:1024"   # prefill-heavy
    [B]="1024:8192"   # decode-heavy
)
read -r -a ORDER <<< "${WORKLOADS:-D E A C}"

if ! curl -sf -m 5 "$BASE_URL/health" >/dev/null; then
    echo "bench-matrix.sh: no server at $BASE_URL -- run ./start-all.sh first" >&2
    exit 1
fi

# max_running_requests caps how many requests are actually in flight; anything
# above it just queues and inflates TTFT rather than measuring more parallelism.
for c in "${CONCS[@]}"; do
    if [ "$c" -gt "$MAX_RUNNING_REQUESTS" ]; then
        echo "bench-matrix.sh: NOTE c=$c exceeds --max-running-requests $MAX_RUNNING_REQUESTS;" >&2
        echo "  those requests queue server-side. Restart with a higher value to measure them." >&2
        break
    fi
done

run_point() {
    local wl="$1" isl="$2" osl="$3" conc="$4"
    local np=$(( conc * 2 ))
    (( np < 16 )) && np=16
    (( np > 128 )) && np=128

    local label="${wl}_in${isl}_out${osl}_c${conc}"
    local json="$OUT_DIR/${label}.jsonl"
    local log="$OUT_DIR/${label}.log"

    if [ -s "$json" ]; then
        echo "[skip] $label already exists"
        return 0
    fi

    echo "===== $label (num-prompts=$np) $(date +%H:%M:%S) ====="
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
        --num-prompts "$np" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --warmup-requests 1 \
        --seed "$BENCH_SEED" \
        --disable-tqdm \
        --output-file "$json" \
        --tag "$label" \
        > "$log" 2>&1

    local rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
        echo "[FAIL] $label (rc=$rc) -- see $log" >&2
        tail -5 "$log" >&2
        rm -f "$json"
        return 1
    fi
    grep -E "Request throughput|Total token throughput|Output token throughput|Mean TTFT|Mean TPOT|Successful requests" "$log" || true
}

for wl in "${ORDER[@]}"; do
    spec="${WL[$wl]:-}"
    if [ -z "$spec" ]; then
        echo "bench-matrix.sh: unknown workload '$wl' (have: ${!WL[*]})" >&2
        continue
    fi
    IFS=: read -r isl osl <<< "$spec"
    echo
    echo "########## Workload $wl: ${isl}/${osl} ##########"
    for c in "${CONCS[@]}"; do
        run_point "$wl" "$isl" "$osl" "$c" || true
    done
done

echo
echo "########## matrix complete $(date +%H:%M:%S) -> $OUT_DIR ##########"
"$PY" ./summarize.py "$OUT_DIR"
