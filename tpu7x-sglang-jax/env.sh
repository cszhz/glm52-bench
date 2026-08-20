#!/bin/bash
# Shared configuration for the 2-node GLM-5.2-FP8 baseline runs.
# Source this; every value can be overridden by exporting it beforehand.

# ---- paths -------------------------------------------------------------
export REPO="${REPO:-/zzlfs/baseline/sglang-jax}"
export VENV="${VENV:-/zzlfs/venv}"
export PY="${PY:-$VENV/bin/python3}"
# This directory, resolved from env.sh's own location -- start-all.sh ssh's
# `cd $SCRIPT_DIR` onto NODE1, so the checkout must live at the same path there.
export SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
export RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results}"

export MODEL_PATH="${MODEL_PATH:-/gcs/models/GLM-5.2-FP8}"
# bench_serving loads the tokenizer with a bare AutoTokenizer, which rejects
# this checkpoint's `tokenizer_class: TokenizersBackend`. This is a re-export of
# the very same tokenizer via sglang's own loader (identical token ids, verified)
# with that field normalised. Regenerate with ./make-tokenizer.sh.
export BENCH_TOKENIZER="${BENCH_TOKENIZER:-$SCRIPT_DIR/tokenizer}"

# ---- cluster -----------------------------------------------------------
export NODE0="${NODE0:-10.128.0.149}"
export NODE1="${NODE1:-10.128.0.150}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-$NODE0}"
export DIST_PORT="${DIST_PORT:-25000}"
export PORT="${PORT:-30000}"

# ---- runtime env -------------------------------------------------------
# The baseline tree is NOT the one installed into $VENV (that one points at
# /zzlfs/glm/sglang-jax-private), so PYTHONPATH has to shadow it.
export PYTHONPATH="$REPO/python"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"
# Separate cache dir so baseline compilations don't mix with other rounds.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/zzlfs/jax_cache_baseline}"
# Pin TPU clocks to the top P-state, same as the other GLM-5.2 runs here.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_dvfs_p_state=7}"
# Escape hatch for the fused_v2 SMEM budget (narrows the routing scratch from
# padded_top_k=128 to top_k=8). NOT needed once the extend bucket hits a tuned
# block config -- see README "fused_v2 SMEM". Export =1 to re-enable.
export FUSED_MOE_V2_ROUTE_SMEM_TOPK_ONLY="${FUSED_MOE_V2_ROUTE_SMEM_TOPK_ONLY:-0}"

# ---- model / parallelism knobs ----------------------------------------
export TP_SIZE="${TP_SIZE:-16}"
export DP_SIZE="${DP_SIZE:-16}"
export EP_SIZE="${EP_SIZE:-16}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-135168}"
export PAGE_SIZE="${PAGE_SIZE:-64}"
# 2048 (as originally specified) makes the global extend bucket 2048x16=32768,
# which has no entry in the fused_v2 tuned block table -> falls back to
# DEFAULT (bt=32) -> 64 SMEM banks -> compile-time SMEM OOM. 1024x16=16384 is
# the largest tuned GLM-5.2/ep16 shape. Still a multiple of page_size 64.
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
# 0.90 (as originally specified) OOMs: it leaves only 94.75 GB/chip for HLO
# temporaries while the extend precompile needs 101.84 GB. See README.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
# Precompile buckets are GLOBAL (summed over DP ranks):
#   tokens 16384 = chunked_prefill_size 1024 x dp 16
#   bs        32 = 2 per DP rank, and fused-MoE needs bs >= 2 * tp_size
export PRECOMPILE_BS_PADDINGS="${PRECOMPILE_BS_PADDINGS:-32}"
export PRECOMPILE_TOKEN_PADDINGS="${PRECOMPILE_TOKEN_PADDINGS:-16384}"
export RANDOM_SEED="${RANDOM_SEED:-3}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$JAX_COMPILATION_CACHE_DIR"

# ---- helpers -----------------------------------------------------------
# Resolve this host's node rank from its primary IP.
detect_rank() {
    local ips
    ips="$(hostname -I)"
    case " $ips " in
        *" $NODE0 "*) echo 0; return 0 ;;
        *" $NODE1 "*) echo 1; return 0 ;;
    esac
    echo "env.sh: cannot map local IPs ($ips) to NODE0=$NODE0 / NODE1=$NODE1" >&2
    return 1
}

# Build the launch_server argv. Kept in one place so start.sh and the README
# can never drift apart.
server_args() {
    local rank="$1"
    cat <<EOF
--model-path $MODEL_PATH
--trust-remote-code
--device tpu
--dtype bfloat16
--kv-cache-dtype bf16
--attention-backend dsa_sparse
--dsa-use-pallas
--page-size $PAGE_SIZE
--chunked-prefill-size $CHUNKED_PREFILL_SIZE
--max-prefill-tokens $MAX_PREFILL_TOKENS
--context-length $CONTEXT_LENGTH
--tp-size $TP_SIZE
--dp-size $DP_SIZE
--dp-schedule-policy round_robin
--ep-size $EP_SIZE
--moe-backend fused_v2
--mem-fraction-static $MEM_FRACTION_STATIC
--max-running-requests $MAX_RUNNING_REQUESTS
--precompile-bs-paddings $PRECOMPILE_BS_PADDINGS
--precompile-token-paddings $PRECOMPILE_TOKEN_PADDINGS
--skip-server-warmup
--random-seed $RANDOM_SEED
--stream-output
--stream-interval 1
--nnodes $WORLD_SIZE
--node-rank $rank
--dist-init-addr $MASTER_ADDR:$DIST_PORT
--host 0.0.0.0
--port $PORT
EOF
}
