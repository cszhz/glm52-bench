# GLM-5.2-FP8 on 8 × TPU7x chips (16 devices) — sglang-jax serving benchmark

`/gcs/models/GLM-5.2-FP8` on 2 nodes × 4 TPU7x chips, sglang-jax tree at
`/zzlfs/baseline/sglang-jax`.

A first working baseline, not a tuned upper bound — nothing here was swept for
throughput. `chunked-prefill-size` is 1024 rather than 2048 because 2048 fails
to compile (fused_v2 SMEM budget), which understates the long-prefill
workloads.

## Environment

| | |
|---|---|
| Hardware | 2 nodes × 4 TPU7x chips = **8 chips / 16 JAX devices** (each chip has 2 TensorCores; `local_device_count 8, global 16`) |
| Nodes | `zzl-tpu7x-slice-mig-0813-271m` 10.128.0.149 (rank 0, master) · `zzl-tpu7x-slice-mig-0813-2dvp` 10.128.0.150 (rank 1) |
| Stack | sglang-jax at `/zzlfs/baseline/sglang-jax`, jax 0.10.2, venv `/zzlfs/venv` |
| Model | `GLM-5.2-FP8` — block-wise FP8 weights (128×128, e4m3), 256 routed + 1 shared expert |
| Parallelism | `tp16 / dp16 / ep16` |
| Kernels | `--attention-backend dsa_sparse`, `--moe-backend fused_v2` |
| KV cache | bf16 |
| Key limits | `chunked-prefill-size 1024`, `mem-fraction-static 0.82`, `max-running-requests 32` |

## Usage

```bash
cd /zzlfs/baseline/glm52-bench/tpu7x-sglang-jax

./start-all.sh            # start both ranks, block until /health answers
./bench-matrix.sh         # concurrency matrix -> results/<date>-matrix/
./bench.sh 4096 1024 32   # ad-hoc single point: isl osl concurrency
./stop-all.sh             # kill both ranks, release the TPU

python3 summarize.py results/20260819-matrix   # regenerate any summary.json
```

Everything is parameterised in `env.sh`; override by exporting first, e.g.
`MEM_FRACTION_STATIC=0.85 ./start-all.sh`. `start-all.sh` ssh's into rank 1 and
`cd`s to `$SCRIPT_DIR`, which defaults to this directory — so the checkout has
to sit at the same path on both nodes.

| file | what it does |
|---|---|
| `env.sh` | all paths / cluster / model knobs, plus `server_args()` — the single source of truth for the launch argv |
| `start.sh` | starts one rank on the local node (rank auto-detected from IP) |
| `start-all.sh` | ssh-starts rank 1, starts rank 0, waits for health |
| `stop.sh` / `stop-all.sh` | terminate and verify the TPU is released |
| `bench-matrix.sh` | **the main benchmark** — workloads A–E × concurrency sweep; resumable |
| `bench.sh` | ad-hoc input-length sweep at fixed concurrency |
| `summarize.py` | aggregates a results directory into tables + `summary.json` |
| `make-tokenizer.sh` | regenerates `tokenizer/` — `bench_serving` needs a local copy |

## Results — concurrency matrix, 2026-08-19

24 points, 0 failures, single run per point (no repeats, no warm-up discard).
Raw: `results/20260819-matrix/`.

| workload | isl/osl | peak tot tok/s | at conc | mean TTFT there |
|---|---|---|---|---|
| D — pure prefill | 1024 / 1 | **7870.4** | 32 | 3.39 s |
| E — pure prefill | 8192 / 1 | **7318.8** | **16** | 17.8 s |
| C — prefill-heavy | 8192 / 1024 | 3425.9 | 32 | 24.7 s |
| A — balanced | 1024 / 1024 | 1201.1 | 32 | 3.50 s |

### D: 1024/1

| conc | req/s | tot tok/s | mean TTFT | p99 TTFT |
|---|---|---|---|---|
| 1 | 0.57 | 580.7 | 1.76 s | 1.81 s |
| 2 | 0.58 | 592.0 | 3.35 s | 3.55 s |
| 4 | 0.78 | 799.8 | 4.79 s | 5.18 s |
| 8 | 1.60 | 1640.9 | 4.67 s | 5.01 s |
| 16 | 5.31 | 5446.4 | 2.95 s | 3.01 s |
| 32 | 7.68 | **7870.4** | 3.39 s | 4.17 s |

### E: 8192/1

| conc | req/s | tot tok/s | mean TTFT | p99 TTFT |
|---|---|---|---|---|
| 1 | 0.06 | 485.1 | 16.9 s | 17.4 s |
| 2 | 0.09 | 751.2 | 21.5 s | 22.1 s |
| 4 | 0.19 | 1539.2 | 21.0 s | 21.4 s |
| 8 | 0.40 | 3270.7 | 19.7 s | 20.1 s |
| 16 | 0.89 | **7318.8** | 17.8 s | 18.0 s |
| 32 | 0.86 | 7028.4 | 29.7 s | 42.4 s |

### A: 1024/1024

| conc | req/s | tot tok/s | mean TTFT | p99 TTFT | TPOT | p99 TPOT |
|---|---|---|---|---|---|---|
| 1 | 0.02 | 42.0 | 1.76 s | 1.82 s | 45.95 ms | 46.01 ms |
| 2 | 0.04 | 82.6 | 1.88 s | 3.14 s | 46.60 ms | 47.87 ms |
| 4 | 0.08 | 160.5 | 2.24 s | 5.10 s | 47.70 ms | 50.40 ms |
| 8 | 0.15 | 317.0 | 2.94 s | 4.94 s | 47.62 ms | 50.23 ms |
| 16 | 0.30 | 617.2 | 3.19 s | 4.75 s | 48.76 ms | 50.55 ms |
| 32 | 0.59 | **1201.1** | 3.50 s | 4.17 s | 49.89 ms | 51.53 ms |

### C: 8192/1024

| conc | req/s | tot tok/s | mean TTFT | p99 TTFT | TPOT | p99 TPOT |
|---|---|---|---|---|---|---|
| 1 | 0.02 | 140.6 | 16.9 s | 17.4 s | 47.56 ms | 47.75 ms |
| 2 | 0.03 | 264.4 | 18.7 s | 22.0 s | 49.90 ms | 53.16 ms |
| 4 | 0.06 | 514.0 | 20.0 s | 21.6 s | 50.57 ms | 54.12 ms |
| 8 | 0.11 | 1046.8 | 19.5 s | 20.1 s | 49.76 ms | 53.70 ms |
| 16 | 0.23 | 2132.1 | 17.7 s | 18.0 s | 50.32 ms | 53.97 ms |
| 32 | 0.37 | **3425.9** | 24.7 s | 32.2 s | 59.95 ms | 70.45 ms |

Workload B (1024/8192) was not run: `WORKLOADS="D E A C B" ./bench-matrix.sh`.

## Results — input-length sweep, 2026-08-19

Fixed c=32, extending input length past the matrix's 8192. Separate `bench.sh`
run (`num_prompts = 4 × conc`, seed 3, vs the matrix's `clamp(2 × conc, 16,
128)`, seed 42), so not directly comparable to the matrix. Raw:
`results/20260819-lensweep/`.

| isl/osl | conc | tot tok/s | in tok/s | out tok/s | mean TTFT | TPOT |
|---|---|---|---|---|---|---|
| 1024/1024 | 1 | 42.0 | 21.0 | 21.0 | 1.78 s | 45.96 ms |
| 1024/1024 | 8 | 322.5 | 161.2 | 161.2 | 2.24 s | 47.47 ms |
| 1024/1024 | 32 | 1205.3 | 602.7 | 602.7 | 3.50 s | 49.71 ms |
| 4096/1024 | 32 | 2440.1 | 1952.1 | 488.0 | 12.4 s | 53.49 ms |
| 16384/1024 | 32 | 4134.8 | 3891.5 | 243.2 | 58.3 s | 74.64 ms |
| 32768/1024 | 32 | 3906.2 | 3787.8 | 118.4 | 160.2 s | 113.96 ms |
