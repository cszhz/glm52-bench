# GLM-5.2-FP8 serving benchmarks — TPU7x

`GLM-5.2-FP8` against a fixed workload matrix on [`tpu7x-sglang-jax/`](tpu7x-sglang-jax/).

| | |
|---|---|
| hardware | 8 × TPU7x chips = 16 JAX devices (2 nodes × 4 chips) |
| stack | sglang-jax (JAX / Pallas) |
| parallelism | tp16 / dp16 / ep16 |
| run | baseline matrix + input-length sweep |
| headline | untuned first baseline; peak 7.9k tok/s (D), 7.3k (E), all at `--max-running-requests 32` |
| date | 2026-08-19 |

## Workloads

| workload | isl / osl | stresses |
|---|---|---|
| A | 1024 / 1024 | balanced |
| B | 1024 / 8192 | decode-heavy |
| C | 8192 / 1024 | prefill-heavy |
| D | 1024 / 1 | pure prefill, short |
| E | 8192 / 1 | pure prefill, long |

## Peak total token throughput

| workload | isl/osl | 8 × TPU7x, sglang-jax, bf16 KV |
|---|---|---|
| D | 1024 / 1 | 7,870 |
| E | 8192 / 1 | 7,319 |
| C | 8192 / 1024 | 3,426 |
| A | 1024 / 1024 | 1,201 |
| B | 1024 / 8192 | not run |

**Nothing here is tuned.** This is a first baseline — the long-prefill numbers
are held back by `chunked-prefill-size 1024` and `mem-fraction-static 0.82`,
both forced rather than chosen.

## Result layout

```
tpu7x-sglang-jax/results/<date>-<run>/[<workload>_]in<isl>_out<osl>_c<conc>.jsonl
                                     summary.json
```

Each `.jsonl` is the raw benchmark record — full `server_info` with every
effective server argument, plus per-request timings. `summary.json` is
generated, never hand-edited; regenerate with `summarize.py`. Points outside
the A–E matrix (the length sweep) carry no workload letter.
