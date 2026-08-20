#!/usr/bin/env python3
"""Aggregate a results directory into a per-workload table + summary.json.

Mirrors the aggregation in the reference harness's bench_all.sh: per workload,
report every concurrency point and flag the one with the highest total token
throughput.

Reads the raw `<WL>_in<isl>_out<osl>_c<conc>.jsonl` points that bench-matrix.sh
writes. The workload letter is optional, so the ad-hoc points from bench.sh
(`in<isl>_out<osl>_c<conc>.jsonl`, no A-E label) aggregate the same way; those
group under an `isl/osl` heading instead of a letter.
"""

import json
import pathlib
import re
import sys

NAME_RE = re.compile(r"^(?:(?P<wl>[A-Z])_)?in(?P<isl>\d+)_out(?P<osl>\d+)_c(?P<c>\d+)$")


def load(out_dir: pathlib.Path):
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        m = NAME_RE.match(f.stem)
        if not m:
            continue  # anything hand-dropped in here
        # bench_serving appends, so take the last complete record.
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not recs:
            continue
        d = recs[-1]
        rows.append(
            {
                "workload": m["wl"] or f"{m['isl']}/{m['osl']}",
                "input_len": int(m["isl"]),
                "output_len": int(m["osl"]),
                "concurrency": int(m["c"]),
                "completed": d.get("completed"),
                "duration_s": d.get("duration"),
                "request_throughput": d.get("request_throughput"),
                "total_token_throughput": d.get("total_throughput"),
                "input_throughput": d.get("input_throughput"),
                "output_throughput": d.get("output_throughput"),
                "mean_ttft_ms": d.get("mean_ttft_ms"),
                "median_ttft_ms": d.get("median_ttft_ms"),
                "p99_ttft_ms": d.get("p99_ttft_ms"),
                "mean_tpot_ms": d.get("mean_tpot_ms"),
                "p99_tpot_ms": d.get("p99_tpot_ms"),
                "mean_e2e_latency_ms": d.get("mean_e2e_latency_ms"),
                "p99_e2e_latency_ms": d.get("p99_e2e_latency_ms"),
            }
        )
    rows.sort(key=lambda r: (r["workload"], r["concurrency"]))
    return rows


def group_key(rows_by_wl, wl):
    """Letter workloads first in A-E order, then unlabelled ones by length."""
    r = rows_by_wl[wl][0]
    return (0, wl, 0, 0) if len(wl) == 1 else (1, "", r["input_len"], r["output_len"])


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    out_dir = pathlib.Path(sys.argv[1])
    rows = load(out_dir)
    if not rows:
        print("(no results)")
        return

    cols = [
        ("conc", "concurrency", 0),
        ("done", "completed", 0),
        ("req/s", "request_throughput", 2),
        ("tot tok/s", "total_token_throughput", 1),
        ("in tok/s", "input_throughput", 1),
        ("out tok/s", "output_throughput", 1),
        ("TTFT ms", "mean_ttft_ms", 1),
        ("p99 TTFT", "p99_ttft_ms", 1),
        ("TPOT ms", "mean_tpot_ms", 2),
        ("p99 TPOT", "p99_tpot_ms", 2),
    ]

    best = {}
    for r in rows:
        t = r["total_token_throughput"]
        if t is None:
            continue
        cur = best.get(r["workload"])
        if cur is None or t > cur["total_token_throughput"]:
            best[r["workload"]] = r

    by_wl = {}
    for r in rows:
        by_wl.setdefault(r["workload"], []).append(r)

    for wl in sorted(by_wl, key=lambda w: group_key(by_wl, w)):
        sub = by_wl[wl]
        isl, osl = sub[0]["input_len"], sub[0]["output_len"]
        title = f"Workload {wl}: {isl}/{osl}" if len(wl) == 1 else f"{isl}/{osl}"
        print(f"\n### {title}")
        hdr = "".join(f"{c[0]:>11}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for r in sub:
            line = "".join(f"{fmt(r[k], nd):>11}" for _, k, nd in cols)
            print(line + ("   <- best tok/s" if r is best.get(wl) else ""))

    print("\n### Peak total token throughput per workload")
    for wl in sorted(best, key=lambda w: group_key(by_wl, w)):
        b = best[wl]
        label = f"{wl} ({b['input_len']}/{b['output_len']})" if len(wl) == 1 else wl
        print(
            f"  {label}: {b['total_token_throughput']:.1f} tok/s "
            f"at concurrency {b['concurrency']}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "results": rows,
                "best_per_workload": {k: v for k, v in best.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
