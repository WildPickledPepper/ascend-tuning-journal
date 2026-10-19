"""Read-only analysis of exported profiler CSVs and the documented compute CTE."""
import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import statistics
from decimal import Decimal

p = argparse.ArgumentParser()
p.add_argument("profiles", type=pathlib.Path)
p.add_argument("output", type=pathlib.Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

MACRO = """WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)"""
SQL = MACRO + """
SELECT op_type, COUNT(*) AS calls, ROUND(SUM(duration_ns)/1000000.0,3) AS total_ms
FROM compute_view GROUP BY op_type ORDER BY total_ms DESC LIMIT 20;
"""


def read(folder, name):
    with (folder / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ns(value):
    return int(Decimal(value.strip()) * 1000)


def number(value):
    return float(value) if value not in ("", "N/A", "NULL", None) else 0.0


def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if not result or lo > result[-1][1]:
            result.append([lo, hi])
        else:
            result[-1][1] = max(result[-1][1], hi)
    return result


results = {}
for label in ["baseline", "mtp"]:
    folder = next((a.profiles / ("profiles-" + label)).glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    kernels = read(folder, "kernel_details.csv")
    tasks = read(folder, "task_time.csv")
    for row in kernels:
        row["_lo"] = ns(row["Start Time(us)"])
        row["_hi"] = row["_lo"] + ns(row["Duration(us)"])
    markers = sorted(ns(r["task_start(us)"]) for r in tasks if r["kernel_type"] == "MODEL_EXECUTE")
    assert len(markers) == (31 if label == "baseline" else 17)
    lo, hi = markers[0], max(r["_hi"] for r in kernels)
    decode = [r for r in kernels if lo <= r["_lo"] < hi]
    span = hi - lo
    intervals = union([(max(lo, r["_lo"]), min(hi, r["_hi"])) for r in decode])
    busy = sum(b - s for s, b in intervals)
    gaps = [(s2 - e1, e1 - lo) for (_, e1), (s2, _) in zip(intervals, intervals[1:])]
    groups = collections.defaultdict(list)
    for row in decode:
        groups[(row["Type"], row["Input Shapes"])].append(row)
    summaries = []
    for (typ, shape), rows in groups.items():
        duration = sum(r["_hi"] - r["_lo"] for r in rows)
        summaries.append(dict(type=typ, input_shape=shape, calls=len(rows),
                              total_ms=duration / 1e6, mean_us=duration / len(rows) / 1000,
                              decode_span_pct=100 * duration / span,
                              model_ids=dict(collections.Counter(r["Model ID"] for r in rows)),
                              mte2_activity_pct=100 * sum(number(r["aic_mte2_ratio"]) * (r["_hi"] - r["_lo"]) for r in rows) / duration,
                              mac_activity_pct=100 * sum(number(r["aic_mac_ratio"]) * (r["_hi"] - r["_lo"]) for r in rows) / duration,
                              mte1_activity_pct=100 * sum(number(r["aic_mte1_ratio"]) * (r["_hi"] - r["_lo"]) for r in rows) / duration))
    summaries.sort(key=lambda r: -r["total_ms"])
    ffn_shapes = {f'{1 if label == "baseline" else 2},5120;34816,5120',
                  f'{1 if label == "baseline" else 2},17408;5120,17408'}
    categories = collections.defaultdict(lambda: dict(calls=0, total_ms=0.0))
    for row in decode:
        if row["Type"] == "MatMulV2":
            shape = row["Input Shapes"].strip('"')
            if shape in ffn_shapes:
                category = "FFN gate-up and down (includes draft FFN when enabled)"
            elif "248320,5120" in shape:
                category = "Vocabulary projection (main and draft)"
            else:
                category = "Other matrix multiplications"
        else:
            category = "Other device kernels"
        categories[category]["calls"] += 1
        categories[category]["total_ms"] += (row["_hi"] - row["_lo"]) / 1e6
    for v in categories.values():
        v["decode_span_pct"] = 100 * v["total_ms"] / (span / 1e6)
    cycles = []
    for start, end in zip(markers, markers[1:] + [hi]):
        rs = [r for r in decode if start <= r["_lo"] < end]
        iv = union([(max(start, r["_lo"]), min(end, r["_hi"])) for r in rs])
        active = sum(b - s for s, b in iv)
        model = [r for r in rs if r["Model ID"] != "4294967295"]
        ext = [r for r in rs if r["Model ID"] == "4294967295"]
        graph_end = max(r["_hi"] for r in model)
        after_graph = [r for r in rs if r["_lo"] >= graph_end]
        after_active = sum(b - s for s, b in union([
            (max(graph_end, r["_lo"]), min(end, r["_hi"])) for r in after_graph]))
        cycles.append(dict(span_ms=(end - start) / 1e6, compute_union_ms=active / 1e6,
                           no_compute_ms=(end - start - active) / 1e6,
                           graph_kernel_sum_ms=sum(r["_hi"] - r["_lo"] for r in model) / 1e6,
                           outside_graph_kernel_sum_ms=sum(r["_hi"] - r["_lo"] for r in ext) / 1e6,
                           graph_window_ms=(graph_end - start) / 1e6,
                           after_graph_window_ms=(end - graph_end) / 1e6,
                           after_graph_compute_ms=after_active / 1e6,
                           after_graph_no_compute_ms=(end - graph_end - after_active) / 1e6))
    db = next(folder.glob("ascend_pytorch_profiler*.db"))
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        top = [dict(r) for r in connection.execute(SQL)]
    results[label] = dict(profile_output_tokens=32, decode_rounds=len(markers),
                          decode_span_ms=span / 1e6, compute_union_ms=busy / 1e6,
                          no_compute_ms=(span - busy) / 1e6, no_compute_pct=100 * (span - busy) / span,
                          median_round_ms=statistics.median(c["span_ms"] for c in cycles),
                          complete_round_medians={k: statistics.median(c[k] for c in cycles[:-1])
                                                  for k in cycles[0]},
                          categories=dict(categories), top_decode_groups=summaries[:20],
                          total_decode_kernel_calls=len(decode), cycles=cycles,
                          largest_compute_gaps_ms=[dict(duration_ms=d / 1e6, offset_ms=o / 1e6)
                                                   for d, o in sorted(gaps, reverse=True)[:20]],
                          whole_profile_compute_top=top, sql_rows=len(top))
result = dict(stages=results,
              phase_boundary="First hardware MODEL_EXECUTE to last compute completion; prefill excluded, final draft work included.",
              sql=SQL, query_channel="Read-only local sqlite3; msprof MCP SQL tool unavailable.",
              limitations=["Short profiled request, separate from unprofiled speed benchmark.",
                           "PMU ratios are pipeline activity, not peak bandwidth utilization.",
                           "No-compute intervals are gaps in exported compute tasks; they are not automatically CPU bottleneck time.",
                           "Graph model IDs distinguish captured tasks from outside-graph work, not exact semantic ownership.",
                           "Categories are cumulative kernel durations; overlapping kernels can make sums differ from interval union."])
(a.output / "mtp-bottleneck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
(a.output / "compute-top.sql").write_text(SQL, encoding="utf-8")
for label, stage in results.items():
    print(label, json.dumps({k: stage[k] for k in ["decode_span_ms", "compute_union_ms", "no_compute_ms", "no_compute_pct", "median_round_ms", "categories", "sql_rows"]}, ensure_ascii=False))
    for row in stage["top_decode_groups"][:8]:
        print(json.dumps(row, ensure_ascii=False))
