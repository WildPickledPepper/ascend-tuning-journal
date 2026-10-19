"""Read existing native profiler exports and the skill's unchanged compute CTE."""
import argparse
import collections
import csv
import json
import pathlib
import sqlite3
from decimal import Decimal

p = argparse.ArgumentParser()
p.add_argument("profiles", type=pathlib.Path)
p.add_argument("output", type=pathlib.Path)
a = p.parse_args()

SQL = """WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
SELECT op_type, COUNT(*) AS calls, ROUND(SUM(duration_ns)/1000000.0,3) AS total_ms
FROM compute_view GROUP BY op_type ORDER BY total_ms DESC LIMIT 20;
"""


def read(folder, name):
    with (folder / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ns(value):
    return int(Decimal(value.strip()) * 1000)


def logical_shape(shape, formats):
    # This run has aligned BF16 weights whose exported NZ shape is [K/16,N/16,16,16].
    # Keep the physical shape too; do not treat the four-dimensional storage as a new GEMM.
    if formats != "ND;FRACTAL_NZ":
        return shape
    pieces = shape.split(";")
    if len(pieces) != 2:
        return shape
    x = [int(v) for v in pieces[0].split(",")]
    w = [int(v) for v in pieces[1].split(",")]
    if len(x) != 2 or len(w) != 4 or w[2:] != [16, 16] or x[1] != w[0] * 16:
        return shape
    return f"{x[0]},{x[1]};{w[1] * 16},{x[1]}"


def merged(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([lo, hi])
    return out


results = {}
for mode in ("nd", "nz"):
    folders = list((a.profiles / ("profiles-" + mode)).glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    assert len(folders) == 1, folders
    folder = folders[0]
    kernels = read(folder, "kernel_details.csv")
    tasks = read(folder, "task_time.csv")
    for r in kernels:
        r["_start"] = ns(r["Start Time(us)"])
        r["_end"] = r["_start"] + ns(r["Duration(us)"])
    markers = sorted(ns(r["task_start(us)"]) for r in tasks if r["kernel_type"] == "MODEL_EXECUTE")
    assert markers, "No hardware graph marker; investigate graph execution before comparison"
    lo, hi = markers[0], max(r["_end"] for r in kernels)
    decode = [r for r in kernels if lo <= r["_start"] < hi]
    busy = sum(e - s for s, e in merged([(r["_start"], r["_end"]) for r in decode]))
    groups = collections.defaultdict(list)
    for r in decode:
        groups[(r["Type"], r["Input Shapes"].strip('"'), r["Input Formats"])].append(r)
    rows = []
    for (typ, shape, fmt), rs in groups.items():
        duration = sum(r["_end"] - r["_start"] for r in rs)
        names = sorted(set(r["Name"] for r in rs))
        metrics = {}
        for key in ("aic_mte2_ratio", "aic_mac_ratio", "aic_mte1_ratio"):
            metrics[key] = sum(float(r.get(key) or 0) * (r["_end"] - r["_start"]) for r in rs) / duration
        rows.append(dict(type=typ, input_shape=shape, logical_input_shape=logical_shape(shape, fmt), input_formats=fmt,
                         names=names, calls=len(rs), total_ms=duration / 1e6,
                         mean_us=duration / len(rs) / 1000,
                         decode_window_pct=100 * duration / (hi - lo),
                         model_ids=dict(collections.Counter(r["Model ID"] for r in rs)),
                         activity_ratios=metrics))
    rows.sort(key=lambda r: -r["total_ms"])
    matmuls = [r for r in rows if "matmul" in r["type"].lower()
               or any("matmul" in s.lower() for s in r["names"])]
    ffn = [r for r in matmuls if r["logical_input_shape"] in
           {"2,5120;34816,5120", "2,17408;5120,17408", "1,5120;34816,5120", "1,17408;5120,17408"}]
    dbs = list(folder.glob("ascend_pytorch_profiler*.db"))
    assert len(dbs) == 1, dbs
    with sqlite3.connect(dbs[0].as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        sql_rows = [dict(r) for r in conn.execute(SQL)]
    results[mode] = dict(profile=folder.parent.name,
                         decode_graph_rounds=len(markers), decode_window_ms=(hi - lo) / 1e6,
                         compute_union_ms=busy / 1e6, no_compute_ms=(hi - lo - busy) / 1e6,
                         no_compute_pct=100 * (hi - lo - busy) / (hi - lo),
                         ffn_total_ms=sum(r["total_ms"] for r in ffn), ffn_groups=ffn,
                         matrix_total_ms=sum(r["total_ms"] for r in matmuls),
                         top_decode_groups=rows[:20], matmul_groups=matmuls,
                         sql_scope="entire profiled request, includes prefill",
                         sql_returned_rows=len(sql_rows), sql_rows=sql_rows)
a.output.mkdir(parents=True, exist_ok=True)
(a.output / "native-profile-comparison.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(a.output / "compute-top.sql").write_text(SQL, encoding="utf-8")
for mode, r in results.items():
    print(json.dumps(dict(mode=mode, **{k: r[k] for k in
                      ("decode_graph_rounds", "decode_window_ms", "ffn_total_ms", "no_compute_pct", "ffn_groups")})))
