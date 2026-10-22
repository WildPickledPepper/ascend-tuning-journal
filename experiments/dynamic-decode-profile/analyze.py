"""Read-only decode analysis of the optimized full-model captures.

Usage: python analyze.py PRIVATE_CAPTURE_ROOT OUTPUT_JSON
The original Compute/Communication CTEs are used unchanged. SQLite is a local
fallback because this session exposes no msprof SQL MCP tool. All trace files
stay outside Git; only bounded aggregates and source checksums are exported.
"""
import collections
import bisect
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

COMPUTE = '''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)'''
COMM = '''WITH comm_view AS (
    SELECT ROUND(c.endNs - c.startNs) AS duration_ns, n.value AS op_name, t.value AS op_type, g.value AS group_name
    FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id = c.opName
    LEFT JOIN STRING_IDS t ON t.id = c.opType
    LEFT JOIN STRING_IDS g ON g.id = c.groupName
)'''


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def ns(value):
    return int(Decimal(str(value).strip()) * 1000)


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


def width(intervals):
    return sum(hi-lo for lo, hi in union(intervals))


def stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p95": values[min(len(values)-1, int(.95*len(values)))]}


def category(row):
    name, kind, shape = row["Name"].lower(), row["Type"], row["Input Shapes"].strip('"')
    phase = row.get("graph_phase", "outside_graph")
    if phase == "outside_graph":
        if kind == "TopKV2" and shape.endswith("248320;"):
            return "dynamic_target_topk"
        if kind == "TopK" and shape == "248320;":
            return "dynamic_pool_topk_aicpu"
        if kind == "Sort" and shape == "4096":
            return "dynamic_id_sort_aicpu"
        if kind == "IndexFill" and shape.startswith("248320;"):
            return "dynamic_recency_indexfill"
        if kind == "GatherV3" and shape == "248320,5120;4096;1":
            return "dynamic_weight_gather"
        if kind == "GatherV3" and shape == "248320;4096;1":
            return "dynamic_score_gather"
        if "4096" in shape and kind != "MatMulV2":
            return "dynamic_other_buffer_ops"
    if "gateup_packed" in name:
        return "main_ffn_gate_up"
    if "down_m8_prefetch" in name:
        return row.get("graph_phase", "unassigned")+"_ffn_down"
    if kind == "SwiGlu":
        return row.get("graph_phase", "outside_graph")+"_ffn_activation"
    if kind == "MatMulV2":
        if shape.endswith(";248320,5120"):
            return "main_vocabulary_projection"
        if shape.endswith(";4096,5120"):
            return "draft_small_vocabulary_projection"
        if shape == "8,5120;34816,5120" or shape == "1,5120;34816,5120":
            return "draft_ffn_gate_up"
        if shape == "8,17408;5120,17408" or shape == "1,17408;5120,17408":
            return "draft_ffn_down"
        return row.get("graph_phase", "outside_graph")+"_other_linear"
    if kind == "RecurrentGatedDeltaRule":
        return "main_gdn_recurrence"
    if any(s in (kind+name).lower() for s in ["attention", "gateddelta", "gated_delta", "recurrent"]):
        return "attention_kernels"
    return "other_compute"


def exclusive(rows_, lo, hi):
    events = collections.defaultdict(list)
    for i, row in enumerate(rows_):
        a, b = max(lo, row["lo"]), min(hi, row["hi"])
        if b > a:
            events[a].append((i, row["category"], 1))
            events[b].append((i, row["category"], -1))
    events[lo]; events[hi]
    active, result, previous = {}, collections.Counter(), lo
    for t, changes in sorted(events.items()):
        groups = set(active.values())
        group = "no_compute" if not groups else next(iter(groups)) if len(groups) == 1 else "overlapping_categories"
        result[group] += t-previous
        for i, cat, delta in changes:
            if delta > 0:
                active[i] = cat
            else:
                active.pop(i)
        previous = t
    assert not active and sum(result.values()) == hi-lo
    return {key: {"exclusive_ms": value/1e6, "window_percent": 100*value/(hi-lo)}
            for key, value in result.items()}


def main(root, destination):
    result = {"model": "Qwen3.8-27B", "device": "Ascend910B2", "physical_device": 7,
              "dtype": "BF16 weights/conv, FP32 SSM state", "TP": 1, "concurrency": 1, "MTP": 7,
              "optimized_paths": ["M8 gate/up packed", "M8 down prefetch", "direct packed prefill", "dynamic 4096 draft vocabulary"],
              "kernel_changes_this_run": False, "captures": {},
              "method": "Decode begins at the first captured main-model MODEL_EXECUTE; prompt/eager work before it is excluded. Native attention projections identify the main model ID. Custom kernels have sentinel Model ID, so graph membership is reconstructed from MODEL_EXECUTE boundaries and the native graph endpoints, never from the custom Model ID. Window ends at last compute task. CPU API durations are never added to device time.",
              "limits": ["Profiler timings are instrumented, not new unprofiled throughput results.",
                         "Pipeline activity is not peak bandwidth utilization.",
                         "No-compute intervals may contain DMA, synchronization or host work.",
                         "Kernel totals include overlaps; exclusive window categories are reported separately.",
                         "A short single-concurrency sample does not establish all batch/context workloads."]}
    for metric in ("PipeUtilization", "L2Cache", "Memory", "LongPipeUtilization"):
        receipt = load(root/"results"/(metric+".json"))
        path = root/receipt["profile"]
        info = load(next(path.glob("profiler_info*.json")))
        assert info["end_info"]["collectionTimeEnd"], "Capture not normally stopped"
        output = path/"ASCEND_PROFILER_OUTPUT"
        assert (output/"trace_view.json").exists()
        database = next(output.glob("ascend_pytorch*.db"))
        kernels, tasks = rows(output/"kernel_details.csv"), rows(output/"task_time.csv")
        for row in kernels:
            row["lo"] = ns(row["Start Time(us)"])
            row["hi"] = row["lo"]+ns(row["Duration(us)"])
        kernels.sort(key=lambda x: x["lo"])
        packed = [r for r in kernels if "gateup_packed" in r["Name"]]
        assert packed, "Packed kernel not visible: inspect exported names before analysis"
        count = len(packed)
        assert count % 64 == 0
        projection = [r for r in kernels if r["Type"] == "MatMulV2"
                      and r["Input Shapes"].strip('"') == "8,5120;16384,5120"]
        main_id = collections.Counter(r["Model ID"] for r in projection).most_common(1)[0][0]
        draft_gate = [r for r in kernels if r["Type"] == "MatMulV2"
                      and r["Input Shapes"].strip('"') == "8,5120;34816,5120"]
        draft_id = collections.Counter(r["Model ID"] for r in draft_gate).most_common(1)[0][0]
        assert main_id != draft_id and main_id != "4294967295" and draft_id != "4294967295"
        markers_all = sorted([r for r in tasks if r["kernel_type"] == "MODEL_EXECUTE"],
                             key=lambda r: ns(r["task_start(us)"]))
        marker_starts = [ns(r["task_start(us)"]) for r in markers_all]
        graph_windows = []
        for i, start in enumerate(marker_starts):
            end = marker_starts[i+1] if i+1 < len(marker_starts) else max(r["hi"] for r in kernels)
            native = [r for r in kernels if start <= r["lo"] < end and r["Model ID"] in (main_id, draft_id)]
            identities = set(r["Model ID"] for r in native)
            assert len(identities) == 1, identities
            model = next(iter(identities))
            graph_windows.append({"lo": start, "hi": max(r["hi"] for r in native),
                                  "phase": "main" if model == main_id else "draft"})
        assert sum(w["phase"] == "main" for w in graph_windows) == count//64
        assert sum(w["phase"] == "draft" for w in graph_windows) == count//64
        for row in kernels:
            i = bisect.bisect_right(marker_starts, row["lo"])-1
            row["graph_phase"] = (graph_windows[i]["phase"] if i >= 0 and row["lo"] < graph_windows[i]["hi"] else "outside_graph")
            row["category"] = category(row)
        for window in graph_windows:
            rs = [r for r in kernels if window["lo"] <= r["lo"] < window["hi"]]
            n_gate = sum("gateup_packed" in r["Name"] for r in rs)
            n_down = sum("down_m8_prefetch" in r["Name"] for r in rs)
            assert (n_gate,n_down) == ((64,64) if window["phase"] == "main" else (0,7)), (window,n_gate,n_down)
        lo = next(w["lo"] for w in graph_windows if w["phase"] == "main")
        hi = max(r["hi"] for r in kernels)
        decode = [r for r in kernels if r["lo"] >= lo]
        assert not any("packed_prefill" in r["Name"] for r in decode), "Prefill leaked into decode window"
        by_group, by_shape = collections.defaultdict(list), collections.defaultdict(list)
        for row in decode:
            by_group[row["category"]].append(row)
            by_shape[row["Name"], row["Type"], row["Input Shapes"], row["Model ID"]].append(row)
        groups = {}
        for name, rs in by_group.items():
            fields = [k for k in rs[0] if k.startswith(("aic_", "aiv_", "l2_", "main_mem", "ub_"))]
            pmu = {}
            for field in fields:
                numeric = []
                for r in rs:
                    try:
                        numeric.append((float(r[field]), r["hi"]-r["lo"]))
                    except (ValueError, TypeError):
                        pass
                if numeric:
                    pmu[field] = {"valid": len(numeric), "mean": statistics.mean(v for v,_ in numeric),
                                  "duration_weighted_mean": sum(v*d for v,d in numeric)/sum(d for _,d in numeric)}
            groups[name] = {"kernel_us": stats([(r["hi"]-r["lo"])/1000 for r in rs]), "pmu": pmu}
            if metric == "L2Cache":
                def val(r, field):
                    try:
                        return float(r.get(field,0) or 0)
                    except (ValueError, TypeError):
                        return 0
                hit = sum(val(r,"aic_r0_read_cache_hit")+val(r,"aic_r1_read_cache_hit") for r in rs)
                miss = sum(val(r,"aic_r0_read_cache_miss_allocate")+val(r,"aic_r1_read_cache_miss_allocate") for r in rs)
                groups[name]["l2_read_counters"] = {"hits": hit, "misses": miss,
                                                      "hit_rate": hit/(hit+miss) if hit+miss else None}
        sql = COMPUTE+'''\nSELECT op_name, op_type, COUNT(*) AS calls, SUM(duration_ns) AS total_ns
FROM compute_view GROUP BY op_name, op_type ORDER BY total_ns DESC LIMIT 20;'''
        with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            top = [dict(r) for r in con.execute(sql)]
            for row in top:
                selected = [r for r in kernels if r["Name"] == row["op_name"] and r["Type"] == row["op_type"]]
                assert len(selected) == row["calls"]
                assert abs(sum(r["hi"]-r["lo"] for r in selected)-row["total_ns"]) < 300
            # Keep the original macro unchanged. The outer LIMIT -1 is an
            # optimization barrier: without it SQLite chooses a quadratic TASK
            # scan because the native index starts with startNs, not task ID.
            decode_sql = COMPUTE+'''\nSELECT op_name, op_type, COUNT(*) AS calls, SUM(duration_ns) AS total_ns
FROM (SELECT * FROM compute_view LIMIT -1)
WHERE globalTaskId IN (SELECT globalTaskId FROM TASK WHERE startNs >= ?)
GROUP BY op_name, op_type ORDER BY total_ns DESC LIMIT 20;'''
            decode_top = [dict(r) for r in con.execute(decode_sql, (lo,))]
            for row in decode_top:
                selected = [r for r in decode if r["Name"] == row["op_name"] and r["Type"] == row["op_type"]]
                assert len(selected) == row["calls"] and abs(sum(r["hi"]-r["lo"] for r in selected)-row["total_ns"]) < 300
            comm = [dict(r) for r in con.execute(COMM+"\nSELECT COUNT(*) AS calls, SUM(duration_ns) AS total_ns FROM comm_view;")] if "COMMUNICATION_OP" in tables else []
            pmu_names = [dict(r) for r in con.execute('''SELECT s.value AS metric, COUNT(*) AS entries
FROM TASK_PMU_INFO p LEFT JOIN STRING_IDS s ON s.id=p.name
GROUP BY s.value ORDER BY entries DESC LIMIT 60;''')]
        shaped = sorted(by_shape.items(), key=lambda kv: sum(r["hi"]-r["lo"] for r in kv[1]), reverse=True)
        no_compute = []
        previous = lo
        for a,b in union([(r["lo"], r["hi"]) for r in decode]):
            if a > previous:
                no_compute.append((previous,a))
            previous = b
        model_groups = collections.defaultdict(list)
        for r in decode:
            model_groups[r["graph_phase"]].append(r)
        model_stats = {key: {"kernels": len(rs), "cumulative_compute_ms": sum(r["hi"]-r["lo"] for r in rs)/1e6,
                              "compute_union_ms": width([(r["lo"],r["hi"]) for r in rs])/1e6}
                       for key,rs in model_groups.items()}
        markers = [r for r in tasks if r["kernel_type"] == "MODEL_EXECUTE" and ns(r["task_stop(us)"]) > lo]
        result["captures"][metric] = {
            "input_tokens": receipt["request"]["usage"]["prompt_tokens"],
            "output_tokens": receipt["request"]["usage"]["completion_tokens"],
            "question_id": receipt["question_id"],
            "valid_stop": True, "config": info["config"], "token_sequence_equal": receipt["token_ids_equal_to_unprofiled"],
            "sequence_sha256": receipt["sequence_sha256"], "kernel_csv_sha256": hashlib.sha256((output/"kernel_details.csv").read_bytes()).hexdigest(),
            "all_kernels": len(kernels), "decode_kernels": len(decode), "main_model_id": main_id,
            "draft_model_id": draft_id,
            "verification_rounds": count//64, "decode_window_ms": (hi-lo)/1e6,
            "exclusive": exclusive(decode, lo, hi), "groups": groups,
            "no_compute_gap_us": stats([(b-a)/1000 for a,b in no_compute]),
            "model_groups": model_stats, "graph_execute_markers": len(markers),
            "graph_windows_ms": {phase: stats([(w["hi"]-w["lo"])/1e6 for w in graph_windows if w["phase"] == phase])
                                 for phase in ("main","draft")},
            "csv_columns": list(kernels[0].keys()), "pmu_names": pmu_names,
            "top_shape_groups": [{"name": name, "type": kind, "shape": shape, "model_id": model,
                                  "duration_us": stats([(r["hi"]-r["lo"])/1000 for r in rs])}
                                 for (name,kind,shape,model),rs in shaped[:20]],
            "sql": sql, "sql_top20_whole_capture": top, "communication": comm,
            "decode_sql": decode_sql, "sql_top20_decode": decode_top,
            "graph_markers": [{"start_relative_ms": (ns(r["task_start(us)"])-lo)/1e6,
                               "duration_ms": (ns(r["task_stop(us)"])-ns(r["task_start(us)"]))/1e6}
                              for r in markers[:4]]}
        if metric in ("PipeUtilization", "LongPipeUtilization"):
            trace = load(output/"trace_view.json")
            events = trace.get("traceEvents",[]) if isinstance(trace,dict) else trace
            scope_groups = collections.defaultdict(list)
            cpu_names = {"dynamic_vocab.update", "dynamic_vocab.repack", "dynamic_vocab.reset", "dynamic_decode.sample_and_update", "draft_token", "sample_token"}
            for event in events:
                if event.get("ph") == "X" and event.get("name") in cpu_names:
                    a = ns(event["ts"])
                    b = a+ns(event.get("dur",0))
                    if lo <= a < hi:
                        scope_groups[event["name"]].append((a,b))
            result["captures"][metric]["host_scopes"] = {
                name:{"cpu_wall_us":stats([(b-a)/1000 for a,b in ranges]),
                      "clipped_union_ms":width([(max(a,lo),min(b,hi)) for a,b in ranges])/1e6}
                for name,ranges in scope_groups.items()}
            result["captures"][metric]["host_scope_note"] = "CPU wall scopes are nested and asynchronous with device work; never add them to GPU task totals or label synchronization time as CPU computation."
        result["captures"][metric]["accelerator_summary"] = {
            core:{"calls":len(rs),"sum_ms":sum(r["hi"]-r["lo"] for r in rs)/1e6}
            for core,rs in ((core,[r for r in decode if r["Accelerator Core"]==core])
                            for core in sorted({r["Accelerator Core"] for r in decode}))}
        print("ANALYZED", metric, "decode_ms", (hi-lo)/1e6, "rounds", count//64, flush=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: {x:v[x] for x in ["verification_rounds","decode_window_ms","exclusive","no_compute_gap_us"]}
                      for k,v in result["captures"].items()},indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
