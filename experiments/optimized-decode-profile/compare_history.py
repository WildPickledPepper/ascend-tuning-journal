"""Compare saved native/down-prefetch/current profiles, without any NPU work.

Usage: python compare_history.py PRIVATE_DATA_ROOT FILTERED_OUTPUT_JSON
Only existing CSV, DB, request receipts and Insight replies are read. Native
and custom FFNs share graph membership and the same decode-window definition.
"""
import bisect
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from analyze import COMPUTE, category, exclusive, load, ns, rows, stats, width


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_sample(folder, backend_path):
    info = load(next(folder.glob("profiler_info*.json")))
    assert info["end_info"]["collectionTimeEnd"]
    cfg = info["config"]
    assert cfg["common_config"]["record_shapes"]
    assert cfg["experimental_config"]["_aic_metrics"] == "ACL_AICORE_PIPE_UTILIZATION"
    output = folder/"ASCEND_PROFILER_OUTPUT"
    assert (output/"trace_view.json").exists()
    kernels, tasks = rows(output/"kernel_details.csv"), rows(output/"task_time.csv")
    for row in kernels:
        row["lo"] = ns(row["Start Time(us)"])
        row["hi"] = row["lo"]+ns(row["Duration(us)"])
        row["shape"] = row["Input Shapes"].strip('"')
    kernels.sort(key=lambda r:r["lo"])
    main_id = collections.Counter(r["Model ID"] for r in kernels
                                  if r["Type"] == "MatMulV2" and r["shape"] == "8,5120;16384,5120").most_common(1)[0][0]
    draft_id = collections.Counter(r["Model ID"] for r in kernels
                                   if r["Type"] == "MatMulV2" and r["shape"] == "8,5120;34816,5120"
                                   and r["Model ID"] not in (main_id,"4294967295")).most_common(1)[0][0]
    markers = sorted(ns(r["task_start(us)"]) for r in tasks if r["kernel_type"] == "MODEL_EXECUTE")
    windows = []
    for i, start in enumerate(markers):
        end = markers[i+1] if i+1<len(markers) else max(r["hi"] for r in kernels)
        native = [r for r in kernels if start<=r["lo"]<end and r["Model ID"] in (main_id,draft_id)]
        models = set(r["Model ID"] for r in native)
        assert len(models) == 1
        windows.append({"lo":start,"hi":max(r["hi"] for r in native),
                        "phase":"main" if next(iter(models))==main_id else "draft"})
    for row in kernels:
        i = bisect.bisect_right(markers,row["lo"])-1
        phase = windows[i]["phase"] if i>=0 and row["lo"]<windows[i]["hi"] else "outside_graph"
        row["graph_phase"] = phase
        if row["Type"] == "MatMulV2" and row["shape"] == "8,5120;34816,5120":
            row["category"] = phase+"_ffn_gate_up"
        elif row["Type"] == "MatMulV2" and row["shape"] == "8,17408;5120,17408":
            row["category"] = phase+"_ffn_down"
        else:
            row["category"] = category(row)
    for window in windows:
        selected = [r for r in kernels if window["lo"]<=r["lo"]<window["hi"]]
        expected = 64 if window["phase"] == "main" else 7
        for part in ("gate_up","down","activation"):
            assert sum(r["category"]==window["phase"]+"_ffn_"+part for r in selected)==expected
    rounds = sum(w["phase"]=="main" for w in windows)
    assert rounds == sum(w["phase"]=="draft" for w in windows) == 26
    lo = next(w["lo"] for w in windows if w["phase"]=="main")
    hi = max(r["hi"] for r in kernels)
    decode = [r for r in kernels if r["lo"]>=lo]
    assert not any("packed_prefill" in r["Name"] for r in decode)
    span = hi-lo
    exc = exclusive(decode,lo,hi)
    result = {"physical_devices":sorted(set(r["Device_id"] for r in kernels)),"rounds":rounds,
              "config":cfg,"decode_window_ms":span/1e6,"decode_ms_per_round":span/1e6/rounds,
              "exclusive":exc,"ffn":{},"source":{},"graph_windows_ms":{}}
    for phase in ("main","draft"):
        ffn_rows = [r for r in decode if r["category"].startswith(phase+"_ffn_")]
        total = width([(r["lo"],r["hi"]) for r in ffn_rows])/1e6
        result["ffn"][phase] = {"total_ms":total,"ms_per_round":total/rounds,
                                "decode_percent":100*total/(span/1e6),"parts":{}}
        for part in ("gate_up","down","activation"):
            selected = [r for r in ffn_rows if r["category"]==phase+"_ffn_"+part]
            durs = [(r["hi"]-r["lo"])/1000 for r in selected]
            result["ffn"][phase]["parts"][part] = {"kernel_us":stats(durs),
                "total_ms":sum(durs)/1000,"decode_percent":100*sum(durs)/1000/(span/1e6),
                "mte2_activity_mean_us":sum(float(r["aic_mte2_time(us)"]) for r in selected)/len(selected),
                "mte2_activity_sum_ms":sum(float(r["aic_mte2_time(us)"]) for r in selected)/1000,
                "mac_activity_mean_us":sum(float(r["aic_mac_time(us)"]) for r in selected)/len(selected)}
        assert abs(sum(x["total_ms"] for x in result["ffn"][phase]["parts"].values())-total)<1e-8
        result["graph_windows_ms"][phase] = stats([(w["hi"]-w["lo"])/1e6 for w in windows if w["phase"]==phase])
        result["ffn"][phase]["percent_of_own_graph_windows"] = 100*total/result["graph_windows_ms"][phase]["sum"]
    result["ffn"]["all"] = {"total_ms":sum(result["ffn"][p]["total_ms"] for p in ("main","draft"))}
    result["ffn"]["all"]["decode_percent"] = 100*result["ffn"]["all"]["total_ms"]/(span/1e6)
    database = next(output.glob("ascend_pytorch*.db"))
    sql = COMPUTE+'''\nSELECT op_name, op_type, COUNT(*) AS calls, SUM(duration_ns) AS total_ns
FROM (SELECT * FROM compute_view LIMIT -1)
WHERE globalTaskId IN (SELECT globalTaskId FROM TASK WHERE startNs >= ?)
GROUP BY op_name, op_type ORDER BY total_ns DESC LIMIT 20;'''
    with sqlite3.connect(database.as_uri()+"?mode=ro",uri=True) as con:
        con.row_factory = sqlite3.Row
        top = [dict(x) for x in con.execute(sql,(lo,))]
    for entry in top:
        selected = [r for r in decode if r["Name"]==entry["op_name"] and r["Type"]==entry["op_type"]]
        assert len(selected)==entry["calls"] and abs(sum(r["hi"]-r["lo"] for r in selected)-entry["total_ns"])<300
    groups = collections.defaultdict(list)
    for row in kernels:
        groups[row["Type"],row["Name"],row["Accelerator Core"],row["Input Shapes"]].append(row)
    backend = load(backend_path)
    assert backend["result"] and len(backend["body"]["data"])==len(groups)
    for item in backend["body"]["data"]:
        item = item.get("compare",item)
        selected = groups[item["opType"],item["opName"],item["accCore"],item["inputShape"]]
        assert len(selected)==int(item["count"])
        assert abs(sum(r["hi"]-r["lo"] for r in selected)/1000-float(item["totalTime"]))<.011
    result["source"] = {"kernel_csv_sha256":digest(output/"kernel_details.csv"),
                         "task_csv_sha256":digest(output/"task_time.csv"),
                         "info_sha256":digest(next(folder.glob("profiler_info*.json"))),
                         "insight_reply_sha256":digest(backend_path),"insight_groups_verified":len(groups)}
    result["decode_sql"] = sql
    result["sql_top20_decode"] = top
    return result


def compare(before,after):
    changes = {"decode_window_reduction_percent":100*(1-after["decode_window_ms"]/before["decode_window_ms"]),"ffn":{}}
    for phase in ("main","draft","all"):
        a,b = before["ffn"][phase],after["ffn"][phase]
        item = {"saved_ms":a["total_ms"]-b["total_ms"],"time_reduction_percent":100*(1-b["total_ms"]/a["total_ms"]),
                "share_change_percentage_points":b["decode_percent"]-a["decode_percent"]}
        if phase != "all":
            item["parts"] = {part:{"saved_ms":a["parts"][part]["total_ms"]-b["parts"][part]["total_ms"],
                                    "time_reduction_percent":100*(1-b["parts"][part]["total_ms"]/a["parts"][part]["total_ms"])}
                             for part in ("gate_up","down","activation")}
        changes["ffn"][phase] = item
    return changes


def main(root,destination):
    old = root/"prefetch-experiment-20261021"
    current = root/"optimized-decode-profile-20261022"
    receipt = load(current/"results/PipeUtilization.json")
    ids = receipt["request"]["token_ids"]
    request = receipt["request"]["request"]
    for name in ("baseline","optimized"):
        previous = load(old/"model-run"/(name+"-profile.json"))
        assert previous["response"]["choices"][0]["token_ids"] == ids
        assert previous["request"]["messages"] == request["messages"]
        assert previous["response"]["usage"]["prompt_tokens"] == 121
        assert previous["response"]["usage"]["completion_tokens"] == 128
    result = {"analyzed_at_beijing":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "analysis_source_sha256":digest(Path(__file__)),
              "offline_only":True,"new_npu_requests":0,"input_tokens":121,"output_tokens":128,
              "all_output_tokens_identical":True,"sequence_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              "ffn_definition":"Gate/up matrix multiplication + SwiGlu + down matrix multiplication; main and draft separately.",
              "window":"First main MODEL_EXECUTE to last compute end; exclude earlier prompt/eager work; include main/draft/logits/postprocessing.",
              "limits":["Historical native and down-prefetch profiles used card 6; latest used card 7 on the next day. Observed differences are not a same-card controlled isolation of packing.",
                        "Profiler metrics match PipeUtilization and Level1, but op_attr and source instrumentation settings differ.",
                        "Latest profile includes both gate packing and direct prefill; there is no full-model profile from the immediately preceding same-card packed-control E2E run.",
                        "Pipeline activities overlap and cannot be added or interpreted as exclusive memory wait time."],"runs":{}}
    samples = [("native",old/"baseline-profile",old/"prefetch-baseline-backend-operators.json"),
               ("down_prefetch",old/"optimized-profile",old/"prefetch-optimized-backend-operators.json"),
               ("current",current/receipt["profile"],current/"insight/optimized-pipe-backend-operators.json")]
    for label,folder,backend in samples:
        result["runs"][label] = analyze_sample(folder,backend)
        print(label,json.dumps({k:result["runs"][label][k] for k in ["physical_devices","decode_window_ms","ffn"]}),flush=True)
    for label,name in [("native","baseline"),("down_prefetch","optimized")]:
        audit = load(old/"model-run"/(name+"-graph-audit-after.json"))["results"][0]
        assert audit["num_speculative_tokens"]==7 and audit["tp_size"]==1 and audit["dtype"]=="torch.bfloat16"
        assert audit["draft_use_cuda_graph"] and not audit["draft_enforce_eager"]
        assert audit["gpu_memory_utilization"]==.97 and audit["main_capture_sizes"]==[8]
    recorded = load(current/"summary-local.json")["captures"]["PipeUtilization"]
    actual = result["runs"]["current"]
    assert actual["exclusive"]==recorded["exclusive"] and actual["decode_window_ms"]==recorded["decode_window_ms"]
    result["comparisons"] = {"native_to_down_prefetch":compare(result["runs"]["native"],result["runs"]["down_prefetch"]),
                             "down_prefetch_to_current":compare(result["runs"]["down_prefetch"],result["runs"]["current"]),
                             "native_to_current":compare(result["runs"]["native"],result["runs"]["current"])}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["comparisons"],indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
