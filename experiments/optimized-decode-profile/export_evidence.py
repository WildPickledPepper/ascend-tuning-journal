"""Verify private capture artifacts and export small, path-free evidence."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

from analyze import ns


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csvrows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(root, target):
    target.mkdir(parents=True, exist_ok=True)
    summary = read(root/"summary-local.json")
    assert summary == read(root/"summary-server.json")
    before, after = [read(root/"results"/(label+"-audit.json"))["results"][0] for label in ("before","after")]
    for audit in (before,after):
        assert audit["tp_size"] == 1 and audit["dtype"] == "torch.bfloat16"
        assert audit["num_speculative_tokens"] == 7 and audit["draft_use_cuda_graph"]
        assert not audit["draft_enforce_eager"]
        layout = audit["layout_experiment"]
        assert layout["enabled"] and layout["packed_count"] == 64 and layout["prefill_direct_enabled"]
        checks = layout["prefill_real_checks"]
        assert len(checks) == 64 and all(x["different"] == 0 and x["finite"] for x in checks.values())
        assert all(any(e["captured"] for e in w["entries"]) for w in audit["wrappers"])
    delta = {k: after["layout_experiment"]["counts"].get(k,0)-before["layout_experiment"]["counts"].get(k,0)
             for k in set(before["layout_experiment"]["counts"]) | set(after["layout_experiment"]["counts"])}
    assert delta.get("prefill_direct_host_calls") == 192 and delta.get("gate_unpack_fallback_host_calls",0) == 0
    reference = read(root/"results/unprofiled-reference.json")
    output = {k:summary[k] for k in ["model","device","physical_device","dtype","TP","concurrency","MTP",
                                     "optimized_paths","kernel_changes_this_run","method","limits"]}
    output["captures"] = {}
    source_records, request_records = [], []
    for metric, cap in summary["captures"].items():
        receipt = read(root/"results"/(metric+".json"))
        folder = root/receipt["profile"]
        info = read(next(folder.glob("profiler_info*.json")))
        assert info["end_info"]["collectionTimeEnd"]
        row = receipt["request"]
        assert row["token_ids"] == reference["token_ids"] and len(row["token_ids"]) == 128
        assert row["usage"]["prompt_tokens"] == 121
        assert cap["verification_rounds"] == 26 and cap["graph_execute_markers"] == 52
        request_records.append({"metric":metric, "input_tokens":121, "output_tokens":128,
                                "same_output_as_unprofiled":True, "sequence_sha256":receipt["sequence_sha256"],
                                "profiled_e2e_s":row["e2e_s"], "profiled_ttft_s":row["ttft_s"],
                                "started_utc":receipt["started_utc"], "completed_utc":receipt["completed_utc"]})
        output["captures"][metric] = {k:cap[k] for k in ["valid_stop","config","verification_rounds","decode_window_ms",
                                                        "exclusive","no_compute_gap_us","graph_windows_ms","model_groups",
                                                        "kernel_csv_sha256","decode_kernels","sql_top20_decode"]}
        output["captures"][metric]["groups"] = {}
        for name, data in cap["groups"].items():
            small = {"kernel_us": data["kernel_us"]}
            if metric == "PipeUtilization":
                small["pmu"] = {k:v for k,v in data["pmu"].items() if k in ["aic_mte2_ratio","aic_mac_ratio","aic_scalar_ratio"]}
            elif metric == "L2Cache":
                small["l2_read_counters"] = data["l2_read_counters"]
            else:
                small["raw_pmu_do_not_interpret_as_system_hbm_bw"] = {
                    k:v for k,v in data["pmu"].items() if k in ["aic_l1_read_bw(GB/s)","aic_main_mem_read_bw(GB/s)"]}
            output["captures"][metric]["groups"][name] = small
        for path in [next(folder.glob("profiler_info*.json"))]+list((folder/"ASCEND_PROFILER_OUTPUT").glob("*")):
            if path.is_file():
                source_records.append({"metric":metric,"artifact":path.name,"bytes":path.stat().st_size,"sha256":sha(path)})
    pipe_receipt = read(root/"results/PipeUtilization.json")
    pipe = root/pipe_receipt["profile"]/"ASCEND_PROFILER_OUTPUT"
    kernels, tasks = csvrows(pipe/"kernel_details.csv"), csvrows(pipe/"task_time.csv")
    lo = min(ns(r["task_start(us)"]) for r in tasks if r["kernel_type"] == "MODEL_EXECUTE")
    decode = [r for r in kernels if ns(r["Start Time(us)"]) >= lo]
    assert not any(r["Name"] == "packed_prefill" for r in decode)
    draft_id = summary["captures"]["PipeUtilization"]["draft_model_id"]
    vocab = [r for r in decode if r["Type"] == "MatMulV2" and r["Input Shapes"].strip('"').endswith(";248320,5120")]
    lm = {}
    for label, selected in [("draft",[r for r in vocab if r["Model ID"] == draft_id]),
                            ("main",[r for r in vocab if r["Model ID"] != draft_id])]:
        total = sum(ns(r["Duration(us)"]) for r in selected)/1e6
        lm[label] = {"calls":len(selected),"total_ms":total,"mean_us":1000*total/len(selected),
                     "decode_window_percent":100*total/summary["captures"]["PipeUtilization"]["decode_window_ms"]}
    assert lm["draft"]["calls"] == 182 and lm["main"]["calls"] == 26
    lm["weight_bytes"] = 248320*5120*2
    lm["logical_weight_bytes_per_main_round"] = lm["weight_bytes"]*8
    lm["traffic_note"] = "Logical matrix bytes, not measured HBM bus traffic or cache-line bytes."
    lm["draft_head_percent_of_draft_graph_window"] = 100*lm["draft"]["total_ms"]/summary["captures"]["PipeUtilization"]["graph_windows_ms"]["draft"]["sum"]
    output["vocabulary_projection_detail"] = lm
    other = collections.defaultdict(lambda:[0,0])
    for row in decode:
        if row["Type"] not in ["MatMulV2","gateup_packed","down_m8_prefetch","SwiGlu","RecurrentGatedDeltaRule","FusedInferAttentionScore"]:
            item = other[row["Name"]]
            item[0] += 1; item[1] += ns(row["Duration(us)"])
    output["other_task_top15"] = [{"name":k,"calls":v[0],"total_ms":v[1]/1e6} for k,v in sorted(other.items(), key=lambda kv:kv[1][1],reverse=True)[:15]]
    groups = collections.defaultdict(list)
    for row in kernels:
        groups[row["Type"],row["Name"],row["Accelerator Core"],row["Input Shapes"]].append(row)
    backend = read(root/"insight/optimized-pipe-backend-operators.json")
    assert backend["result"] and len(backend["body"]["data"]) == len(groups)
    for item in backend["body"]["data"]:
        item = item.get("compare",item)
        selected = groups[item["opType"],item["opName"],item["accCore"],item["inputShape"]]
        assert len(selected) == int(item["count"])
        assert abs(sum(ns(r["Duration(us)"])/1000 for r in selected)-float(item["totalTime"])) < .011
    output["verification"] = {"server_local_analysis_identical":True,"insight_groups_verified":len(groups),
                              "sqlite_top20_verified_for_each_capture_and_decode":True,
                              "prefill_real_activation_layers_bit_equal":64,"formal_host_count_delta":delta,
                              "no_hccl_kernel_observed":not any("hccl" in r["Name"].lower() or "hcom" in r["Name"].lower() for r in decode),
                              "prefill_excluded_from_decode":True,
                              "custom_model_id_sentinel_handled_by_timeline":True}
    output["unprofiled_reference"] = {"input_tokens":121,"output_tokens":128,"e2e_s":reference["e2e_s"],"ttft_s":reference["ttft_s"],
                                       "purpose":"One same-request token and instrumentation control, not a repeated historical baseline benchmark."}
    output["requests"] = request_records
    cleanup = read(root/"results/cleanup.json")
    assert cleanup["no_device_process"] and cleanup["port_8047_free"]
    archive = root/"profiles-and-sources.tar.gz"
    expected = (root/"profiles-and-sources.tar.gz.sha256").read_text().split()[0]
    assert sha(archive) == expected
    sources = {"archive_sha256":expected,"archive_bytes":archive.stat().st_size,
               "runtime":read(root/"results/provenance.json"),"artifacts":source_records,
               "analysis_sources":[{"artifact":p.name,"sha256":sha(p)} for p in sorted(Path(__file__).parent.glob("*.py"))]}
    assert all(not x["status"] for x in sources["runtime"]["runtime"].values())
    for filename,content in [("optimized-decode-profile.json",output),("optimized-decode-profile-sources.json",sources),
                             ("optimized-decode-profile-cleanup.json",cleanup)]:
        (target/filename).write_text(json.dumps(content,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"insight_groups":len(groups),"captured_requests":3,"verified_decode_rounds":78,
                      "archive_sha256":expected,"vocabulary":lm,"cleanup":cleanup},indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
