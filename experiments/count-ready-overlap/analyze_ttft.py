"""Offline TTFT audit. Raw requests/traces remain outside the repository.

Usage: python analyze_ttft.py PRIVATE_STAGE3_ROOT OUTPUT_JSON
No model, NPU API, or database writes are used. Optional runtime timing source
snapshots must be fetched separately with read-only access.
"""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import sys


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(" ", 1)
        result[key] = float(value)
    return result


def fingerprint(items):
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def trace_summary(profile):
    infos = list(profile.glob("profiler_info*.json"))
    assert len(infos) == 1
    info = load(infos[0])
    assert info["end_info"]
    out = profile / "ASCEND_PROFILER_OUTPUT"
    assert (out / "trace_view.json").is_file() and (out / "kernel_details.csv").is_file()
    assert list(out.glob("*profiler*.db"))
    trace = load(out / "trace_view.json")
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    pids = {e["pid"] for e in events if e.get("ph") == "M"
            and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Python"}
    assert len(pids) == 1
    pid = next(iter(pids))
    ops = sorted((e for e in events if e.get("ph") == "X"
                  and e.get("pid") == pid and e.get("cat") == "cpu_op"),
                 key=lambda e: (Decimal(str(e["ts"])), -e["dur"]))
    first = {}
    for name in ("prepare input", "forward", "sample_token", "draft_token", "async_state_update"):
        first[name] = next(e for e in ops if e["name"] == name)
    forward = first["forward"]
    a = Decimal(str(forward["ts"]))
    b = a + Decimal(str(forward["dur"]))
    names = [e["name"] for e in ops if a <= Decimal(str(e["ts"]))
             and Decimal(str(e["ts"])) + Decimal(str(e["dur"])) <= b]
    sample_start = Decimal(str(first["sample_token"]["ts"]))
    clones = [e for e in ops if e["name"] == "aten::clone"
              and b <= Decimal(str(e["ts"])) < sample_start]
    with (out / "kernel_details.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    head = next(r for r in rows if r["Type"] == "MatMulV2"
                and r["Input Shapes"].strip('"') == "1,5120;248320,5120")
    end = Decimal(head["Start Time(us)"].strip()) + Decimal(head["Duration(us)"])
    pre = [r for r in rows if Decimal(r["Start Time(us)"].strip()) < end]
    shapes = [(r["Name"], r["Type"], r["Input Shapes"]) for r in pre]
    return {
        "valid_framework_stop": True,
        "profiler_config": info["config"],
        "trace_sha256": sha(out / "trace_view.json"),
        "kernel_csv_sha256": sha(out / "kernel_details.csv"),
        "first_cpu_scopes_ms": {k: e["dur"] / 1000 for k, e in first.items()},
        "prefill_forward_cpu_op_count": len(names),
        "prefill_forward_cpu_op_sequence_sha256": fingerprint(names),
        "main_prefill_through_full_head_kernel_count": len(pre),
        "main_prefill_through_full_head_kernel_sequence_sha256": fingerprint(shapes),
        "main_prefill_through_full_head_kernel_duration_sum_ms": sum(float(r["Duration(us)"]) for r in pre) / 1000,
        "snapshot_clones_before_first_sampling": len(clones),
        "snapshot_clone_cpu_scope_sum_ms": sum(e["dur"] for e in clones) / 1000,
        "sample_start_to_draft_submission_end_ms": float(
            Decimal(str(first["draft_token"]["ts"])) + Decimal(str(first["draft_token"]["dur"])) - sample_start) / 1000,
        "boundary_note": "The sampled request is question 0, not the unprofiled question-2 outlier. CPU scopes include submission; kernel duration sums are not wall time or a full TTFT decomposition.",
    }


def main(root, output):
    results = root / "results"
    evidence = {"model": "Qwen3.8-27B", "physical_card": 6, "TP": 1,
                "concurrency": 1, "MTP": 7, "dynamic_vocabulary": 4096,
                "weights": "BF16", "ssm_state": "FP32", "new_npu_requests": 0,
                "analysis_source_sha256": sha(Path(__file__)), "arms": {},
                "private_input_sha256": {}}
    metric_names = ("time_to_first_token_seconds", "request_queue_time_seconds",
                    "request_prefill_time_seconds", "request_decode_time_seconds",
                    "request_inference_time_seconds", "e2e_request_latency_seconds")
    rows_by_label = {}
    for label in ("control4", "fast4"):
        path = results / (label + ".json")
        rows = load(path)
        assert len(rows) == 4
        rows_by_label[label] = rows
        before = metrics(results / (label + "-before-metrics.txt"))
        after = metrics(results / (label + "-after-metrics.txt"))
        delta = {}
        for name in metric_names:
            values = {suffix: sum(v - before.get(k, 0) for k, v in after.items()
                                  if k.startswith("vllm:" + name + suffix + "{"))
                      for suffix in ("_count", "_sum")}
            assert values["_count"] == 4
            delta[name] = {"count": 4, "sum_s": values["_sum"], "mean_ms": values["_sum"] * 250}
        ab = load(results / (label + "-before-audit.json"))["results"][0]["layout_experiment"]["counts"]
        aa = load(results / (label + "-after-audit.json"))["results"][0]["layout_experiment"]["counts"]
        layout_delta = {k: v - ab.get(k, 0) for k, v in aa.items() if v != ab.get(k, 0)}
        assert layout_delta["prefill_direct_host_calls"] == 256
        assert layout_delta.get("gate_unpack_fallback_host_calls", 0) == 0
        arm = {"per_request": [], "metric_deltas": delta, "layout_dispatch_deltas": layout_delta}
        guards = {}
        for name in ("request_success_total", "prompt_tokens_total", "prompt_tokens_cached_total",
                     "num_preemptions_total", "request_prefill_kv_computed_tokens_sum"):
            guards[name] = sum(v - before.get(k, 0) for k, v in after.items()
                               if k.startswith("vllm:" + name + "{"))
        assert guards["request_success_total"] == 4
        assert guards["prompt_tokens_total"] == guards["request_prefill_kv_computed_tokens_sum"] == 410
        assert guards["prompt_tokens_cached_total"] == guards["num_preemptions_total"] == 0
        arm["counter_guards"] = guards
        for row in rows:
            role = next(c["elapsed_s"] for c in row["chunks"]
                        if any(x.get("delta", {}).get("role") for x in c["response"].get("choices", [])))
            arm["per_request"].append({"question_id": row["question_id"],
                "input_tokens": row["usage"]["prompt_tokens"], "output_tokens": len(row["token_ids"]),
                "ttft_ms": row["ttft_s"] * 1000, "role_chunk_ms": role * 1000,
                "role_to_first_token_ms": (row["ttft_s"] - role) * 1000, "e2e_s": row["e2e_s"]})
        arm["client_ttft_mean_ms"] = statistics.mean(x["ttft_s"] * 1000 for x in rows)
        evidence["arms"][label] = arm
        for suffix in (".json", "-before-metrics.txt", "-after-metrics.txt", "-before-audit.json", "-after-audit.json"):
            p = results / (label + suffix)
            evidence["private_input_sha256"][p.name] = sha(p)
    assert all(a["token_ids"] == b["token_ids"] and a["usage"]["prompt_tokens"] == b["usage"]["prompt_tokens"]
               for a, b in zip(rows_by_label["control4"], rows_by_label["fast4"]))
    evidence["same_outputs_and_prompt_lengths"] = True
    evidence["same_layout_dispatch_counts"] = evidence["arms"]["control4"]["layout_dispatch_deltas"] == evidence["arms"]["fast4"]["layout_dispatch_deltas"]
    additional = {}
    for name in ("check-0-base1", "check-0-base2", "check-0-ready3", "check-2-base1", "check-2-base2", "check-2-ready3", "baseline-unprofiled-prefix", "ready3-unprofiled-prefix"):
        row = load(results / (name + ".json"))
        additional[name] = {"ttft_ms": row["ttft_s"] * 1000, "e2e_s": row["e2e_s"]}
    later = load(results / "ready3.json")
    assert all(later[i]["token_ids"] == rows_by_label["fast4"][i]["token_ids"] for i in range(4))
    additional["later32_first4"] = [{"question_id": r["question_id"], "ttft_ms": r["ttft_s"] * 1000} for r in later[:4]]
    additional["later32_first4_mean_ms"] = statistics.mean(r["ttft_s"] * 1000 for r in later[:4])
    additional["limitations"] = "Later32 is same service/stage with diagnostics off, but a later sequential run, not a randomized repeated A/B. Preflight uses diagnostics and is not a benchmark. Prefix requests have different output limits."
    evidence["additional_existing_observations"] = additional
    evidence["sampled_question0"] = {}
    for label in ("baseline", "ready3"):
        request = load(results / (label + "-profile.json"))
        profile = root / request["profile"]
        evidence["sampled_question0"][label] = trace_summary(profile)
        evidence["sampled_question0"][label]["client_ttft_ms"] = request["request"]["ttft_s"] * 1000
    a, b = evidence["sampled_question0"].values()
    for key in ("prefill_forward_cpu_op_sequence_sha256", "main_prefill_through_full_head_kernel_sequence_sha256"):
        assert a[key] == b[key]
    evidence["timing_metric_definition"] = {
        "queue": "first SCHEDULED timestamp minus first QUEUED timestamp",
        "prefill": "first NEW_TOKEN engine timestamp minus first SCHEDULED timestamp; includes first-iteration preparation, submission, sampling/draft and output processing, not just model kernels",
        "server_ttft": "API output processor iteration timestamp minus request arrival timestamp",
        "client_ttft": "request start to first SSE choice containing token_ids; empty role event is excluded",
    }
    src = root / "timing-runtime-sources-private.json"
    ver = root / "timing-runtime-verification-private.json"
    if src.exists() and ver.exists():
        sources, verified = load(src), load(ver)
        assert verified["commit"] == "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac" and verified["tracked_diff"] == ""
        assert {k: v["sha256"] for k, v in sources.items()} == verified["files"]
        assert "prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts" in sources["v1/metrics/stats.py"]["text"]
        evidence["timing_runtime_source_verification"] = verified
    evidence["limitations"] = [
        "Only four-request aggregate server histograms were saved; the question-2 server prefill duration is not individually known.",
        "No unprofiled request-stage timestamps, worker OS scheduling/GC trace, or device counters were saved for the 499ms outlier.",
        "Existing detailed traces are profiled question 0; cannot identify the exact source of the unprofiled question-2 extra 132ms.",
        "Variation of the same optimized request supports a transient component, but does not prove absence of a systematic first-token regression.",
        "Early-count scheduling also runs on the first/prefill iteration; only the fast metadata builder is decode-only. Clone CPU scope durations do not measure total asynchronous impact.",
    ]
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TTFT audit passed: four-request metrics, identical outputs/layouts, first-round sequences, and source boundaries verified.")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
