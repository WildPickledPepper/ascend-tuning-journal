"""Export the 32-request comparison without prompts, answers, IDs or addresses."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

from analyze import main, summarize


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(raw, destination):
    root = raw / "results"
    main(root, "dynamic4096")
    result = read(root / "dynamic4096-summary.json")
    cohorts = [read(root / f"{label}-cohort.json") for label in ("full-vocab", "dynamic4096")]
    assert cohorts[0] == cohorts[1] and cohorts[0]["count"] == 32
    assert len(set(cohorts[0]["question_sha256"])) == 32
    result["cohort"] = cohorts[0]
    result["request_statistics"] = {}
    result["audits"] = {}
    arm_rows = {}
    for label in ("full-vocab", "dynamic4096"):
        _, rows = summarize(root, label)
        arm_rows[label] = rows
        assert len(rows) == 32 and [r["question_id"] for r in rows] == list(range(32))
        assert all(not r["request"]["ignore_eos"] and r["request"]["temperature"] == 0 for r in rows)
        speeds = [r["output_tok_s"] for r in rows]
        result["request_statistics"][label] = {
            "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
            "capped_question_ids": [r["question_id"] for r in rows if r["finish_reason"] == "length"],
            "input_range": [min(r["usage"]["prompt_tokens"] for r in rows), max(r["usage"]["prompt_tokens"] for r in rows)],
            "output_range": [min(len(r["token_ids"]) for r in rows), max(len(r["token_ids"]) for r in rows)],
            "mean_request_output_tok_s": statistics.mean(speeds),
            "median_request_output_tok_s": statistics.median(speeds),
            "wall_window": read(root / f"{label}-window.json"),
            "per_request": [{"question_id": r["question_id"],
                             "input_tokens": r["usage"]["prompt_tokens"], "output_tokens": len(r["token_ids"]),
                             "e2e_s": r["e2e_s"], "ttft_s": r["ttft_s"],
                             "output_tok_s": r["output_tok_s"], "finish_reason": r["finish_reason"]}
                            for r in rows],
        }
        before = read(root / f"{label}-before-audit.json")["results"][0]
        after = read(root / f"{label}-after-audit.json")["results"][0]
        assert after["dtype"] == "torch.bfloat16" and after["gdn_state"]["layer_count"] == 48
        assert after["gdn_state"]["all_ssm_fp32"] and after["gdn_state"]["all_conv_bf16"]
        assert after["layout_experiment"]["packed_count"] == 64
        dynamic = dict(after["dynamic_vocabulary"])
        if dynamic["enabled"]:
            previous = before["dynamic_vocabulary"]
            assert all(dynamic[k] == previous[k] for k in ("ids_pointer", "packed_pointer"))
            dynamic["updates_in_measured_group"] = dynamic["updates"] - previous["updates"]
            assert dynamic["updates_in_measured_group"] > 0
            assert dynamic["capacity"] == 4096 and dynamic["packed_bytes"] == 40 * 1024 * 1024
            dynamic["buffers_stable"] = True
        for key in ("ids_pointer", "packed_pointer"):
            dynamic.pop(key, None)
        result["audits"][label] = {key: after[key] for key in (
            "dtype", "mamba_cache_dtype", "mamba_ssm_cache_dtype", "tp_size", "max_num_seqs",
            "context_length", "num_speculative_tokens", "prefix_caching", "main_capture_sizes",
            "draft_use_cuda_graph", "draft_enforce_eager", "gpu_memory_utilization")}
        state = after["gdn_state"]
        result["audits"][label].update(
            state_summary={key: state[key] for key in ("layer_count", "all_ssm_fp32", "all_conv_bf16", "allocated_bytes", "reserved_bytes")},
            dynamic_vocabulary=dynamic, graph_wrappers=after["wrappers"],
            layout={key: after["layout_experiment"][key] for key in ("packed_count", "prefill_direct_enabled", "down_prefetch_in_both_arms")})
    result["paired_timing"] = {
        "candidate_faster_requests": sum(c["e2e_s"] < b["e2e_s"]
                                         for b, c in zip(arm_rows["full-vocab"], arm_rows["dynamic4096"])),
        "requests": 32,
        "note": "Descriptive paired count; arms ran sequentially, not interleaved; no confidence interval implied."}
    result["exported_utc"] = datetime.now(timezone.utc).isoformat()
    result["hardware"] = {"soc": "Ascend910B2", "physical_card": 7}
    result["private_summary_sources"] = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.iterdir()) if p.is_file() and p.suffix in (".json", ".txt")]
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
