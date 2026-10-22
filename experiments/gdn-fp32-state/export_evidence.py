"""Compact summaries only; raw requests, answers, traces and addresses stay private."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(raw, destination):
    root = raw / "results"
    summary = read(root / "dynamic4096-summary.json")
    diagnostics = read(root / "trace-analysis.json")
    same_process = [read(root / (label + "-audit.json"))["results"][0]
                    for label in ("full-a", "masked", "full-b")]
    controls = {"same_" + key: len({x[key] for x in same_process}) == 1
                for key in ("main_head_ptr", "draft_head_ptr", "mask_ptr", "draft_graph_wrapper_id")}
    assert all(controls.values())
    controls["main_head_shapes"] = [x["main_head_shape"] for x in same_process]
    controls["main_head_versions"] = [x["main_head_version"] for x in same_process]
    audits = {}
    dynamic_before = read(root / "dynamic4096-before-audit.json")["results"][0]["dynamic_vocabulary"]
    dynamic_after = read(root / "dynamic4096-after-audit.json")["results"][0]["dynamic_vocabulary"]
    candidate_controls = {"same_" + key: dynamic_before[key] == dynamic_after[key]
                          for key in ("ids_pointer", "packed_pointer")}
    candidate_controls["updates_in_measured_group"] = dynamic_after["updates"] - dynamic_before["updates"]
    assert all(candidate_controls.values())
    assert dynamic_after["capacity"] == 4096 and dynamic_after["packed_bytes"] == 40 * 1024 * 1024
    for label, filename in (("diagnostic", "state-audit.json"),
                            ("full-vocab", "full-vocab-after-audit.json"),
                            ("dynamic4096", "dynamic4096-after-audit.json")):
        item = read(root / filename)["results"][0]
        assert item["dtype"] == "torch.bfloat16"
        assert item["gdn_state"]["all_ssm_fp32"]
        assert item["gdn_state"]["all_conv_bf16"]
        dynamic = dict(item["dynamic_vocabulary"])
        for key in ("ids_pointer", "packed_pointer"):
            dynamic.pop(key, None)
        layout = item["layout_experiment"]
        audits[label] = {key: item[key] for key in (
            "dtype", "mamba_cache_dtype", "mamba_ssm_cache_dtype", "num_speculative_tokens",
            "draft_use_cuda_graph", "draft_enforce_eager", "main_graph_mode", "main_capture_sizes",
            "tp_size", "context_length", "max_num_seqs", "prefix_caching", "gpu_memory_utilization")}
        audits[label].update(gdn_state=item["gdn_state"], dynamic_vocabulary=dynamic,
                             graph_wrappers=item["wrappers"], layout={
                                 "packed_count": layout["packed_count"],
                                 "prefill_direct_enabled": layout["prefill_direct_enabled"],
                                 "down_prefetch_in_both_arms": layout["down_prefetch_in_both_arms"],
                                 "counts": layout["counts"]})
    repeat_checks = {}
    repeat_timing = {}
    for label in ("full-vocab", "dynamic4096"):
        rows = read(root / (label + ".json"))
        checks = []
        for qid in range(4):
            pair = [r for r in rows if r["question_id"] == qid]
            assert len(pair) == 2
            checks.append({"question_id": qid,
                           "tokens_equal": pair[0]["token_ids"] == pair[1]["token_ids"],
                           "output_tokens": [len(r["token_ids"]) for r in pair]})
        repeat_checks[label] = checks
        repeat_timing[label] = []
        for repeat in range(2):
            group = [r for r in rows if r["repeat"] == repeat]
            tokens = sum(len(r["token_ids"]) for r in group)
            elapsed = sum(r["e2e_s"] for r in group)
            repeat_timing[label].append({"repeat": repeat, "output_tokens": tokens,
                                         "e2e_total_s": elapsed, "output_tok_s": tokens / elapsed})
    summary.update(diagnostics=diagnostics, same_process_controls=controls,
                   small_head_controls=candidate_controls,
                   audits=audits, repeat_checks=repeat_checks, repeat_timing=repeat_timing,
                   exported_utc=datetime.now(timezone.utc).isoformat(),
                   hardware={"physical_card": 7, "soc": "Ascend910B2", "profiler": False},
                   scope="BF16 weights/conv, FP32 SSM cache; four existing natural-EOS questions twice, no training or new NPU kernel.")
    if (root / "earlier-output-check.json").exists():
        summary["earlier_output_checks"] = read(root / "earlier-output-check.json")
    sources = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in (".json", ".txt"):
            sources.append({"name": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary["private_summary_sources"] = sources
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
