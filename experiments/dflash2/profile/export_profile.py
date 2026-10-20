"""Export small path-free evidence; raw traces remain in the private data area."""
import argparse
import hashlib
import json
import pathlib
import statistics

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("repo")
a = p.parse_args()
data = pathlib.Path(a.data)
repo = pathlib.Path(a.repo)
evidence = repo / "evidence/2026-10-20"
eos = set(json.loads((evidence / "dflash2-comparison.json").read_text())["eos_token_ids"])


def prefix(tokens):
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[:i + 1]
    return tokens


captures = {}
normal = {}
checks = []
for mode in ["mtp", "dflash2"]:
    rows = json.loads((data / f"results/{mode}-requests.json").read_text())
    benchmarks = {r["label"]: r for r in rows if r["label"].startswith("benchmark_")}
    normal[mode] = []
    for label, r in benchmarks.items():
        chunks = r["token_chunks"]
        elapsed = chunks[-1]["time_s"] - chunks[0]["time_s"]
        count = len(r["token_ids"]) - chunks[0]["tokens"]
        interval = elapsed / (len(chunks) - 1)
        tokens_per_return = count / (len(chunks) - 1)
        assert abs(tokens_per_return / interval - r["decode_tokens_per_s"]) < 1e-8
        normal[mode].append({"label": label, "returned_batches": len(chunks), "decode_tokens_per_s": r["decode_tokens_per_s"],
                             "tokens_per_return_after_first": tokens_per_return, "mean_return_interval_ms": interval * 1000})
    profile_rows = json.loads((data / f"results/{mode}-profile-requests.json").read_text())
    for row in profile_rows:
        if not row["label"].startswith("profile_"):
            continue
        i = row["prompt_index"]
        label = f"{mode}-p{i}"
        s = json.loads((data / f"analysis/{label}.json").read_text())
        for ref in [r for r in benchmarks.values() if r["label"].startswith(f"benchmark_p{i}_")]:
            checks.append({"capture": label, "reference": ref["label"],
                           "raw_ids_equal": row["token_ids"] == ref["token_ids"],
                           "eos_prefix_ids_equal": prefix(row["token_ids"]) == prefix(ref["token_ids"])})
        captures[label] = {k: s[k] for k in ["decode_rounds", "decode_span_ms", "compute_union_ms", "no_compute_ms", "no_compute_pct",
            "phase_device_totals", "cpu_inclusive_scopes", "phase_device_types", "cycle_medians", "top_decode_groups", "whole_profile_compute_top",
            "sql", "db_tables", "flow_linked_hardware_events", "boundary", "limits"]}
        captures[label]["profiled_request_elapsed_s"] = row["elapsed_s"]
        captures[label]["normal_elapsed_s"] = [r["elapsed_s"] for r in benchmarks.values() if r["label"].startswith(f"benchmark_p{i}_")]
        for stage in ["draft_token", "DFlash2::lm_head_and_topk", "DFlash2::selector_walk"]:
            if stage in s["phase_device_totals"]:
                item = s["phase_device_totals"][stage]
                item["calls_per_round"] = item["calls"] / s["decode_rounds"]
                item["kernel_ms_per_round"] = item["kernel_sum_ms"] / s["decode_rounds"]
        captures[label]["phase_device_totals"] = s["phase_device_totals"]
assert all(c["eos_prefix_ids_equal"] for c in checks)
result = {
    "model": "Qwen3.8-27B", "dtype": "BF16 target and draft", "TP": 1, "concurrency": 1,
    "framework": "vLLM 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac / Ascend 3b31886237ed65c435e0965b10e8002ff5769766",
    "sampling": "T=0, seed42, thinking off, forced 256 tokens", "draft_execution": "eager for both",
    "candidates": {"mtp": 1, "dflash2": 7}, "captures": captures,
    "unprofiled_return_batches": normal, "profile_vs_normal_output_checks": checks,
    "backend_validation": json.loads((data / "analysis/backend-validation.json").read_text()),
    "limits": ["No-compute is absence of compute tasks, not a direct CPU-idle or bandwidth measurement.",
               "Profiler shape recording increases overhead, especially for many eager draft operations.",
               "CPU scopes are inclusive and overlap; kernel sums, graph windows and CPU scopes must not be added together.",
               "SSE return batches and hardware verification rounds can differ by one at the capture boundary.",
               "No SQL MCP was available; the fixed compute macro ran on local read-only SQLite and was cross-checked against CSV."],
}
(evidence / "dflash2-profile-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("EXPORTED", len(captures), "captures", len(checks), "prefix checks")
