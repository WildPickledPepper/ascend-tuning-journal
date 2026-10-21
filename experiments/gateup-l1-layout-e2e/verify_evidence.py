"""Recompute this study from filtered JSON; never imports NPU libraries."""
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "evidence/2026-10-21"


def load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def stats(rows):
    elapsed = sum(r["e2e_s"] for r in rows)
    tokens = sum(r["usage"]["completion_tokens"] for r in rows)
    return {"requests": len(rows), "output_tokens": tokens,
            "mean_e2e_s": elapsed / len(rows), "total_e2e_s": elapsed,
            "mean_ttft_s": statistics.mean(r["ttft_s"] for r in rows),
            "median_e2e_s": statistics.median(r["e2e_s"] for r in rows),
            "output_tok_s": tokens / elapsed,
            "weighted_tpot_ms": 1000 * sum(r["e2e_s"] - r["ttft_s"] for r in rows) / (tokens - len(rows))}


requests = load("gateup-l1-layout-e2e-requests.json")
summary = load("gateup-l1-layout-e2e.json")
assert summary["tokens_equal"]
hashes = {}
for label, rows in requests.items():
    assert len(rows) == 10
    for row in rows:
        key = row["kind"], row["question_id"]
        hashes.setdefault(key, set()).add(row["output_token_ids_sha256"])
    for kind in ("fixed512", "natural"):
        values = stats([r for r in rows if r["kind"] == kind])
        for key, value in values.items():
            same(value, summary["metrics"][kind][label][key])
    audit = summary["audits"][label]
    assert audit["tp_size"] == 1 and audit["dtype"] == "torch.bfloat16"
    assert audit["num_speculative_tokens"] == 7 and audit["draft_use_cuda_graph"]
    assert audit["layout_experiment"]["packed_count"] == (64 if label == "packed" else 0)
    assert len(audit["layout_experiment"]["eligible"]) == 64
    for draft in (False, True):
        assert any(w["is_draft_runnable"] == draft and any(e["captured"] for e in w["entries"]) for w in audit["wrappers"])
assert all(len(value) == 1 for value in hashes.values())
assert all(value == summary["speculative_counter_deltas"]["baseline-a"] for value in summary["speculative_counter_deltas"].values())
for kind in ("fixed512", "natural"):
    baseline = stats([r for label in ("baseline-a", "baseline-b") for r in requests[label] if r["kind"] == kind])
    candidate = stats([r for r in requests["packed"] if r["kind"] == kind])
    comp = summary["comparison"][kind]
    for key, value in baseline.items():
        same(value, comp["baseline_pooled"][key])
    same(100 * (1 - candidate["mean_e2e_s"] / baseline["mean_e2e_s"]), comp["e2e_reduction_pct"])
    same(100 * (candidate["output_tok_s"] / baseline["output_tok_s"] - 1), comp["throughput_increase_pct"])
    same(100 * (candidate["mean_ttft_s"] / baseline["mean_ttft_s"] - 1), comp["ttft_change_pct"])
    same(100 * (1 - candidate["weighted_tpot_ms"] / baseline["weighted_tpot_ms"]), comp["tpot_reduction_pct"])

micro = load("gateup-l1-layout.json")
timings = load("gateup-l1-layout-timings.json")
for layout in ("native", "nz", "packed"):
    for scope in ("gate", "interleaved"):
        values = [v for suffix in ("a", "b") for v in timings[f"{layout}-{suffix}"]["timing"][scope]["samples_ms"]]
        assert len(values) == 80
        same(statistics.median(values) * 1000 / 64, micro["timing"][layout][scope]["per_layer_us"])
    gate = micro["l2"][layout]["gate"]
    same(gate["read_hits"] / (gate["read_hits"] + gate["read_misses"]), gate["read_hit_rate"])
validation = load("gateup-l1-layout-validation.json")
assert validation["validation_passed"] and validation["native_bit_equal"]
for value in validation["profile_weight_restore_checks"].values():
    assert value["layers"] == 64 and value["all_output_bit_equal"] and value["all_weight_restore_exact"]
for item in load("gateup-l1-layout-sources.json")["copied_sources"]:
    assert hashlib.sha256((REPO / item["archived_path"]).read_bytes()).hexdigest() == item["archived_sha256"], item["archived_path"]
assert load("gateup-l1-layout-cleanup.json")["card7_no_running_processes"]
print("PASS: 30 request timings; exact output hashes; MTP counters; main/draft graphs; 80 samples/layout; L2 ratios; real-weight checks; archived source hashes; card-7 cleanup")
