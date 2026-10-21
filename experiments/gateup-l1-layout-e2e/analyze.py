"""Recompute client timing and exact token comparisons from archived records."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent / "results"
labels = ["baseline-a", "packed", "baseline-b"]
data = {label: json.loads((ROOT / f"{label}.json").read_text()) for label in labels}


def stats(rows):
    total = sum(r["e2e_s"] for r in rows)
    tokens = sum(r["usage"]["completion_tokens"] for r in rows)
    return {"requests": len(rows), "output_tokens": tokens,
            "mean_e2e_s": total / len(rows), "total_e2e_s": total,
            "mean_ttft_s": statistics.mean(r["ttft_s"] for r in rows),
            "median_e2e_s": statistics.median(r["e2e_s"] for r in rows),
            "output_tok_s": tokens / total,
            "weighted_tpot_ms": 1000 * sum(r["e2e_s"] - r["ttft_s"] for r in rows) / (tokens - len(rows))}


out = {"protocol": "One physical card 7, sequential baseline/packed/baseline; same down-prefetch, MTP7 and main/draft graphs; no profiler. Client perf_counter includes HTTP streaming overhead. Startup/weight packing and warmups excluded; runtime unpack included.",
       "metrics": {}, "comparison": {}, "tokens_equal": True, "token_checks": []}
out["audits"] = {}
out["speculative_counter_deltas"] = {}
for label in labels:
    out["audits"][label] = json.loads((ROOT / f"{label}-after-audit.json").read_text())["results"][0]
    deltas = {}
    for phase, sign in (("before", -1), ("after", 1)):
        for line in (ROOT / f"{label}-{phase}-metrics.txt").read_text().splitlines():
            if line.startswith("vllm:spec_decode") and "_total{" in line:
                key, value = line.rsplit(" ", 1)
                deltas[key] = deltas.get(key, 0) + sign * float(value)
    out["speculative_counter_deltas"][label] = deltas
for kind in ("fixed512", "natural"):
    grouped = {label: [r for r in rows if r["kind"] == kind] for label, rows in data.items()}
    out["metrics"][kind] = {label: stats(rows) for label, rows in grouped.items()}
    base = stats(grouped["baseline-a"] + grouped["baseline-b"])
    candidate = stats(grouped["packed"])
    out["comparison"][kind] = {
        "baseline_pooled": base, "packed": candidate,
        "e2e_reduction_pct": 100 * (1 - candidate["mean_e2e_s"] / base["mean_e2e_s"]),
        "throughput_increase_pct": 100 * (candidate["output_tok_s"] / base["output_tok_s"] - 1),
        "ttft_change_pct": 100 * (candidate["mean_ttft_s"] / base["mean_ttft_s"] - 1),
        "tpot_reduction_pct": 100 * (1 - candidate["weighted_tpot_ms"] / base["weighted_tpot_ms"]),
    }
    for i in sorted({r["question_id"] for r in grouped["packed"]}):
        rows = [(label, r) for label, rr in grouped.items() for r in rr if r["question_id"] == i]
        expected = rows[0][1]["token_ids"]
        equal = all(r["token_ids"] == expected for _, r in rows)
        out["tokens_equal"] &= equal
        out["token_checks"].append({"kind": kind, "question_id": i, "all_equal": equal,
            "cases": [{"label": label, "repeat": r["repeat"], "tokens": len(r["token_ids"]),
                       "sha256": hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()}
                      for label, r in rows]})
(ROOT / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k not in {"token_checks", "audits"}}, indent=2))
