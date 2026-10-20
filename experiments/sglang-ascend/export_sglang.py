"""Export filtered four-question results, keeping login paths and full outputs private."""
import argparse
import ast
import hashlib
import json
import math
import pathlib

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("repo")
a = p.parse_args()
data = pathlib.Path(a.data)
repo = pathlib.Path(a.repo)
receipts = {}
reference_requests = None
answers = {}
for mode in ["ar", "dflash2", "mtp"]:
    d = json.loads((data / f"{mode}-sglang-requests.json").read_text())
    assert len(d["records"]) == d["example_count"] == 4
    requests = [r["request"] for r in d["records"]]
    if reference_requests is None:
        reference_requests = requests
    else:
        assert requests == reference_requests, mode
    assert d["output_tokens"] == sum(r["response"]["usage"]["completion_tokens"] for r in d["records"])
    assert math.isclose(d["throughput_tok_s"], d["output_tokens"] / d["latency_s"], rel_tol=1e-12)
    rows = []
    answers[mode] = []
    for i, r in enumerate(d["records"]):
        response = r["response"]
        choice = response["choices"][0]
        text = choice["message"].get("content") or ""
        answers[mode].append(text)
        meta = response.get("meta_info") or {}
        selected = {k: meta[k] for k in ["spec_verify_ct", "spec_accept_length", "completion_tokens", "cached_tokens", "prompt_tokens", "e2e_latency"] if isinstance(meta, dict) and k in meta}
        rows.append({"question_index": i, "prompt_sha256": hashlib.sha256(json.dumps(r["request"]["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                     "started_utc": r["started_utc"], "elapsed_s": r["elapsed_s"], "usage": response["usage"],
                     "finish_reason": choice["finish_reason"], "answer_sha256": hashlib.sha256(text.encode()).hexdigest(), "meta": selected})
    receipts[mode] = {k: v for k, v in d.items() if k not in {"records", "observer"}}
    receipts[mode]["requests"] = rows
state = json.loads((data / "sglang-state.json").read_text())
assert "error" not in state and state.get("restored_health") == {"8038": 200, "8039": 200}
runtime = json.loads((data / "runtime-facts.json").read_text())
for facts in runtime.values():
    facts["resolved_fields"] = {k: ast.literal_eval(v) for k, v in facts["resolved_fields"].items()}
result = {
    "model": "Qwen3.8-27B", "target_and_draft_dtype": "BF16", "physical_card_count": 2, "hardware": "Ascend 910B2", "TP": 2,
    "concurrency": 1, "protocol": "GSM8K zero-shot chat, first4 test questions, greedy T0/top_p1/top_k1, max2048, RadixCache off, no profiler",
    "same_request_payloads": True, "methods": receipts,
    "dflash2_over_ar": receipts["dflash2"]["throughput_tok_s"] / receipts["ar"]["throughput_tok_s"],
    "mtp_over_ar": receipts["mtp"]["throughput_tok_s"] / receipts["ar"]["throughput_tok_s"],
    "dflash2_over_mtp": receipts["dflash2"]["throughput_tok_s"] / receipts["mtp"]["throughput_tok_s"],
    "answer_content_equal_to_ar": {m: [x == y for x, y in zip(answers["ar"], answers[m])] for m in ["dflash2", "mtp"]},
    "started_utc": state["started_utc"], "restored_utc": state["restored_utc"], "restored_health": state["restored_health"],
    "restored_configuration": json.loads((data / "restored-checks.json").read_text()),
    "sources": json.loads((data / "source-receipt.json").read_text()),
    "runtime_facts": runtime,
    "limits": ["Four examples are a smoke benchmark, not a full accuracy or performance evaluation.",
               "TP2 results cannot be treated as a framework-only speedup over previous TP1 runs.",
               "The upstream Ascend PR used 910C; this run uses 910B2.",
               "Upstream PR does not publish a complete dependency lockfile; source is pinned to its merge and kernel packages to the contemporaneous release.",
               "OpenAI answer-content hashes do not include reasoning_content and are not full token-ID comparisons."],
}
out = repo / "evidence/2026-10-20/sglang-ascend-four-questions.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"methods": {m: {k: receipts[m][k] for k in ["throughput_tok_s", "output_tokens", "latency_s", "accuracy"]} for m in receipts},
                  "dflash2_over_ar": result["dflash2_over_ar"], "dflash2_over_mtp": result["dflash2_over_mtp"]}, indent=2))
