"""Compare saved same-card ND/NZ MTP receipts; performs no inference."""
import argparse
import datetime
import json
import pathlib
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument("receipts", type=pathlib.Path)
p.add_argument("output", type=pathlib.Path)
p.add_argument("--reference", type=pathlib.Path)
a = p.parse_args()


def load(name):
    return json.loads((a.receipts / name).read_text(encoding="utf-8"))


def relevant(rows):
    return {r["label"]: r for r in rows
            if r["label"].startswith(("benchmark_", "output_check_"))}


def compare(left, right):
    aa, bb = relevant(left), relevant(right)
    assert aa.keys() == bb.keys()
    result = []
    for label, first in aa.items():
        second = bb[label]
        x, y = first["token_ids"], second["token_ids"]
        index = next((i for i, (v, w) in enumerate(zip(x, y)) if v != w), None)
        if index is None and len(x) != len(y):
            index = min(len(x), len(y))
        result.append(dict(label=label, token_ids_equal=x == y,
                           text_equal=first["response_text"] == second["response_text"],
                           first_difference_index=index,
                           reference_tokens=len(x), candidate_tokens=len(y)))
    return result


def metrics(text):
    values = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode_") or "_created" in line:
            continue
        key = line.split("{", 1)[0].split()[0]
        match = re.search(r"\s([-+0-9.eE]+)(?:\s|$)", line)
        if match:
            values[key] = values.get(key, 0) + float(match.group(1))
    return values


rows = {mode: load(mode + "-requests.json") for mode in ("nd", "nz")}
summaries = {mode: load(mode + "-summary.json") for mode in rows}
for mode in rows:
    bench = [r for r in rows[mode] if r["label"].startswith("benchmark_")]
    assert len(bench) == 8
    assert all(r["usage"]["completion_tokens"] == 128 for r in bench)
    assert summaries[mode]["median_decode_tokens_per_s"] == statistics.median(
        r["decode_tokens_per_s"] for r in bench)

acceptance = {}
for mode in rows:
    before = metrics((a.receipts / (mode + "-metrics-before.txt")).read_text(encoding="utf-8"))
    after = metrics((a.receipts / (mode + "-metrics-benchmark-end.txt")).read_text(encoding="utf-8"))
    delta = {k: v - before.get(k, 0) for k, v in after.items()}
    drafts = delta["vllm:spec_decode_num_drafts_total"]
    tokens = delta["vllm:spec_decode_num_draft_tokens_total"]
    accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
    assert drafts > 0 and tokens > 0
    acceptance[mode] = dict(counters_delta=delta, accepted_fraction=accepted / tokens,
                            mean_tokens_per_verification=1 + accepted / drafts,
                            scope="warmups, 8 benchmarks and 3 output checks; excludes profiler")

comparison = compare(rows["nd"], rows["nz"])
result = dict(model="Qwen3.8-27B", architecture="Qwen3_5ForConditionalGeneration",
              dtype="BF16", physical_device=6, TP=1, concurrency=1,
              main_graph_mode="FULL_DECODE_ONLY", capture_sizes=[2],
              speculative_config=dict(method="qwen3_5_mtp", num_speculative_tokens=1,
                                      enforce_eager=True),
              weight_nz_mode=dict(nd=1, nz=2),
              custom_kernel_enabled=False, profiler_during_benchmarks=False,
              summaries=summaries,
              decode_speed_gain_pct=100 * (summaries["nz"]["median_decode_tokens_per_s"] /
                                            summaries["nd"]["median_decode_tokens_per_s"] - 1),
              request_time_change_pct=100 * (summaries["nz"]["median_elapsed_s"] /
                                              summaries["nd"]["median_elapsed_s"] - 1),
              samples={m: [r["decode_tokens_per_s"] for r in rs
                            if r["label"].startswith("benchmark_")] for m, rs in rows.items()},
              acceptance=acceptance, output_comparisons=comparison,
              all_tested_token_ids_equal=all(x["token_ids_equal"] for x in comparison),
              all_tested_text_equal=all(x["text_equal"] for x in comparison),
              completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              limitations=["Same card, sequential ND then NZ; no randomized crossover.",
                           "Two short prompts, 4 repetitions each; not a broad workload or quality evaluation.",
                           "Token equality covers tested prompts only; BF16 is not a universal bitwise guarantee.",
                           "Separate 32-token profiling requests excluded from reported speed.",
                           "Stream timestamps include chunked arrival of accepted MTP tokens."])
if a.reference:
    reference = json.loads(a.reference.read_text(encoding="utf-8"))
    result["nd_vs_prior_non_mtp_baseline"] = compare(reference, rows["nd"])
    result["nz_vs_prior_non_mtp_baseline"] = compare(reference, rows["nz"])
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items()
                  if k in ("decode_speed_gain_pct", "request_time_change_pct",
                           "all_tested_token_ids_equal", "summaries", "acceptance")}, ensure_ascii=False))
