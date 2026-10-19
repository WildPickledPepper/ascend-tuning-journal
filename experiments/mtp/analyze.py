"""Summarize saved receipts without starting inference or changing the runtime."""
import argparse
import datetime
import json
import pathlib
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument("run")
a = p.parse_args()
root = pathlib.Path(a.run)


def metrics(text):
    values = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode_") or "_created" in line:
            continue
        name = line.split("{", 1)[0].split()[0]
        match = re.search(r"\s([-+0-9.eE]+)(?:\s|$)", line)
        if match:
            values[name] = values.get(name, 0) + float(match.group(1))
    return values


data = {m: json.loads((root / (m + "-requests.json")).read_text()) for m in ["baseline", "mtp"]}
summaries = {m: json.loads((root / (m + "-summary.json")).read_text()) for m in data}
comparisons = []
per_prompt = {}
for row in data["baseline"]:
    if not row["label"].startswith(("benchmark_", "output_check_")):
        continue
    other = next(r for r in data["mtp"] if r["label"] == row["label"])
    aa, bb = row["token_ids"], other["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(aa, bb)) if x != y), None)
    if first is None and len(aa) != len(bb):
        first = min(len(aa), len(bb))
    comparisons.append(dict(label=row["label"], token_ids_equal=aa == bb,
                            text_equal=row["response_text"] == other["response_text"],
                            baseline_tokens=len(aa), mtp_tokens=len(bb), first_difference_index=first))
for i in range(2):
    series = {m: [r["decode_tokens_per_s"] for r in rows
                  if r["label"].startswith(f"benchmark_p{i}_")] for m, rows in data.items()}
    medians = {m: statistics.median(v) for m, v in series.items()}
    per_prompt[str(i)] = dict(samples=series, medians=medians,
                              gain_pct=100 * (medians["mtp"] / medians["baseline"] - 1))
before = metrics((root / "mtp-metrics-before.txt").read_text())
after = metrics((root / "mtp-metrics-benchmark-end.txt").read_text())
delta = {k: v - before.get(k, 0) for k, v in after.items()}
drafts = delta.get("vllm:spec_decode_num_drafts_total", 0)
draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
assert drafts > 0 and draft_tokens > 0, delta
result = dict(model="Qwen3.8-27B", architecture="Qwen3_5ForConditionalGeneration",
              dtype="BF16", physical_devices={"baseline": 7, "mtp": 6}, TP=1, concurrency=1,
              graph_mode="FULL_DECODE_ONLY", capture_sizes={"baseline": [1], "mtp": [2]},
              speculative_config=dict(method="qwen3_5_mtp", num_speculative_tokens=1, enforce_eager=True),
              custom_kernel_enabled=False, summaries=summaries, per_prompt=per_prompt,
              speed_gain_pct=100 * (summaries["mtp"]["median_decode_tokens_per_s"] /
                                    summaries["baseline"]["median_decode_tokens_per_s"] - 1),
              acceptance=dict(scope="warmups, benchmarks and output checks, before profiler",
                              counters_delta=delta, draft_acceptance_rate=accepted / draft_tokens,
                              mean_tokens_per_verification=1 + accepted / drafts),
              output_comparisons=comparisons,
              all_tested_token_ids_equal=all(c["token_ids_equal"] for c in comparisons),
              completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              limitations=["Short deterministic single-user requests, not a broad quality evaluation.",
                           "Timing uses streamed token timestamps; accepted tokens can arrive in chunks.",
                           "Acceptance counters include warmups and output checks; timing summary uses eight benchmarks only.",
                           "Different rented 910B2 devices: baseline on 7, MTP on 6; performance requests staggered, no randomized crossover.",
                           "Weight-byte amortization is a model of reuse, not an HBM counter measurement."])
(root / "mtp-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
