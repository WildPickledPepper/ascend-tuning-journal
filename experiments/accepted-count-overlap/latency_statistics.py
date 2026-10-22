"""Detailed client-side distributions, including MTP's grouped token delivery."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys


def describe(values, unit):
    values = sorted(float(x) for x in values)
    assert values
    def percentile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {"unit": unit, "n": len(values), "mean": statistics.mean(values),
            "min": values[0], "max": values[-1], "p50": percentile(.5),
            "p75": percentile(.75), "p90": percentile(.9),
            "p95": percentile(.95), "p99": percentile(.99)}


def summarize_rows(rows, window):
    assert len(rows) == 32
    first_times, last_times, tpot_delivery = [], [], []
    token_gaps, block_gaps, block_sizes, blocks_per_request = [], [], [], []
    checks = []
    for row in rows:
        events = []
        arrival_times = []
        ids = []
        for chunk in row["chunks"]:
            event_ids = [t for c in chunk["response"].get("choices", [])
                         for t in (c.get("token_ids") or [])]
            if event_ids:
                events.append(chunk["elapsed_s"])
                block_sizes.append(len(event_ids))
                arrival_times.extend([chunk["elapsed_s"]] * len(event_ids))
                ids.extend(event_ids)
        assert ids == row["token_ids"]
        assert len(ids) == row["usage"]["completion_tokens"] and len(ids) > 1
        assert arrival_times == sorted(arrival_times)
        assert arrival_times[0] == row["ttft_s"] and arrival_times[-1] <= row["e2e_s"]
        first_times.append(arrival_times[0])
        last_times.append(arrival_times[-1])
        tpot_delivery.append(1000 * (arrival_times[-1] - arrival_times[0]) / (len(ids) - 1))
        token_gaps.extend(1000 * (b - a) for a, b in zip(arrival_times, arrival_times[1:]))
        block_gaps.extend(1000 * (b - a) for a, b in zip(events, events[1:]))
        blocks_per_request.append(len(events))
        checks.append({"question_id": row["question_id"], "events": len(events),
                       "tokens": len(ids), "last_token_arrival_s": arrival_times[-1],
                       "stream_end_tail_ms": 1000 * (row["e2e_s"] - arrival_times[-1])})
    outputs = sum(len(r["token_ids"]) for r in rows)
    inputs = sum(r["usage"]["prompt_tokens"] for r in rows)
    elapsed = sum(r["e2e_s"] for r in rows)
    distributions = {
        "E2E": describe([r["e2e_s"] * 1000 for r in rows], "ms"),
        "TTFT": describe([r["ttft_s"] * 1000 for r in rows], "ms"),
        "TPOT_stream_end": describe([r["tpot_ms"] for r in rows], "ms/token"),
        "TPOT_last_token": describe(tpot_delivery, "ms/token"),
        "client_token_delivery_interval": describe(token_gaps, "ms"),
        "nonempty_stream_event_interval": describe(block_gaps, "ms"),
        "InputTokens": describe([r["usage"]["prompt_tokens"] for r in rows], "token"),
        "OutputTokens": describe([len(r["token_ids"]) for r in rows], "token"),
        "request_output_throughput": describe([r["output_tok_s"] for r in rows], "token/s"),
        "tokens_per_nonempty_stream_event": describe(block_sizes, "token/event"),
        "nonempty_events_per_request": describe(blocks_per_request, "event"),
    }
    return {"distributions": distributions,
            "common": {"successful_requests": len(rows), "failed_requests": 0,
                       "concurrency": 1, "input_tokens": inputs, "output_tokens": outputs,
                       "request_e2e_sum_s": elapsed, "benchmark_window_s": window["wall_s"],
                       "client_inter_request_gap_s": window["wall_s"] - elapsed,
                       "output_throughput_by_request_e2e": outputs / elapsed,
                       "request_throughput_by_window": len(rows) / window["wall_s"],
                       "input_throughput_by_window": inputs / window["wall_s"],
                       "output_throughput_by_window": outputs / window["wall_s"],
                       "total_token_throughput_by_window": (inputs + outputs) / window["wall_s"],
                       "weighted_TPOT_stream_end_ms": 1000 * sum(r["e2e_s"] - r["ttft_s"] for r in rows) / (outputs - len(rows)),
                       "weighted_TPOT_last_token_ms": sum(token_gaps) / len(token_gaps),
                       "zero_delivery_intervals": sum(x == 0 for x in token_gaps),
                       "finish_reasons": dict(Counter(r["finish_reason"] for r in rows))},
            "delivery_validation": checks}


def main(root):
    result = {"definitions": {
        "quantiles": "Linear interpolation at (n-1)*p; request percentile estimates use 32 observations.",
        "TPOT_stream_end": "(HTTP stream E2E - first token arrival)/(output tokens-1); matches prior client reports.",
        "TPOT_last_token": "(last token arrival - first token arrival)/(output tokens-1).",
        "client_token_delivery_interval": "Adjacent token arrival timestamps; all IDs in one SSE event share that event's timestamp, hence zero gaps within a group. Not device-internal token generation ITL.",
        "nonempty_stream_event_interval": "Gap between consecutive SSE events containing token IDs; not divided by group size. Not single-token ITL.",
        "common": "Window throughput includes inter-request client serialization/file-writing gaps; request-E2E throughput excludes those gaps. Startup, warmup, audits and metrics excluded.",
    }, "arms": {}}
    for label in ("full-vocab", "dynamic4096"):
        rows = json.loads((root / f"{label}.json").read_text())
        window = json.loads((root / f"{label}-window.json").read_text())
        result["arms"][label] = summarize_rows(rows, window)
    (root / "detailed-statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["common"] for k, v in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
