"""Paired throughput experiment; no profiler; all selection/gather work timed."""
import json
from pathlib import Path
import time

import requests
from client import request, snapshot, DATA, ROOT, URL


def metrics(path):
    # Metrics are snapshots, not part of the request E2E stopwatch.
    response = requests.get(URL + "/metrics", timeout=30)
    response.raise_for_status()
    path.write_text(response.text)


def benchmark(label):
    questions = [json.loads(line)["question"] for line in DATA.read_text().splitlines()[:4]]
    request("What is 1 + 1?", 64, True)
    request(questions[0], 512, True)
    snapshot(label + "-before")
    records = []
    groups = {
        "fixed512": [(i, repeat, q, 512, True)
                     for repeat in range(2) for i, q in enumerate(questions[:2])],
        "natural": [(i, 0, q, 2048, False) for i, q in enumerate(questions)],
        "long-prefill": [(0, 0,
                          "Background fact: a box contains ten red balls and ten blue balls.\n" * 96
                          + "Ignore the background facts. What is 7 times 8? Explain briefly.", 128, True)],
    }
    for kind, cases in groups.items():
        metrics(ROOT / "results" / f"{label}-{kind}-before-metrics.txt")
        for i, repeat, question, limit, fixed in cases:
            row = request(question, limit, fixed)
            row.update(kind=kind, repeat=repeat, question_id=i)
            records.append(row)
            (ROOT / "results" / f"{label}.json").write_text(json.dumps(records))
            print("REQUEST", label, kind, i, repeat,
                  "tokens", row["usage"]["completion_tokens"],
                  "e2e", row["e2e_s"], "ttft", row["ttft_s"], flush=True)
        time.sleep(0.3)
        metrics(ROOT / "results" / f"{label}-{kind}-after-metrics.txt")
    snapshot(label + "-after")
    print("BENCHMARK_COMPLETE", label, flush=True)
