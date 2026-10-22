"""First 32 test questions once per arm, natural EOS, sequential requests."""
import hashlib
import json
import time
import requests
from client import DATA, ROOT, URL, request, snapshot


def metrics(path):
    response = requests.get(URL + "/metrics", timeout=30)
    response.raise_for_status()
    path.write_text(response.text)


def benchmark(label):
    lines = DATA.read_text().splitlines()
    assert len(lines) >= 32
    questions = [json.loads(line)["question"] for line in lines[:32]]
    assert len(set(questions)) == 32
    (ROOT / "results" / f"{label}-cohort.json").write_text(json.dumps({
        "source_split": "test", "selection": "first 32 rows, original order",
        "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "question_sha256": [hashlib.sha256(q.encode()).hexdigest() for q in questions],
        "count": 32, "concurrency": 1, "max_output_tokens": 2048,
        "ignore_eos": False, "diagnostics": False, "profiler": False}, indent=2))
    request("What is 1 + 1?", 64, False)
    request(questions[0], 2048, False)
    audit = snapshot(label + "-before")
    state = audit["results"][0]["gdn_state"]
    assert state["all_ssm_fp32"] and state["all_conv_bf16"], state
    assert audit["results"][0]["dtype"] == "torch.bfloat16"
    metrics(ROOT / "results" / f"{label}-natural-before-metrics.txt")
    records = []
    window_start = time.perf_counter()
    for i, question in enumerate(questions):
        row = request(question, 2048, False)
        row.update(kind="natural", repeat=0, question_id=i)
        records.append(row)
        (ROOT / "results" / f"{label}.json").write_text(json.dumps(records))
        print("REQUEST", label, i, "tokens", len(row["token_ids"]),
              "e2e", row["e2e_s"], "ttft", row["ttft_s"],
              "finish", row["finish_reason"], flush=True)
    window = time.perf_counter() - window_start
    tokens = sum(len(r["token_ids"]) for r in records)
    (ROOT / "results" / f"{label}-window.json").write_text(json.dumps({
        "wall_s": window, "output_tokens": tokens, "wall_output_tok_s": tokens / window,
        "note": "Includes client inter-request serialization and file-write gaps; excludes startup, warmup and metrics/audits."}, indent=2))
    metrics(ROOT / "results" / f"{label}-natural-after-metrics.txt")
    snapshot(label + "-after")
    print("BENCHMARK_COMPLETE", label, flush=True)
