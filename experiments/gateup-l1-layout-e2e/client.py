"""Client-visible streaming latency, with token sequences retained for comparison."""
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8047"
DATA = Path(os.environ["GSM8K_TEST_PATH"])


def snapshot(label):
    r = requests.post(URL + "/collective_rpc", json={"method": "graph_audit"}, timeout=60)
    r.raise_for_status()
    audit = r.json()
    (ROOT / "results" / f"{label}-audit.json").write_text(json.dumps(audit, indent=2))
    r = requests.get(URL + "/metrics", timeout=30)
    r.raise_for_status()
    (ROOT / "results" / f"{label}-metrics.txt").write_text(r.text)
    item = audit["results"][0]
    assert item["num_speculative_tokens"] == 7 and item["draft_use_cuda_graph"]
    assert not item["draft_enforce_eager"] and item["tp_size"] == 1
    assert len(item["layout_experiment"]["eligible"]) == 64
    assert item["layout_experiment"]["packed_count"] in (0, 64)
    assert any(w["is_draft_runnable"] and any(e["captured"] for e in w["entries"]) for w in item["wrappers"])
    assert any(not w["is_draft_runnable"] and any(e["captured"] for e in w["entries"]) for w in item["wrappers"])
    return audit


def request(question, max_tokens, fixed):
    payload = {
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user", "content": f"Question: {question}\nAnswer:"}],
        "temperature": 0, "top_p": 1, "top_k": 1, "seed": 42,
        "max_tokens": max_tokens, "ignore_eos": fixed,
        "stream": True, "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    start = time.perf_counter()
    first = None
    token_ids, chunks, usage = [], [], None
    with requests.post(URL + "/v1/chat/completions", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1):
            if not line or not line.startswith(b"data: "):
                continue
            if line == b"data: [DONE]":
                break
            now = time.perf_counter() - start
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item)
            for choice in item.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids and first is None:
                    first = now
                token_ids.extend(ids)
            if item.get("usage"):
                usage = item["usage"]
            chunks.append({"elapsed_s": now, "response": item})
    elapsed = time.perf_counter() - start
    assert usage and first is not None, chunks[-3:]
    assert len(token_ids) == usage["completion_tokens"], (len(token_ids), usage)
    if fixed:
        assert len(token_ids) == max_tokens
    return {"request": payload, "e2e_s": elapsed, "ttft_s": first,
            "tpot_ms": (elapsed - first) * 1000 / (len(token_ids) - 1),
            "output_tok_s": len(token_ids) / elapsed,
            "usage": usage, "token_ids": token_ids, "chunks": chunks}


def benchmark(label):
    questions = [json.loads(line)["question"] for line in DATA.read_text().splitlines()[:4]]
    request("What is 1 + 1?", 64, True)
    request(questions[0], 512, True)
    snapshot(label + "-before")
    records = []
    for repeat in range(3):
        for i, question in enumerate(questions[:2]):
            row = request(question, 512, True)
            row.update(kind="fixed512", repeat=repeat, question_id=i)
            records.append(row)
            (ROOT / "results" / f"{label}.json").write_text(json.dumps(records))
            print("REQUEST", label, len(records), row["e2e_s"], row["ttft_s"], flush=True)
    for i, question in enumerate(questions):
        row = request(question, 2048, False)
        row.update(kind="natural", repeat=0, question_id=i)
        records.append(row)
        (ROOT / "results" / f"{label}.json").write_text(json.dumps(records))
        print("REQUEST", label, len(records), row["e2e_s"], row["ttft_s"], flush=True)
    snapshot(label + "-after")
    print("BENCHMARK_COMPLETE", label, flush=True)


if __name__ == "__main__":
    import sys
    benchmark(sys.argv[1])
