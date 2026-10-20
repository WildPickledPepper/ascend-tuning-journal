"""Run the public SGLang zero-shot GSM8K evaluator on 128 examples."""
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import requests

root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
os.environ["OPENAI_API_KEY"] = "EMPTY"
package = importlib.import_module("sglang.test")
for name in ["simple_eval_common", "simple_eval_gsm8k"]:
    fullname = "sglang.test." + name
    spec = importlib.util.spec_from_file_location(fullname, root / "upstream" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    setattr(package, name, module)
    spec.loader.exec_module(module)

from sglang.test.simple_eval_common import ChatCompletionSampler
from sglang.test.simple_eval_gsm8k import GSM8KEval

url = "http://127.0.0.1:8040"
warmup = requests.post(url + "/v1/chat/completions", json={
    "model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "Question: What is 1 + 1?\nAnswer:"}],
    "temperature": 0, "top_p": 1, "top_k": 1, "max_tokens": 64,
}, timeout=600)
warmup.raise_for_status()

def snapshot(phase):
    response = requests.post(url + "/collective_rpc", json={"method":"graph_audit"}, timeout=60)
    response.raise_for_status()
    audit = response.json()
    (root / f"graph-audit-{phase}.json").write_text(json.dumps(audit, indent=2))
    assert len(audit["results"]) == 1
    item = audit["results"][0]
    assert item["num_speculative_tokens"] == 7 and item["draft_use_cuda_graph"]
    assert not item["draft_enforce_eager"] and item["tp_size"] == 1
    assert item["mamba_ssm_cache_dtype"] == "bfloat16"
    assert any(w["is_draft_runnable"] and any(e["captured"] for e in w["entries"]) for w in item["wrappers"])
    response = requests.get(url + "/metrics", timeout=15)
    response.raise_for_status()
    (root / f"metrics-{phase}.txt").write_text(response.text)

snapshot("before")
evaluator = GSM8KEval(num_examples=128, num_threads=1, num_shots=0, data_path=str(root / "gsm8k-test.jsonl"))
sampler = ChatCompletionSampler(base_url=url + "/v1", model="Qwen3.8-27B", temperature=0,
                                top_p=1, max_tokens=2048, extra_body={"top_k": 1, "return_meta_info": True})
original = sampler.client.chat.completions.create
records = []
progress = dict(phase='running', total=128, completed=0, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root / 'progress.json').write_text(json.dumps(progress, indent=2))
expected_messages = [sampler._pack_message(content=f"Question: {row['question']}\nAnswer:", role='user') for row in evaluator._lines]
assert len(expected_messages)==128



def observed(**kwargs):
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        response = original(**kwargs)
    except Exception as exc:
        progress.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        (root / 'progress.json').write_text(json.dumps(progress, indent=2))
        raise
    elapsed = time.perf_counter() - started
    index = len(records)
    assert kwargs['messages'] == [expected_messages[index]]
    record = {"started_utc": started_utc, "elapsed_s": elapsed,
              "request": kwargs, "response": response.model_dump(mode="json")}
    records.append(record)
    with (root / 'requests.jsonl').open('a', encoding='utf-8') as receipt:
        receipt.write(json.dumps(record, ensure_ascii=False) + '\n')
    progress.update(completed=len(records), output_tokens=sum(r['response']['usage']['completion_tokens'] for r in records), last_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root / 'progress.json').write_text(json.dumps(progress, indent=2))
    print("REQUEST_COMPLETE", mode, len(records), response.usage.completion_tokens, elapsed, flush=True)
    return response


sampler.client.chat.completions.create = observed
start = time.perf_counter()
result = evaluator(sampler)
latency = time.perf_counter() - start
snapshot("after")
assert len(records) == 128 and all(r["response"].get("usage") for r in records)
total = sum(r["response"]["usage"]["completion_tokens"] for r in records)
receipt = {
    "mode": mode, "protocol": "SGLang PR35629: GSM8K zero-shot/chat, greedy T0/top_p1/top_k1, 2048 output limit, concurrency1; 128 examples.",
    "evaluator_source_commit": "bea986b2f91432e0bdf2c283527427b0e63fcda1",
    "evaluator_source_sha256": hashlib.sha256((root / "upstream/simple_eval_gsm8k.py").read_bytes()).hexdigest(),
    "example_count": 128, "warmup_requests": 1, "warmup_max_tokens": 64,
    "latency_s": latency, "output_tokens": total, "throughput_tok_s": total / latency,
    "accuracy": result.score, "records": records,
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "observer": "Public evaluator and sampler unchanged; normal non-streaming API capture plus one incremental JSONL receipt and progress update per question (outside per-request timer, inside whole-evaluation elapsed).",
}
(root / f"results/{mode}-sglang-requests.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print("BENCHMARK_RESULT", json.dumps({k: v for k, v in receipt.items() if k not in {"records"}}, ensure_ascii=False), flush=True)

progress.update(phase='complete', completed=128, throughput_tok_s=total / latency, accuracy=result.score, finished_utc=receipt['finished_utc'])
(root / 'progress.json').write_text(json.dumps(progress, indent=2))
