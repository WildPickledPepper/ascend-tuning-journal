"""Natural EOS, four existing questions twice; diagnostics/profiler disabled."""
import json
import requests
from client import DATA, ROOT, URL, request, snapshot


def metrics(path):
    response = requests.get(URL + "/metrics", timeout=30)
    response.raise_for_status()
    path.write_text(response.text)


def benchmark(label):
    questions = [json.loads(line)["question"] for line in DATA.read_text().splitlines()[:4]]
    request("What is 1 + 1?", 64, False)
    request(questions[0], 2048, False)
    audit = snapshot(label + "-before")
    state = audit["results"][0]["gdn_state"]
    assert state["all_ssm_fp32"] and state["all_conv_bf16"], state
    assert audit["results"][0]["dtype"] == "torch.bfloat16"
    metrics(ROOT / "results" / f"{label}-natural-before-metrics.txt")
    records = []
    for repeat in range(2):
        for i, question in enumerate(questions):
            row = request(question, 2048, False)
            row.update(kind="natural", repeat=repeat, question_id=i)
            records.append(row)
            (ROOT / "results" / f"{label}.json").write_text(json.dumps(records))
            print("REQUEST", label, i, repeat, "tokens", len(row["token_ids"]),
                  "e2e", row["e2e_s"], "ttft", row["ttft_s"], flush=True)
    metrics(ROOT / "results" / f"{label}-natural-after-metrics.txt")
    snapshot(label + "-after")
    print("BENCHMARK_COMPLETE", label, flush=True)
