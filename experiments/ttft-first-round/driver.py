"""First-round diagnosis and bounded natural-output measurements on our service."""
import json
import os
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parent
sys.path.append(os.environ["COUNT_READY_BASE"])
import client

client.ROOT = ROOT
OUT = ROOT / "results"
URL = client.URL
questions = [json.loads(s)["question"] for s in client.DATA.read_text().splitlines()[:32]]
reference = {r["question_id"]: r for r in json.loads(Path(os.environ["REFERENCE_RESULTS"]).read_text())}


def save(name, obj):
    (OUT / (name + ".json")).write_text(json.dumps(obj, indent=2) + "\n")


def rpc(method, *args):
    if method == "configure_ttft":
        enabled = args[0] == "early_output"
        if enabled:
            args = ("ready3", *args[1:])
        result = raw_rpc(method, *args)
        result["first_output_copy"] = raw_rpc("configure_first_copy", enabled)
        return result
    return raw_rpc(method, *args)


def raw_rpc(method, *args):
    r = requests.post(URL + "/collective_rpc", json={"method": method, "args": list(args)}, timeout=180)
    r.raise_for_status()
    return r.json()["results"][0]


def metrics():
    r = requests.get(URL + "/metrics", timeout=30)
    r.raise_for_status()
    return r.text


def measured(qid, max_tokens=2048):
    row = client.request(questions[qid], max_tokens, False)
    row["question_id"] = qid
    assert row["token_ids"] == reference[qid]["token_ids"][:max_tokens], (qid, "output mismatch")
    return row


action = sys.argv[1]
if action == "warmup":
    rpc("configure_ttft", "ready3", False)
    client.request("What is 1 + 1?", 64, False)
    measured(0); measured(2)
    print("WARMUP_COMPLETE", flush=True)
elif action in ("diagnose", "diagnose_fast"):
    rpc("configure_ttft", "ready3", False)
    client.request("What is 1 + 1?", 64, False)
    measured(0); measured(2)
    trials = []
    orders = (("baseline", "ready3", "prefill_original"), ("prefill_original", "ready3", "baseline"))
    if action == "diagnose_fast":
        selected = sys.argv[2] if len(sys.argv) > 2 else "ready3"
        orders = ((selected,), (selected,))
    for repeat, modes in enumerate(orders):
        for qid in (0, 2):
            for mode in modes:
                config = rpc("configure_ttft", mode, True)
                before = metrics()
                row = measured(qid)
                after = metrics()
                audit = rpc("first_round_audit")
                item = {"repeat": repeat, "mode": mode, "config": config, "request": row,
                        "audit": audit, "before_metrics": before, "after_metrics": after}
                trials.append(item); save("diagnosis", trials)
                print("DIAG", repeat, qid, mode, round(row["ttft_s"] * 1000, 3),
                      "gc_events", len(audit["gc_events"]), flush=True)
    rpc("configure_ttft", "ready3", False)
    print("DIAGNOSIS_COMPLETE", flush=True)
elif action == "chain":
    chains = []
    for mode in ("baseline", "ready3", "prefill_original"):
        rpc("configure_ttft", mode, True)
        before = metrics()
        rows = [measured(qid) for qid in range(4)]
        after = metrics()
        item = {"mode": mode, "requests": rows, "audit": rpc("first_round_audit"),
                "overlap": rpc("overlap_audit"), "before_metrics": before, "after_metrics": after}
        chains.append(item); save("diagnosis-chains", chains)
        print("CHAIN", mode, [round(r["ttft_s"] * 1000, 3) for r in rows], flush=True)
    rpc("configure_ttft", "ready3", False)
    print("CHAIN_COMPLETE", flush=True)
elif action == "benchmark":
    mode, label = sys.argv[2:4]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    save(label + "-config", rpc("configure_ttft", mode, False))
    measured(0)
    save(label + "-before-audit", rpc("graph_audit"))
    (OUT / (label + "-before-metrics.txt")).write_text(metrics())
    rows = []; start = time.perf_counter()
    for qid in range(count):
        row = measured(qid); rows.append(row); save(label, rows)
        print("REQUEST", label, qid, round(row["ttft_s"] * 1000, 3), round(row["e2e_s"], 6), flush=True)
    save(label + "-window", {"wall_s": time.perf_counter() - start})
    (OUT / (label + "-after-metrics.txt")).write_text(metrics())
    save(label + "-after-audit", rpc("graph_audit"))
    save(label + "-overlap-audit", rpc("overlap_audit"))
    save(label + "-first-copy-audit", rpc("first_copy_audit"))
    print("BENCHMARK_COMPLETE", label, flush=True)
elif action == "validate":
    mode = sys.argv[2]
    rows = []
    for qid in (0, 2):
        group = []
        for name in ("baseline", mode, "baseline"):
            rpc("configure_ttft", name, False, True)
            row = measured(qid)
            audit = rpc("overlap_audit")
            entry = {"question_id": qid, "mode": name, "request": row, "audit": audit,
                     "first_copy": rpc("first_copy_audit")}
            rows.append(entry); group.append(entry)
            save("validation-" + mode, rows)
        for key in ("candidate_set_sequence_sha256", "draft_sequence_sha256", "sampled_sequence_sha256"):
            if group[0]["audit"][key] == group[2]["audit"][key]:
                assert group[0]["audit"][key] == group[1]["audit"][key], (qid, key, "trajectory mismatch")
    rpc("configure_ttft", mode, False)
    print("VALIDATION_COMPLETE", mode, flush=True)
else:
    raise ValueError(action)
