"""Export small evidence; never copy server logs, paths, prompts or raw IDs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

raw, dest = map(Path, sys.argv[1:3])
summary = json.loads((raw / "results/dynamic4096-summary.json").read_text())
audits = {}
for label in ("full-vocab", "dynamic4096"):
    item = json.loads((raw / f"results/{label}-after-audit.json").read_text())["results"][0]
    dynamic = item["dynamic_vocabulary"]
    for key in ("ids_pointer", "packed_pointer"):
        dynamic.pop(key, None)
    layout = item["layout_experiment"]
    audits[label] = {
        key:item[key] for key in ("num_speculative_tokens", "draft_use_cuda_graph", "draft_enforce_eager",
                                 "main_graph_mode", "main_capture_sizes", "dtype", "tp_size", "context_length",
                                 "max_num_seqs", "prefix_caching", "gpu_memory_utilization")}
    audits[label].update(dynamic_vocabulary=dynamic, layout={
        "packed_count":layout["packed_count"], "prefill_direct_enabled":layout["prefill_direct_enabled"],
        "down_prefetch_in_both_arms":layout["down_prefetch_in_both_arms"], "counts":layout["counts"]},
        graph_wrappers=item["wrappers"])
summary["audit"] = audits
summary["exported_utc"] = datetime.now(timezone.utc).isoformat()
summary["hardware"] = {"card":7, "soc":"Ascend910B2", "profiler":False}
dest.mkdir(parents=True, exist_ok=True)
(dest / "dynamic-draft-vocab.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")
manifest = []
for name in ("dynamic_vocab.py", "prepare_seed.py", "model_extension.py", "start.sh", "serve.py",
             "controller.py", "client.py", "bench.py", "analyze.py", "seed-private.json"):
    path = raw / name
    manifest.append({"name":name, "bytes":path.stat().st_size,
                     "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
sources = {"files":manifest, "selection_source":"64 independent GSM8K train model-generated records; no test fitting"}
(dest / "dynamic-draft-vocab-sources.json").write_text(json.dumps(sources,indent=2), encoding="utf-8")
print("Exported summary and source digests; raw data remains private")
