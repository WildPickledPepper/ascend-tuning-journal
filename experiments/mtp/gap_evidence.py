"""Inspect CPU activity overlapping three largest gaps, without claiming root cause for all gaps."""
import argparse
import csv
import json
import pathlib
from decimal import Decimal

p = argparse.ArgumentParser()
p.add_argument("profile", type=pathlib.Path)
p.add_argument("output", type=pathlib.Path)
a = p.parse_args()
with (a.profile / "task_time.csv").open(encoding="utf-8-sig", newline="") as f:
    first = min(Decimal(r["task_start(us)"].strip()) for r in csv.DictReader(f)
                if r["kernel_type"] == "MODEL_EXECUTE")
diagnosis = json.loads((a.output / "mtp-bottleneck.json").read_text(encoding="utf-8"))
trace = json.loads((a.profile / "trace_view.json").read_text(encoding="utf-8"))
events = trace["traceEvents"] if isinstance(trace, dict) else trace
result = []
for gap in diagnosis["stages"]["mtp"]["largest_compute_gaps_ms"][:3]:
    start = first + Decimal(str(gap["offset_ms"])) * 1000
    end = start + Decimal(str(gap["duration_ms"])) * 1000
    matches = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op" or not event.get("dur"):
            continue
        lo = Decimal(str(event["ts"]))
        hi = lo + Decimal(str(event["dur"]))
        overlap = min(end, hi) - max(start, lo)
        if overlap > 0:
            matches.append(dict(name=event["name"], overlap_us=float(overlap)))
    result.append(dict(gap=gap, overlapping_cpu_ops=sorted(matches, key=lambda x: -x["overlap_us"])[:12]))
(a.output / "mtp-gap-evidence.json").write_text(json.dumps(dict(
    samples=result,
    interpretation="Largest compute gaps coincide with small CPU copy/conversion/indexing operations.",
    limitations=["Nested CPU scopes overlap; do not sum their durations.",
                 "Three samples do not establish causal attribution of all no-compute time.",
                 "A host/profiled eager path has additional measurement overhead."]),
    ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved", len(result), "bounded gap samples")
