"""Summarize native msOpProf layout probes; retain raw inputs outside the repo."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys

root, output = map(Path, sys.argv[1:3])
result = {"mode": "device", "input": "application", "device": "Ascend910B2",
          "dtype": "BF16", "M": 8, "captures": {}, "launches": {}}
for spec in ["nd-up", "nd-down", "nz-up", "nz-down"]:
    folder, = (root / (spec + "-free")).glob("OPPROF_*")
    log = (root / (spec + "-free.log")).read_text()
    assert "COMPLETE_NATIVE" in log and "[ERROR]" not in log
    tables = {}
    for filename in ["OpBasicInfo", "PipeUtilization", "L2Cache", "Memory", "ResourceConflictRatio"]:
        tables[filename] = list(csv.DictReader((folder / (filename + ".csv")).open()))
    pipe = tables["PipeUtilization"]
    assert len(pipe) == 23
    metrics = ["aic_time(us)", "aic_mte2_time(us)", "aic_mte1_time(us)", "aic_cube_time(us)"]
    positive = [r for r in pipe if float(r["aic_mte2_active_bw(GB/s)"]) > 0]
    capture = {
        "source": folder.name,
        "basic": tables["OpBasicInfo"],
        "mean_across_23_blocks": {k: statistics.mean(float(r[k]) for r in pipe) for k in metrics},
        "mte2_active_bw_coverage": {
            "nonzero_blocks": [int(r["block_id"]) for r in positive],
            "zero_blocks": [int(r["block_id"]) for r in pipe if float(r["aic_mte2_active_bw(GB/s)"]) == 0],
            "nonzero_mean_GB_s": statistics.mean(float(r["aic_mte2_active_bw(GB/s)"]) for r in positive),
            "status": "partial coverage or mapping anomaly; zero rows are not physical zero bandwidth",
        },
        "GM_to_L1_request_KiB": sum(float(r["GM_to_L1_datas(KB)"]) for r in tables["Memory"]),
        "L2_read_hit_percent_mean": statistics.mean(float(r["aic_read_hit_rate(%)"]) for r in tables["L2Cache"]),
        "csv_sha256": {name: hashlib.sha256((folder / (name + ".csv")).read_bytes()).hexdigest()
                       for name in tables},
    }
    # Basic info includes run identifiers; filter them before publishing.
    capture["basic"] = [{k: v for k, v in row.items() if k.lower() not in {"pid", "device id", "device_id"}}
                        for row in capture["basic"]]
    result["captures"][spec] = capture
for fmt in ["nd", "nz"]:
    filename, = (root / ("launch-fixed-" + fmt)).glob("*.jsonl")
    row = json.loads(filename.read_text().splitlines()[0])
    raw = bytes.fromhex(row["args_hex"])[row["tiling_data_offset"]:]
    assert len(raw) == 160
    result["launches"][fmt] = {"key": row["key"], "blocks": row["blocks"],
                               "tiling_words": list(struct.unpack("<40I", raw)),
                               "first152_sha256": hashlib.sha256(raw[:152]).hexdigest()}
assert result["launches"]["nd"]["first152_sha256"] == result["launches"]["nz"]["first152_sha256"]
result["limits"] = [
    "Independent native probes, not real 64-layer model measurements or a speed benchmark.",
    "One profiled call per layout/shape; no distribution or end-to-end improvement claimed.",
    "Pipeline activity times overlap and cannot be added to form wall time.",
    "ND/NZ total-time difference includes layout and memory-transaction changes, not pure conversion time.",
    "Current data do not split physical HBM service, L2 lookup and ND2NZ execution into additive times.",
    "Failed captures with memory allocation error 207001 are excluded from effective metrics.",
]
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v["mean_across_23_blocks"] for k, v in result["captures"].items()}, indent=2))
