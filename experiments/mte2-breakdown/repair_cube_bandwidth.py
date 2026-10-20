"""Recover 910B Cube bandwidth from preserved msOpProf 26.1 DBI records.

Fixes the observed 3*b+2 -> b key mismatch in host-side statistics only.
Writes separate corrected CSVs and never overwrites original measurements.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import re

root, dest = map(Path, sys.argv[1:3])
dest.mkdir(parents=True, exist_ok=True)
record = struct.Struct("<6QHHBB2x")  # Common::MemRecord, #pragma pack(4)
assert record.size == 56
result = {"source_commit": "56632dc4a944410c791538451dc2156a151bd105",
          "dtype": "BF16", "M": 8, "method": "host-side DBI key correction", "captures": {}}
launches = {}
for spec in ["full-up", "alias64-up", "full-down", "alias64-down"]:
    folder, = (root / (spec + "-profile")).glob("OPPROF_*")
    raw, = (root / (spec + "-raw")).glob("MemoryChart-*.bin")
    blob = raw.read_bytes()
    assert len(blob) % record.size == 0
    raw_bytes = collections.Counter()
    direction_bytes = collections.Counter()
    for item in record.iter_unpack(blob):
        source, target = item[8:10]
        if source == 6 and target in [0, 1, 2]:
            subblock = item[6]
            assert subblock % 3 == 2
            if target == 0:
                raw_bytes[subblock] += item[2]
            direction_bytes[(subblock // 3, target)] += item[2]
    assert set(raw_bytes) == {3 * i + 2 for i in range(23)}
    with (folder / "PipeUtilization.csv").open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        pipe = list(reader)
    memory = list(csv.DictReader((folder / "Memory.csv").open()))
    l2 = list(csv.DictReader((folder / "L2Cache.csv").open()))
    basic = list(csv.DictReader((folder / "OpBasicInfo.csv").open()))[0]
    corrected = []
    for row in pipe:
        b = int(row["block_id"])
        time = float(row["aic_mte2_time(us)"])
        old = raw_bytes[b] / time * 1e6 / 1024**3
        assert abs(old - float(row["aic_mte2_active_bw(GB/s)"])) < 0.0001
        size = sum(direction_bytes[b, direction] for direction in [0, 1, 2])
        bw = size / time * 1e6 / 1024**3
        assert bw > 0
        fixed = dict(row)
        fixed["aic_mte2_active_bw(GB/s)"] = f"{bw:.6f}"
        corrected.append(fixed)
    with (dest / (spec + "-PipeUtilization.corrected.csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(corrected)
    pmu_bytes = sum(float(r["GM_to_L1_datas(KB)"]) for r in memory) * 1024
    assert sum(raw_bytes.values()) == pmu_bytes
    hits = sum(int(r["aic_r0_read_cache_hit"]) + int(r["aic_r1_read_cache_hit"]) for r in l2)
    misses = sum(int(r["aic_r0_read_cache_miss_allocate"]) + int(r["aic_r1_read_cache_miss_allocate"]) for r in l2)
    logs = (root / (spec + ".log")).read_text()
    assert "[ERROR]" not in logs and "VMM_COMPLETE" in logs
    metadata = json.loads(next(l.split(" ", 1)[1] for l in logs.splitlines() if l.startswith("VMM_METADATA")))
    launch_file, = (root / (spec + "-raw")).glob("launch-*.jsonl")
    launch = json.loads(launch_file.read_text().splitlines()[0])
    args = bytes.fromhex(launch["args_hex"])
    tiling = args[launch["tiling_data_offset"]:]
    launches[spec] = {"key": launch["key"], "blocks": launch["blocks"],
                      "tiling_first152_sha256": hashlib.sha256(tiling[:152]).hexdigest(),
                      "weight_address": metadata["address"]}
    result["captures"][spec] = {
        "source": folder.name,
        "kernel_us": float(basic["Task Duration(us)"]),
        "mte2_mean_us": statistics.mean(float(r["aic_mte2_time(us)"]) for r in pipe),
        "cube_mean_us": statistics.mean(float(r["aic_cube_time(us)"]) for r in pipe),
        "l2_read_hit_rate": hits / (hits + misses),
        "l2_hits": hits, "l2_misses": misses,
        "logical_weight_bytes": metadata["logical_bytes"],
        "physical_allocation_bytes": metadata["physical_bytes"],
        "GM_to_L1_source_bytes": sum(raw_bytes.values()),
        "raw_records": len(blob) // record.size,
        "raw_sha256": hashlib.sha256(blob).hexdigest(),
        "old_formula_matches_all_23_rows": True,
        "corrected_positive_blocks": 23,
        "corrected_active_bandwidth_GiB_s": [float(r["aic_mte2_active_bw(GB/s)"]) for r in corrected],
        "launch": {k: v for k, v in launches[spec].items() if k != "weight_address"},
        "valid_output": json.loads(next(l.split(" ", 1)[1] for l in logs.splitlines() if l.startswith("VMM_COMPLETE"))),
    }
for shape in ["up", "down"]:
    a, b = (launches[fmt + "-" + shape] for fmt in ["full", "alias64"])
    assert a == b, (shape, a, b)
result["same_virtual_weight_address_and_tiling_verified"] = True
result["repeat_controls"] = {}
for spec in ["full-nd-repeat", "alias64-nd-repeat", "full-nz-repeat", "alias64-nz-repeat"]:
    calls = []
    for filename in sorted((root / (spec + "-profile")).rglob("OpBasicInfo*.csv")):
        base = list(csv.DictReader(filename.open()))[0]
        rows = list(csv.DictReader(next(filename.parent.glob("PipeUtilization*.csv")).open()))
        cache = list(csv.DictReader(next(filename.parent.glob("L2Cache*.csv")).open()))
        hits = sum(int(r["aic_r0_read_cache_hit"]) + int(r["aic_r1_read_cache_hit"]) for r in cache)
        misses = sum(int(r["aic_r0_read_cache_miss_allocate"]) + int(r["aic_r1_read_cache_miss_allocate"]) for r in cache)
        calls.append({"kernel_us": float(base["Task Duration(us)"]),
                      "mte2_mean_us": statistics.mean(float(r["aic_mte2_time(us)"]) for r in rows),
                      "l2_read_hit_rate": hits / (hits + misses)})
    assert len(calls) == 3
    log = (root / (spec + ".log")).read_text()
    assert "VMM_COMPLETE" in log and "[ERROR]" not in log
    result["repeat_controls"][spec] = {"calls": calls,
        "kernel_mean_us": statistics.mean(c["kernel_us"] for c in calls),
        "kernel_median_us": statistics.median(c["kernel_us"] for c in calls)}
result["smmu_controls"] = {}
for mode in ["full", "alias64"]:
    folder = root / ("smmu-" + mode)
    info = json.loads(next(folder.rglob("profiler_info.json")).read_text())
    assert info["end_info"]["collectionTimeEnd"] > 0 and list(folder.rglob("end_info.6"))
    kernels = [r for r in csv.DictReader(next(folder.rglob("kernel_details.csv")).open()) if "MatMul" in r["Name"]]
    assert len(kernels) == 3
    mapping = {(int(r["Stream ID"]), int(r["Task ID"])): r for r in kernels}
    events = []
    for filename in folder.rglob("socpmu.data.*"):
        if not re.search(r"\.slice_\d+$", filename.name):
            continue
        blob = filename.read_bytes()
        assert len(blob) % 80 == 0
        for row in struct.iter_unpack("<8B4H8Q", blob):
            if row[11] != 2 or (row[9], row[10]) not in mapping:
                continue
            values = row[12:]
            assert values[0] + values[3] + values[4] == values[1]
            events.append({"requests": values[1], "misses": values[0], "key": (row[9], row[10])})
    exported = list(csv.DictReader(next(folder.rglob("soc_pmu.csv")).open()))
    assert len(events) == len(exported) == 3
    for event in events:
        row = next(r for r in exported if (int(r["Stream Id"]), int(r["Task Id"])) == event["key"])
        assert abs(event["misses"] / event["requests"] - float(row["TLB Miss Rate"])) < 0.00000051
    result["smmu_controls"][mode] = {
        "calls": 3, "requests": sum(r["requests"] for r in events), "misses": sum(r["misses"] for r in events),
        "raw_export_crosscheck": True, "valid_end_info": True,
        "kernel_durations_us": [float(r["Duration(us)"]) for r in kernels],
        "warning": "Incorrect schedule warning; complete end metadata and all three kernel/counter groups verified.",
    }
result["insight_validation"] = {}
for spec, capture in result["captures"].items():
    data = json.loads((root / (spec + "-all-memory.json")).read_text())
    assert data["base"]["result"] and len(data["cores"]) == 23
    assert abs(float(data["base"]["body"]["compare"]["duration"]) - capture["kernel_us"]) < 0.00001
    folder, = (root / (spec + "-profile")).glob("OPPROF_*")
    rows = list(csv.DictReader((folder / "L2Cache.csv").open()))
    for core, row in zip(sorted(data["cores"], key=lambda r: int(r["blockId"])), rows):
        assert int(core["l2Cache"]["compare"]["hit"]) == (int(row["aic_r0_read_cache_hit"])
            + int(row["aic_r1_read_cache_hit"]) + int(row["aic_write_cache_hit"]))
    result["insight_validation"][spec] = {"base_duration_matches_csv": True,
        "l2_read_plus_write_hits_match_all_23_blocks": True}
state = json.loads((root / "state.json").read_text())
stop = json.loads((root / "card7-stop-receipt.json").read_text())
result["service_actions"] = {"card6_restored_utc": state["restored_utc"],
    "card6_health_at_restore": state["restored_health"]["8039"],
    "card7_stopped_utc": stop["completed_utc"], "remaining_owned_card7_processes": stop["remaining_owned_device7_processes"]}
result["limitations"] = [
    "Constant synthetic weights and VMM aliases are diagnostic controls, not a model deployment.",
    "Cold-cache native probes do not directly split a real-model 282 us into additive stages.",
    "Changing physical backing also changes physical addressing; cache-hit improvement is measured.",
    "GB/s in the original field uses binary GiB/s according to its source formula.",
    "Host correction is a separate post-processor; the installed msOpProf binary is unchanged.",
]
(dest / "repair-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: {f: r[f] for f in ["kernel_us", "mte2_mean_us", "l2_read_hit_rate", "corrected_positive_blocks"]}
                  for k, r in result["captures"].items()}, indent=2))
