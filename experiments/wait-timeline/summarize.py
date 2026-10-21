"""Read private msOpProf outputs and Insight replies; publish no addresses/PIDs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()

def read_json(name):
    return json.loads((a.input / name).read_text(encoding="utf-8"))

def read_csv(root, name):
    files = list(root.rglob(name))
    assert len(files) == 1, (root, name, files)
    with files[0].open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

result = {
    "mode": "device",
    "soc": "Ascend910B2",
    "dtype": "BF16",
    "weight_layout": "ND",
    "physical_device": 6,
    "native_kernel_unchanged": True,
    "model_service_started": False,
    "default_start_utc": (a.input / "start.txt").read_text().strip(),
    "default_end_utc": (a.input / "end.txt").read_text().strip(),
    "tool_version": (a.input / "version.txt").read_text().strip(),
    "instruction_request": {
        "metric": "instrTimeLine",
        "accepted": False,
        "error": (a.input / "instruction-up.log").read_text().strip(),
        "note": "The command returns exit status 0 despite its error; log content determines validity.",
    },
    "profiles": {},
}
fields = [
    "aic_time(us)", "aic_cube_time(us)", "aic_mte1_time(us)",
    "aic_mte2_time(us)", "aic_scalar_cube_stall_time(us)",
    "aic_scalar_wait_time(us)", "aic_scalar_mte1_stall_time(us)",
    "aic_scalar_mte2_stall_time(us)",
]
for label in ["default-up", "default-down", "source-down"]:
    root = a.input / label
    basic = read_csv(root, "OpBasicInfo.csv")[0]
    pipe = read_csv(root, "PipeUtilization.csv")
    core0 = next(r for r in pipe if r["block_id"] == "0" and r["sub_block_id"] == "cube0")
    base = read_json(label + "-source-details-baseInfo.json")
    memory = read_json(label + "-source-details-memoryGraph.json")
    dynamic = read_json(label + "-source-api-instructions-dynamic.json")
    assert base["result"] and memory["result"] and dynamic["result"]
    assert abs(float(base["body"]["compare"]["duration"]) - float(basic["Task Duration(us)"])) < 0.00001
    assert len(dynamic["body"]["Instructions"]) == 0
    graph = memory["body"]["coreMemory"][0]
    assert abs(float(graph["cube"]["compare"]["ratio"]) / 100 - float(core0["aic_cube_ratio"])) < 0.00001
    binary = next(root.rglob("visualize_data.bin"))
    result["profiles"][label] = {
        "kernel": basic["Op Name"],
        "task_duration_us": float(basic["Task Duration(us)"]),
        "block_dim": int(basic["Block Dim"]),
        "core0_pmu_us": {k: float(core0[k]) for k in fields},
        "core0_l2_read_write_hit_percent": float(graph["l2Cache"]["compare"]["hitRatio"]),
        "insight_duration_and_core0_cube_ratio_match_csv": True,
        "insight_dynamic_instruction_count": 0,
        "visualize_bin_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
source_log = (a.input / "source-down.log").read_text()
result["source_limitations"] = [line for line in source_log.splitlines() if "[WARN]" in line]
result["capability_source_commit"] = "56632dc4a944410c791538451dc2156a151bd105"
result["conclusion"] = {
    "exact_wait_device_timeline_obtained": False,
    "specific_wait_producer_transfer_pair_measured": False,
    "software_serialization_defect_proven": False,
    "TimelineDetail": "Device data plus SIMULATED instruction timing, not measured WAIT timing.",
    "instrTimeLine_PipeTimeline_PcSampling": "Restricted to Ascend950 in matching tool source.",
    "scalar_cube_stall": "Scalar stalls on full CUBE IQ; NOT Cube waiting for data.",
    "scalar_wait_zero": "Does not establish zero WAIT_FLAG stalls inside other pipelines.",
    "scope": "One profiled launch per shape; synthetic constants, no model workload or throughput benchmark.",
}
result["cleanup"] = {"checked_utc": "2026-10-21T02:08:08+00:00", "card6_active_processes": 0, "card7_active_processes": 0}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v["task_duration_us"] for k, v in result["profiles"].items()}))
