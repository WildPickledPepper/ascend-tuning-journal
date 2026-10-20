"""Cross-check all Insight operator groups and the fixed SQL macro against CSV."""
import argparse
import collections
import csv
import json
import pathlib
from decimal import Decimal

p = argparse.ArgumentParser()
p.add_argument("data")
a = p.parse_args()
root = pathlib.Path(a.data)
results = {}
for mode in ["mtp", "dflash2"]:
    folders = sorted((root / f"profiles-{mode}").glob("*/ASCEND_PROFILER_OUTPUT"))
    assert len(folders) == 2
    for i, folder in enumerate(folders):
        label = f"{mode}-p{i}"
        groups = collections.defaultdict(lambda: [0, Decimal(0)])
        types = collections.defaultdict(lambda: [0, Decimal(0)])
        with (folder / "kernel_details.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["Type"], row["Name"], row["Accelerator Core"], row["Input Shapes"])
                duration = Decimal(row["Duration(us)"])
                groups[key][0] += 1
                groups[key][1] += duration
                types[row["Type"]][0] += 1
                types[row["Type"]][1] += duration
        response = json.loads((root / f"backend/{label}-backend-operators.json").read_text())
        assert response["result"]
        backend = response["body"]["data"]
        assert len(backend) == response["body"]["total"] == len(groups)
        largest_us_error = Decimal(0)
        for item in backend:
            row = item["compare"]
            key = (row["opType"], row["opName"], row["accCore"], row["inputShape"])
            count, duration = groups[key]
            assert int(row["count"]) == count, (label, key, count)
            error = abs(Decimal(row["totalTime"]) - duration)
            largest_us_error = max(largest_us_error, error)
            assert error <= Decimal("0.001"), (label, key, error)
        analysis = json.loads((root / f"analysis/{label}.json").read_text())
        for row in analysis["whole_profile_compute_top"]:
            count, duration = types[row["op_type"]]
            assert count == row["calls"], (label, row)
            assert abs(duration / 1000 - Decimal(str(row["total_ms"]))) <= Decimal("0.000001")
        results[label] = {"groups_checked": len(groups), "sql_types_checked": len(analysis["whole_profile_compute_top"]),
                          "csv_kernel_count": sum(v[0] for v in groups.values()), "max_total_time_error_us": str(largest_us_error),
                          "insight_csv_counts_and_totals_match": True, "sql_csv_counts_and_totals_match": True}
out = root / "analysis/backend-validation.json"
out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2))
