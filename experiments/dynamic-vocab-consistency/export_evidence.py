"""Export compact consistency evidence; leave prompts, IDs, pointers and traces private."""
import hashlib
import json
from pathlib import Path
import sys


def export(root, destination):
    result = {"trace_analysis": json.loads((root / "trace-analysis.json").read_text()),
              "state_segmentation": json.loads((root / "state-segmentation.json").read_text())}
    audits = [json.loads((root / (label + "-audit.json")).read_text())["results"][0]
              for label in ("full-a", "masked", "full-b")]
    result["same_process_controls"] = {
        "same_" + key: len({item[key] for item in audits}) == 1
        for key in ("main_head_ptr", "draft_head_ptr", "mask_ptr", "draft_graph_wrapper_id")
    }
    result["same_process_controls"].update(
        main_head_shapes=[x["main_head_shape"] for x in audits],
        main_head_versions=[x["main_head_version"] for x in audits])
    for key, value in result["same_process_controls"].items():
        if key.startswith("same_"):
            assert value, key
    sources = []
    for path in sorted(root.glob("*.json")):
        sources.append({"name": path.name, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result["private_summary_sources"] = sources
    result["scope"] = "Single process full-head mask diagnostic, not a speed benchmark. Operator FP32 is not a full-model fix test."
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
