"""Filter private first-round telemetry into timings, without request content."""
import json
from pathlib import Path
import sys


def main(source, output):
    trials = json.loads(source.read_text())
    result = []
    for t in trials:
        audit = t["audit"]
        gc_pairs, stack = [], {}
        for e in audit["gc_events"]:
            gen = e["generation"]
            if e["phase"] == "start":
                stack[gen] = e["wall_ns"]
            elif gen in stack:
                gc_pairs.append((stack.pop(gen), e["wall_ns"], gen))
        scopes = []
        for e in sorted(audit["events"], key=lambda x: x["start_ns"]):
            wall = (e["end_ns"] - e["start_ns"]) / 1e6
            gc = sum(max(0, min(b, e["end_ns"]) - max(a, e["start_ns"])) / 1e6 for a, b, _ in gc_pairs)
            scopes.append({"name": e["name"], "wall_ms": wall, "cpu_ms": e["cpu_ms"],
                           "wall_minus_cpu_ms": wall - e["cpu_ms"], "gc_overlap_ms": gc})
        row = {"repeat": t["repeat"], "mode": t["mode"],
               "question_id": t["request"]["question_id"], "ttft_ms": t["request"]["ttft_s"] * 1000,
               "e2e_s": t["request"]["e2e_s"], "first_round_scopes": scopes,
               "device_event_ms": audit["execute_device_event_ms"],
               "gc_pause_max_ms": max(((b - a) / 1e6 for a, b, _ in gc_pairs), default=0)}
        result.append(row)
        print(row["repeat"], row["question_id"], row["mode"], "TTFT", round(row["ttft_ms"], 3),
              "GCmax", round(row["gc_pause_max_ms"], 3),
              [(s["name"], round(s["wall_ms"], 2), round(s["cpu_ms"], 2), round(s["gc_overlap_ms"], 2))
               for s in scopes if s["name"] in ("execute_model", "_prepare_inputs", "sample_tokens", "propose_draft_token_ids")])
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
