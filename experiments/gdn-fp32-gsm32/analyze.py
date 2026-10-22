"""Offline reconstruction of measured throughput and acceptance counters."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def counters(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r'(vllm:spec_decode_\w+_total)(\{[^}]*\})?\s+(\S+)', line)
        if match:
            key = match[1] + (match[2] or "")
            result[key] = float(match[3])
    return result


def acceptance(root, label, kind):
    before = counters(root / f"{label}-{kind}-before-metrics.txt")
    after = counters(root / f"{label}-{kind}-after-metrics.txt")
    delta = {k:after[k] - before.get(k, 0) for k in after}
    def get(name):
        return sum(v for k,v in delta.items() if k.split("{")[0] == "vllm:" + name)
    rounds = get("spec_decode_num_drafts_total")
    drafted = get("spec_decode_num_draft_tokens_total")
    accepted = get("spec_decode_num_accepted_tokens_total")
    assert rounds > 0 and drafted == 7 * rounds
    assert 0 <= accepted <= drafted
    per_position = [v for k,v in delta.items() if "num_accepted_tokens_per_pos_total" in k]
    assert len(per_position) == 7 and sum(per_position) == accepted
    return {"rounds": rounds, "drafted": drafted, "accepted": accepted,
            "accepted_per_round": accepted / rounds if rounds else None,
            "acceptance_fraction": accepted / drafted if drafted else None,
            "one_plus_accepted_per_round": 1 + accepted / rounds if rounds else None,
            "raw_counter_deltas": delta}


def summarize(root, label):
    rows = json.loads((root / f"{label}.json").read_text())
    result = {}
    for kind in sorted({x["kind"] for x in rows}):
        group = [x for x in rows if x["kind"] == kind]
        output = sum(x["usage"]["completion_tokens"] for x in group)
        e2e = sum(x["e2e_s"] for x in group)
        acc = acceptance(root, label, kind)
        result[kind] = {
            "requests": len(group), "input_tokens": [x["usage"]["prompt_tokens"] for x in group],
            "output_tokens": output, "e2e_total_s": e2e,
            "e2e_mean_s": e2e / len(group), "output_tok_s": output / e2e,
            "ttft_mean_ms": statistics.mean(x["ttft_s"] for x in group) * 1000,
            "weighted_tpot_ms": sum(x["e2e_s"] - x["ttft_s"] for x in group) * 1000 / (output - len(group)),
            "acceptance": acc,
            "client_decode_ms_per_draft_round":
                sum(x["e2e_s"] - x["ttft_s"] for x in group) * 1000 / acc["rounds"],
        }
    return result, rows


def main(root, candidate):
    base, br = summarize(root, "full-vocab")
    cand, cr = summarize(root, candidate)
    assert len(br) == len(cr)
    pairs = []
    for b,c in zip(br,cr):
        assert (b["kind"], b["question_id"], b["repeat"]) == (c["kind"],c["question_id"],c["repeat"])
        assert b["request"] == c["request"]
        pairs.append({"kind":b["kind"], "question_id":b["question_id"], "repeat":b["repeat"],
                      "baseline_output":len(b["token_ids"]), "candidate_output":len(c["token_ids"]),
                      "tokens_equal": b["token_ids"] == c["token_ids"],
                      "first_different_token_index":next((i for i,(u,v) in enumerate(zip(b["token_ids"],c["token_ids"])) if u != v),None),
                      "baseline_ids_sha256":hashlib.sha256(json.dumps(b["token_ids"]).encode()).hexdigest(),
                      "candidate_ids_sha256":hashlib.sha256(json.dumps(c["token_ids"]).encode()).hexdigest()})
    comparison = {}
    for kind,b in base.items():
        c = cand[kind]
        comparison[kind] = {"throughput_change_pct": (c["output_tok_s"] / b["output_tok_s"] - 1) * 100,
                            "e2e_change_pct": (c["e2e_total_s"] / b["e2e_total_s"] - 1) * 100,
                            "accepted_per_round_change_pct":
                                (c["acceptance"]["accepted_per_round"] / b["acceptance"]["accepted_per_round"] - 1) * 100}
    output = {"baseline":base, "candidate":cand, "comparison":comparison,
              "paired_output_check":pairs, "all_output_sequences_equal":all(p["tokens_equal"] for p in pairs),
              "timing": "HTTP streaming E2E; output tokens divided by sum of request E2E; no profiler",
              "acceptance_note": "accepted_per_round excludes one target correction/bonus token; counters include terminal wasted draft work"}
    matched = [(b,c) for b,c in zip(br,cr) if b["kind"] == "natural" and b["token_ids"] == c["token_ids"]]
    if matched:
        tokens = sum(len(b["token_ids"]) for b,c in matched)
        bt = sum(b["e2e_s"] for b,c in matched)
        ct = sum(c["e2e_s"] for b,c in matched)
        output["matched_natural_subset"] = {"requests":len(matched), "question_ids":[b["question_id"] for b,c in matched],
                                            "output_tokens":tokens, "baseline_e2e_s":bt, "candidate_e2e_s":ct,
                                            "baseline_output_tok_s":tokens/bt, "candidate_output_tok_s":tokens/ct,
                                            "throughput_change_pct":(bt/ct-1)*100,
                                            "note":"Post-hoc subset selected by exact token equality; not the full benchmark score"}
    (root / f"{candidate}-summary.json").write_text(json.dumps(output,indent=2))
    print(json.dumps({k:output[k] for k in ("baseline","candidate","comparison","all_output_sequences_equal")}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "dynamic4096")
