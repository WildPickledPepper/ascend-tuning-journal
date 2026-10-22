"""Compare a single matched control and candidate from private request logs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
parser.add_argument("--output",type=Path)
args=parser.parse_args()
root=args.root
labels=("packed-control","prefill-direct")
data={label:json.loads((root/f"results/{label}.json").read_text()) for label in labels}


def stats(rows):
    elapsed=sum(r["e2e_s"] for r in rows)
    tokens=sum(r["usage"]["completion_tokens"] for r in rows)
    return {"requests":len(rows),"output_tokens":tokens,
            "input_tokens":[r["usage"]["prompt_tokens"] for r in rows],
            "total_e2e_s":elapsed,"mean_e2e_s":elapsed/len(rows),
            "mean_ttft_s":statistics.mean(r["ttft_s"] for r in rows),
            "output_tok_s":tokens/elapsed,
            "weighted_tpot_ms":1000*sum(r["e2e_s"]-r["ttft_s"] for r in rows)/(tokens-len(rows))}


out={"comparison":{},"all_tokens_equal":True,"token_checks":[],"audits":{},"counter_deltas":{}}
for kind in ("fixed512","natural","long-prefill"):
    grouped={label:[r for r in rows if r["kind"]==kind] for label,rows in data.items()}
    control,candidate=(stats(grouped[label]) for label in labels)
    out["comparison"][kind]={"control":control,"candidate":candidate,
        "e2e_reduction_pct":100*(1-candidate["mean_e2e_s"]/control["mean_e2e_s"]),
        "ttft_reduction_pct":100*(1-candidate["mean_ttft_s"]/control["mean_ttft_s"]),
        "ttft_saved_ms":1000*(control["mean_ttft_s"]-candidate["mean_ttft_s"]),
        "throughput_increase_pct":100*(candidate["output_tok_s"]/control["output_tok_s"]-1)}
    kind_equal=True
    for q in sorted({r["question_id"] for r in grouped[labels[0]]}):
        matching=[(label,r) for label,rows in grouped.items() for r in rows if r["question_id"]==q]
        expected=matching[0][1]["token_ids"]
        same=all(r["token_ids"]==expected for _,r in matching)
        out["all_tokens_equal"] &= same
        kind_equal &= same
        out["token_checks"].append({"kind":kind,"question_id":q,"all_equal":same,
            "cases":[{"label":label,"repeat":r["repeat"],"tokens":len(r["token_ids"]),
                      "sha256":hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()}
                     for label,r in matching]})
    out["comparison"][kind]["tokens_equal"]=kind_equal
for label in labels:
    before=json.loads((root/f"results/{label}-before-audit.json").read_text())["results"][0]
    after=json.loads((root/f"results/{label}-after-audit.json").read_text())["results"][0]
    out["audits"][label]=after
    bc,ac=(value["layout_experiment"]["counts"] for value in (before,after))
    out["counter_deltas"][label]={k:ac.get(k,0)-bc.get(k,0) for k in ac.keys()|bc.keys()}
    values={}
    for phase,sign in (("before",-1),("after",1)):
        for line in (root/f"results/{label}-{phase}-metrics.txt").read_text().splitlines():
            if line.startswith("vllm:spec_decode") and "_total{" in line:
                key,value=line.rsplit(" ",1)
                values[key]=values.get(key,0)+sign*float(value)
    out.setdefault("speculative_counter_deltas",{})[label]=values
out["speculative_counters_equal"]=out["speculative_counter_deltas"][labels[0]]==out["speculative_counter_deltas"][labels[1]]
path=args.output or root/"results/summary.json"
path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({k:v for k,v in out.items() if k not in {"audits","token_checks","speculative_counter_deltas"}},indent=2))
