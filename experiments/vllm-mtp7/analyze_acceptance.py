"""Recompute MTP prefix-survival and conditional acceptance; no NPU requests.

Usage: python analyze_acceptance.py REPO SOURCE_AUDIT_JSON OUTPUT_JSON
The K-prefix rows describe prefixes of the existing K7 rounds, not new K runs.
"""
import json
from pathlib import Path
import sys

repo, audit_file, output = map(Path,sys.argv[1:4])
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
full=read(repo/"evidence/2026-10-20/vllm-mtp7-gsm128-result.json")
four=read(repo/"evidence/2026-10-21/dflash2-vllm-graph.json")["acceptance"]["mtp7"]
def summarize(rounds,counts):
    n=int(rounds)
    vals=[int(counts[str(i)]) for i in range(7)]
    assert n>=vals[0] and all(a>=b for a,b in zip(vals,vals[1:]))
    reached=[n]+vals[:-1]
    histogram=[n-vals[0]]+[a-b for a,b in zip(vals,vals[1:])]+[vals[-1]]
    assert sum(histogram)==n and sum(i*v for i,v in enumerate(histogram))==sum(vals)
    return {"rounds":n,"accepted":sum(vals),"proposed":n*7,
      "candidate_acceptance":sum(vals)/(n*7),"counter_tokens_per_round":1+sum(vals)/n,
      "per_position":[{"position":i+1,"accepted_prefix_rounds":v,
        "prefix_survival":v/n,"conditional_acceptance_given_prior_accepted":v/reached[i]}
        for i,v in enumerate(vals)],
      "accepted_prefix_length_histogram_capped_at_7":histogram,
      "existing_round_prefixes":[{"prefix_length":k,
        "prefix_candidate_acceptance":sum(vals[:k])/(n*k),
        "counter_tokens_per_round":1+sum(vals[:k])/n} for k in range(1,8)]}
a=summarize(full["speculation"]["verification_rounds"],full["speculation"]["accepted_per_position"])
b=summarize(four["verification_rounds"],four["accepted_per_position"])
assert a['accepted']==full['speculation']['accepted_tokens'] and b['accepted']==four['accepted']
result={"source_audit":read(audit_file),"gsm128":a,"four_question":b,
 "limits":["Counters include EOS/tail conventions, not exact HTTP output per round.",
 "Conditional acceptance is on surviving prefixes; selection and tail effects remain.",
 "Prefix rows reuse K7 rounds: changing K changes round boundaries, so these are not counterfactual K benchmark results.",
 "Profiler/counters do not contain draft logits, entropy, top-2 margins, target ranks or per-round confidence.",
 "No confidence-policy improvement, training benefit or new speed has been measured."]}
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"gsm128":a,"source_audit":result['source_audit']},ensure_ascii=False,indent=2))
