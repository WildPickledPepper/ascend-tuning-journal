"""Read-only checks of the filtered prefill experiment, without NPU imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics

repo=Path(__file__).resolve().parents[2]
evidence=repo/"evidence/2026-10-21"
load=lambda name:json.loads((evidence/name).read_text(encoding="utf-8"))
data=load("packed-prefill-requests.json")
summary=load("packed-prefill-e2e.json")
assert summary["all_tokens_equal"]
hashes={}
for label,rows in data.items():
    assert len(rows)==11
    for row in rows:
        hashes.setdefault((row["kind"],row["question_id"]),set()).add(row["output_token_ids_sha256"])
    for kind in ("fixed512","natural","long-prefill"):
        selected=[r for r in rows if r["kind"]==kind]
        elapsed=sum(r["e2e_s"] for r in selected)
        tokens=sum(r["usage"]["completion_tokens"] for r in selected)
        expected=summary["comparison"][kind]["candidate" if label=="prefill-direct" else "control"]
        values={"requests":len(selected),"output_tokens":tokens,"total_e2e_s":elapsed,
                "mean_e2e_s":elapsed/len(selected),"output_tok_s":tokens/elapsed,
                "mean_ttft_s":statistics.mean(r["ttft_s"] for r in selected),
                "weighted_tpot_ms":1000*sum(r["e2e_s"]-r["ttft_s"] for r in selected)/(tokens-len(selected))}
        for key,value in values.items():
            assert math.isclose(value,expected[key],rel_tol=1e-12,abs_tol=1e-12),(label,kind,key)
assert all(len(value)==1 for value in hashes.values())
for comp in summary["comparison"].values():
    a,b=comp["control"],comp["candidate"]
    assert math.isclose(100*(b["output_tok_s"]/a["output_tok_s"]-1),comp["throughput_increase_pct"],abs_tol=1e-12)
    assert math.isclose(1000*(a["mean_ttft_s"]-b["mean_ttft_s"]),comp["ttft_saved_ms"],abs_tol=1e-12)
    assert comp["tokens_equal"]
control,candidate=(summary["counter_deltas"][name] for name in ("packed-control","prefill-direct"))
assert control["gate_unpack_fallback_host_calls"]==832
assert candidate["prefill_direct_host_calls"]==832 and candidate.get("gate_unpack_fallback_host_calls",0)==0
assert candidate["packed_M512"]==128 and candidate["packed_M489"]==64
audit=summary["audits"]["prefill-direct"]
checks=audit["layout_experiment"]["prefill_real_checks"]
assert len(checks)==64 and all(v["different"]==0 and v["finite"] for v in checks.values())
assert audit["layout_experiment"]["packed_count"]==64
assert audit["layout_experiment"]["prefill_direct_enabled"]
for draft in (False,True):
    assert any(w["is_draft_runnable"]==draft and any(e["captured"] for e in w["entries"]) for w in audit["wrappers"])
for item in load("packed-prefill-sources.json")["sources"]:
    assert hashlib.sha256((repo/item["archived_path"]).read_bytes()).hexdigest()==item["archived_sha256"],item["archived_path"]
validation=load("packed-prefill-validation.json")
assert len(validation["cases"])==20 and not validation["all_bit_equal"]
assert all(r["different"]==0 for r in validation["cases"] if r["M"] in (17,65,83,92,114,121))
cleanup=load("packed-prefill-cleanup.json")
assert cleanup["card7_released"] and cleanup["port_8047_closed"]
print("PASS: 22 request statistics/hashes; 64 real-activation checks; 832 direct calls/no timed unpack; graphs; source hashes; precision boundary; card-7 cleanup")
