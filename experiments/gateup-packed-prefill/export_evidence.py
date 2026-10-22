"""Export small filtered evidence; raw prompts, token arrays and states stay private."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--private-root",type=Path,required=True)
parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args()
raw=args.private_root
repo=args.repo_root
out=repo/"evidence/2026-10-21"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")


summary=load(raw/"results/summary.json")
write("packed-prefill-e2e.json",summary)
write("packed-prefill-validation.json",load(raw/"results/validation.json"))
filtered={}
for label in ("packed-control","prefill-direct"):
    filtered[label]=[]
    for row in load(raw/f"results/{label}.json"):
        filtered[label].append({**{k:row[k] for k in ("kind","repeat","question_id","e2e_s","ttft_s","tpot_ms","output_tok_s","usage")},
            "output_token_ids_sha256":hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest()})
write("packed-prefill-requests.json",filtered)
environment=load(raw/"results/environment.json")
write("packed-prefill-cleanup.json",{k:environment[k] for k in ("recorded_utc","card7_released","port_8047_closed")})
sources=[]
for folder,names in (("kernels/gateup-packed-prefill",("kernel.asc","bridge.cpp","build.py","validate.py")),
                     ("experiments/gateup-packed-prefill",("client.py","controller.py","model_extension.py","start.sh","graph_audit_base.py","serve.py","analyze.py"))):
    for name in names:
        archive_path=repo/folder/name
        sources.append({"artifact":name,"runtime_source_sha256":digest(raw/name),
                        "archived_path":archive_path.relative_to(repo).as_posix(),
                        "archived_sha256":digest(archive_path)})
write("packed-prefill-sources.json",{"sources":sources,"archive_sha256":load(raw/"manifest.json")["archive_sha256"],
    "summary_sha256":digest(raw/"results/summary.json"),
    "failures":[{"attempt":1,"reason":"Fixed 128-row source stride used for a shorter L0C result; failed correctness, excluded from timing."},
                {"attempt":2,"reason":"m=1 MMAD input interpretation did not match padded matrix storage; fixed by 16-row internal alignment and revalidated."}],
    "precision_boundary":"Actual short-request M65/83/92/114/121 are bit equal in the isolated tested cases; M1/128/257/512 are not universally bit equal. See per-case validation and full-model token checks."})
print("EXPORTED: 5 filtered evidence files; no raw prompts/token arrays, private paths or other-process state")
