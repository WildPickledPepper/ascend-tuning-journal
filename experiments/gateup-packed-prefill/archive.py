"""Archive bounded raw experiment evidence; no checkpoints or cache trees."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
root=args.root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True)
    return {"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}


state=command(["npu-smi","info"])
env={"recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "npu_smi_final":state,"port_check":command(["ss","-ltnp"])}
assert "No running processes found in NPU 7" in state["stdout"],"Do not archive as released while card 7 is in use"
env["card7_released"]=True
env["port_8047_closed"]=not any(":8047 " in line for line in env["port_check"]["stdout"].splitlines())
(root/"results/environment.json").write_text(json.dumps(env,indent=2)+"\n")
files=[p for p in root.iterdir() if p.is_file() and p.suffix in {".py",".asc",".cpp",".sh",".log",".md",".so"}]
files += [p for p in (root/"results").iterdir() if p.is_file()]
entries=[]
with tarfile.open(root/"evidence.tar.gz","w:gz") as archive:
    for p in sorted(files):
        name=p.relative_to(root).as_posix()
        entries.append({"path":name,"bytes":p.stat().st_size,"sha256":digest(p)})
        archive.add(p,arcname=name)
manifest={"files":entries,"archive_sha256":digest(root/"evidence.tar.gz")}
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"files":len(entries),"archive_bytes":(root/"evidence.tar.gz").stat().st_size,
                  "archive_sha256":manifest["archive_sha256"]}))
