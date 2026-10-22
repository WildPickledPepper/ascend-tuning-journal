"""Read-only completion, card-7 release and unchanged-serving-source receipts."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def collect(root, serving):
    process = subprocess.check_output(["npu-smi", "info", "-t", "proc-mem", "-i", "7"], text=True)
    port = subprocess.check_output(["ss", "-ltn", "sport = :8047"], text=True)
    now = datetime.now(timezone.utc)
    log = (root / "benchmark-controller.log").read_text()
    receipt = {"recorded_utc": now.isoformat(),
               "recorded_beijing": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
               "card7_released": "No process in device" in process,
               "port8047_closed": len(port.strip().splitlines()) == 1,
               "process_query": process, "port_query": port,
               "benchmark_completed": "ALL_COMPLETE" in log,
               "full_completed": "BENCHMARK_COMPLETE full-vocab" in log,
               "dynamic_completed": "BENCHMARK_COMPLETE dynamic4096" in log}
    assert all(receipt[k] for k in ("card7_released", "port8047_closed", "benchmark_completed", "full_completed", "dynamic_completed"))
    (root / "cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n")
    sources = {"benchmark_sources": [digest(p) for p in sorted(root.glob("*.py"))],
               "serving_sources": [digest(serving / name) for name in (
                   "start.sh", "model_extension.py", "dynamic_vocab.py", "graph_audit_base.py", "serve.py")],
               "private_seed": digest(root / "seed-private.json"),
               "serving_entry": "../gdn-fp32-state/start.sh; unchanged source and separate existing full/dynamic caches",
               "discarded_launch_attempt": "Windows CRLF prevented bash launch before serving or timing; normalized to LF and restarted."}
    (root / "source-manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    collect(Path(sys.argv[1]), Path(sys.argv[2]))
