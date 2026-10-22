"""Read-only receipts after owned controllers exit; never list other cards."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def collect(root):
    process = subprocess.check_output(["npu-smi", "info", "-t", "proc-mem", "-i", "7"], text=True)
    port = subprocess.check_output(["ss", "-ltn", "sport = :8047"], text=True)
    now = datetime.now(timezone.utc)
    receipt = {"recorded_utc": now.isoformat(),
               "recorded_beijing": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
               "card7_released": "No process in device" in process,
               "port8047_closed": len(port.strip().splitlines()) == 1,
               "process_query": process, "port_query": port,
               "diagnostic_completed": "DIAGNOSTICS_COMPLETE" in (root / "controller.log").read_text(),
               "benchmark_completed": "ALL_COMPLETE" in (root / "benchmark-controller.log").read_text()}
    assert all(receipt[x] for x in ("card7_released", "port8047_closed", "diagnostic_completed", "benchmark_completed")), receipt
    (root / "cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n")
    sources = {"runtime_sources": [digest(p) for p in sorted(root.iterdir())
                                    if p.suffix in (".py", ".sh") and "private" not in p.name],
               "private_seed": digest(root / "seed-private.json"),
               "traces": [digest(p) for p in sorted((root / "results").glob("*-trace.pt"))]}
    (root / "source-manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    collect(Path(sys.argv[1]))
