"""Capture optimized model only, physical card 7; always stop our own service."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests
from client import request, snapshot, URL, DATA

ROOT = Path(__file__).resolve().parent
ACTIVE = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    (ROOT / "results" / (name + ".json")).write_text(json.dumps(value, indent=2))


def interrupt(sig, frame):
    raise SystemExit(128 + sig)


def rpc(method, args=()):
    response = requests.post(URL + "/collective_rpc", json={"method": method, "args": list(args)}, timeout=120)
    response.raise_for_status()
    return response.json()


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupt)
try:
    requests.get(URL + "/health", timeout=2)
except requests.ConnectionError:
    pass
else:
    raise RuntimeError("Port occupied; refusing to replace service")

(ROOT / "results").mkdir(exist_ok=True)
save("started", {"utc": now(), "physical_device": 7, "optimized_only": True})
log = open(ROOT / "results/server.log", "w")
try:
    ACTIVE = subprocess.Popen(["bash", str(ROOT / "start.sh"), "1"],
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save("service", {"pid": ACTIVE.pid, "utc": now()})
    start = time.monotonic()
    while True:
        if ACTIVE.poll() is not None:
            raise RuntimeError("Server exited; inspect server.log")
        try:
            ready = requests.get(URL + "/health", timeout=2).status_code == 200
        except requests.RequestException:
            ready = False
        if ready:
            break
        if time.monotonic() - start > 1500:
            raise TimeoutError("server startup")
        time.sleep(4)
    print("READY", now(), time.monotonic() - start, flush=True)
    question = json.loads(DATA.read_text().splitlines()[0])["question"]
    for label, prompt, count in [("warmup65", "What is 1 + 1?", 64),
                                 ("warmup512", question, 512),
                                 ("unprofiled-reference", question, 128)]:
        row = request(prompt, count, True)
        save(label, row)
        print("REQUEST", label, row["e2e_s"], row["usage"], flush=True)
    reference = row["token_ids"]
    snapshot("before")
    for metric in ("PipeUtilization", "L2Cache", "Memory"):
        configured = rpc("configure_decode_profile", (metric,))
        previous = set((ROOT / "profiles").glob("*_ascend_pt"))
        started = now()
        response = requests.post(URL + "/start_profile", json={}, timeout=900)
        response.raise_for_status()
        try:
            row = request(question, 128, True)
        finally:
            response = requests.post(URL + "/stop_profile", json={}, timeout=900)
            response.raise_for_status()
        created = set((ROOT / "profiles").glob("*_ascend_pt")) - previous
        assert len(created) == 1, created
        folder = next(iter(created))
        deadline = time.monotonic() + 300
        while not (folder / "ASCEND_PROFILER_OUTPUT/kernel_details.csv").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("profiler export")
            time.sleep(2)
        record = {"metric": metric, "configured": configured,
                  "started_utc": started, "completed_utc": now(),
                  "profile": str(folder.relative_to(ROOT)),
                  "token_ids_equal_to_unprofiled": row["token_ids"] == reference,
                  "sequence_sha256": hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest(),
                  "request": row}
        save(metric, record)
        assert record["token_ids_equal_to_unprofiled"]
        print("PROFILE_COMPLETE", metric, folder.name, flush=True)
    snapshot("after")
    save("complete", {"utc": now(), "success": True})
finally:
    if ACTIVE is not None:
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            ACTIVE.wait(timeout=45)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        ACTIVE.wait(timeout=20)
    log.close()
    time.sleep(6)
    save("stopped", {"utc": now(), "own_service_stopped": True})
    print("SERVICE_STOPPED", now(), flush=True)
