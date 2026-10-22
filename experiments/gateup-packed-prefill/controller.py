"""One packed-fallback control and one direct-prefill candidate on card 7."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests
from client import benchmark, URL

ROOT = Path(__file__).resolve().parent
ACTIVE = None


def stop():
    global ACTIVE
    if ACTIVE is None:
        return
    try:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        ACTIVE.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(ACTIVE.pid, signal.SIGKILL)
        ACTIVE.wait(timeout=15)
    # The parent can exit before workers; only signal our same process group.
    try:
        os.killpg(ACTIVE.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    ACTIVE = None
    time.sleep(8)


def interrupted(sig, frame):
    raise SystemExit(128 + sig)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    requests.get(URL + "/health", timeout=2)
except requests.ConnectionError:
    pass
else:
    raise RuntimeError("Experiment port already occupied; refusing to replace service")

(ROOT / "controller.pid").write_text(str(os.getpid()))
phases = [("packed-control", "0"), ("prefill-direct", "1")]
if len(sys.argv) > 1:
    phases = [(sys.argv[1], sys.argv[2])]
try:
    for label, enable in phases:
        log = open(ROOT / "results" / f"{label}-server.log", "w")
        started = time.time()
        ACTIVE = subprocess.Popen(["bash", str(ROOT / "start.sh"), enable], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / "results" / f"{label}-pid.json").write_text(json.dumps({"pid": ACTIVE.pid, "started_unix": started, "device": 7}))
        print("SERVER_START", label, ACTIVE.pid, flush=True)
        while True:
            if ACTIVE.poll() is not None:
                raise RuntimeError(f"{label} exited: {ACTIVE.returncode}; see server log")
            try:
                ready = requests.get(URL + "/health", timeout=2).status_code == 200
            except requests.RequestException:
                ready = False
            if ready:
                break
            if time.time() - started > 1500:
                raise TimeoutError(label)
            time.sleep(5)
        print("SERVER_READY", label, time.time() - started, flush=True)
        benchmark(label)
        stop()
        log.close()
    print("ALL_COMPLETE", flush=True)
finally:
    stop()
