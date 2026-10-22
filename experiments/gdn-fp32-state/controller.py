"""One startup, same-process arm isolation, only the divergent third question."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests
from client import ROOT, URL, DATA, request

ACTIVE = None


def rpc(method, *args):
    response = requests.post(URL + "/collective_rpc",json={"method":method,"args":list(args)},timeout=180)
    response.raise_for_status()
    return response.json()


def stop():
    global ACTIVE
    if ACTIVE is None:
        return
    try:
        os.killpg(ACTIVE.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        ACTIVE.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(ACTIVE.pid,signal.SIGKILL)
        ACTIVE.wait(timeout=15)
    try:
        os.killpg(ACTIVE.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    ACTIVE = None


def interrupted(sig, frame):
    raise SystemExit(128+sig)


signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
(ROOT/"controller.pid").write_text(str(os.getpid()))
try:
    try:
        requests.get(URL+"/health",timeout=2)
    except requests.ConnectionError:
        pass
    else:
        raise RuntimeError("Port occupied; refusing to replace service")
    with open(ROOT/"results/server.log","w") as log:
        env = dict(os.environ, STATE_DIAGNOSTIC_ENABLE="1")
        ACTIVE = subprocess.Popen(["bash",str(ROOT/"start.sh"),"0"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        print("SERVER_START",ACTIVE.pid,flush=True)
        started = time.monotonic()
        while True:
            if ACTIVE.poll() is not None:
                raise RuntimeError("Server exited; see server.log")
            try:
                if requests.get(URL+"/health",timeout=2).status_code==200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic()-started>1500:
                raise TimeoutError("startup")
            time.sleep(5)
        print("READY",time.monotonic()-started,flush=True)
        request("What is 1 + 1?",64,True)
        audit = rpc("graph_audit")
        (ROOT/"results/state-audit.json").write_text(json.dumps(audit,indent=2))
        assert audit["results"][0]["gdn_state"]["all_ssm_fp32"]
        question = json.loads(DATA.read_text().splitlines()[2])["question"]
        for mode,label in [("full","full-a"),("masked","masked"),("full","full-b")]:
            audit = rpc("diag_select",mode,label)
            (ROOT/"results"/(label+"-audit.json")).write_text(json.dumps(audit,indent=2))
            row = request(question,640,False)
            (ROOT/"results"/(label+"-request.json")).write_text(json.dumps(row))
            print("REQUEST",label,len(row["token_ids"]),flush=True)
            print("TRACE",rpc("diag_save"),flush=True)
        print("DIAGNOSTICS_COMPLETE",flush=True)
finally:
    stop()
    print("SERVER_STOPPED",flush=True)
