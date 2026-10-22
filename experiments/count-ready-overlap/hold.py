"""Own one guarded card-6 service while bounded offline-driven phases run."""
from datetime import datetime,timezone
import json,os,signal,subprocess,time
from pathlib import Path
import requests
ROOT=Path(__file__).parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
active=None
def now():return datetime.now(timezone.utc).isoformat()
def save(name,data):(OUT/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
def interrupt(sig,frame):raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
try:
    proc=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i','6','-c','0'],text=True)
    assert 'No process in device' in proc,proc
    assert len(subprocess.check_output(['ss','-ltn','sport = :8047'],text=True).strip().splitlines())==1
    save('started',{'utc':now(),'physical_card':6,'no_prior_process':True})
    log=open(OUT/'server.log','w')
    active=subprocess.Popen(['bash',str(ROOT/'start.sh'),'1'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save('service',{'pid':active.pid,'utc':now()})
    start=time.monotonic()
    while True:
        if active.poll() is not None:raise RuntimeError('Service exited')
        try:
            if requests.get('http://127.0.0.1:8047/health',timeout=2).status_code==200:break
        except requests.RequestException:pass
        if time.monotonic()-start>1500:raise TimeoutError('startup')
        time.sleep(4)
    save('ready',{'utc':now(),'startup_s':time.monotonic()-start});print('READY',now(),flush=True)
    start=time.monotonic()
    while not (ROOT/'STOP').exists():
        if active.poll() is not None:raise RuntimeError('Service exited')
        if time.monotonic()-start>3600:raise TimeoutError('bounded session lifetime')
        time.sleep(2)
finally:
    if active is not None:
        try:os.killpg(active.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:active.wait(timeout=45)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(active.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        active.wait(timeout=20);log.close();time.sleep(6)
        save('stopped',{'utc':now(),'own_service_stopped':True});print('STOPPED',now(),flush=True)
