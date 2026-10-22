"""Single owned card-6 service: correctness, same-card A/B and Pipe profiles."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import threading
import requests
from client import ROOT,URL,DATA,request,snapshot

ACTIVE=None
MONITOR_STOP=threading.Event()
RESULTS=ROOT/'results'
RESULTS.mkdir(exist_ok=True)


def now():return datetime.now(timezone.utc).isoformat()
def save(name,obj):(RESULTS/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
def rpc(method,*args):
    r=requests.post(URL+'/collective_rpc',json={'method':method,'args':list(args)},timeout=180)
    r.raise_for_status();return r.json()
def mode(enabled,record=False):
    result=rpc('configure_overlap',enabled,record)
    print('MODE',enabled,record,result,flush=True);return result
def audit(label):
    result=rpc('overlap_audit');save(label,result);return result['results'][0]
def metrics(label):
    r=requests.get(URL+'/metrics',timeout=30);r.raise_for_status();(RESULTS/(label+'.txt')).write_text(r.text)
def interrupted(sig,frame):raise SystemExit(128+sig)
def monitor():
    with (RESULTS/'device6-memory-observations.jsonl').open('w') as f:
        while not MONITOR_STOP.is_set():
            proc=subprocess.run(['npu-smi','info','-t','proc-mem','-i','6','-c','0'],capture_output=True,text=True)
            usage=subprocess.run(['npu-smi','info','-t','usages','-i','6','-c','0'],capture_output=True,text=True)
            f.write(json.dumps({'utc':now(),'proc':proc.stdout,'usage':usage.stdout})+'\n');f.flush()
            MONITOR_STOP.wait(3)
signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)


try:
    response=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i','6','-c','0'],text=True)
    assert 'No process in device' in response,response
    port=subprocess.check_output(['ss','-ltn','sport = :8047'],text=True)
    assert len(port.strip().splitlines())==1,port
    save('started',{'utc':now(),'physical_card':6,'no_prior_device_process':True,'port_free':True})
    threading.Thread(target=monitor,daemon=True).start()
    log=open(RESULTS/'server.log','w')
    ACTIVE=subprocess.Popen(['bash',str(ROOT/'start.sh'),'1'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save('service',{'pid':ACTIVE.pid,'utc':now()})
    started=time.monotonic()
    while True:
        if ACTIVE.poll() is not None:raise RuntimeError('Service exited; inspect private log')
        try:
            if requests.get(URL+'/health',timeout=2).status_code==200:break
        except requests.RequestException:pass
        if time.monotonic()-started>1500:raise TimeoutError('startup')
        time.sleep(5)
    print('READY',now(),time.monotonic()-started,flush=True)
    questions=[json.loads(line)['question'] for line in DATA.read_text().splitlines()[:32]]
    prior=json.loads(Path(os.environ['REFERENCE_RESULTS']).read_text())
    refs={r['question_id']:r for r in prior}
    assert len(prior)==32
    save('warmup65',request('What is 1 + 1?',64,False))
    save('warmup-natural',request(questions[0],2048,False))
    snapshot('startup')
    # Record exactly the selected global IDs per target round, only in these
    # untimed correctness requests. Main output and candidate set must match.
    checks={}
    for qid in (0,2):
        pair=[]
        for enabled,label in ((False,'baseline'),(True,'early')):
            mode(enabled,True)
            row=request(questions[qid],2048,False);save(f'check-{qid}-{label}',row)
            item=audit(f'check-{qid}-{label}-audit')
            pair.append((row,item))
            print('CHECK',qid,label,len(row['token_ids']),flush=True)
        same_output=pair[0][0]['token_ids']==pair[1][0]['token_ids']==refs[qid]['token_ids']
        same_candidates=pair[0][1]['candidate_set_sequence_sha256']==pair[1][1]['candidate_set_sequence_sha256']
        checks[str(qid)]={'outputs_identical':same_output,'same_round_candidate_sequences_identical':same_candidates,
                          'output_tokens':len(pair[0][0]['token_ids']),
                          'target_rounds_including_prefill':len(pair[0][1]['candidate_set_sequence_sha256'])}
        save('correctness',checks)
        assert same_output and same_candidates,checks
    # One same-card control is needed: historical card-7 throughput is not the
    # primary comparison for a card-6 scheduling change.
    for enabled,label in ((False,'baseline'),(True,'early')):
        mode(enabled,False)
        request(questions[0],2048,False)
        snapshot(label+'-before');metrics(label+'-before-metrics')
        records=[];start=time.perf_counter()
        for qid,question in enumerate(questions):
            row=request(question,2048,False)
            row.update(question_id=qid,kind='natural')
            assert row['token_ids']==refs[qid]['token_ids'],(label,qid,'output mismatch')
            records.append(row);save(label,records)
            print('REQUEST',label,qid,len(row['token_ids']),row['e2e_s'],row['ttft_s'],flush=True)
        save(label+'-window',{'wall_s':time.perf_counter()-start})
        snapshot(label+'-after');metrics(label+'-after-metrics');audit(label+'-audit')
        print('BENCHMARK_COMPLETE',label,flush=True)
    # Instrumented latency is separate from the unprofiled benchmark above.
    for enabled,label in ((False,'baseline'),(True,'early')):
        mode(enabled,False)
        configured=rpc('configure_decode_profile','PipeUtilization')
        previous=set((ROOT/'profiles').glob('*_ascend_pt'))
        r=requests.post(URL+'/start_profile',json={},timeout=300);r.raise_for_status()
        try:row=request(questions[0],128,False)
        finally:
            r=requests.post(URL+'/stop_profile',json={},timeout=900);r.raise_for_status()
        folders=set((ROOT/'profiles').glob('*_ascend_pt'))-previous
        assert len(folders)==1
        folder=next(iter(folders));assert row['token_ids']==refs[0]['token_ids'][:128]
        save(label+'-profile',{'profile':str(folder.relative_to(ROOT)),'request':row,'configured':configured,
                              'utc':now(),'output_prefix_identical':True})
        print('PROFILE_COMPLETE',label,folder.name,flush=True)
    snapshot('final');save('complete',{'utc':now(),'success':True})
    print('ALL_COMPLETE',now(),flush=True)
finally:
    MONITOR_STOP.set()
    if ACTIVE is not None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(timeout=45)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(ACTIVE.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        ACTIVE.wait(timeout=20)
        log.close();time.sleep(6)
        save('stopped',{'utc':now(),'own_service_stopped':True})
        print('SERVICE_STOPPED',now(),flush=True)
