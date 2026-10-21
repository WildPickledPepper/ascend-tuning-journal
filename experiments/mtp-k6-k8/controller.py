"""Run exactly one native-MTP four-question candidate and release its processes.

Usage: python controller.py RUN_FOLDER K
RUN_FOLDER contains model-root/runtime/client-runtime/upstream/data symlinks.
"""
import datetime,json,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path

run=Path(sys.argv[1]).resolve();k=int(sys.argv[2]);assert k in (6,8)
base=(run/'model-root').resolve()
state={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'phase':'preflight','k':k,'evaluation_examples':4,'restore_services':False,
       'native_checkpoint':True,'baseline_reused':True}
def save(**kw):
    state.update(kw);(run/'state.json').write_text(json.dumps(state,indent=2))
def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8040/health',timeout=2) as response:return response.status==200
    except OSError:return False
def procs():
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            s=(p/'stat').read_text().rsplit(')',1)[1].split()
            result[int(p.name)]=(int(s[1]),s[19],s[0])
        except OSError:pass
    return result
def cleanup(proc):
    table=procs();picked={proc.pid}
    while True:
        new={pid for pid,(parent,_,_) in table.items() if parent in picked}
        if new<=picked:break
        picked|=new
    snapshot={pid:table[pid][1] for pid in picked if pid in table}
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    for attempt in range(35):
        current=procs();live=[pid for pid,start in snapshot.items() if pid in current and current[pid][1]==start and current[pid][2]!='Z']
        if not live:break
        if attempt in (10,25):
            for pid in live:
                try:os.kill(pid,signal.SIGTERM if attempt==10 else signal.SIGKILL)
                except ProcessLookupError:pass
        time.sleep(1)
    proc.wait(timeout=15)
    current=procs()
    assert not [pid for pid,start in snapshot.items() if pid in current and current[pid][1]==start and current[pid][2]!='Z']

server=None
try:
    assert 'No process in device' in subprocess.check_output(['npu-smi','info','-t','proc-mem','-i','6'],text=True)
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8040))!=0,'Port occupied'
    lib=base.parent/'qwen35-9b/runtime/python/python3.12.13/lib'
    env=dict(os.environ,LD_LIBRARY_PATH=str(lib),QWEN_MODEL_ROOT=str(base),DFLASH_RUN=str(run),MTP_K=str(k))
    with (run/'server-private.log').open('w') as log:
        server=subprocess.Popen(['bash',str(run/'start.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(phase='starting',server_pid=server.pid)
    deadline=time.monotonic()+1500
    while not health():
        assert server.poll() is None,'Server exited'
        if time.monotonic()>deadline:raise TimeoutError('Server startup')
        time.sleep(3)
    save(phase='evaluating_4',ready_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (run/'card6-ready-private.txt').write_text(subprocess.check_output(['npu-smi','info','-t','proc-mem','-i','6'],text=True))
    subprocess.run([str(run/'client-runtime/venv/bin/python'),'-u',str(run/'client.py'),str(run),f'k{k}'],
                   env=dict(env,TORCH_DEVICE_BACKEND_AUTOLOAD='0'),check=True,timeout=600)
    save(phase='measured')
except BaseException as error:
    save(phase='failed',error=type(error).__name__+': '+str(error));raise
finally:
    if server is not None:cleanup(server)
    cards={}
    for card in (6,7):
        text=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',str(card)],text=True)
        cards[str(card)]='No process in device' in text
        (run/f'card{card}-after-private.txt').write_text(text)
    save(phase='complete' if state['phase']=='measured' else state['phase'],
         ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cards_empty=cards)
    print(json.dumps(state),flush=True)
