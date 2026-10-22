"""Bounded CANN dynamic capture of our existing worker, without Torch CPU trace."""
import json,os,subprocess,time
from pathlib import Path
import requests
from client import ROOT,URL,DATA,request

pid=int(os.environ['OWN_WORKER_PID'])
assert Path(f'/proc/{pid}').exists()
questions=[json.loads(s)['question'] for s in DATA.read_text().splitlines()[:32]]
refs={r['question_id']:r for r in json.loads(Path(os.environ['REFERENCE_RESULTS']).read_text())}
for stage,label in ((0,'baseline'),(3,'ready3')):
    r=requests.post(URL+'/collective_rpc',json={'method':'configure_count_stage','args':[stage,False]},timeout=180);r.raise_for_status()
    out=ROOT/'lite'/label;out.mkdir(parents=True,exist_ok=True)
    with (out/'collect.log').open('w') as log:
        p=subprocess.Popen(['msprof','--dynamic=on',f'--pid={pid}',f'--output={out}',
            '--duration=8','--task-time=on','--ai-core=off','--ascendcl=off','--runtime-api=off'],stdout=log,stderr=subprocess.STDOUT)
        time.sleep(2)
        if p.poll() is not None and p.returncode!=0:raise RuntimeError('Dynamic profiler failed to attach')
        row=request(questions[0],128,False)
        assert row['token_ids']==refs[0]['token_ids'][:128]
        (ROOT/'results'/f'{label}-lite.json').write_text(json.dumps({'request':row,'output_prefix_identical':True},indent=2)+'\n')
        code=p.wait(timeout=120)
        assert code==0,code
    profiles=[p for p in out.rglob('PROF_*') if p.is_dir()]
    assert len(profiles)==1,profiles
    with (out/'export.log').open('w') as log:
        p=subprocess.run(['msprof','--export=on',f'--output={profiles[0]}'],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert p.returncode==0
    print('LITE_COMPLETE',label,profiles[0].name,flush=True)
