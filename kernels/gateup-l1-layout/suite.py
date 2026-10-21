"""Sequential bounded full-working-set comparisons. No model service mutations."""
import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=True)
commands=[]
for label in ['native-a','nz-a','packed-a','packed-b','nz-b','native-b']:
    layout=label.split('-')[0]
    cmd=[sys.executable,str(root/'run.py'),'bench','--layout',layout,'--layers','64',
         '--include-down','--rounds','40','--output',str(out/(label+'.json'))]
    commands.append({'label':label,'command':cmd})
(out/'commands.json').write_text(json.dumps(commands,indent=2))
for c in commands:
    print('START',c['label'],time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    with (out/(c['label']+'.log')).open('w') as f:
        done=subprocess.run(c['command'],stdout=f,stderr=subprocess.STDOUT,timeout=600)
    if done.returncode:
        print((out/(c['label']+'.log')).read_text()[-6000:],flush=True)
        raise RuntimeError(c['label'])
    data=json.loads((out/(c['label']+'.json')).read_text())
    print('FINISH',c['label'],json.dumps({k:v['per_layer_us'] for k,v in data['timing'].items()}),flush=True)
print('SUITE_COMPLETE',flush=True)
