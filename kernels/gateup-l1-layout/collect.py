"""Collect L2 counters and actual device-7 HBM traffic in separate runs."""
import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent;out=root/'results'
assert json.loads((out/'validation.json').read_text())['native_bit_equal']
for metric in ['l2','hbm']:
    for layout in ['native','nz','packed']:
        label=layout+'-'+metric
        cmd=[sys.executable,str(root/'run.py'),'bench','--layout',layout,'--layers','64',
             '--include-down','--rounds','5','--output',str(out/(label+'.json'))]
        if metric=='l2':cmd+=['--profile',str(out/(label+'-profile')),'--metric','L2Cache']
        else:
            cmd+=['--seconds','6']
            cmd=['msprof','--output='+str(out/(label+'-sys')),'--sys-devices=7',
                 '--sys-hardware-mem=on','--sys-hardware-mem-freq=100','--ai-core=off',
                 '--task-time=off','--ascendcl=off','--application='+' '.join(cmd)]
        (out/(label+'-command.json')).write_text(json.dumps(cmd,indent=2))
        print('COLLECT_START',label,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        with (out/(label+'.log')).open('w') as f:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        if p.returncode:
            print((out/(label+'.log')).read_text()[-6000:],flush=True)
            raise RuntimeError(label)
        print('COLLECT_DONE',label,flush=True)
