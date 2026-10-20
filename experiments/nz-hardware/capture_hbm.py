"""Device-6-only system HBM sampling around labelled native-operator phases."""
import datetime,json,pathlib,subprocess,sys,time,urllib.request

root=pathlib.Path(sys.argv[1]);assert root.is_absolute()
tag=sys.argv[2] if len(sys.argv)>2 else 'hbm'
period=int(sys.argv[3]) if len(sys.argv)>3 else 36
def rpc(kwargs):
    payload={'method':'nz_hardware_probe','kwargs':kwargs,'timeout':180}
    req=urllib.request.Request('http://127.0.0.1:8039/collective_rpc',
      data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as f:return json.loads(f.read())['results'][0]

def telemetry():
    return {kind:subprocess.run(['npu-smi','info','-t',kind,'-i','6','-c','0'],
           capture_output=True,text=True,timeout=10).stdout
            for kind in ['sensors','memory','power']}

command=['msprof','--output='+str(root/(tag+'-profile')),'--sys-devices=6','--sys-period='+str(period),
         '--sys-hardware-mem=on','--sys-hardware-mem-freq=100']
receipt={'command':command,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'profiler_scope':'physical device 6 only','phases':[]}
with (root/(tag+'-msprof.log')).open('w') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        time.sleep(2)
        if process.poll() is not None:raise RuntimeError('System profiler exited before phases')
        for i,order in enumerate(['small','full','small','full']):
            before=telemetry()
            phase=rpc({'mode':'hbm','order':order,'label':f'{tag}-{order}-{i}','seconds':2.0})
            phase['telemetry_before']=before;phase['telemetry_after']=telemetry()
            receipt['phases'].append(phase)
            (root/(tag+'-phases.json')).write_text(json.dumps(receipt,indent=2))
            print(json.dumps({k:phase[k] for k in ['label','active_seconds','graphs','active_start_ns','active_end_ns']}),flush=True)
            time.sleep(.4)
        receipt['msprof_exit_code']=process.wait(timeout=90)
    finally:
        if process.poll() is None:
            process.terminate();process.wait(timeout=15)
receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/(tag+'-phases.json')).write_text(json.dumps(receipt,indent=2))
