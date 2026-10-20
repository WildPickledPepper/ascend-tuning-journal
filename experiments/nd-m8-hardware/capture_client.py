import json,pathlib,sys,time,datetime,subprocess,urllib.request
root=pathlib.Path(sys.argv[1]);url='http://127.0.0.1:8040'
def post(path,payload,timeout=900):
 req=urllib.request.Request(url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())
def rpc(method,kwargs=None):return post('/collective_rpc',{'method':method,'kwargs':kwargs or {},'timeout':900})['results'][0]
def save(name,value):(root/name).write_text(json.dumps(value,indent=2))
print('CLIENT_START',flush=True)
save('graph-audit.json',rpc('graph_audit'))
save('scope-audit.json',rpc('enable_profile_scopes'))
source=root.parent/'vllm-mtp7-gsm128-20261020/results/mtp-sglang-requests.json'
records=json.loads(source.read_text())['records']
requests=[]
try:
 for index in [0,2]:
  payload=dict(records[index]['request']);payload.update(payload.pop('extra_body',{}));payload['max_tokens']=128;payload['return_token_ids']=True
  rpc('mark_phase',{'label':f'ND_M8_REQUEST_{index}_BEGIN'})
  start=time.time_ns();response=post('/v1/chat/completions',payload);end=time.time_ns()
  rpc('mark_phase',{'label':f'ND_M8_REQUEST_{index}_END'})
  assert response['usage']['completion_tokens']==128,response.get('usage')
  requests.append(dict(question_index=index,start_ns=start,end_ns=end,request=payload,response=response))
  save('model-requests.json',requests)
  print('MODEL_REQUEST_COMPLETE',index,flush=True)
finally:
 save('boot-stop.json',rpc('stop_boot_profile'))
print('MODEL_PROFILE_STOPPED',flush=True)
for metric,tag,orders in [('PipeUtilization','tlb',['small','full','sweep2','adjacent']),('L2Cache','l2',['small','full','sweep2','adjacent']),('MemoryAccess','memory',['small','full','adjacent']),('ResourceConflictRatio','conflict',['full']),('Memory','bandwidth',['full']),('MemoryL0','l0',['full'])]:
 for order in orders:
  result=rpc('nz_hardware_probe',dict(mode='tlb',order=order,label=tag+'-'+order,metric_name=metric,repeats=6,l2_enabled=True))
  print('PROBE_COMPLETE',tag,order,result,flush=True)
for order in ['small','full','sweep2','adjacent']:
 result=rpc('nz_hardware_probe',dict(mode='timing',order=order,label='timing-'+order))
 print('TIMING',order,result['median_us_per_call'],flush=True)
cmd=['msprof','--output='+str(root/'hbm-profile'),'--sys-devices=6','--sys-period=36','--sys-hardware-mem=on','--sys-hardware-mem-freq=100']
receipt={'command':cmd,'phases':[]}
with (root/'hbm-msprof.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 try:
  time.sleep(2)
  assert proc.poll() is None
  for i,order in enumerate(['small','full','small','full']):
   before=subprocess.check_output(['npu-smi','info','-t','memory','-i','6','-c','0'],text=True)
   result=rpc('nz_hardware_probe',dict(mode='hbm',order=order,label=f'hbm-{order}-{i}',seconds=2.0))
   after=subprocess.check_output(['npu-smi','info','-t','memory','-i','6','-c','0'],text=True)
   result['telemetry_before']={'memory':before};result['telemetry_after']={'memory':after}
   receipt['phases'].append(result);save('hbm-phases.json',receipt)
   print('HBM_PHASE',i,order,flush=True)
  receipt['msprof_exit_code']=proc.wait(timeout=90)
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
save('hbm-phases.json',receipt)
print('CAPTURE_COMPLETE',flush=True)
