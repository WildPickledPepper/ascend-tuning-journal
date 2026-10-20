"""Export filtered experiment evidence; run on the experiment host."""
import datetime,hashlib,json,pathlib,re,subprocess,sys,urllib.request

r=pathlib.Path(sys.argv[1]);out=r/'summary';out.mkdir(exist_ok=True)
def portable(value):
    if isinstance(value,str):return value.replace(str(r),'$DFLASH_RUN').replace(str(r.parent.parent),'$QWEN_MODEL_ROOT')
    if isinstance(value,list):return [portable(x) for x in value]
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    return value
def load(name):return json.loads((r/name).read_text())
def health(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as f:return f.status
    except OSError:return 0
states=[]
for p in sorted(r.glob('attempt-*-state.json'))+[r/'experiment-state.json']:
    s=json.loads(p.read_text())
    states.append({'receipt':p.name,**{k:s[k] for k in ['started_utc','error','restored_health','restored_utc'] if k in s}})
logs=[]
for p in sorted((r/'logs').glob('*server.log')):
    body=p.read_text(errors='replace')
    selected=[]
    for line in body.splitlines():
        if any(t in line for t in ['Loading model weights took','Available KV cache memory:',
             'ValueError: To serve at least one request','GPU KV cache size:',
             'Maximum concurrency for','Graph capturing finished','Graph memory usage']):
            line=line.replace(str(r),'$DFLASH_RUN')
            line=re.sub(r'pid=\d+','pid=REDACTED',line)
            selected.append(line)
    logs.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_lines':selected})
s=load('experiment-state.json')
mem=[]
for m in s.get('modes',[]):
    raw=m.get('process_memory','')
    vals=re.findall(r'(?:HBM Memory Usage\(MB\)|HBM Usage\(MB\)|Process memory\(MB\))\s*:\s*(\d+)',raw)
    mem.append({'mode':m['mode'],'ready_utc':m.get('ready_utc'),'reported_process_hbm_mib':[int(x) for x in vals]})
current={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'mtp_8039_health':health(8039),'baseline_8038_health':health(8038),'experimental_8040_health':health(8040)}
for port,key in [(8039,'mtp'),(8038,'baseline')]:
    if current[f'{key}_{port}_health']!=200:continue
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=5) as f:
        models=json.load(f)
    current[key+'_served_model_ids']=[m['id'] for m in models['data']]
data={'attempts':states,'log_evidence':logs,'ready_memory':mem,'final_service_state':current,
 'raw_result_files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in sorted((r/'results').iterdir()) if p.is_file()],
 'privacy':'Raw server logs, login information, process IDs and weights remain on the host.'}
data=portable(data)
(out/'dflash2-runtime-evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(data,ensure_ascii=False))
