import argparse,datetime,json,pathlib,statistics,time,urllib.request,urllib.error
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--run',required=True);p.add_argument('--profile',action='store_true');p.add_argument('--port',type=int,default=8038);p.add_argument('--device',type=int,default=7)
a=p.parse_args();root=pathlib.Path(a.run);root.mkdir(parents=True,exist_ok=True)
receipt=root/(a.mode+'-requests.json');assert not receipt.exists(),receipt
base=f'http://127.0.0.1:{a.port}'
def post(path,body=None,timeout=180):
    req=urllib.request.Request(base+path,data=json.dumps(body or {}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read() or b'{}')
deadline=time.monotonic()+1800
while True:
    try:
        with urllib.request.urlopen(base+'/health',timeout=3) as r:
            if r.status==200:break
    except (OSError,urllib.error.URLError):pass
    if time.monotonic()>deadline:raise TimeoutError('Model readiness timeout')
    time.sleep(3)
def metrics():
    with urllib.request.urlopen(base+'/metrics',timeout=15) as r:return r.read().decode()
(root/(a.mode+'-metrics-before.txt')).write_text(metrics())
results=[]
def save():receipt.write_text(json.dumps(results,ensure_ascii=False,indent=2))
def generate(label,prompt,tokens=128,ignore_eos=True):
    payload={'model':'Qwen3.8-27B','messages':[{'role':'user','content':prompt}],
      'max_tokens':tokens,'temperature':0,'seed':42,'stream':True,
      'return_token_ids':True,'stream_options':{'include_usage':True},
      'chat_template_kwargs':{'enable_thinking':False},'ignore_eos':ignore_eos}
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    metrics_before=metrics()
    started=time.perf_counter();text=[];ids=[];times=[];chunks=[];usage={};first_content=None;last_content=None
    with urllib.request.urlopen(req,timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':break
            item=json.loads(raw);now=time.perf_counter()-started
            if item.get('usage'):usage=item['usage']
            for choice in item.get('choices',[]):
                delta=choice.get('delta',{});piece=delta.get('content')
                chunkids=choice.get('token_ids') or delta.get('token_ids') or []
                if chunkids:
                    ids.extend(chunkids);times.extend([now]*len(chunkids));chunks.append(len(chunkids))
                if piece:
                    text.append(piece);first_content=now if first_content is None else first_content;last_content=now
    elapsed=time.perf_counter()-started
    metrics_after=metrics()
    if ignore_eos:assert usage.get('completion_tokens')==tokens,usage
    assert ids and len(ids)==usage['completion_tokens'],(len(ids),usage)
    assert len(times)>1
    row={'mode':a.mode,'label':label,'prompt':prompt,'request':payload,'usage':usage,
      'response_text':''.join(text),'token_ids':ids,'token_times_s':times,'chunk_token_counts':chunks,
      'elapsed_s':elapsed,'ttft_s':times[0],'decode_tokens_per_s':(len(ids)-1)/(times[-1]-times[0]),
      'content_ttft_s':first_content,'content_last_s':last_content,
      'metrics_before':metrics_before,'metrics_after':metrics_after,
      'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    results.append(row);save()
    print(json.dumps({k:row[k] for k in ['mode','label','usage','elapsed_s','ttft_s','decode_tokens_per_s']}),flush=True)
    return row
prompts=['请用中文简要解释什么是矩阵乘法，以及它为什么常用于神经网络。',
         '请解释HBM带宽为什么会影响大语言模型逐token推理速度，并给出一个通俗的比喻。']
generate('warmup_16',prompts[0],16)
generate('warmup_128',prompts[1],128)
for repeat in range(4):
    for index,prompt in enumerate(prompts):generate(f'benchmark_p{index}_r{repeat}',prompt)
if a.mode!='baseline-a2':
    checks=['计算 37×24，并简要展示计算过程。','用Python写一个判断素数的函数，并解释边界条件。',
            '把这句话翻译成英语：我们通过复用输入数据减少了重复搬运，但仍需要验证整个模型的速度和输出。']
    for i,prompt in enumerate(checks):generate('output_check_'+str(i),prompt,96,False)
# Statistics are published periodically; wait outside the measured requests.
time.sleep(6)
(root/(a.mode+'-metrics-benchmark-end.txt')).write_text(metrics())
if a.profile:
    active=False
    try:
        post('/start_profile',timeout=120);active=True
        generate('profiled_32',prompts[0],32)
    finally:
        if active:post('/stop_profile',timeout=600)
time.sleep(6)
(root/(a.mode+'-metrics-after.txt')).write_text(metrics())
bench=[r for r in results if r['label'].startswith('benchmark_')]
summary={'mode':a.mode,'requests':len(bench),'dtype':'BF16','physical_device':a.device,'graph':'FULL_DECODE_ONLY',
 'profiler_during_benchmark':False,'median_decode_tokens_per_s':statistics.median(r['decode_tokens_per_s'] for r in bench),
 'median_ttft_s':statistics.median(r['ttft_s'] for r in bench),'median_elapsed_s':statistics.median(r['elapsed_s'] for r in bench),
 'per_prompt':{str(i):{'median_decode_tokens_per_s':statistics.median(r['decode_tokens_per_s'] for r in bench if r['label'].startswith(f'benchmark_p{i}_'))} for i in range(2)}}
(root/(a.mode+'-summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('MODEL_BENCHMARK_COMPLETE '+json.dumps(summary),flush=True)
