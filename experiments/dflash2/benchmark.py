"""Paired BF16 single-user comparison. No profiler runs during timing."""
import argparse, datetime, json, pathlib, statistics, time, urllib.request

p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--run',required=True)
p.add_argument('--port',type=int,default=8040);a=p.parse_args()
root=pathlib.Path(a.run);root.mkdir(parents=True,exist_ok=True)
receipt=root/(a.mode+'-requests.json');assert not receipt.exists(),receipt
base=f'http://127.0.0.1:{a.port}';rows=[]

def metrics():
    with urllib.request.urlopen(base+'/metrics',timeout=10) as f:return f.read().decode()

def generate(label,prompt,tokens=256,ignore_eos=True):
    payload={'model':'Qwen3.8-27B','messages':[{'role':'user','content':prompt}],
      'max_tokens':tokens,'temperature':0,'seed':42,'stream':True,'return_token_ids':True,
      'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False},
      'ignore_eos':ignore_eos}
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),
                              headers={'Content-Type':'application/json'})
    before=metrics();start=time.perf_counter();ids=[];chunks=[];text=[];usage={}
    with urllib.request.urlopen(req,timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':break
            item=json.loads(raw);elapsed=time.perf_counter()-start
            if item.get('usage'):usage=item['usage']
            for choice in item.get('choices',[]):
                delta=choice.get('delta',{});new_ids=choice.get('token_ids') or delta.get('token_ids') or []
                if new_ids:ids.extend(new_ids);chunks.append({'time_s':elapsed,'tokens':len(new_ids)})
                if delta.get('content'):text.append(delta['content'])
                if choice.get('finish_reason')=='error':raise RuntimeError(item)
    elapsed=time.perf_counter()-start
    assert ids and len(ids)==usage['completion_tokens'],(len(ids),usage)
    if ignore_eos:assert len(ids)==tokens,(len(ids),tokens)
    assert len(chunks)>1
    span=chunks[-1]['time_s']-chunks[0]['time_s']
    row={'mode':a.mode,'label':label,'request':payload,'usage':usage,'token_ids':ids,
      'response_text':''.join(text),'token_chunks':chunks,'ttft_s':chunks[0]['time_s'],
      'elapsed_s':elapsed,'decode_tokens_per_s':(len(ids)-chunks[0]['tokens'])/span,
      'legacy_first_token_metric':(len(ids)-1)/span,
      'metrics_before':before,'metrics_after':metrics(),
      'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    rows.append(row);receipt.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ['mode','label','usage','elapsed_s','ttft_s','decode_tokens_per_s']}),flush=True)

(root/(a.mode+'-metrics-before.txt')).write_text(metrics())
prompts=['请用中文简要解释什么是矩阵乘法，以及它为什么常用于神经网络。',
         '请解释HBM带宽为什么会影响大语言模型逐token推理速度，并给出一个通俗的比喻。']
generate('warmup_16',prompts[0],16)
generate('warmup_128',prompts[1],128)
for repeat in range(2):
    for index,prompt in enumerate(prompts):generate(f'benchmark_p{index}_r{repeat}',prompt)
for i,prompt in enumerate(['计算 37×24，并简要展示计算过程。','用Python写一个判断素数的函数，并解释边界条件。',
                          '把这句话翻译成英语：我们通过复用输入数据减少了重复搬运，但仍需要验证整个模型的速度和输出。']):
    generate('output_check_'+str(i),prompt,96,False)
time.sleep(6)
(root/(a.mode+'-metrics-after.txt')).write_text(metrics())
bench=[r for r in rows if r['label'].startswith('benchmark_')]
result={'mode':a.mode,'requests':len(bench),'tokens_per_benchmark':256,'dtype':'BF16','TP':1,
 'physical_device':6,'concurrency':1,'profiler_during_benchmark':False,
 'median_decode_tokens_per_s':statistics.median(r['decode_tokens_per_s'] for r in bench),
 'median_ttft_s':statistics.median(r['ttft_s'] for r in bench),
 'median_elapsed_s':statistics.median(r['elapsed_s'] for r in bench),
 'metric':'Exclude all tokens in the first streamed token chunk; divide remaining tokens by first-to-last chunk time.',
 'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/(a.mode+'-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print('BENCHMARK_COMPLETE',json.dumps(result),flush=True)
