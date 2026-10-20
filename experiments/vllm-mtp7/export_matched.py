"""Export filtered, matched GSM8K MTP1/MTP7 receipts; no raw replies."""
import argparse,hashlib,json,math,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
hash_json=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
old=load(a.old/'requests.json');new=load(a.new/'requests.json')
assert len(old['records'])==len(new['records'])==4
assert old['evaluator_source_sha256']==new['evaluator_source_sha256']
assert [x['request'] for x in old['records']]==[x['request'] for x in new['records']]
def counters(root):
 def read(name):
  d={}
  for l in (root/name).read_text().splitlines():
   if l.startswith('vllm:spec_decode_') and '_total{' in l:
    key=l.split('{')[0]
    if 'per_pos' not in key:d[key]=d.get(key,0)+float(l.rsplit(' ',1)[1])
  return d
 b=read('metrics-before.txt');c=read('metrics-after.txt')
 d={k:c.get(k,0)-b.get(k,0) for k in c.keys()|b.keys()}
 rounds=d['vllm:spec_decode_num_drafts_total'];proposed=d['vllm:spec_decode_num_draft_tokens_total'];accepted=d['vllm:spec_decode_num_accepted_tokens_total']
 return dict(rounds=rounds,proposed=proposed,accepted=accepted,candidate_acceptance_rate=accepted/proposed,counter_tokens_per_round=1+accepted/rounds)
def extract(raw,root):
 assert sum(r['response']['usage']['completion_tokens'] for r in raw['records'])==raw['output_tokens']==844
 assert math.isclose(raw['throughput_tok_s'],raw['output_tokens']/raw['latency_s'],rel_tol=1e-12)
 audit=load(root/'graph-audit-before.json')['results'][0]
 keys=['num_speculative_tokens','draft_use_cuda_graph','draft_enforce_eager','tp_size','context_length','dtype','mamba_cache_dtype','mamba_ssm_cache_dtype','gpu_memory_utilization','max_num_seqs','prefix_caching','main_capture_sizes']
 return dict(benchmark={k:raw[k] for k in ['latency_s','output_tokens','throughput_tok_s','accuracy','warmup_requests','warmup_max_tokens','evaluator_source_sha256']},configuration={k:audit[k] for k in keys},counters=counters(root),requests=[dict(started_utc=r['started_utc'],elapsed_s=r['elapsed_s'],output_tokens=r['response']['usage']['completion_tokens'],prompt_tokens=r['response']['usage']['prompt_tokens'],request_sha256=hash_json(r['request']),message_sha256=hash_json(r['response']['choices'][0]['message'])) for r in raw['records']])
x=extract(old,a.old);y=extract(new,a.new)
assert x['configuration']['num_speculative_tokens']==1 and x['configuration']['draft_enforce_eager']
assert y['configuration']['num_speculative_tokens']==7 and y['configuration']['draft_use_cuda_graph']
assert x['counters']['rounds']==448 and y['counters']['rounds']==171
r=dict(model='Qwen3.8-27B',hardware='Ascend 910B2',physical_devices=[6],weights_dtype='BF16',profiler=False,same_request_payloads=True,old=x,new=y,throughput_ratio=y['benchmark']['throughput_tok_s']/x['benchmark']['throughput_tok_s'],total_latency_reduction=1-y['benchmark']['latency_s']/x['benchmark']['latency_s'],verification_round_reduction=1-y['counters']['rounds']/x['counters']['rounds'],per_question_equal_token_count=[o['output_tokens']==n['output_tokens'] for o,n in zip(x['requests'],y['requests'])],per_question_equal_message=[o['message_sha256']==n['message_sha256'] for o,n in zip(x['requests'],y['requests'])],final_content_equal=[o['response']['choices'][0]['message']['content']==n['response']['choices'][0]['message']['content'] for o,n in zip(old['records'],new['records'])],limits=['Four questions, one normal timed run per configuration; not a general performance ceiling or broad precision evaluation.','Question 3 has identical final text and token count but different reasoning text; do not claim bitwise full generation equality.','Speculation count, draft graphs, SSM state dtype, memory budget and main capture size changed together; their independent contributions were not ablated.','End-to-end timing includes prefill. Counter tokens/round may include discarded EOS/tail tokens; do not equate it with exact returned output.'])
state=load(a.old/'state.json')
assert state['phase']=='finished' and state['baseline_api_unchanged'] and state['restored_health']=={'8038':200,'8039':200} and state['experiment_health']==0
r['restoration']={k:state[k] for k in ['restored_utc','restored_health','baseline_api_unchanged','experiment_health']}
a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,ensure_ascii=False,indent=2))
