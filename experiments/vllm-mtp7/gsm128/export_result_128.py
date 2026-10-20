"""Validate all 128 receipts and export filtered benchmark metrics."""
import argparse,ast,collections,hashlib,json,math,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
hash_json=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
raw=load(a.data/'results/mtp-sglang-requests.json');state=load(a.data/'state.json');preflight=load(a.data/'preflight.json');progress=load(a.data/'progress.json')
assert state['phase']=='finished' and 'error' not in state and state['baseline_api_unchanged'] and state['restored_health']=={'8038':200,'8039':200} and state['experiment_health']==0
assert progress['phase']=='complete' and progress['completed']==128
records=raw['records'];lines=[json.loads(s) for s in (a.data/'gsm8k-test.jsonl').read_text().splitlines()][:128]
assert len(records)==len(lines)==128 and raw['example_count']==128
assert records==[json.loads(s) for s in (a.data/'requests.jsonl').read_text().splitlines()]
assert hashlib.sha256((a.data/'gsm8k-test.jsonl').read_bytes()).hexdigest()==preflight['dataset_sha256']
assert hashlib.sha256((a.data/'simple_eval_gsm8k.py').read_bytes()).hexdigest()==raw['evaluator_source_sha256']
node=next(n for n in ast.parse((a.data/'simple_eval_gsm8k.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_answer_value')
ns={'ast':ast,'re':re,'INVALID':-9999999};exec(compile(ast.Module(body=[node],type_ignores=[]),'fixed_public_answer_parser','exec'),ns);parse=ns['get_answer_value']
questions=[];finishes=collections.Counter()
for index,(r,example) in enumerate(zip(records,lines),1):
 assert r['request']['messages']==[{'role':'user','content':f"Question: {example['question']}\nAnswer:"}]
 assert r['request']['temperature']==0 and r['request']['top_p']==1 and r['request']['max_tokens']==2048 and r['request']['extra_body']['top_k']==1
 response=r['response'];usage=response['usage'];c=response['choices'][0];m=c['message'];pred=parse(m.get('content') or '');expected=parse(example['answer']);reason=c['finish_reason'];finishes[reason]+=1
 assert usage['prompt_tokens']+2048<=4096 and usage['completion_tokens']>0
 if reason=='length':assert usage['completion_tokens']==2048
 questions.append(dict(question=index,started_utc=r['started_utc'],elapsed_s=r['elapsed_s'],prompt_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],throughput_tok_s=usage['completion_tokens']/r['elapsed_s'],finish_reason=reason,correct=pred==expected,parsed_answer=pred,reference_answer=expected,final_content_present=bool(m.get('content')),request_sha256=hash_json(r['request']),message_sha256=hash_json(m)))
correct=sum(q['correct'] for q in questions);assert correct==121 and correct/128==raw['accuracy']
tokens=sum(q['output_tokens'] for q in questions);assert tokens==raw['output_tokens']==40767
assert math.isclose(tokens/raw['latency_s'],raw['throughput_tok_s'],rel_tol=1e-12)
assert finishes=={'stop':124,'length':4}
def metrics(file):
 d={};positions={}
 for l in file.read_text().splitlines():
  if not l.startswith('vllm:'):continue
  m=re.match(r'([^ {]+)(?:\{([^}]*)\})?\s+([-+\deE.]+)',l)
  if not m:continue
  name,labels,value=m.groups();v=float(value);d[name]=d.get(name,0)+v
  if name=='vllm:spec_decode_num_accepted_tokens_per_pos_total':
   pos=re.search(r'position="(\d+)"',labels or '');assert pos;positions[int(pos.group(1))]=v
 return d,positions
b,pb=metrics(a.data/'metrics-before.txt');e,pe=metrics(a.data/'metrics-after.txt');delta={k:e.get(k,0)-b.get(k,0) for k in e.keys()|b.keys()}
assert delta['vllm:time_to_first_token_seconds_count']==128
rounds=delta['vllm:spec_decode_num_drafts_total'];proposed=delta['vllm:spec_decode_num_draft_tokens_total'];accepted=delta['vllm:spec_decode_num_accepted_tokens_total'];assert proposed==7*rounds
positions={k:pe.get(k,0)-pb.get(k,0) for k in sorted(pe.keys()|pb.keys())};assert sum(positions.values())==accepted
audits={}
for phase in ['before','after']:
 audit=load(a.data/f'graph-audit-{phase}.json')['results'][0]
 for k,v in [('num_speculative_tokens',7),('draft_use_cuda_graph',True),('draft_enforce_eager',False),('dtype','torch.bfloat16'),('mamba_ssm_cache_dtype','bfloat16'),('tp_size',1),('gpu_memory_utilization',.97),('main_capture_sizes',[8]),('max_num_seqs',1),('prefix_caching',False)]:assert audit[k]==v
 assert any(w['is_draft_runnable'] and any(e['captured'] for e in w['entries']) for w in audit['wrappers']);audits[phase]=audit
memory=[int(x) for x in re.findall(r'Process memory\(MB\):(\d+)',(a.data/'logs/card6-ready.txt').read_text())];assert len(memory)==1
manual=[dict(question=8,classification='answer_extraction_false_negative',correct_answer_explicit_in_content=160,last_number_extracted=40,explanation='Answer gives 160 minutes followed by equivalent 2 hours 40 minutes; the public parser takes the last number.'),dict(question=37,classification='answer_extraction_false_negative',correct_answer_explicit_in_content=75,last_number_extracted=30,explanation='Answer states $75.00 and then repeats the 30-day interval; parser takes 30.'),dict(question=99,classification='answer_extraction_false_negative',correct_answer_explicit_in_content=5,last_number_extracted=15,explanation='Answer states 5 cars and repeats first 15 minutes; parser takes 15.')]
assert [q['question'] for q in questions if not q['correct']]==[8,37,86,88,94,99,120]
assert [q['question'] for q in questions if q['finish_reason']=='length']==[86,88,94,120]
assert all(not q['final_content_present'] for q in questions if q['finish_reason']=='length')
result=dict(model='Qwen3.8-27B',hardware='Ascend 910B2',physical_devices=[6],dtype='BF16 weights and SSM state',profiler=False,protocol={k:raw[k] for k in ['protocol','evaluator_source_commit','evaluator_source_sha256','example_count','warmup_requests','warmup_max_tokens']},benchmark=dict(output_tokens=tokens,elapsed_s=raw['latency_s'],throughput_tok_s=raw['throughput_tok_s'],accuracy=raw['accuracy'],correct_questions=correct,question_count=128,finish_reasons=dict(finishes),mean_server_ttft_ms=1000*delta['vllm:time_to_first_token_seconds_sum']/128,ttft_measurement='Server Prometheus sum/count difference after warmup; not client streaming TTFT.',ttft_delta_sum_s=delta['vllm:time_to_first_token_seconds_sum'],ttft_delta_count=128,request_latency_sum_s=sum(q['elapsed_s'] for q in questions),evaluator_observer_overhead_s=raw['latency_s']-sum(q['elapsed_s'] for q in questions),ready_process_memory_mib=sum(memory),ready_process_memory_gib=sum(memory)/1024,memory_peak_measured=False),speculation=dict(verification_rounds=rounds,proposed_tokens=proposed,accepted_tokens=accepted,candidate_acceptance_rate=accepted/proposed,counter_tokens_per_round=1+accepted/rounds,accepted_per_position=positions,per_position_acceptance={k:v/rounds for k,v in positions.items()}),questions=questions,manual_review=dict(original_score_unchanged=True,answer_extraction_false_negatives=manual,length_limit_no_final_answer=[86,88,94,120],interpretation='The 7 officially unscored items contain 3 extraction false negatives and 4 truncations; do not label all 7 mathematical errors or silently replace the official score.'),graph_audit=audits,preflight=preflight,evaluation_started_utc=progress['started_utc'],evaluation_finished_utc=raw['finished_utc'],restoration={k:state[k] for k in ['restored_utc','restored_health','baseline_api_unchanged','experiment_health']},source_sha256={name:hashlib.sha256((a.data/name).read_bytes()).hexdigest() for name in ['results/mtp-sglang-requests.json','requests.jsonl','metrics-before.txt','metrics-after.txt','graph-audit-before.json','graph-audit-after.json','state.json']},limitations=['First 128 GSM8K test examples, zero-shot, greedy, concurrency1, 2048-token cap; not full 1319-example test set.','Generated-token counts include reasoning and truncated reasoning, not just final visible answer tokens.','Ready NPU process memory snapshot is not inference peak usage.','No matched non-speculative 128-question accuracy comparison; do not attribute failures to speculation or SSM dtype.','Speculation counters can include discarded EOS/tail candidates; tokens-per-round is a counter convention.','Incremental receipt/progress writes are outside per-request timers but inside whole-evaluation elapsed.'])
a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(benchmark=result['benchmark'],speculation=result['speculation'],manual_review=result['manual_review'],restoration=result['restoration']),ensure_ascii=False,indent=2))
