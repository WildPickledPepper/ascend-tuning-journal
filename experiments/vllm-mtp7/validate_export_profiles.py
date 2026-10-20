"""Validate every Insight group and fixed compute-CTE result, then export filtered phase statistics."""
import argparse,collections,csv,hashlib,json
from decimal import Decimal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
results={};manifest={}
for mode in ['old','new']:
 folders=sorted((a.data/mode/'profiles-mtp').glob('*/ASCEND_PROFILER_OUTPUT'));assert len(folders)==2
 prof=load(a.data/mode/'profile-requests.json');normal=load(a.data/mode/'profile-prefix-normal.json')
 for i,folder in enumerate(folders):
  label=f'{mode}-p{i}';d=load(a.data/mode/'analysis'/f'{label}.json')
  groups=collections.defaultdict(lambda:[0,Decimal(0)]);types=collections.defaultdict(lambda:[0,Decimal(0)])
  draft_up=0
  with (folder/'kernel_details.csv').open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):
    key=tuple(r[k] for k in ['Type','Name','Accelerator Core','Input Shapes']);v=Decimal(r['Duration(us)']);groups[key][0]+=1;groups[key][1]+=v;types[r['Type']][0]+=1;types[r['Type']][1]+=v
    if r['Model ID'] in d['draft_graph_model_ids'] and r['Type']=='MatMulV2' and r['Input Shapes'].strip(chr(34)).endswith(',5120;34816,5120'):draft_up+=1
  assert draft_up==(7*d['decode_rounds'] if mode=='new' else 0)
  backend=load(a.data/'backend'/f'{label}-backend-operators.json');rows=backend['body']['data']
  assert backend['result'] and len(rows)==backend['body']['total']==len(groups)
  maxerror=Decimal(0)
  for item in rows:
   row=item.get('compare',item);key=tuple(row[k] for k in ['opType','opName','accCore','inputShape']);count,total=groups[key]
   assert int(row['count'])==count
   error=abs(Decimal(row['totalTime'])-total);assert error<=Decimal('0.001');maxerror=max(maxerror,error)
  for row in d['whole_profile_compute_top']:
   count,total=types[row['op_type']];assert row['calls']==count;assert abs(Decimal(str(row['total_ms']))-total/1000)<=Decimal('0.000001')
  assert prof[i]['question_index']==normal[i]['question_index']
  assert prof[i]['request']==normal[i]['request']
  assert prof[i]['response']['usage']['completion_tokens']==normal[i]['response']['usage']['completion_tokens']==128
  token_ids=lambda r:r['response']['choices'][0]['token_ids']
  assert token_ids(prof[i])==token_ids(normal[i])
  ffns=[r for r in d['all_decode_groups'] if r['main_graph'] and r['type']=='MatMulV2' and ('34816' in r['input_shape'] or '17408' in r['input_shape'])]
  assert len(ffns)==2 and all(r['calls']==64*d['decode_rounds'] for r in ffns)
  summary={k:d[k] for k in ['decode_rounds','decode_span_ms','graph_replay_commands','target_ffn_calls_per_round','main_graph_model_id','draft_graph_model_ids','compute_union_ms','no_compute_ms','no_compute_pct','cycle_medians','phase_device_totals','cpu_inclusive_scopes','phase_device_types','whole_profile_compute_top','sql','boundary','limits']}
  summary.update(question_index=prof[i]['question_index'],output_tokens=128,profile_elapsed_s=prof[i]['elapsed_s'],normal_prefix_elapsed_s=normal[i]['elapsed_s'],profile_started_utc=prof[i]['started_utc'],normal_profile_token_ids_equal=True,main_ffn_groups=ffns,main_ffn_total_ms=sum(r['total_ms'] for r in ffns),main_ffn_ms_per_output_token=sum(r['total_ms'] for r in ffns)/128,all_compute_ms_per_output_token=sum(v['kernel_sum_ms'] for v in d['phase_device_totals'].values())/128,backend_validation=dict(groups_checked=len(groups),kernel_count=sum(v[0] for v in types.values()),sql_types_checked=len(d['whole_profile_compute_top']),max_time_error_us=str(maxerror),counts_and_totals_match=True,draft_ffn_calls=draft_up))
  results[label]=summary
  for name in ['kernel_details.csv','task_time.csv','trace_view.json','ascend_pytorch_profiler_0.db']:
   manifest[f'{label}/{name}']=hashlib.sha256((folder/name).read_bytes()).hexdigest()
# Same requests between configurations; record generation equality without imposing it.
for i in range(2):
 o=load(a.data/'old/profile-requests.json')[i];n=load(a.data/'new/profile-requests.json')[i];assert o['request']==n['request']
 results[f'new-p{i}']['old_new_profile_token_ids_equal']=o['response']['choices'][0]['token_ids']==n['response']['choices'][0]['token_ids']
r=dict(model='Qwen3.8-27B',hardware='Ascend 910B2',physical_devices=[6],weights_dtype='BF16',profiler='PyTorch Level1/PipeUtilization CPU+NPU, record_shapes=True',query_channel='Compiled Insight backend WebSocket operator/statistic; readonly local SQLite fallback for fixed skill CTE (no SQL MCP exposed).',profiles=results,source_sha256=manifest,limits=['The four-question unprofiled receipts determine the speedup; these 128-token prefixes diagnose the phases.','New target and merged draft graphs are separated; every target replay has 64 FFN up/down calls, every draft replay 7.','First new normal prefix took 3.731s versus profiled 2.298s, so it includes an unisolated startup effect; do not claim negative profiler overhead or compare that pair as steady-state performance.','No-compute means no recorded compute kernel: it can include DMA/control/wait and is not a direct CPU idle metric.','No isolated graph-only or SSM-dtype ablation; do not assign the entire idle-gap change solely to graph mode.','Model weights are BF16 in both; SSM cache changes FP32 to BF16. Four correct math answers do not establish broad precision equivalence.'])
a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:{x:v[x] for x in ['decode_rounds','main_ffn_total_ms','main_ffn_ms_per_output_token','all_compute_ms_per_output_token','backend_validation']} for k,v in results.items()},indent=2))
