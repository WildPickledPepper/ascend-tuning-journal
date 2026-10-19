import argparse,datetime,hashlib,json,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--optimized',default='optimized-d');a=p.parse_args();root=pathlib.Path(a.run)
optimized=a.optimized
modes=['baseline-a','baseline-a2',optimized]
data={m:json.loads((root/(m+'-requests.json')).read_text()) for m in modes}
summaries={m:json.loads((root/(m+'-summary.json')).read_text()) for m in modes}
status=json.loads((root/(optimized+'-status-after.json')).read_text())['results'][0]
assert status['submitted']>0 and len(status['layers'])==64,status
baseline_rows=[r for r in data['baseline-a2'] if r['label'].startswith('benchmark_') and int(r['label'].rsplit('r',1)[1])>=2]
assert len(baseline_rows)==4
baseline_tps=statistics.median(r['decode_tokens_per_s'] for r in baseline_rows)
comparisons=[]
for row in data['baseline-a']:
    if not row['label'].startswith(('benchmark_','output_check_')):continue
    other=next(r for r in data[optimized] if r['label']==row['label'])
    aa,bb=row['token_ids'],other['token_ids']
    first=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
    if first is None and len(aa)!=len(bb):first=min(len(aa),len(bb))
    comparisons.append({'label':row['label'],'baseline_tokens':len(aa),'optimized_tokens':len(bb),
                        'token_ids_equal':aa==bb,'text_equal':row['response_text']==other['response_text'],
                        'first_different_token_index':first})
drift=[]
for row in data['baseline-a']:
    if not row['label'].startswith('benchmark_'):continue
    other=next(r for r in data['baseline-a2'] if r['label']==row['label'])
    drift.append({'label':row['label'],'token_ids_equal':row['token_ids']==other['token_ids']})
by_prompt={}
for i in range(2):
    series={m:[r['decode_tokens_per_s'] for r in (baseline_rows if m=='baseline-a2' else data[m]) if r['label'].startswith(f'benchmark_p{i}_')] for m in modes}
    med={m:statistics.median(s) for m,s in series.items()}
    base=med['baseline-a2']
    by_prompt[str(i)]={'samples':series,'medians':med,'reference_baseline_tps':base,
      'optimized_vs_reference_pct':100*(med[optimized]/base-1),
      'baseline_drift_pct':100*(med['baseline-a2']/med['baseline-a']-1)}
base=baseline_tps
r={'model':'Qwen3.8-27B','dtype':'BF16','physical_device':7,'TP':1,'concurrency':1,
 'graph_mode':'FULL_DECODE_ONLY','modified_scope':'single-token 17408->5120 BF16 down projections only',
 'order':['baseline-a','optimized-b (invalid routing, excluded)','baseline-a2','optimized-c (K order mismatch)',optimized],
 'benchmark_output_tokens':128,'benchmark_requests_per_stage':8,
 'reference_baseline':'baseline-a2 final 4 requests, after isolated dispatch test completed',
 'reference_baseline_samples':4,
 'profiler_during_benchmark':False,'summaries':summaries,'reference_baseline_tps':base,
 'reference_baseline_ttft_s':statistics.median(r['ttft_s'] for r in baseline_rows),
 'optimized_vs_reference_pct':100*(summaries[optimized]['median_decode_tokens_per_s']/base-1),
 'verified_custom_submissions':status['submitted'],'eligible_layers':len(status['layers']),
 'per_prompt':by_prompt,'output_comparisons':comparisons,'baseline_repeat_output_comparisons':drift,
 'all_tested_outputs_equal':all(c['token_ids_equal'] for c in comparisons),
 'all_baseline_repeats_equal':all(c['token_ids_equal'] for c in drift),
 'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'limitations':['Short deterministic single-user requests; not a broad model quality evaluation.',
  'Initial optimized-b attempt did not execute custom kernels and is excluded from optimization claims.',
  'An isolated dispatch verification overlapped early baseline-a2 requests; only its final 4 clean requests are the primary reference. Full baseline-a is also retained.',
  'Four optimized performance requests per prompt; stage ordering and baseline drift are reported explicitly.',
  'Raw profiler captures are separate from performance requests.']}
(root/'model-ab-summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False))
