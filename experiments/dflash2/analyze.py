"""Summarize paired receipts; retain mismatches instead of assuming precision."""
import argparse,hashlib,json,pathlib,re,statistics

p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('output')
p.add_argument('--generation-config',required=True);a=p.parse_args()
root=pathlib.Path(a.results);out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
generation_file=pathlib.Path(a.generation_config)
eos=json.loads(generation_file.read_text())['eos_token_id']
if isinstance(eos,int):eos=[eos]
def response_prefix(ids):
    end=next((i+1 for i,t in enumerate(ids) if t in eos),len(ids))
    return ids[:end]
def prefix_rate(row):
    n=len(response_prefix(row['token_ids']));used=0;chunks=[]
    for chunk in row['token_chunks']:
        take=min(chunk['tokens'],n-used)
        if take>0:chunks.append({'time_s':chunk['time_s'],'tokens':take});used+=take
        if used>=n:break
    assert used==n and len(chunks)>1
    return (n-chunks[0]['tokens'])/(chunks[-1]['time_s']-chunks[0]['time_s'])
def read(name):return json.loads((root/name).read_text())
def counters(text):
    result={}
    for line in text.splitlines():
        if not line.startswith('vllm:spec_decode_') or '_created' in line:continue
        name=line.split('{')[0].split()[0];match=re.search(r'\s([-+0-9.eE]+)(?:\s|$)',line)
        if match:result[name]=result.get(name,0)+float(match.group(1))
    return result
def positions(text):
    result={}
    for line in text.splitlines():
        if not line.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{'):continue
        pos=re.search(r'position="(\d+)"',line);value=re.search(r'\s([-+0-9.eE]+)(?:\s|$)',line)
        if pos and value:result[int(pos.group(1))]=result.get(int(pos.group(1)),0)+float(value.group(1))
    return result
summaries={mode:read(mode+'-summary.json') for mode in ['mtp','dflash2']}
rows={mode:read(mode+'-requests.json') for mode in summaries}
checks=[]
for mtp in rows['mtp']:
    if not mtp['label'].startswith(('benchmark_','output_check_')):continue
    other=next(r for r in rows['dflash2'] if r['label']==mtp['label'])
    x,y=mtp['token_ids'],other['token_ids']
    first=next((i for i,(aa,bb) in enumerate(zip(x,y)) if aa!=bb),None)
    if first is None and len(x)!=len(y):first=min(len(x),len(y))
    checks.append({'label':mtp['label'],'mtp_tokens':len(x),'dflash2_tokens':len(y),
                   'token_ids_equal':x==y,'text_equal':mtp['response_text']==other['response_text'],
                   'first_difference_index':first,
                   'mtp_first_eos_index':next((i for i,t in enumerate(x) if t in eos),None),
                   'dflash2_first_eos_index':next((i for i,t in enumerate(y) if t in eos),None),
                   'through_first_eos_or_output_limit_equal':response_prefix(x)==response_prefix(y)})
acceptance={}
for mode in summaries:
    before=counters((root/(mode+'-metrics-before.txt')).read_text())
    after=counters((root/(mode+'-metrics-after.txt')).read_text())
    delta={key:value-before.get(key,0) for key,value in after.items()}
    n=delta.get('vllm:spec_decode_num_drafts_total',0)
    k=delta.get('vllm:spec_decode_num_draft_tokens_total',0)
    accepted=delta.get('vllm:spec_decode_num_accepted_tokens_total',0)
    assert n>0 and k>0,(mode,delta)
    acceptance[mode]={'counters_delta':delta,'draft_token_acceptance_rate':accepted/k,
                      'mean_tokens_per_verification':1+accepted/n,
                      'scope':'Warmups, four timed requests and three output checks; profiler disabled.'}
    pb=positions((root/(mode+'-metrics-before.txt')).read_text())
    pa=positions((root/(mode+'-metrics-after.txt')).read_text())
    acceptance[mode]['accepted_per_position']=[{'position_zero_based':i,'accepted_count':v-pb.get(i,0),
      'fraction_of_rounds':(v-pb.get(i,0))/n} for i,v in sorted(pa.items())]
result={'model':'Qwen3.8-27B','target_dtype':'BF16','draft_dtype':'BF16','physical_device':6,'TP':1,
 'concurrency':1,'runtime':'Same vLLM 0.27.1 / vLLM-Ascend 0.27.1rc1 code extracted from the official image; native private CANN/PyTorch runtime.',
 'speculative_configs':{'mtp':{'method':'qwen3_5_mtp','num_speculative_tokens':1,'enforce_eager':True},
                       'dflash2':{'method':'dflash','num_speculative_tokens':7,'enforce_eager':True}},
 'summaries':summaries,'acceptance':acceptance,'output_comparisons':checks,
 'all_tested_token_ids_equal':all(x['token_ids_equal'] for x in checks),
 'all_response_prefixes_equal':all(x['through_first_eos_or_output_limit_equal'] for x in checks),
 'eos_token_ids':eos,'generation_config_sha256':hashlib.sha256(generation_file.read_bytes()).hexdigest(),
 'decode_gain_pct':100*(summaries['dflash2']['median_decode_tokens_per_s']/summaries['mtp']['median_decode_tokens_per_s']-1),
 'limits':['Short deterministic single-user text requests; this is not a broad precision evaluation.',
           'The new-version MTP arm controls the framework change; historical v0.26 results use different output lengths.',
           'Chunk-normalized continuous decode metric excludes the entire first token chunk for both methods.',
           'Fixed-length benchmarks ignore EOS; inspect first-EOS and first-difference indices before interpreting forced continuation mismatches.',
           'Streaming times include HTTP and scheduler effects and are not kernel timings.']}
paired=[]
for x in rows['mtp']:
    if not x['label'].startswith('benchmark_'):continue
    y=next(v for v in rows['dflash2'] if v['label']==x['label'])
    paired.append({'label':x['label'],'mtp_tokens_per_s':x['decode_tokens_per_s'],
                   'dflash2_tokens_per_s':y['decode_tokens_per_s'],
                   'decode_gain_pct':100*(y['decode_tokens_per_s']/x['decode_tokens_per_s']-1),
                   'mtp_eos_or_limit_prefix_tokens_per_s':prefix_rate(x),
                   'dflash2_eos_or_limit_prefix_tokens_per_s':prefix_rate(y)})
result['paired_benchmark_requests']=paired
result['eos_or_limit_prefix_median_tokens_per_s']={mode:statistics.median(prefix_rate(x) for x in rows[mode]
 if x['label'].startswith('benchmark_')) for mode in summaries}
ep=result['eos_or_limit_prefix_median_tokens_per_s']
result['eos_or_limit_prefix_gain_pct']=100*(ep['dflash2']/ep['mtp']-1)
(out/'dflash2-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['summaries','acceptance','all_tested_token_ids_equal','decode_gain_pct']},ensure_ascii=False))
