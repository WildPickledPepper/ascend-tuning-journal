"""Compare four-question K6/K8 receipts with the existing K7 baseline.

Usage: python analyze.py PRIVATE_RUN OLD_PREFETCH_PRIVATE FILTERED_JSON
No requests are issued. Responses, token IDs, paths and process IDs stay private.
"""
import json,re,statistics,sys
from pathlib import Path

root,old,output=map(Path,sys.argv[1:4])
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def metrics(p):
    totals={};positions={}
    for line in p.read_text().splitlines():
        if line.startswith('#'):continue
        m=re.match(r'([^{} ]+)(?:\{([^}]*)\})?\s+(\S+)',line)
        if not m:continue
        name,labels,number=m.groups()
        if name.endswith('_created') or name.endswith('_bucket'):continue
        if 'spec_decode_num_accepted_tokens_per_pos_total' in name:
            pos=re.search(r'position="(\d+)"',labels)[1];positions[pos]=positions.get(pos,0)+float(number)
        else:totals[name]=totals.get(name,0)+float(number)
    return totals,positions
def counters(before,after,k):
    a,pa=metrics(before);b,pb=metrics(after)
    delta={name:b.get(name,0)-a.get(name,0) for name in a.keys()|b.keys()}
    pos={n:pb.get(n,0)-pa.get(n,0) for n in pa.keys()|pb.keys()}
    rounds=delta['vllm:spec_decode_num_drafts_total'];proposed=delta['vllm:spec_decode_num_draft_tokens_total'];accepted=delta['vllm:spec_decode_num_accepted_tokens_total']
    assert rounds>0 and proposed==k*rounds and accepted==sum(pos.values())
    assert len(pos)==k and 0<=accepted<=proposed
    assert delta['vllm:time_to_first_token_seconds_count']==4
    assert delta['vllm:request_decode_time_seconds_count']==4
    return {'verification_rounds':rounds,'proposed':proposed,'accepted':accepted,
        'acceptance_rate':accepted/proposed,'counter_tokens_per_round':1+accepted/rounds,
        'accepted_per_position':pos,
        'mean_server_ttft_ms':250*delta['vllm:time_to_first_token_seconds_sum'],
        'server_decode_total_s':delta['vllm:request_decode_time_seconds_sum'],
        'server_decode_ms_per_verification_round':1000*delta['vllm:request_decode_time_seconds_sum']/rounds}

baseline=read(old/'model-run/results/baseline-sglang-requests.json')
rates=[read(old/f'model-run/results/{n}-sglang-requests.json')['throughput_tok_s'] for n in ['baseline-r1','baseline-r2','baseline']]
summary={'k7':{'k':7,'throughput_tok_s':statistics.median(rates),'three_pass_rates':rates,
    'output_tokens':baseline['output_tokens'],'accuracy':baseline['accuracy'],
    'counters':counters(old/'model-run/baseline-metrics-before.txt',old/'model-run/baseline-metrics-after.txt',7)}}
for k in (6,8):
    d=root/f'k{k}';receipt=read(d/f'results/k{k}-sglang-requests.json')
    assert receipt['example_count']==4 and len(receipt['records'])==4
    assert receipt['evaluator_source_sha256']==baseline['evaluator_source_sha256']
    assert [x['request'] for x in receipt['records']]==[x['request'] for x in baseline['records']]
    audits=[]
    for phase in ['before','after']:
        audit,=read(d/f'graph-audit-{phase}.json')['results'];audits.append(audit)
        for key,value in {'num_speculative_tokens':k,'draft_use_cuda_graph':True,'draft_enforce_eager':False,
            'main_graph_mode':'FULL_DECODE_ONLY','main_capture_sizes':[k+1],'dtype':'torch.bfloat16',
            'tp_size':1,'context_length':4096,'max_num_seqs':1,'prefix_caching':False,
            'mamba_ssm_cache_dtype':'bfloat16','mamba_cache_dtype':'bfloat16','gpu_memory_utilization':.97}.items():
            assert audit[key]==value,(k,key,audit[key])
        for is_draft in [False,True]:
            assert any(w['is_draft_runnable']==is_draft and any(e['captured'] and f'num_tokens={k+1},' in e['batch_descriptor'] for e in w['entries']) for w in audit['wrappers'])
    state=read(d/'state.json');assert state['phase']=='complete' and all(state['cards_empty'].values()) and state['native_checkpoint']
    questions=[]
    for i,(a,b) in enumerate(zip(baseline['records'],receipt['records']),1):
        ac,bc=a['response']['choices'][0],b['response']['choices'][0]
        assert bc.get('token_ids') and len(bc['token_ids'])==b['response']['usage']['completion_tokens']
        questions.append({'question':i,'baseline_output_tokens':a['response']['usage']['completion_tokens'],
            'output_tokens':b['response']['usage']['completion_tokens'],'elapsed_s':b['elapsed_s'],
            'token_ids_equal_to_k7':ac['token_ids']==bc['token_ids'],
            'content_equal_to_k7':ac['message']['content']==bc['message']['content'],'finish_reason':bc['finish_reason']})
    mem=re.findall(r'Process memory\(MB\):(\d+)',(d/'card6-ready-private.txt').read_text())
    summary[f'k{k}']={'k':k,'benchmark':{key:value for key,value in receipt.items() if key!='records'},
        'throughput_tok_s':receipt['throughput_tok_s'],'over_k7_median_ratio':receipt['throughput_tok_s']/statistics.median(rates),
        'counters':counters(d/'metrics-before.txt',d/'metrics-after.txt',k),
        'main_and_draft_graphs_verified':True,'capture_sizes':[k+1],
        'ready_memory_mib':sum(map(int,mem)),'questions':questions,
        'cleanup':{key:value for key,value in state.items() if key!='server_pid'}}
result={'model':'Qwen3.8-27B','dtype':'BF16','hardware':'Ascend910B2','TP':1,'physical_card':6,
    'native_mtp_only':True,'new_k7_baseline_requests':0,'evaluation_examples_per_candidate':4,
    'summary':summary,'fastest_observed_k':max(summary.values(),key=lambda x:x['throughput_tok_s'])['k'],
    'limits':['K7 reuses an earlier three-pass median; K6 and K8 each have one four-question pass, not a simultaneous new A/B.',
        'Throughput includes prefill and client evaluator overhead; server decode/round is an aggregate wall-time ratio, not a measured kernel-cycle duration.',
        'No profiler during timing, no kernel rewrite, no trained MTP checkpoint.',
        'Candidate count changes round boundaries and can change numerical paths and output lengths.',
        'This tests K6/7/8 on four questions, not a global optimum or a full accuracy evaluation.']}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:{'throughput_tok_s':v['throughput_tok_s'],'counters':v['counters']} for key,v in summary.items()},indent=2))
