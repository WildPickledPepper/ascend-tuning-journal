"""Compare DFlash2 graph receipts to existing native MTP7, without new requests.

Usage: python analyze.py NEW_PRIVATE_ROOT MTP_PRIVATE_ROOT FILTERED_JSON
The MTP root is the prior prefetch experiment, including baseline-profile,
model-run receipts and prefixed compiled Insight responses. New root holds the
downloaded DFlash2 run and its profile/Insight exports.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

root, old, output = map(Path, sys.argv[1:4])


def read(file):
    return json.loads(file.read_text(encoding='utf-8'))


def metrics(file):
    totals, positions = {}, {}
    for line in file.read_text().splitlines():
        if line.startswith('#'):
            continue
        match = re.match(r'([^{} ]+)(?:\{([^}]*)\})?\s+(\S+)',line)
        if not match:
            continue
        name, labels, number = match.groups()
        if 'spec_decode_num_accepted_tokens_per_pos_total' in name:
            position = re.search(r'position="(\d+)"',labels)[1]
            positions[position] = positions.get(position,0)+float(number)
        elif 'spec_decode_' in name or 'time_to_first_token_seconds_' in name:
            totals[name] = totals.get(name,0)+float(number)
    return totals, positions


def acceptance(before,after):
    a,pa = metrics(before)
    b,pb = metrics(after)
    delta = {name:b.get(name,0)-a.get(name,0) for name in a.keys()|b.keys() if not name.endswith('_created')}
    positions = {key:pb.get(key,0)-pa.get(key,0) for key in pa.keys()|pb.keys()}
    rounds = delta['vllm:spec_decode_num_drafts_total']
    proposed = delta['vllm:spec_decode_num_draft_tokens_total']
    accepted = delta['vllm:spec_decode_num_accepted_tokens_total']
    assert rounds>0 and proposed==7*rounds and 0<=accepted<=proposed
    assert sum(positions.values()) == accepted
    count = delta['vllm:time_to_first_token_seconds_count']
    assert count==4
    return {'verification_rounds':rounds, 'proposed':proposed, 'accepted':accepted,
            'acceptance_rate':accepted/proposed,'counter_tokens_per_round':1+accepted/rounds,
            'accepted_per_position':positions,
            'unconditional_per_position_acceptance':{k:v/rounds for k,v in positions.items()},
            'mean_server_ttft_ms':1000*delta['vllm:time_to_first_token_seconds_sum']/count}


baseline = read(old/'model-run/results/baseline-sglang-requests.json')
candidate = read(root/'results/dflash2-sglang-requests.json')
assert [r['request'] for r in baseline['records']] == [r['request'] for r in candidate['records']]
assert baseline['evaluator_source_sha256'] == candidate['evaluator_source_sha256']
profile_a = read(old/'model-run/baseline-profile.json')
profile_b = read(root/'dflash2-profile.json')
assert profile_a['request'] == profile_b['request']
profile_ids_equal = profile_a['response']['choices'][0]['token_ids'] == profile_b['response']['choices'][0]['token_ids']
assert profile_ids_equal
assert profile_b['response']['usage']['completion_tokens'] == 128
comparison = []
for index,(a,b) in enumerate(zip(baseline['records'],candidate['records'])):
    ac,bc = a['response']['choices'][0], b['response']['choices'][0]
    aid,bid = ac['token_ids'],bc['token_ids']
    first = next((i for i,(x,y) in enumerate(zip(aid,bid)) if x!=y),None)
    if first is None and len(aid)!=len(bid):
        first=min(len(aid),len(bid))
    comparison.append({'question':index+1,'mtp_tokens':len(aid),'dflash2_tokens':len(bid),
        'mtp_elapsed_s':a['elapsed_s'],'dflash2_elapsed_s':b['elapsed_s'],
        'token_ids_equal':aid==bid,'first_token_difference':first,
        'final_content_equal':ac['message']['content']==bc['message']['content'],
        'mtp_finish':ac['finish_reason'],'dflash2_finish':bc['finish_reason']})

audits = {}
for label,base,file in [('mtp7',old/'model-run','baseline-graph-audit-after.json'),
                         ('dflash2',root,'dflash2-graph-audit-after.json')]:
    audit, = read(base/file)['results']
    for key,expected in [('num_speculative_tokens',7),('draft_use_cuda_graph',True),
                         ('draft_enforce_eager',False),('dtype','torch.bfloat16'),
                         ('tp_size',1),('context_length',4096),('max_num_seqs',1),
                         ('prefix_caching',False),('mamba_cache_dtype','bfloat16'),
                         ('mamba_ssm_cache_dtype','bfloat16'),('gpu_memory_utilization',.97)]:
        assert audit[key]==expected,(label,key,audit[key])
    assert any(w['is_draft_runnable'] and any(e['captured'] for e in w['entries']) for w in audit['wrappers'])
    audits[label]=audit
assert audits['dflash2']['selector_enabled'] and audits['dflash2']['selector_top_k']==16
assert audits['dflash2']['proposer_class']=='AscendDflash2Proposer'

profiles = {}
for label,base,folder,prefix in [('mtp7',old,'baseline-profile','prefetch-baseline'),
                                  ('dflash2',root,'dflash2-profile','dflash2-graph')]:
    capture=base/folder
    infofile,=capture.glob('profiler_info*.json')
    info=read(infofile)
    assert info['end_info']['collectionTimeEnd']
    assert info['config']['common_config']['record_shapes']
    assert info['config']['experimental_config']['_aic_metrics']=='ACL_AICORE_PIPE_UTILIZATION'
    exported=capture/'ASCEND_PROFILER_OUTPUT'
    assert (exported/'trace_view.json').exists() and list(exported.glob('*.db'))
    with (exported/'kernel_details.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    groups=collections.defaultdict(list)
    for row in rows:
        groups[row['Type'],row['Name'],row['Accelerator Core'],row['Input Shapes']].append(row)
    backend=read(base/(prefix+'-backend-operators.json'))
    assert backend['result'] and len(backend['body']['data'])==len(groups)
    for item in backend['body']['data']:
        item=item.get('compare',item)
        selected=groups[item['opType'],item['opName'],item['accCore'],item['inputShape']]
        assert len(selected)==int(item['count'])
        assert abs(sum(float(r['Duration(us)']) for r in selected)-float(item['totalTime']))<.011
    phase=read(root/(label+'-profile-phases.json'))
    # Validate the existing skill-macro TOP20 result against all CSV operator types.
    csv_types=collections.defaultdict(list)
    for row in rows:csv_types[row['Type']].append(row)
    for item in phase['whole_profile_compute_top']:
        selected=csv_types[item['op_type']]
        assert len(selected)==item['calls']
        assert abs(sum(float(r['Duration(us)']) for r in selected)/1000-item['total_ms'])<.000011
    profiles[label]={'profiler_complete':True,'insight_groups_verified':len(groups),
        'sql_top20_verified':True,'decode_rounds':phase['decode_rounds'],
        'decode_span_ms':phase['decode_span_ms'],'cycle_medians':phase['cycle_medians'],
        'device_phase_totals':phase['phase_device_totals'],
        'cpu_inclusive_scopes':phase['cpu_inclusive_scopes'],
        'target_ffn_groups':[g for g in phase['all_decode_groups'] if g['main_graph'] and g['input_shape'] in ['8,5120;34816,5120','8,17408;5120,17408']],
        'merged_draft_kernel_sum_ms_per_round':phase['phase_device_totals']['merged_draft_graph']['kernel_sum_ms']/phase['decode_rounds'],
        'no_compute_percent':phase['no_compute_pct'],
        'kernel_csv_sha256':hashlib.sha256((exported/'kernel_details.csv').read_bytes()).hexdigest()}

rates=[read(old/f'model-run/results/{name}-sglang-requests.json')['throughput_tok_s'] for name in ['baseline-r1','baseline-r2','baseline']]
accept={
    'mtp7':acceptance(old/'model-run/baseline-metrics-before.txt',old/'model-run/baseline-metrics-after.txt'),
    'dflash2':acceptance(root/'dflash2-metrics-before.txt',root/'dflash2-metrics-after.txt')}
memory=re.findall(r'Process memory\(MB\):(\d+)',(root/'card6-ready-private.txt').read_text())
result={'model':'Qwen3.8-27B','hardware':'Ascend910B2','dtype':'BF16','single_physical_card':6,
    'mtp_baseline_reused':True,'mtp_baseline_three_pass_tok_s':rates,
    'mtp_baseline_median_tok_s':statistics.median(rates),
    'dflash2_benchmark':{k:v for k,v in candidate.items() if k!='records'},
    'dflash2_over_mtp_median_ratio':candidate['throughput_tok_s']/statistics.median(rates),
    'acceptance':accept,'graph_audit':audits,'questions':comparison,'profiles':profiles,
    'ready_process_memory_mib':sum(map(int,memory)),
    'matched_profile_tokens_equal':profile_ids_equal,
    'compatibility':{'capture_adapter':'Unmasked noncausal draft graph uses sparse_mode=0 and unrestricted pre/next, matching installed eager branch.',
        'scope':'Process-local Python adapter; no target causal attention, verification policy, checkpoint or NPU kernel changes.',
        'failed_attempts':['Capture rejected with 561002: sparseMode must be 0 when mask absent.',
            'Graphs captured; HTTP client failed before requests because libpython3.12 library path absent.'],
        'client_fix':'Provide private Python library path; disable torch device backend autoload in HTTP evaluator.'},
    'cleanup':{k:v for k,v in read(root/'state.json').items() if k!='server_pid'},
    'limits':['Existing MTP baseline reused, not a simultaneous newly measured A/B.',
        'Four GSM8K questions; generated lengths may differ. Include prefill in benchmark timing.',
        'Acceptance is after 64-token warmup and before separate profiling; no acceptance thresholds changed.',
        'Counter tokens per round includes candidate/tail counting conventions, not exact HTTP output per round.',
        'Profile timings include overhead; CPU scopes are inclusive and not added to device time.',
        'DFlash2 profile includes host labels enabled after normal benchmark; reused MTP profile has no such added labels. No-compute and host timing differences cannot be treated as unprofiled costs or a proven normal-run bottleneck.',
        'Different draft shapes change numerical execution; report token differences and score without claiming universal equivalence.']}
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['mtp_baseline_median_tok_s','dflash2_benchmark','dflash2_over_mtp_median_ratio','acceptance','questions','profiles']},ensure_ascii=False,indent=2))
