"""Verify measured sources, then export path-free numeric results."""
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from analyze import load,rows
from latency_statistics import summarize_rows,describe
from summarize import acceptance


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def arm(root,label):
    p=root/'results';data=load(p/f'{label}.json');window=load(p/f'{label}-window.json')
    if len(data)==32:s=summarize_rows(data,window)
    else:
        elapsed=sum(x['e2e_s'] for x in data);n=sum(len(x['token_ids']) for x in data)
        s={'common':{'successful_requests':len(data),'output_tokens':n,'request_e2e_sum_s':elapsed,
            'output_throughput_by_request_e2e':n/elapsed,'benchmark_window_s':window['wall_s']},
           'distributions':{'TTFT':describe([x['ttft_s']*1000 for x in data],'ms'),
                            'TPOT_stream_end':describe([x['tpot_ms'] for x in data],'ms/token')}}
    assert all(x['finish_reason']=='stop' and len(x['token_ids'])<2048 for x in data)
    s['acceptance']=acceptance(p,label)
    s['per_request']=[{k:x[k] for k in ('question_id','e2e_s','ttft_s','tpot_ms','output_tok_s','finish_reason')}|
                      {'output_tokens':len(x['token_ids']),'sequence_sha256':hashlib.sha256(json.dumps(x['token_ids']).encode()).hexdigest()} for x in data]
    return s,data


def pair(a,b,rawa,rawb):
    assert len(rawa)==len(rawb) and all(x['token_ids']==y['token_ids'] for x,y in zip(rawa,rawb))
    return {'all_sequences_equal':True,'requests':len(rawa),
            'throughput_change_pct':100*(b['common']['output_throughput_by_request_e2e']/a['common']['output_throughput_by_request_e2e']-1),
            'e2e_change_pct':100*(b['common']['request_e2e_sum_s']/a['common']['request_e2e_sum_s']-1),
            'faster_requests':sum(y['e2e_s']<x['e2e_s'] for x,y in zip(rawa,rawb))}


def validate_profiles(root,prefix):
    s=load(root/'summary-server.json');assert s==load(root/'summary-local.json')
    h=load(root/'prepare-input-server.json');assert h==load(root/'prepare-input-local.json')
    backend={};manifest=[]
    for label,c in s['captures'].items():
        receipt=load(root/'results'/f'{label}-profile.json');f=root/receipt['profile']
        info=load(next(f.glob('profiler_info*.json')))
        assert info['end_info']['collectionTimeEnd'] and receipt['output_prefix_identical']
        assert c['output_tokens']==128 and c['graph_execute_markers']==2*c['verification_rounds']
        assert info['config']['common_config']['record_shapes']
        assert info['config']['experimental_config']['_profiler_level']=='Level1'
        c.pop('config',None)
        output=f/'ASCEND_PROFILER_OUTPUT'
        artifacts=[output/n for n in ('trace_view.json','kernel_details.csv','task_time.csv')]+list(output.glob('ascend_pytorch*.db'))
        assert len(artifacts)==4 and all(p.stat().st_size for p in artifacts)
        manifest += [{'capture':label,'artifact':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in artifacts]
        g=collections.defaultdict(lambda:[0,Decimal(0)])
        for r in rows(output/'kernel_details.csv'):
            key=r['Type'],r['Name'],r['Accelerator Core'],r['Input Shapes']
            g[key][0]+=1;g[key][1]+=Decimal(r['Duration(us)'])
        b=load(root/'insight'/f'{prefix}-{label}-backend-operators.json')
        assert b['result'] and len(b['body']['data'])==b['body']['total']==len(g)
        for item in b['body']['data']:
            r=item.get('compare',item);count,duration=g[r['opType'],r['opName'],r['accCore'],r['inputShape']]
            assert count==int(r['count']) and abs(duration-Decimal(r['totalTime']))<=Decimal('0.0051')
        backend[label]={'groups':len(g),'count_and_time_match_csv':True}
    return {'profiling':s,'prepare_input':h,'insight':backend},manifest


def audit_model(root,labels):
    pointers=set()
    for label in labels:
        a=load(root/'results'/f'{label}-audit.json')['results'][0]
        assert a['dtype']=='torch.bfloat16' and a['tp_size']==1 and a['num_speculative_tokens']==7
        assert a['gdn_state']['layer_count']==48 and a['gdn_state']['all_ssm_fp32'] and a['gdn_state']['all_conv_bf16']
        assert a['layout_experiment']['packed_count']==64 and a['layout_experiment']['prefill_direct_enabled']
        assert a['draft_use_cuda_graph'] and not a['draft_enforce_eager']
        assert all(any(e['captured'] for e in w['entries']) for w in a['wrappers'])
        v=a['dynamic_vocabulary'];assert v['enabled'] and v['capacity']==4096
        pointers.add((v['ids_pointer'],v['packed_pointer']))
    assert len(pointers)==1


def main(r2,r3,target):
    b2,rawb=arm(r2,'baseline');o2,raw2=arm(r2,'ready2')
    b4,raw4=arm(r3,'control4');o4,rawf4=arm(r3,'fast4');o3,raw3=arm(r3,'ready3')
    assert len(raw3)==32 and all(x['token_ids']==y['token_ids'] for x,y in zip(rawb,raw3))
    p2,m2=validate_profiles(r2,'count2');p3,m3=validate_profiles(r3,'count3')
    audit_model(r2,['baseline-before','baseline-after','ready2-before','ready2-after'])
    audit_model(r3,['control4-before','control4-after','fast4-before','fast4-after','ready3-before','ready3-after'])
    for root,label in ((r2,'ready2'),(r3,'ready3')):
        a=load(root/'results'/f'{label}-audit.json')['results'][0]['counts']
        assert a['sample_calls']==a['early_vocab_updates']==a['valid_count_primed']==a['valid_count_prepare_reused']==a['valid_count_copy_reused']
    checks={}
    for q in (0,2):
        ma=load(r3/'results'/f'check-{q}-metadata-validation.json')['results'][0]
        assert len(ma['validated_builders'])==3 and all(x['all_equal'] and x['fields_checked']==30 for x in ma['validated_builders'])
        checks[str(q)]={'validated_builders':ma['validated_builders'],'counts':ma['counts']}
    correctness={}
    for name,root in (('stage2',r2),('stage3',r3)):
        correctness[name]=load(root/'results/correctness.json')
        assert all(all(v.values()) for v in correctness[name].values())
    calibration={}
    for label in ('baseline','ready3'):
        a=load(r3/'results'/f'{label}-unprofiled-prefix.json')
        b=load(r3/'results'/f'{label}-profile.json')['request']
        assert a['token_ids']==b['token_ids'] and len(a['token_ids'])==128
        au=load(r3/'results'/f'{label}-unprofiled-prefix-audit.json')['results'][0]
        calibration[label]={'outputs_equal':True,'output_tokens':128,'profiled_e2e_s':b['e2e_s'],
            'unprofiled_e2e_s':a['e2e_s'],'profiled_decode_s':b['e2e_s']-b['ttft_s'],
            'unprofiled_decode_s':a['e2e_s']-a['ttft_s'],'unprofiled_sample_calls_including_prefill':au['counts']['sample_calls'],
            'profiled_verification_rounds':p3['profiling']['captures'][label]['verification_rounds']}
    cleanup={};runtime={}
    for name,root in (('stage2',r2),('stage3',r3)):
        cleanup[name]=load(root/'results/cleanup.json')
        assert cleanup[name]['own_service_stopped'] and cleanup[name]['recorded_own_process_ids_absent'] and cleanup[name]['port8047_closed']
        runtime[name]=load(root/'runtime-provenance-before.json')
        assert runtime[name]==load(root/'runtime-provenance-after.json') and not any(runtime[name]['worktrees'].values())
    if (r3/'results/cleanup-recheck.json').exists():
        cleanup['completion_recheck']=load(r3/'results/cleanup-recheck.json')
        assert cleanup['completion_recheck']['recorded_own_process_ids_absent'] and cleanup['completion_recheck']['port8047_closed']
    code=Path(__file__).parent
    operational=('early_counts.py','fast_metadata.py','model_extension.py','dynamic_vocab.py','graph_audit_base.py','start.sh','hold.py','client.py','profile_factory.py','profile_scopes.py')
    assert all(sha(code/n)==sha(r3/n) for n in operational)
    assert (code/'dynamic_vocab.py').read_bytes()==(code.parent/'accepted-count-overlap/dynamic_vocab.py').read_bytes()
    for root,manifest in ((r2,m2),(r3,m3)):
        for p in sorted((root/'results').glob('*.json')):manifest.append({'capture':'request_or_audit','artifact':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    evidence={'model':'Qwen3.8-27B','physical_card':6,'TP':1,'concurrency':1,'MTP':7,
        'weights_and_conv':'BF16','ssm_state':'FP32','same_round_dynamic_vocabulary':4096,
        'stage2':{'baseline32':b2,'ready2_32':o2,'comparison':pair(b2,o2,rawb,raw2),**p2},
        'stage3':{'paired4_baseline':b4,'paired4_optimized':o4,'paired4_comparison':pair(b4,o4,raw4,rawf4),
                  'optimized32':o3,'recent32_reference_comparison':{'same32_outputs':True,'cross_service':True,
                  'throughput_change_pct':100*(o3['common']['output_throughput_by_request_e2e']/b2['common']['output_throughput_by_request_e2e']-1)},**p3},
        'correctness':correctness,'real_npu_metadata_checks':checks,'profiler_calibration':calibration,
        'validation':{'server_local_analysis_identical':True,'all48_ssm_fp32':True,'all64_gate_up_packed':True,
                      'same_graphs_and_dynamic_buffer_addresses':True,'installed_sources_unchanged':True},
        'limitations':['One sequential arm each; shared-host timing variation; no stable speedup guarantee.',
           'Stage2 baseline CPU profiling was unusually slow (30.91% gap); retain rather than discard it.',
           'Final32 reuses recent same-card reference across service startups; final same-service A/B has four requests.',
           'Same128 profiler on/off calibration has a one-sample-call difference in the baseline; not an exact attribution of every millisecond.',
           'Dynamic msprof attach rejected the running PID; no lighter device-gap measurement is claimed.',
           'Only first-question 128-output profiling; do not extrapolate to all contexts/batches.',
           'Final outputs equal does not prove every formal draft/candidate trajectory equal.']}
    archives={'stage2':(r2/'stage2-profiles-and-sources.tar.gz','fb25f939e2ee2ce0919920cbe44de2ed8a5ffa4a94c864a9b4c1b06e149cab92'),
              'stage3':(r3/'profiles-and-sources.tar.gz','86c54e3b830ec2375a4b8d9d2d14dd46ce289f5afeae2d4ce31ca77243c8eed2')}
    for p,expected in archives.values():assert sha(p)==expected
    sources={'archives':{k:{'sha256':expected,'bytes':p.stat().st_size} for k,(p,expected) in archives.items()},
             'runtime':runtime,'stage2_artifacts':m2,'stage3_artifacts':m3,
             'stage2_runtime_code':[{'artifact':n,'sha256':sha(r2/n)} for n in operational if (r2/n).exists()],
             'code':[{'artifact':p.name,'sha256':sha(p)} for p in sorted(code.iterdir()) if p.is_file() and p.suffix in ('.py','.sh','.cjs')]}
    target.mkdir(parents=True,exist_ok=True)
    for name,obj in (('count-ready-overlap.json',evidence),('count-ready-overlap-sources.json',sources),('count-ready-overlap-cleanup.json',cleanup)):
        (target/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'paired4':evidence['stage3']['paired4_comparison'],'recent32':evidence['stage3']['recent32_reference_comparison'],
                      'gap_percent':{k:v['exclusive']['no_compute']['window_percent'] for k,v in p3['profiling']['captures'].items()}},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
