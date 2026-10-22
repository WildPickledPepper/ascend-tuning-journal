"""Validate private results and export numeric, path-free evidence only."""
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from analyze import load, rows


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main(root,target):
    s=load(root/'summary-local.json')
    assert s==load(root/'summary-server.json')
    assert load(root/'prepare-input-local.json')==load(root/'prepare-input-server.json')
    runtime=load(root/'runtime-provenance-before.json')
    assert runtime==load(root/'runtime-provenance-after.json')
    assert not any(runtime['worktrees'].values())
    local=Path(__file__).parent
    names=('start.sh','serve.py','model_extension.py','graph_audit_base.py','dynamic_vocab.py',
           'profile_factory.py','profile_scopes.py','early_counts.py','run.py','client.py')
    assert all(sha(root/n)==sha(local/n) for n in names)
    assert (local/'dynamic_vocab.py').read_bytes()==(local.parent/'dynamic-decode-profile/dynamic_vocab.py').read_bytes()
    bench=load(root/'benchmark-summary.json')
    assert bench['comparison']['all32_sequences_equal']
    assert all(x['outputs_identical'] and x['same_round_candidate_sequences_identical'] for x in bench['correctness'].values())
    graphs=[]
    for label in ('baseline-before','baseline-after','early-before','early-after','final'):
        a=load(root/'results'/f'{label}-audit.json')['results'][0]
        assert a['dtype']=='torch.bfloat16' and a['tp_size']==1 and a['num_speculative_tokens']==7
        assert a['draft_use_cuda_graph'] and not a['draft_enforce_eager']
        assert a['gdn_state']['layer_count']==48 and a['gdn_state']['all_ssm_fp32'] and a['gdn_state']['all_conv_bf16']
        assert a['layout_experiment']['packed_count']==64 and a['layout_experiment']['prefill_direct_enabled']
        assert all(any(e['captured'] for e in w['entries']) for w in a['wrappers'])
        assert a['dynamic_vocabulary']['capacity']==4096
        graphs.append(a)
    assert len({(a['dynamic_vocabulary']['ids_pointer'],a['dynamic_vocabulary']['packed_pointer']) for a in graphs})==1
    manifest=[];backend={};validation={}
    for label,c in s['captures'].items():
        receipt=load(root/'results'/f'{label}-profile.json')
        folder=root/receipt['profile']
        info=load(next(folder.glob('profiler_info*.json')))
        assert info['end_info']['collectionTimeEnd'] and receipt['output_prefix_identical']
        conf=info['config'];common=conf['common_config'];experimental=conf['experimental_config']
        assert common['record_shapes'] and not common['profile_memory'] and not common['with_stack']
        assert experimental['_profiler_level']=='Level1' and set(experimental['_export_type'])=={'text','db'}
        validation[label]={'type':'PyTorch framework profiler','normal_stop':True,'text_and_db_complete':True,
                           'record_shapes':True,'profile_memory':False,'with_stack':False,'level':'Level1',
                           'aic_metrics':experimental['_aic_metrics']}
        assert c['graph_execute_markers']==2*c['verification_rounds']
        output=folder/'ASCEND_PROFILER_OUTPUT'
        artifacts=[output/n for n in ('trace_view.json','kernel_details.csv','task_time.csv')]+list(output.glob('ascend_pytorch*.db'))
        assert len(artifacts)==4
        for p in artifacts:
            assert p.stat().st_size>0
            manifest.append({'capture':label,'artifact':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
        groups=collections.defaultdict(lambda:[0,Decimal(0)])
        for r in rows(output/'kernel_details.csv'):
            key=(r['Type'],r['Name'],r['Accelerator Core'],r['Input Shapes'])
            groups[key][0]+=1;groups[key][1]+=Decimal(r['Duration(us)'])
        b=load(root/'insight'/f'{label}-backend-operators.json')
        assert b['result'] and len(b['body']['data'])==b['body']['total']==len(groups)
        errors=[]
        for item in b['body']['data']:
            r=item.get('compare',item)
            count,duration=groups[r['opType'],r['opName'],r['accCore'],r['inputShape']]
            assert count==int(r['count'])
            error=abs(duration-Decimal(r['totalTime']));errors.append(error)
            assert error<=Decimal('0.0051'),(label,r['opName'],error)
        backend[label]={'groups_checked':len(groups),'calls_and_totals_match':True,'max_rounding_error_us':str(max(errors))}
        # Do not export config fields containing private profiler paths.
        c.pop('config',None)
    cleanup=load(root/'results/cleanup.json')
    assert cleanup['own_service_stopped'] and cleanup['recorded_own_process_ids_absent'] and cleanup['port8047_closed']
    for p in sorted((root/'results').glob('*.json')):
        manifest.append({'capture':'request_or_audit','artifact':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    for n in ('analyze.py','prepare_input_breakdown.py','summarize.py','latency_statistics.py'):
        assert sha(root/n)==sha(local/n)
    evidence={'benchmark':bench,'profiling':s,'profile_validation':validation,'prepare_input':load(root/'prepare-input-local.json'),
              'verification':{'same_round_candidate_policy_unchanged':True,'runtime_installed_sources_unchanged':True,
                              'same_service_graphs_and_buffer_addresses':True,'all_gdn_ssm_48_fp32':True,
                              'main_gate_up_64_packed':True,'server_local_analysis_identical':True,'insight':backend}}
    sources={'artifacts':manifest,'code':[{'artifact':n,'sha256':sha(local/n)} for n in names],
             'analysis_code':[{'artifact':p.name,'sha256':sha(p)} for p in sorted(local.glob('*.py'))],
             'archive_sha256':sha(root/'profiles-and-sources.tar.gz'),
             'runtime':runtime}
    target.mkdir(parents=True,exist_ok=True)
    for name,obj in (('accepted-count-overlap.json',evidence),('accepted-count-overlap-sources.json',sources),('accepted-count-overlap-cleanup.json',cleanup)):
        (target/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'insight':backend,'cleanup':cleanup},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
