"""Validate captured artifacts and export only path-free numeric evidence."""
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from analyze import load, rows


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def main(root,target):
    s=load(root/'summary-local.json')
    assert s==load(root/'summary-server.json')
    local=Path(__file__).parent
    operational=('start.sh','serve.py','model_extension.py','graph_audit_base.py',
                 'dynamic_vocab.py','profile_factory.py','profile_scopes.py','capture.py','client.py')
    assert all(sha(root/name)==sha(local/name) for name in operational)
    original=local.parent/'gdn-fp32-state'
    assert (local/'dynamic_vocab.py').read_bytes()==(original/'dynamic_vocab.py').read_bytes()
    assert (local/'graph_audit_base.py').read_bytes()==(original/'graph_audit_base.py').read_bytes()
    assert (local/'model_extension.py').read_bytes().startswith((original/'model_extension.py').read_bytes())
    before,after=[load(root/'results'/f'{x}-audit.json')['results'][0] for x in ('before','after')]
    for a in (before,after):
        assert a['dtype']=='torch.bfloat16' and a['tp_size']==1 and a['num_speculative_tokens']==7
        assert a['draft_use_cuda_graph'] and not a['draft_enforce_eager']
        assert a['gdn_state']['layer_count']==48 and a['gdn_state']['all_ssm_fp32'] and a['gdn_state']['all_conv_bf16']
        assert a['layout_experiment']['packed_count']==64 and a['layout_experiment']['prefill_direct_enabled']
        checks=a['layout_experiment']['prefill_real_checks']
        assert len(checks)==64 and all(x['different']==0 and x['finite'] for x in checks.values())
        assert all(any(e['captured'] for e in w['entries']) for w in a['wrappers'])
        assert a['dynamic_vocabulary']['enabled'] and a['dynamic_vocabulary']['capacity']==4096
    for k in ('ids_pointer','packed_pointer'):
        assert before['dynamic_vocabulary'][k]==after['dynamic_vocabulary'][k]
    delta={k:after['layout_experiment']['counts'].get(k,0)-before['layout_experiment']['counts'].get(k,0)
           for k in set(before['layout_experiment']['counts'])|set(after['layout_experiment']['counts'])}
    assert delta['prefill_direct_host_calls']==256 and delta.get('gate_unpack_fallback_host_calls',0)==0
    ref=load(root/'results/reference.json')['sequences']
    evidence={k:s[k] for k in ('model','device','physical_device','dtype','TP','concurrency','MTP','optimized_paths','kernel_changes_this_run','method','limits')}
    evidence['captures']={}
    manifest=[]
    for label,c in s['captures'].items():
        receipt=load(root/'results'/f'{label}.json')
        r=receipt['request']; n=receipt['max_output_tokens']
        assert r['token_ids']==ref[str(receipt['question_id'])][:n] and len(r['token_ids'])==n
        folder=root/receipt['profile']
        info=load(next(folder.glob('profiler_info*.json')))
        assert info['end_info']['collectionTimeEnd']
        assert c['graph_execute_markers']==c['verification_rounds']*2
        for name in ('trace_view.json','kernel_details.csv','task_time.csv'):
            p=folder/'ASCEND_PROFILER_OUTPUT'/name
            assert p.stat().st_size>0
            manifest.append({'capture':label,'artifact':name,'bytes':p.stat().st_size,'sha256':sha(p)})
        db=next((folder/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch*.db'))
        manifest.append({'capture':label,'artifact':'ascend_pytorch.db','bytes':db.stat().st_size,'sha256':sha(db)})
        evidence['captures'][label]={k:c[k] for k in (
            'input_tokens','output_tokens','question_id','valid_stop','token_sequence_equal','sequence_sha256',
            'all_kernels','decode_kernels','verification_rounds','decode_window_ms','exclusive','groups',
            'no_compute_gap_us','model_groups','graph_execute_markers','graph_windows_ms','top_shape_groups',
            'sql','sql_top20_whole_capture','decode_sql','sql_top20_decode','communication','accelerator_summary')}
        if 'host_scopes' in c:
            evidence['captures'][label]['host_scopes']=c['host_scopes']
            evidence['captures'][label]['host_scope_note']=c['host_scope_note']
        evidence['captures'][label]['started_utc']=receipt['started_utc']
        evidence['captures'][label]['completed_utc']=receipt['completed_utc']
    backend={}
    for label,short in (('PipeUtilization','dynamic-pipe'),('LongPipeUtilization','dynamic-longpipe')):
        receipt=load(root/'results'/f'{label}.json')
        folder=root/receipt['profile']/'ASCEND_PROFILER_OUTPUT'
        groups=collections.defaultdict(lambda:[0,Decimal(0)])
        for r in rows(folder/'kernel_details.csv'):
            key=(r['Type'],r['Name'],r['Accelerator Core'],r['Input Shapes'])
            groups[key][0]+=1;groups[key][1]+=Decimal(r['Duration(us)'])
        b=load(root/'insight'/f'{short}-backend-operators.json')
        assert b['result'] and len(b['body']['data'])==b['body']['total']==len(groups)
        errors=[]
        for item in b['body']['data']:
            r=item.get('compare',item)
            count,duration=groups[r['opType'],r['opName'],r['accCore'],r['inputShape']]
            assert count==int(r['count'])
            error=abs(duration-Decimal(r['totalTime']));errors.append(error)
            # Insight rounds displayed totals to 0.01 us, CSV to 0.001 us.
            assert error<=Decimal('0.0051'),(short,r['opName'],error)
        backend[label]={'groups_checked':len(groups),'calls_and_totals_match':True,
                        'max_rounding_error_us':str(max(errors))}
    evidence['gap_analysis']=load(root/'gap-analysis.json')
    for filename,key in (('gap-comparison.json','historical_gap_comparison'),
                         ('gap-launch-comparison.json','gap_following_launch_phase_comparison'),
                         ('prepare-input-breakdown.json','prepare_input_deep_dive')):
        if (root/filename).exists():evidence[key]=load(root/filename)
    if (root/'prepare-source-reference.json').exists():
        evidence['prepare_input_source_reference']=json.loads((root/'prepare-source-reference.json').read_text(encoding='utf-8-sig'))
    evidence['verification']={'server_local_analysis_identical':True,'all_four_output_prefixes_match_existing32':True,
        'nine_capture_source_files_match_private_archive':True,
        'dynamic_vocab_and_base_graph_audit_unchanged':True,'base_model_extension_retained_before_profiler_append':True,
        'prefill_excluded':True,'custom_kernels_assigned_by_graph_boundaries':True,
        'sql_top20_csv_match_each_capture_and_decode':True,'insight':backend,
        'main_packed_layers':64,'fp32_ssm_layers':48,'conv_bf16':True,
        'formal_host_count_delta':delta,'dynamic_buffer_addresses_unchanged':True,
        'no_hccl_kernel_observed':all(not c['communication'] or not c['communication'][0]['calls'] for c in s['captures'].values()),
        'new_unprofiled_performance_benchmark':False}
    archive=root/'profiles-and-sources.tar.gz'
    assert sha(archive)=='56935dd9f3b21d9f7efff5ffb26c26035653effe7b879d2646fa932fd9d5cca4'
    sources={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'artifacts':manifest,
             'runtime_sources':load(root/'source-manifest.json'),
             'analysis_sources':[{'artifact':p.name,'sha256':sha(p)} for p in sorted(Path(__file__).parent.glob('*.py'))]}
    cleanup=load(root/'results/cleanup.json')
    assert cleanup['own_service_stopped'] and cleanup['recorded_own_process_ids_absent'] and cleanup['port8047_closed']
    target.mkdir(parents=True,exist_ok=True)
    for name,obj in [('dynamic-decode-profile.json',evidence),('dynamic-decode-profile-sources.json',sources),('dynamic-decode-profile-cleanup.json',cleanup)]:
        (target/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'insight':backend,'archive_verified':True,'rounds':sum(c['verification_rounds'] for c in s['captures'].values()),'cleanup':cleanup},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
