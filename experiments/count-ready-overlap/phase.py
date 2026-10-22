"""Bounded validation/benchmark/profile actions against our owned service."""
from datetime import datetime,timezone
import json,os,sys,time
from pathlib import Path
import requests
from client import ROOT,URL,DATA,request,snapshot
OUT=ROOT/'results'
def save(name,obj):(OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
def rpc(method,*args):
    r=requests.post(URL+'/collective_rpc',json={'method':method,'args':list(args)},timeout=180);r.raise_for_status();return r.json()
def mode(stage,record=False):
    result=rpc('configure_count_stage',stage,record);print('MODE',result,flush=True);return result
def audit(label):
    result=rpc('overlap_audit');save(label,result);return result['results'][0]
def metrics(label):
    r=requests.get(URL+'/metrics',timeout=30);r.raise_for_status();(OUT/(label+'.txt')).write_text(r.text)
questions=[json.loads(s)['question'] for s in DATA.read_text().splitlines()[:32]]
refs={r['question_id']:r for r in json.loads(Path(os.environ['REFERENCE_RESULTS']).read_text())}
action=sys.argv[1]
if action=='validate':
    opt_stage=int(sys.argv[2]) if len(sys.argv)>2 else 2
    request('What is 1 + 1?',64,False);request(questions[0],2048,False);snapshot('startup')
    checks={}
    for qid in (0,2):
        items=[]
        for stage,label in ((0,'base1'),(0,'base2'),(opt_stage,f'ready{opt_stage}')):
            mode(stage,True)
            if stage==3:rpc('metadata_control',True,True)
            row=request(questions[qid],2048,False);save(f'check-{qid}-{label}',row)
            if stage==3:
                ma=rpc('metadata_audit')
                save(f'check-{qid}-metadata-validation',ma)
                assert ma['results'][0]['validated_builders'] and all(x['all_equal'] for x in ma['results'][0]['validated_builders'])
                rpc('metadata_control',True,False)
            a=audit(f'check-{qid}-{label}-audit');items.append((row,a))
            assert row['token_ids']==refs[qid]['token_ids']
            print('CHECK',qid,label,len(row['token_ids']),flush=True)
        checks[str(qid)]={'outputs_identical':all(x[0]['token_ids']==items[0][0]['token_ids'] for x in items),
             'baseline_candidates_repeat':items[0][1]['candidate_set_sequence_sha256']==items[1][1]['candidate_set_sequence_sha256'],
             'early_candidates_equal':items[0][1]['candidate_set_sequence_sha256']==items[2][1]['candidate_set_sequence_sha256'],
             'baseline_drafts_repeat':items[0][1]['draft_sequence_sha256']==items[1][1]['draft_sequence_sha256'],
             'early_drafts_equal':items[0][1]['draft_sequence_sha256']==items[2][1]['draft_sequence_sha256']}
        save('correctness',checks);print('CORRECTNESS',checks[str(qid)],flush=True)
        assert checks[str(qid)]['outputs_identical']
        if checks[str(qid)]['baseline_candidates_repeat'] and checks[str(qid)]['baseline_drafts_repeat']:
            assert checks[str(qid)]['early_candidates_equal'] and checks[str(qid)]['early_drafts_equal'],checks
    print('VALIDATION_COMPLETE',flush=True)
elif action=='benchmark':
    stage=int(sys.argv[2]);label=sys.argv[3]
    count=int(sys.argv[4]) if len(sys.argv)>4 else 32
    mode(stage,False);request(questions[0],2048,False);snapshot(label+'-before');metrics(label+'-before-metrics')
    data=[];start=time.perf_counter()
    for qid,q in enumerate(questions[:count]):
        row=request(q,2048,False);row.update(question_id=qid,kind='natural')
        assert row['token_ids']==refs[qid]['token_ids'],(label,qid,'output mismatch')
        data.append(row);save(label,data);print('REQUEST',label,qid,len(row['token_ids']),row['e2e_s'],flush=True)
    save(label+'-window',{'wall_s':time.perf_counter()-start});snapshot(label+'-after');metrics(label+'-after-metrics');audit(label+'-audit')
    if stage==3:save(label+'-metadata',rpc('metadata_audit'))
    print('BENCHMARK_COMPLETE',label,flush=True)
elif action=='profile':
    stage=int(sys.argv[2]);label=sys.argv[3]
    mode(stage,False);rpc('configure_decode_profile','PipeUtilization')
    old=set((ROOT/'profiles').glob('*_ascend_pt'))
    r=requests.post(URL+'/start_profile',json={},timeout=300);r.raise_for_status()
    try:row=request(questions[0],128,False)
    finally:
        r=requests.post(URL+'/stop_profile',json={},timeout=900);r.raise_for_status()
    new=set((ROOT/'profiles').glob('*_ascend_pt'))-old
    assert len(new)==1 and row['token_ids']==refs[0]['token_ids'][:128]
    save(label+'-profile',{'profile':str(next(iter(new)).relative_to(ROOT)),'request':row,'output_prefix_identical':True,
                          'utc':datetime.now(timezone.utc).isoformat()})
    if stage==3:save(label+'-profile-metadata',rpc('metadata_audit'))
    print('PROFILE_COMPLETE',label,flush=True)
elif action=='calibrate':
    for stage,label in ((0,'baseline'),(3,'ready3')):
        mode(stage,False);metrics(label+'-prefix-before-metrics')
        row=request(questions[0],128,False)
        assert row['token_ids']==refs[0]['token_ids'][:128]
        save(label+'-unprofiled-prefix',row)
        metrics(label+'-prefix-after-metrics');audit(label+'-unprofiled-prefix-audit')
        print('UNPROFILED_PREFIX',label,row['e2e_s'],row['ttft_s'],flush=True)
else:raise ValueError(action)
