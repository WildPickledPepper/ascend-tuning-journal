"""Offline proof of wait exposure using CPU->device copy flow endpoints."""
import collections
import json
from pathlib import Path
import sys
from analyze import load,rows,ns,stats


def inspect(root,label):
    receipt=load(root/'results'/f'{label}-profile.json')
    folder=root/receipt['profile']/'ASCEND_PROFILER_OUTPUT'
    trace=load(folder/'trace_view.json');ev=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    preps=sorted([e for e in ev if e.get('name')=='prepare input' and e['ph']=='X'],key=lambda e:ns(e['ts']))
    thread=(preps[0]['pid'],preps[0]['tid'])
    cpu=[e for e in ev if e.get('ph')=='X' and (e['pid'],e['tid'])==thread]
    flows={ns(e['ts']):e for e in ev if e.get('name')=='torch_to_npu' and e['ph']=='s' and (e['pid'],e['tid'])==thread}
    ends={e['id']:e for e in ev if e.get('name')=='torch_to_npu' and e['ph']=='f'}
    copies={(e['pid'],e['tid'],ns(e['ts'])):e for e in ev if e.get('name')=='MEMCPY_ASYNC' and e['ph']=='X'}
    markers=sorted(ns(t['task_start(us)']) for t in rows(folder/'task_time.csv') if t['kernel_type']=='MODEL_EXECUTE')
    rounds=len(markers)//2
    assert len(markers)==2*rounds and len(preps)==rounds+2
    result=[]
    for i in range(rounds-1):
        previous,p=preps[i+1],preps[i+2]
        a,b=ns(p['ts']),ns(p['ts'])+ns(p['dur']);anchor=markers[2*i]
        sync=sorted([e for e in cpu if e['name']=='Event::synchronize' and a<=ns(e['ts'])<b],key=lambda e:ns(e['ts']))
        assert len(sync)==3
        pending=[e for e in cpu if e['name']=='acl_memcpy_device_to_host' and ns(previous['ts'])+ns(previous['dur'])<=ns(e['ts'])<a]
        assert len(pending)==3,(label,i,len(pending))
        selected={}
        for c in pending:
            start=ns(c['ts']);f=flows[start];end=ends[f['id']]
            device=copies[end['pid'],end['tid'],ns(end['ts'])]
            parent=[s['name'] for s in cpu if s['name'] in ('async_state_update','draft_token') and ns(s['ts'])<=start<ns(s['ts'])+ns(s['dur'])]
            role='accepted_count' if 'async_state_update' in parent else 'valid_count' if 'draft_token' in parent else 'async_sampled_ids'
            assert role not in selected
            selected[role]={'host_submit_ms':(start-anchor)/1e6,'device_start_ms':(ns(device['ts'])-anchor)/1e6,
                            'device_end_ms':(ns(device['ts'])+ns(device['dur'])-anchor)/1e6,
                            'copy_device_us':ns(device['dur'])/1000,
                            'host_submit_to_device_start_ms':(ns(device['ts'])-start)/1e6}
        assert set(selected)=={'accepted_count','valid_count','async_sampled_ids'}
        item={'round':i,'copies':selected,'syncs':{str(j+1):{'arrive_ms':(ns(e['ts'])-anchor)/1e6,
                          'return_ms':(ns(e['ts'])+ns(e['dur'])-anchor)/1e6,'wait_ms':ns(e['dur'])/1e6} for j,e in enumerate(sync)}}
        item['valid_ready_minus_sync3_arrival_ms']=selected['valid_count']['device_end_ms']-item['syncs']['3']['arrive_ms']
        item['sync3_return_minus_valid_copy_end_ms']=item['syncs']['3']['return_ms']-selected['valid_count']['device_end_ms']
        result.append(item)
    return {'steady_rounds':len(result),'copies':{role:{key:stats([x['copies'][role][key] for x in result]) for key in result[0]['copies'][role]} for role in result[0]['copies']},
            'syncs':{str(j):{key:stats([x['syncs'][str(j)][key] for x in result]) for key in ('arrive_ms','return_ms','wait_ms')} for j in (1,2,3)},
            'valid_ready_minus_sync3_arrival_ms':stats([x['valid_ready_minus_sync3_arrival_ms'] for x in result]),
            'sync3_return_minus_valid_copy_end_ms':stats([x['sync3_return_minus_valid_copy_end_ms'] for x in result]),
            'examples':result[:3],
            'method':'CPU memcpy scope -> torch_to_npu flow -> device MEMCPY_ASYNC; roles from enclosing original scopes/source. All times relative to the preceding main MODEL_EXECUTE. Copy duration excludes queue/dependency waiting.'}


if __name__=='__main__':
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);labels=sys.argv[3:] or ['baseline','early']
    result={label:inspect(root,label) for label in labels}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for label,v in result.items():
        print(label,'valid_copy', {k:x['mean'] for k,x in v['copies']['valid_count'].items()},
              'sync3',{k:x['mean'] for k,x in v['syncs']['3'].items()})
