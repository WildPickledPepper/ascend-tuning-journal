"""Map the following kernel's launch to its existing CPU phase using trace flows.

The phase identifies who submits the next task, not the cause of the wait.
"""
import bisect
import collections
import json
from pathlib import Path
import sys
from analyze import load, rows, ns, stats


def inspect(root):
    receipt=load(root/'results/PipeUtilization.json')
    folder=root/receipt['profile']/'ASCEND_PROFILER_OUTPUT'
    tasks=rows(folder/'task_time.csv');ks=rows(folder/'kernel_details.csv')
    lo=min(ns(t['task_start(us)']) for t in tasks if t['kernel_type']=='MODEL_EXECUTE')
    ks=sorted([r for r in ks if ns(r['Start Time(us)'])>=lo],key=lambda r:ns(r['Start Time(us)']))
    hi=max(ns(r['Start Time(us)'])+ns(r['Duration(us)']) for r in ks)
    trace=load(folder/'trace_view.json');events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    flowstart={e['id']:e for e in events if e.get('ph')=='s' and e.get('name')=='torch_to_npu'}
    flowend={ns(e['ts']):e['id'] for e in events if e.get('ph')=='f' and e.get('name')=='torch_to_npu'}
    phases={'prepare input','draft_token','sample_token','dynamic_vocab.update','dynamic_vocab.repack','postprocess','model_forward'}
    scopes=collections.defaultdict(list)
    for e in events:
        if e.get('ph')=='X' and e.get('name') in phases:
            scopes[e['pid'],e['tid']].append((ns(e['ts']),ns(e['ts'])+ns(e.get('dur',0)),e['name']))
    result=collections.defaultdict(lambda:[0,0]);top=[];previous=lo
    for r in ks:
        a,b=ns(r['Start Time(us)']),ns(r['Start Time(us)'])+ns(r['Duration(us)'])
        if a>previous:
            gap=a-previous;fid=flowend.get(a);start=flowstart.get(fid)
            labels=[]
            if start:
                ts=ns(start['ts'])
                labels=[label for x,y,label in scopes[start['pid'],start['tid']] if x<=ts<=y]
            label=' + '.join(sorted(labels)) or 'unmapped_or_unlabelled'
            result[label][0]+=1;result[label][1]+=gap
            top.append({'next_kernel':r['Name'],'gap_us':gap/1000,'next_launch_phase':label})
        previous=max(previous,b)
    cpu={}
    for label in phases:
        vals=[(y-x)/1000 for ranges in scopes.values() for x,y,k in ranges if k==label and lo<=x<hi]
        if vals:cpu[label]=stats(vals)
    return {'gap_by_following_task_launch_phase':[{'phase':k,'calls':v[0],'gap_ms':v[1]/1e6} for k,v in sorted(result.items(),key=lambda kv:kv[1][1],reverse=True)],
            'largest_gaps':sorted(top,key=lambda x:x['gap_us'],reverse=True)[:10],
            'cpu_scope_wall_us':cpu,
            'note':'Flow-based ownership of the next launch. Does not prove that the CPU phase caused the entire gap or quantify profiler perturbation.'}


if __name__=='__main__':
    result={'historical':inspect(Path(sys.argv[1])),'current':inspect(Path(sys.argv[2]))}
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
