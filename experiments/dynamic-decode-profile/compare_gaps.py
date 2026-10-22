"""Offline, identical CSV interval accounting for historical/current captures."""
import bisect
import collections
import json
from pathlib import Path
import sys
from analyze import load, rows, ns, union, width


def intersect(a,b):
    i=j=total=0
    while i<len(a) and j<len(b):
        x,y=a[i];c,d=b[j]
        total+=max(0,min(y,d)-max(x,c))
        if y<=d:i+=1
        else:j+=1
    return total


def read(root):
    receipt=load(root/'results/PipeUtilization.json')
    folder=root/receipt['profile']/'ASCEND_PROFILER_OUTPUT'
    kernels=rows(folder/'kernel_details.csv');tasks=rows(folder/'task_time.csv')
    lo=min(ns(r['task_start(us)']) for r in tasks if r['kernel_type']=='MODEL_EXECUTE')
    ks=[r for r in kernels if ns(r['Start Time(us)'])>=lo]
    hi=max(ns(r['Start Time(us)'])+ns(r['Duration(us)']) for r in ks)
    covered=union([(ns(r['Start Time(us)']),ns(r['Start Time(us)'])+ns(r['Duration(us)'])) for r in ks])
    gaps=[];prev=lo
    for a,b in covered:
        if a>prev:gaps.append((prev,a))
        prev=b
    ends=sorted((ns(r['Start Time(us)'])+ns(r['Duration(us)']),r['Name']) for r in ks)
    starts=sorted((ns(r['Start Time(us)']),r['Name']) for r in ks)
    et=[x[0] for x in ends];st=[x[0] for x in starts]
    edges=collections.defaultdict(lambda:[0,0]); kinds=collections.defaultdict(list)
    for a,b in gaps:
        key=(ends[bisect.bisect_right(et,a)-1][1],starts[bisect.bisect_left(st,b)][1])
        edges[key][0]+=1;edges[key][1]+=b-a
    for t in tasks:
        a,b=ns(t['task_start(us)']),ns(t['task_stop(us)'])
        if b>lo and a<hi:kinds[t['kernel_type']].append((max(a,lo),min(b,hi)))
    rounds=sum('gateup_packed' in r['Name'] for r in ks)//64
    head=[r for r in ks if r['Type']=='MatMulV2' and r['Input Shapes'].strip('"')=='8,5120;248320,5120']
    return {'rounds':rounds,'decode_ms':(hi-lo)/1e6,'compute_free_ms':width(gaps)/1e6,
            'compute_free_percent':100*width(gaps)/(hi-lo),'compute_free_ms_per_round':width(gaps)/1e6/rounds,
            'main_head_ms_per_round':sum(ns(r['Duration(us)']) for r in head)/1e6/rounds,
            'top_gap_edges':[{'previous':k[0],'next':k[1],'count':v[0],'gap_ms':v[1]/1e6,'gap_ms_per_round':v[1]/1e6/rounds} for k,v in sorted(edges.items(),key=lambda kv:kv[1][1],reverse=True)[:20]],
            'task_type_gap_overlap':[{'task_type':k,'calls':len(v),'overlap_ms':intersect(gaps,union(v))/1e6} for k,v in sorted(kinds.items())],
            'note':'Task types may overlap each other and must not be added. Historical/current configurations differ; this is interval accounting, not causal A/B.'}


if __name__=='__main__':
    result={'historical':read(Path(sys.argv[1])),'current':read(Path(sys.argv[2]))}
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
