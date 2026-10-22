"""Offline CPU interval and host/device flow analysis of existing traces.

No NPU execution. CPU self time includes uninstrumented Python and scheduling;
it is not a direct CPU-busy measurement. Launch-delay bins describe timestamps,
not an assertion that every interval can be removed.
"""
import bisect
import collections
import hashlib
import json
from pathlib import Path
import sys
from analyze import load, rows, ns, union, width, stats


def overlap(a,b):
    i=j=total=0
    while i<len(a) and j<len(b):
        x,y=a[i];c,d=b[j];total+=max(0,min(y,d)-max(x,c))
        if y<=d:i+=1
        else:j+=1
    return total


def inspect(root,label='baseline'):
    receipt=load(root/'results'/f'{label}-profile.json')
    folder=root/receipt['profile']/'ASCEND_PROFILER_OUTPUT'
    tasks=rows(folder/'task_time.csv');ks=rows(folder/'kernel_details.csv')
    lo=min(ns(t['task_start(us)']) for t in tasks if t['kernel_type']=='MODEL_EXECUTE')
    ks=sorted([r for r in ks if ns(r['Start Time(us)'])>=lo],key=lambda r:ns(r['Start Time(us)']))
    rounds=sum('gateup_packed' in r['Name'] for r in ks)//64
    hi=max(ns(r['Start Time(us)'])+ns(r['Duration(us)']) for r in ks)
    trace=load(folder/'trace_view.json');events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    cpu=[];preps=[];all_preps=[];flowstart={};flowend={};cannstart={};cannend={}
    for e in events:
        ph=e.get('ph');name=e.get('name','')
        if ph in ('s','f'):
            if name=='torch_to_npu':
                if ph=='s':flowstart[e['id']]=e
                else:flowend[ns(e['ts'])]=e['id']
            elif e.get('cat')=='HostToDevice':
                if ph=='s':cannstart[name,e['id']]=e
                else:cannend[ns(e['ts'])]=(name,e['id'])
        if ph=='X' and e.get('cat')=='cpu_op':
            a=ns(e['ts']);b=a+ns(e.get('dur',0))
            if name=='prepare input':all_preps.append({'a':a,'b':b,'name':name,'tid':e['tid'],'pid':e['pid']})
            if b>lo and a<hi:
                item={'a':a,'b':b,'name':name,'tid':e['tid'],'pid':e['pid']}
                cpu.append(item)
                if name=='prepare input':preps.append(item)
    mainkey=(preps[0]['pid'],preps[0]['tid'])
    cpu=[e for e in cpu if (e['pid'],e['tid'])==mainkey]
    cpu.sort(key=lambda e:(e['a'],-e['b']))
    preps.sort(key=lambda e:e['a'])
    pstarts=[e['a'] for e in preps]
    prep_events=[[] for p in preps]
    for e in cpu:
        i=bisect.bisect_right(pstarts,e['a'])-1
        if i>=0 and e['b']<=preps[i]['b']:prep_events[i].append(e)
    # Subtract the union of all contained direct children on the same thread.
    # Includes CPU-op wrappers and sync ops; record neither GPU nor other-thread
    # time as an additional CPU cost.
    names=collections.defaultdict(lambda:{'count':0,'inclusive_ns':0,'self_ns':0})
    selfranges=collections.defaultdict(list)
    for group in prep_events:
        stack=[]
        for e in group:
            e['children']=[]
            while stack and e['a']>=stack[-1]['b']:stack.pop()
            if stack:
                assert e['b']<=stack[-1]['b'],('crossing cpu scopes',e['name'],stack[-1]['name'])
                stack[-1]['children'].append((e['a'],e['b']))
            stack.append(e)
        for e in group:
            item=names[e['name']];item['count']+=1;item['inclusive_ns']+=e['b']-e['a']
            item['self_ns']+=e['b']-e['a']-width(e['children'])
            cursor=e['a']
            for a,b in union(e['children']):
                if a>cursor:selfranges[e['name']].append((cursor,a))
                cursor=b
            if cursor<e['b']:selfranges[e['name']].append((cursor,e['b']))
    # Find the most specific aten operator and its enclosing prepare-input
    # interval at each CPU->device flow timestamp, offline.
    starts=sorted((ns(e['ts']),fid,e) for fid,e in flowstart.items()
                  if (e['pid'],e['tid'])==mainkey)
    active=[];index=0;cpu_flow={}
    for t,fid,flow in starts:
        while index<len(cpu) and cpu[index]['a']<=t:
            active.append(cpu[index]);index+=1
        active=[e for e in active if e['b']>=t]
        prep=[e for e in active if e['name']=='prepare input']
        aten=[e for e in active if e['name'].startswith('aten::')]
        if prep:
            parent=min(aten,key=lambda e:e['b']-e['a']) if aten else None
            cpu_flow[fid]=(t,parent)
    bins=collections.Counter();opbins=collections.defaultdict(lambda:collections.Counter())
    mapped=0;gap_total=0;previous=lo;examples=[];gaps=[]
    for r in ks:
        a,b=ns(r['Start Time(us)']),ns(r['Start Time(us)'])+ns(r['Duration(us)'])
        if a>previous:
            gap=a-previous;gap_total+=gap;gaps.append((previous,a))
            fid=flowend.get(a)
            if fid in cpu_flow:
                t,parent=cpu_flow[fid];mapped+=gap
                host=cannstart.get(cannend.get(a))
                c=ns(host['ts']) if host else None
                # Clipping keeps the four bins disjoint and summing to the gap,
                # including CPU-ahead cases where submission precedes the gap.
                opstart=parent['a'] if parent else t
                cuts=[previous]+[max(previous,min(a,x)) for x in (opstart,t,c if c is not None else t)]+[a]
                valid=cuts==sorted(cuts)
                if not valid:
                    bins['unresolved_nonmonotonic']+=gap
                else:
                    keys=('before_next_aten_call','aten_to_cpu_enqueue','enqueue_to_cann_launch','cann_launch_to_device')
                    op=parent['name'] if parent else '(no aten parent)'
                    for k,x,y in zip(keys,cuts,cuts[1:]):bins[k]+=y-x;opbins[op][k]+=y-x
                    opbins[op]['gap_total']+=gap;opbins[op]['count']+=1
                    examples.append({'next_aten':op,'next_kernel':r['Name'],'gap_us':gap/1000,
                                     'before_aten_us':(cuts[1]-cuts[0])/1000,
                                     'enqueue_to_launch_us':(cuts[3]-cuts[2])/1000,
                                     'relative_device_ms':(a-lo)/1e6})
        previous=max(previous,b)
    steady=[group for group in prep_events if sum(e['name']=='aten::sub' for e in group)==23]
    assert sum(v['self_ns'] for v in names.values())==sum(p['b']-p['a'] for p in preps)
    assert sum(bins.values())==mapped
    steady_names=collections.defaultdict(list)
    syncs=collections.defaultdict(list)
    for group in steady:
        counts=collections.Counter(e['name'] for e in group)
        for name,count in counts.items():steady_names[name].append(count)
        sync=[e for e in group if e['name']=='Event::synchronize']
        assert len(sync)==3
        for i,e in enumerate(sync):syncs[i+1].append((e['b']-e['a'])/1e6)
    # Align the sequence of CPU preparations to the captured model pairs.
    # Raw trace has one eager prefill, N decode preparations and one final
    # cancellation/empty step. First decode prep precedes the device ROI.
    all_preps.sort(key=lambda p:p['a'])
    assert len(all_preps)==rounds+2
    markers=sorted(ns(t['task_start(us)']) for t in tasks if t['kernel_type']=='MODEL_EXECUTE')
    assert len(markers)==rounds*2
    summary=load(root/'summary-server.json')['captures'][label]
    nativeids=(summary['main_model_id'],summary['draft_model_id'])
    bounds=[]
    for i,start in enumerate(markers):
        end=markers[i+1] if i+1<len(markers) else hi
        native=[r for r in ks if start<=ns(r['Start Time(us)'])<end and r['Model ID'] in nativeids]
        ids={r['Model ID'] for r in native}
        assert ids=={nativeids[i%2]}
        bounds.append((start,max(ns(r['Start Time(us)'])+ns(r['Duration(us)']) for r in native)))
    afterdraft=[];timelines=[]
    for i in range(rounds-1):
        p=all_preps[i+2];draft=bounds[2*i+1];nextmain=bounds[2*i+2]
        afterdraft.append((draft[1],nextmain[0]))
        timelines.append({'round':i,'main_graph_start_ms':(bounds[2*i][0]-lo)/1e6,
                          'draft_graph_start_ms':(draft[0]-lo)/1e6,'draft_graph_end_ms':(draft[1]-lo)/1e6,
                          'next_prepare_start_ms':(p['a']-lo)/1e6,'next_prepare_end_ms':(p['b']-lo)/1e6,
                          'next_main_start_ms':(nextmain[0]-lo)/1e6,
                          'host_prep_end_minus_draft_end_ms':(p['b']-draft[1])/1e6})
    return {'rounds':rounds,'decode_ms':(hi-lo)/1e6,'all_gaps_ms':gap_total/1e6,
            'prepare_input_following_launch_gap_ms':mapped/1e6,
            'gap_bins_ms':{k:v/1e6 for k,v in bins.items()},
            'next_aten_gap_top20':[{'name':k,**{j:(v if j=='count' else v/1e6) for j,v in d.items()}} for k,d in sorted(opbins.items(),key=lambda kv:kv[1]['gap_total'],reverse=True)[:20]],
            'cpu_self_top20':[{'name':k,'count':v['count'],'inclusive_ms':v['inclusive_ns']/1e6,'self_ms':v['self_ns']/1e6,'self_ms_per_round':v['self_ns']/1e6/rounds} for k,v in sorted(names.items(),key=lambda kv:kv[1]['self_ns'],reverse=True)[:20]],
            'cpu_self_during_gaps_top20':sorted([{'name':k,'gap_overlap_ms':overlap(gaps,union(v))/1e6} for k,v in selfranges.items()],key=lambda e:e['gap_overlap_ms'],reverse=True)[:20],
            'prepare_scope_us':stats([(p['b']-p['a'])/1000 for p in preps]),
            'steady_preparations':len(steady),
            'steady_cpu_op_sequence_hashes':dict(collections.Counter(hashlib.sha256('\n'.join(e['name'] for e in group).encode()).hexdigest() for group in steady)),
            'steady_counts_per_preparation':{k:stats(v) for k,v in steady_names.items() if k in ('aten::sub','aten::slice','aten::copy_','aten::as_strided','aten::fill_','aten::index_select','aten::arange')},
            'steady_sync_duration_ms_by_order':{k:stats(v) for k,v in syncs.items()},
            'steady_prepare_cpu_wall_ms':stats([(group[0]['b']-group[0]['a'])/1e6 for group in steady]),
            'steady_prepare_minus_event_wait_ms':stats([((group[0]['b']-group[0]['a'])-sum(e['b']-e['a'] for e in group if e['name']=='Event::synchronize'))/1e6 for group in steady]),
            'gaps_after_draft_before_next_main_ms':overlap(gaps,union(afterdraft))/1e6,
            'host_prep_end_minus_previous_draft_end_ms':stats([t['host_prep_end_minus_draft_end_ms'] for t in timelines]),
            'first_three_round_timelines':timelines[:3],
            'largest_examples':sorted(examples,key=lambda e:e['gap_us'],reverse=True)[:10],
            'notes':['CPU self wall time includes scheduling/GC/profiler and uninstrumented code, not necessarily busy computation.',
                     'Flow phase attribution identifies who launches the next task, not by itself the cause of the delay.',
                     'Same-service scheduling A/B; profiling overhead is separate from unprofiled throughput.']}


if __name__=='__main__':
    root=Path(sys.argv[1])
    result={label:inspect(root,label) for label in ('baseline','early')}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
