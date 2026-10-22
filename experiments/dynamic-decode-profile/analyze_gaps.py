"""Read-only location of compute-free intervals; no root cause inferred."""
import bisect
import collections
import json
from pathlib import Path
import sys
from analyze import load, rows, ns, union, width, stats


def run(root):
    results = {}
    summary = load(root/'summary-server.json')
    for label in ('PipeUtilization','LongPipeUtilization'):
        receipt = load(root/'results'/f'{label}.json')
        folder = root/receipt['profile']/'ASCEND_PROFILER_OUTPUT'
        kernels, tasks = rows(folder/'kernel_details.csv'), rows(folder/'task_time.csv')
        kernels.sort(key=lambda r:ns(r['Start Time(us)']))
        starts = sorted(ns(t['task_start(us)']) for t in tasks if t['kernel_type']=='MODEL_EXECUTE')
        lo = starts[0]
        kernels = [r for r in kernels if ns(r['Start Time(us)'])>=lo]
        hi = max(ns(r['Start Time(us)'])+ns(r['Duration(us)']) for r in kernels)
        windows=[]
        cap=summary['captures'][label]
        for i,a in enumerate(starts):
            b=starts[i+1] if i+1<len(starts) else hi
            native=[r for r in kernels if a<=ns(r['Start Time(us)'])<b and r['Model ID'] in (cap['main_model_id'],cap['draft_model_id'])]
            windows.append((a,max(ns(r['Start Time(us)'])+ns(r['Duration(us)']) for r in native)))
        covered=union([(ns(r['Start Time(us)']),ns(r['Start Time(us)'])+ns(r['Duration(us)'])) for r in kernels])
        gaps=[]; prev=lo
        for a,b in covered:
            if a>prev:gaps.append((prev,a))
            prev=b
        inside=sum(max(0,min(b,d)-max(a,c)) for a,b in gaps for c,d in windows)
        # Attribute by adjacent device kernel names, rather than CPU scopes that
        # can return before asynchronous device work begins.
        ends=sorted((ns(r['Start Time(us)'])+ns(r['Duration(us)']),r['Name']) for r in kernels)
        begin=sorted((ns(r['Start Time(us)']),r['Name']) for r in kernels)
        end_times=[x[0] for x in ends]; begin_times=[x[0] for x in begin]
        edges=collections.defaultdict(lambda:[0,0])
        for a,b in gaps:
            left=ends[bisect.bisect_right(end_times,a)-1][1]
            right=begin[bisect.bisect_left(begin_times,b)][1]
            item=edges[left,right];item[0]+=1;item[1]+=b-a
        trace=load(folder/'trace_view.json')
        events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
        hosts=collections.defaultdict(list)
        scope_ranges=collections.defaultdict(list)
        kinds=collections.Counter()
        for e in events:
            if e.get('ph')!='X' or not lo<=ns(e.get('ts',0))<hi:continue
            kinds[e.get('cat','')]+=1
            if e.get('name') in ('draft_token','sample_token','dynamic_vocab.update','dynamic_vocab.repack'):
                a=ns(e['ts']);scope_ranges[e['name']].append((a,a+ns(e.get('dur',0))))
            if any(x in e.get('name','').lower() for x in ('synchron','streamwait','item','dtoh','devicetohost')):
                hosts[e['name']].append(float(e.get('dur',0)))
        results[label]={'decode_window_ms':(hi-lo)/1e6,'no_compute_ms':width(gaps)/1e6,
            'inside_model_graph_gap_ms':inside/1e6,'outside_model_graph_gap_ms':(width(gaps)-inside)/1e6,
            'gap_edges_top20':[{'previous':k[0],'next':k[1],'count':v[0],'gap_ms':v[1]/1e6}
                for k,v in sorted(edges.items(),key=lambda x:x[1][1],reverse=True)[:20]],
            'host_sync_candidates_top20':[{'name':k,'cpu_wall_us':stats(v)} for k,v in sorted(hosts.items(),key=lambda x:sum(x[1]),reverse=True)[:20]],
            'gap_overlap_with_host_scope_ms':{name:sum(max(0,min(b,d)-max(a,c)) for a,b in gaps for c,d in union(ranges))/1e6 for name,ranges in scope_ranges.items()},
            'trace_categories':dict(kinds),
            'note':'Adjacent kernel boundaries locate gaps, not their cause. Host scopes are asynchronous, nested and instrumented; sums must not be added to device times.'}
    (root/'gap-analysis.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
