"""Read-only CSV/CTE analysis plus CPU-to-NPU flow attribution."""
import argparse,bisect,collections,csv,json,pathlib,sqlite3,statistics
from decimal import Decimal
p=argparse.ArgumentParser();p.add_argument('capture');p.add_argument('output');a=p.parse_args()
folder=pathlib.Path(a.capture);out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
def ns(v):return int(Decimal(str(v).strip())*1000)
def read(name):
    with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def union(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if hi<=lo:continue
        if not result or lo>result[-1][1]:result.append([lo,hi])
        else:result[-1][1]=max(hi,result[-1][1])
    return result
def width(iv):return sum(hi-lo for lo,hi in union(iv))
kernels=read('kernel_details.csv');tasks=read('task_time.csv')
for k in kernels:k['_lo']=ns(k['Start Time(us)']);k['_hi']=k['_lo']+ns(k['Duration(us)'])
all_markers=sorted(ns(t['task_start(us)']) for t in tasks if t['kernel_type']=='MODEL_EXECUTE')
assert all_markers
# A merged draft has a separate graph: count target replays, not all graphs.
up=lambda k:k['Type']=='MatMulV2' and k['Input Shapes'].strip(chr(34)).endswith(',5120;34816,5120')
graph_up=collections.Counter(k['Model ID'] for k in kernels if up(k) and k['Model ID']!='4294967295')
assert graph_up
main_id=graph_up.most_common(1)[0][0]
first=min((k for k in kernels if k['Model ID']==main_id),key=lambda k:k['_lo'])
signature=lambda k:tuple(k[v] for v in ['Model ID','Stream ID','Task ID','Name'])
starts=sorted(k['_lo'] for k in kernels if signature(k)==signature(first))
markers=[all_markers[bisect.bisect_right(all_markers,s)-1] for s in starts]
assert len(set(markers))==len(markers) and markers==sorted(markers)
lo=markers[0];hi=max(k['_hi'] for k in kernels)
decode=[k for k in kernels if lo<=k['_lo']<hi];span=hi-lo
graph_ids=collections.Counter(k['Model ID'] for k in decode if k['Model ID']!='4294967295')
draft_ids=set(graph_ids)-{main_id}

groups=collections.defaultdict(list)
for k in decode:groups[(k['Type'],k['Input Shapes'].strip('"'),k['Model ID']==main_id)].append(k)
stats=[]
for (typ,shape,main),rs in groups.items():
    duration=sum(k['_hi']-k['_lo'] for k in rs)
    def weighted(field):return sum(float(k.get(field) or 0)*(k['_hi']-k['_lo']) for k in rs)/duration
    stats.append(dict(type=typ,input_shape=shape,main_graph=main,calls=len(rs),total_ms=duration/1e6,
       mean_us=duration/len(rs)/1000,mte2_activity_ratio=weighted('aic_mte2_ratio'),mac_activity_ratio=weighted('aic_mac_ratio')))
stats.sort(key=lambda x:-x['total_ms'])
cycles=[]
for start,end in zip(markers,markers[1:]+[hi]):
    rs=[k for k in decode if start<=k['_lo']<end]
    main=[k for k in rs if k['Model ID']==main_id]
    assert main
    assert sum(up(k) for k in main)==64, "Expected 64 target FFN layers per replay"
    graph_end=max(k['_hi'] for k in main)
    cycles.append(dict(span_ms=(end-start)/1e6,main_graph_window_ms=(graph_end-start)/1e6,
      main_graph_kernel_sum_ms=sum(k['_hi']-k['_lo'] for k in main)/1e6,
      after_graph_window_ms=(end-graph_end)/1e6,
      outside_graph_kernel_sum_ms=sum(k['_hi']-k['_lo'] for k in rs if k['Model ID']!=main_id)/1e6,
      no_compute_ms=((end-start)-width([(max(start,k['_lo']),min(end,k['_hi'])) for k in rs]))/1e6))
print('CSV_READY',len(decode),'kernels',len(markers),'rounds',flush=True)
macro='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)'''
sql=macro+'''
SELECT op_type, COUNT(*) AS calls, ROUND(SUM(duration_ns)/1000000.0,6) AS total_ms
FROM compute_view GROUP BY op_type ORDER BY total_ms DESC LIMIT 20;'''
db=next(folder.glob('ascend_pytorch_profiler*.db'))
with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as con:
    con.row_factory=sqlite3.Row
    top=[dict(x) for x in con.execute(sql)]
    tables=[x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print('DB_READY',len(top),'rows',flush=True)
trace=json.loads((folder/'trace_view.json').read_text(encoding='utf-8'))
events=trace['traceEvents'] if isinstance(trace,dict) else trace
labels=['DFlash2::selector_walk','DFlash2::candidate_selector','DFlash2::lm_head_and_topk',
        'draft_token','sample_token','post process','forward','prepare input','async_state_update']
scopes=collections.defaultdict(list);cpu=[];flow_start={};flow_end={};hardware=[]
for e in events:
    if 'ts' not in e:continue
    ph=e.get('ph');cat=e.get('cat');name=e.get('name','')
    if ph=='X' and e.get('dur'):
        start=ns(e['ts']);end=start+ns(e['dur'])
        if name in labels:scopes[(e['tid'],name)].append((start,end))
        if cat=='cpu_op' and end>lo and start<hi:cpu.append((start,end,name))
        if 'Task Type' in e.get('args',{}) and end>lo and start<hi:hardware.append((start,end,e))
    elif cat=='async_npu':
        if ph=='s':flow_start[e['id']]=(e['tid'],ns(e['ts']))
        elif ph=='f':flow_end[(e['pid'],e['tid'],ns(e['ts']))]=e['id']
indexed={key:(sorted(vals),sorted(x[0] for x in vals)) for key,vals in scopes.items()}
def owner(tid,t):
    for label in labels:
        vals,starts=indexed.get((tid,label),([],[]));i=bisect.bisect_right(starts,t)-1
        if i>=0 and vals[i][0]<=t<vals[i][1]:return label
    return 'unattributed'
attribution={};flow_hits=0
for start,end,e in hardware:
    fid=flow_end.get((e['pid'],e['tid'],start));src=flow_start.get(fid)
    if src:
        args=e['args'];stream=args.get('Physic Stream Id');task=args.get('Task Id')
        attribution[(str(stream),str(task),start)]=owner(*src);flow_hits+=1
categories=collections.defaultdict(lambda:dict(calls=0,kernel_sum_ms=0.0,intervals=[]))
for k in decode:
    label='main_verification_graph' if k['Model ID']==main_id else 'merged_draft_graph' if k['Model ID'] in draft_ids else attribution.get((k['Stream ID'],k['Task ID'],k['_lo']),'unattributed')
    k['_phase']=label;c=categories[label];c['calls']+=1;c['kernel_sum_ms']+=(k['_hi']-k['_lo'])/1e6;c['intervals'].append((k['_lo'],k['_hi']))
for c in categories.values():c['union_ms']=width(c.pop('intervals'))/1e6
busy_iv=union([(k['_lo'],k['_hi']) for k in decode]);busy=sum(b-a for a,b in busy_iv)
gaps=sorted([(b-a,a,b) for (_,a),(b,_) in zip(busy_iv,busy_iv[1:])],reverse=True)
samples=[]
for d,start,end in gaps[:5]:
    matches=sorted([(min(end,b)-max(start,a),name) for a,b,name in cpu if b>start and a<end],reverse=True)[:12]
    samples.append(dict(duration_ms=d/1e6,offset_ms=(start-lo)/1e6,
      overlapping_cpu_scopes=[dict(name=n,overlap_us=v/1000) for v,n in matches]))
phase_types=collections.defaultdict(lambda:dict(calls=0,total_ms=0.0))
for k in decode:
    key=(k['_phase'],k['Type']);phase_types[key]['calls']+=1;phase_types[key]['total_ms']+=(k['_hi']-k['_lo'])/1e6
cpu_stats={label:dict(calls=sum(1 for (tid,n),v in scopes.items() if n==label for s,e in v if s>=lo and s<hi),
    inclusive_ms=sum(max(0,min(e,hi)-max(s,lo)) for (tid,n),v in scopes.items() if n==label for s,e in v)/1e6) for label in labels}
result=dict(graph_replay_commands=len(all_markers), draft_graph_model_ids=sorted(draft_ids), target_ffn_calls_per_round=64, decode_rounds=len(markers),decode_span_ms=span/1e6,main_graph_model_id=main_id,
  compute_union_ms=busy/1e6,no_compute_ms=(span-busy)/1e6,no_compute_pct=100*(span-busy)/span,
  model_id_counts=dict(graph_ids),phase_device_totals=dict(categories),cpu_inclusive_scopes=cpu_stats,
  phase_device_types=[dict(phase=p,type=t,**v) for (p,t),v in sorted(phase_types.items(),key=lambda x:-x[1]['total_ms'])][:40],
  main_graph_kernel_sum_ms=sum(k['_hi']-k['_lo'] for k in decode if k['Model ID']==main_id)/1e6,
  cycle_medians={k:statistics.median(c[k] for c in cycles[:-1]) for k in cycles[0]},cycles=cycles,
  top_decode_groups=stats[:30],all_decode_groups=stats,largest_gaps=samples,
  whole_profile_compute_top=top,sql=sql,db_tables=tables,flow_linked_hardware_events=flow_hits,
  boundary='First hardware MODEL_EXECUTE to last compute completion; prefill excluded; final draft work retained.',
  limits=['Profiler timings are not unprofiled benchmark speed.',
    'CPU scopes are inclusive and overlap; do not add them to device time.',
    'Device groups contain cumulative kernel sums and interval unions; distinct phases may overlap.',
    'No-compute includes DMA/control/wait and is not automatically host idle.',
    'Target graph identity uses FFN up-projections, with exactly 64 per replay; target first-kernel boundaries exclude merged draft replays.'])
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['decode_rounds','decode_span_ms','no_compute_pct','cycle_medians','phase_device_totals']},ensure_ascii=False),flush=True)
