"""Audit inter-layer windows from an existing native MTP7 capture, without NPU work.

Usage: python analyze.py PROFILE_FOLDER PRIVATE_OUTPUT FILTERED_EVIDENCE
CSV supplies task intervals; read-only SQLite aggregates cross-check the export.
Compiled Insight replies are separately verified against the same CSV.
"""
import bisect,collections,csv,hashlib,json,sqlite3,statistics,sys
from decimal import Decimal
from pathlib import Path

profile,private,output=map(Path,sys.argv[1:4]);private.mkdir(parents=True,exist_ok=True)
folder=profile/'ASCEND_PROFILER_OUTPUT'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ns(s):return int(Decimal(str(s).strip())*1000)
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def union(iv):
    result=[]
    for a,b in sorted(iv):
        if b<=a:continue
        if not result or a>result[-1][1]:result.append([a,b])
        else:result[-1][1]=max(result[-1][1],b)
    return result
def width(iv):return sum(b-a for a,b in union(iv))
def stats(vals):
    vals=sorted(vals)
    return {'count':len(vals),'min':min(vals),'median':statistics.median(vals),
            'mean':statistics.mean(vals),'p95':vals[min(len(vals)-1,int(.95*len(vals)))],'max':max(vals),'sum':sum(vals)}

info=read(next(profile.glob('profiler_info*.json')))
assert info['end_info']['collectionTimeEnd'] and info['config']['common_config']['record_shapes']
assert info['config']['experimental_config']['_aic_metrics']=='ACL_AICORE_PIPE_UTILIZATION'
assert (folder/'trace_view.json').exists() and list(folder.glob('ascend_pytorch*.db'))
rows=csvrows(folder/'kernel_details.csv');tasks=csvrows(folder/'task_time.csv')
for k in rows:
    k['lo']=ns(k['Start Time(us)']);k['hi']=k['lo']+ns(k['Duration(us)']);k['shape']=k['Input Shapes'].strip('"')
rows.sort(key=lambda x:x['lo'])
row_starts=[k['lo'] for k in rows]
max_kernel_duration=max(k['hi']-k['lo'] for k in rows)
dma_intervals=[(ns(t['task_start(us)']),ns(t['task_stop(us)'])) for t in tasks if t['kernel_type']=='MEMCPY_ASYNC']
all_markers=sorted(ns(t['task_start(us)']) for t in tasks if t['kernel_type']=='MODEL_EXECUTE')
main_id=collections.Counter(k['Model ID'] for k in rows if k['shape']=='8,17408;5120,17408').most_common(1)[0][0]
markers=[t for t in all_markers if rows[bisect.bisect_left(row_starts,t)]['Model ID']==main_id]
assert len(markers)==26 and len(all_markers)==52
main=[k for k in rows if k['Model ID']==main_id]
down=[k for k in main if k['shape']=='8,17408;5120,17408']
up=[k for k in main if k['shape']=='8,5120;34816,5120']
projections=[k for k in main if k['shape'] in ['8,5120;16384,5120','8,5120;14336,5120']]
assert len(down)==len(up)==len(projections)==64*len(markers)
patterns=collections.Counter();boundaries=[];rounds=[];main_windows=[]
for n,start in enumerate(markers):
    end=markers[n+1] if n+1<len(markers) else max(k['hi'] for k in rows)
    rs=[k for k in main if start<=k['lo']<end]
    ds=[k for k in down if start<=k['lo']<end];us=[k for k in up if start<=k['lo']<end];ps=[k for k in projections if start<=k['lo']<end]
    assert len(ds)==len(us)==len(ps)==64
    assert all(p['shape']==('8,5120;14336,5120' if i%4==3 else '8,5120;16384,5120') for i,p in enumerate(ps))
    main_windows.append((rs[0]['lo'],rs[-1]['hi']))
    for i in range(64):assert ps[i]['lo']<us[i]['lo']<ds[i]['lo']
    for i in range(63):
        a,b=ds[i]['hi'],ps[i+1]['lo'];assert b>=a
        middle=[k for k in rows[bisect.bisect_left(row_starts,a-max_kernel_duration):bisect.bisect_left(row_starts,b)] if k['hi']>a]
        pattern=tuple(k['Type'] for k in middle);patterns[pattern]+=1
        busy=width([(max(a,k['lo']),min(b,k['hi'])) for k in middle])
        # Locate non-compute hardware tasks, while excluding graph-spanning MODEL_EXECUTE markers.
        dma=[(lo,hi) for lo,hi in dma_intervals if lo<b and hi>a]
        boundary={'round':n,'from_layer':i,'to_layer':i+1,'window_us':(b-a)/1000,'compute_us':busy/1000,
            'no_compute_us':((b-a)-busy)/1000,'intervening_types':list(pattern),'dma_tasks':len(dma),
            'dma_overlap_us':width([(max(a,lo),min(b,hi)) for lo,hi in dma])/1000,
            'previous_down_us':ns(ds[i]['Duration(us)'])/1000,
            'next_projection_us':ns(ps[i+1]['Duration(us)'])/1000,
            'next_projection_weight_mib':int(ps[i+1]['shape'].split(';')[1].split(',')[0])*5120*2/2**20,
            'previous_down_mte2_ratio':float(ds[i]['aic_mte2_ratio']),
            'previous_down_mac_ratio':float(ds[i]['aic_mac_ratio'])}
        boundaries.append(boundary)
    rounds.append({'round':n,'main_graph_window_ms':(rs[-1]['hi']-rs[0]['lo'])/1e6,
        'layer_projection_to_last_down_ms':(ds[-1]['hi']-ps[0]['lo'])/1e6})

# Validate all compiled Insight operator groups against exported timing records.
groups=collections.defaultdict(list)
for k in rows:groups[k['Type'],k['Name'],k['Accelerator Core'],k['Input Shapes']].append(k)
backend=read(private/'cross-layer-native-backend-operators.json');assert backend['result']
assert len(backend['body']['data'])==len(groups)
for item in backend['body']['data']:
    item=item.get('compare',item);selected=groups[item['opType'],item['opName'],item['accCore'],item['inputShape']]
    assert len(selected)==int(item['count'])
    assert abs(sum(ns(k['Duration(us)'])/1000 for k in selected)-float(item['totalTime']))<.011

macro='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)'''
sql=macro+'''\nSELECT op_type,COUNT(*) AS calls,SUM(duration_ns) AS total_ns
FROM compute_view GROUP BY op_type ORDER BY total_ns DESC LIMIT 20;'''
with sqlite3.connect(next(folder.glob('ascend_pytorch*.db')).as_uri()+'?mode=ro',uri=True) as con:
    con.row_factory=sqlite3.Row;top=[dict(x) for x in con.execute(sql)]
    for x in top:
        selected=[k for k in rows if k['Type']==x['op_type']]
        assert len(selected)==x['calls'] and abs(sum(k['hi']-k['lo'] for k in selected)-x['total_ns'])<200
    pmu=con.execute('''SELECT s.value AS metric,COUNT(*) AS entries FROM TASK_PMU_INFO p
LEFT JOIN STRING_IDS s ON s.id=p.name GROUP BY s.value ORDER BY entries DESC LIMIT 40''').fetchall()
    tables=[x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

start=markers[0];end=max(k['hi'] for k in rows);decode=[k for k in rows if k['lo']>=start]
span=end-start
end_to_start=stats([x['window_us'] for x in boundaries]);idle=stats([x['no_compute_us'] for x in boundaries])
result={'model':'Qwen3.8-27B','hardware':'Ascend910B2','dtype':'BF16','TP':1,'K':7,
    'source':'Existing native MTP7 profile; no new model requests, training or kernel changes.',
    'profiler_valid':True,'profiler_config':info['config'],'new_npu_requests':0,
    'kernel_csv_sha256':hashlib.sha256((folder/'kernel_details.csv').read_bytes()).hexdigest(),
    'main_model_id':main_id,'verification_rounds':len(markers),'layers_per_round':64,
    'boundaries_analyzed':len(boundaries),'boundary_patterns':[{'types':list(k),'count':v} for k,v in patterns.items()],
    'down_end_to_next_projection_start_us':end_to_start,'no_compute_in_boundary_us':idle,
    'boundary_dma_tasks':sum(x['dma_tasks'] for x in boundaries),
    'boundary_dma_overlap_us':sum(x['dma_overlap_us'] for x in boundaries),
    'next_projection_kernel_us':stats([x['next_projection_us'] for x in boundaries]),
    'previous_down_mte2_ratio':stats([x['previous_down_mte2_ratio'] for x in boundaries]),
    'previous_down_mac_ratio':stats([x['previous_down_mac_ratio'] for x in boundaries]),
    'decode_span_ms':span/1e6,'main_graph_window_sum_ms':sum(b-a for a,b in main_windows)/1e6,
    'boundary_window_percent_of_decode':100*end_to_start['sum']*1000/span,
    'boundary_no_compute_percent_of_decode':100*idle['sum']*1000/span,
    'optimistic_bound':'Only for prefetch confined to the measured down-end to next-projection-start interval: time hidden cannot exceed the interval, even assuming all its HBM bandwidth is available and zero prefetch overhead. This is not a bound on prefetch started earlier inside the down kernel.',
    'projection_weight_mib':sorted(set(x['next_projection_weight_mib'] for x in boundaries)),
    'insight_operator_groups_verified':len(groups),'sql':sql,'sql_top20':top,
    'pmu_metrics_available':[dict(x) for x in pmu],'db_tables':tables,
    'example_first_boundary':boundaries[0],
    'limits':['Layer identities are reconstructed from 64 ordered attention projections and FFN pairs per main graph, not named module scopes.',
        'PIPE_UTILIZATION gives per-task activity totals, not a time-resolved HBM-idle trace.',
        'No-compute is not automatically no-DMA or no-HBM traffic; vector operators themselves read/write data.',
        'No measured speedup: no cross-layer prefetch implementation was run.']}
(private/'boundaries.json').write_text(json.dumps(boundaries,indent=2),encoding='utf-8')
# Align a bounded Insight timeline request to CSV using its first exact-name matmul.
search=read(private/'timeline-search.json');first=search['body']['searchAllSlicesDetails'][0]
first_csv=next(k for k in rows if k['Name']==first['name'])
assert ns(first_csv['Duration(us)'])==first['duration']
origin=first_csv['lo']-first['timestamp']
ds=down[0];ps=projections[1];window={'cardId':read(private/'cross-layer-native-backend-import.json')['body']['result'][0]['rankId'],
    'dbPath':read(private/'cross-layer-native-backend-import.json')['body']['result'][0]['dbPath'],
    'processId':first['pid'],'threadId':ds['Stream ID'],'metaType':first['metaType'],
    'startTime':ds['lo']-origin,'endTime':ps['hi']-origin,'timePerPx':0}
(private/'timeline-window-request.json').write_text(json.dumps(window,indent=2),encoding='utf-8')
(private/'timeline-window-expected.json').write_text(json.dumps([{'name':k['Name'],'startTime':k['lo']-origin,'duration':k['hi']-k['lo']} for k in rows if ds['lo']<=k['lo']<ps['hi']],indent=2),encoding='utf-8')
timeline_path=private/'timeline-boundary.json'
if timeline_path.exists():
    timeline=read(timeline_path);assert timeline['result']
    actual=sorted([{'name':k['name'],'startTime':k['startTime'],'duration':k['duration']}
        for lane in timeline['body']['data'] for k in lane],key=lambda k:k['startTime'])
    expected=read(private/'timeline-window-expected.json')
    assert actual==expected
    result['insight_boundary_timeline_slices_verified']=len(actual)
    result['example_timeline_relative_us']=[{'name':k['name'],'start_us':(k['startTime']-actual[0]['startTime'])/1000,
                                            'duration_us':k['duration']/1000} for k in actual]
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['boundaries_analyzed','boundary_patterns','down_end_to_next_projection_start_us','no_compute_in_boundary_us','decode_span_ms','boundary_window_percent_of_decode','previous_down_mte2_ratio']},indent=2))
