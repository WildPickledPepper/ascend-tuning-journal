"""Decode documented CANN v5 SMMU records and cross-check msprof exports."""
import collections,csv,hashlib,json,pathlib,re,sqlite3,statistics,struct,sys

root=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
initial=csv_rows(next((root/'profile-tlb-full').rglob('kernel_details.csv')))
shapes={name:next(k['Input Shapes'] for k in initial if k['Type']=='MatMulV2' and token in k['Input Shapes'] and k['Input Shapes'].strip(chr(34)).startswith(prefix)) for name,token,prefix in [('up','34816','8,5120;'),('down','17408','8,17408;')]}

def mean(rows,key):return statistics.mean(float(x[key]) for x in rows)
def tail_stable(rows):return rows[len(rows)//3:]

result={'dtype':'BF16','physical_device':6,'M':8,'weight_format':'ND','custom_kernel':False,
 'counter_source':'msprof --l2 / PyTorch Profiler l2_cache=True; native socpmu.data records',
 'smmu_record_format':'<8B4H8Q, 80 bytes, installed CANN platform_version=5',
 'smmu_events':['0x2','0x8a','0x8b','0x8c','0x8d'],
 'formulas':{'requests':'event[1] (0x8a)','misses':'event[0] (0x2)',
             'hits':'event[3]+event[4] (0x8c+0x8d)','miss_rate':'sum(misses)/sum(requests)'},
 'stability_selection':'Six graph replays; exclude first two replays within each shape/position group',
 'profiles':{},'l2_cache':{},'unprofiled_timing':{},'source_hashes':{}}

for order in ['small','full','sweep2','adjacent']:
    p=root/('profile-tlb-'+order)
    kernels=csv_rows(next(p.rglob('kernel_details.csv')))
    mapping={(int(x['Stream ID']),int(x['Task ID'])):x for x in kernels if 'MatMul' in x['Name']}
    assert len(mapping)==(128 if order in ['small','full'] else 256)
    soc=csv_rows(next(p.rglob('soc_pmu.csv')))
    exported=collections.defaultdict(list)
    for row in soc:
        exported[int(row['Stream Id']),int(row['Task Id'])].append(float(row['TLB Miss Rate']))
    by=collections.defaultdict(list);raw_by_key=collections.defaultdict(list)
    for f in p.rglob('socpmu.data.*'):
        if not re.search(r'\.slice_\d+$',f.name):continue
        raw=f.read_bytes();assert len(raw)%80==0
        result['source_hashes'][order+'/'+f.name]=hashlib.sha256(raw).hexdigest()
        for record in struct.iter_unpack('<8B4H8Q',raw):
            if record[11]!=2:continue
            sid,tid=record[9:11];values=record[12:]
            if values[1]:raw_by_key[sid,tid].append(values[0]/values[1])
            if (sid,tid) not in mapping:continue
            kernel=mapping[sid,tid]
            group=(kernel['Input Shapes'],tid%2 if order=='adjacent' else 'all')
            assert sum(values[i] for i in [0,3,4])==values[1],values
            by[group].append(values)
    for key,rates in raw_by_key.items():
        assert len(rates)==len(exported[key]),(order,key,len(rates),len(exported[key]))
        assert all(abs(a-b)<.00000051 for a,b in zip(sorted(rates),sorted(exported[key]))),(order,key)
    data={}
    for name,shape in shapes.items():
        data[name]={}
        for position in ([0,1] if order=='adjacent' else ['all']):
            values=tail_stable(by[shape,position])
            k=[x for x in kernels if x['Input Shapes']==shape and (position=='all' or int(x['Task ID'])%2==position)]
            k=tail_stable(k)
            assert len(values)==len(k)
            data[name][str(position)]={'samples':len(values),
              'requests_per_call':statistics.mean(x[1] for x in values),
              'misses_per_call':statistics.mean(x[0] for x in values),
              'miss_rate':sum(x[0] for x in values)/sum(x[1] for x in values),
              'mean_duration_us':mean(k,'Duration(us)'),
              'mean_mte2_us':mean(k,'aic_mte2_time(us)'),
              'mean_mac_us':mean(k,'aic_mac_time(us)')}
    result['profiles'][order]=data
    result['unprofiled_timing'][order]=read(root/('timing-'+order+'.json'))['median_us_per_call']

    rows=csv_rows(next((root/('profile-l2-'+order)).rglob('kernel_details.csv')))
    cache={}
    for name,shape in shapes.items():
        cache[name]={}
        for position in ([0,1] if order=='adjacent' else ['all']):
            g=tail_stable([x for x in rows if x['Input Shapes']==shape and (position=='all' or int(x['Task ID'])%2==position)])
            hits=sum(float(x['aic_r0_read_cache_hit'])+float(x['aic_r1_read_cache_hit']) for x in g)
            misses=sum(float(x['aic_r0_read_cache_miss_allocate'])+float(x['aic_r1_read_cache_miss_allocate']) for x in g)
            cache[name][str(position)]={'samples':len(g),'read_hit_rate':hits/(hits+misses),
              'hits_per_call':hits/len(g),'misses_per_call':misses/len(g)}
    result['l2_cache'][order]=cache

# All orders retain the same tensor addresses for a given actual model layer.
plans={order:read(root/('tlb-'+order+'.json')) for order in ['small','full','sweep2','adjacent']}
fullmap={v['name']:v for v in plans['full']['tensors']}
for order,plan in plans.items():
    for v in plan['tensors']:
        assert all(v[key]==fullmap[v['name']][key] for key in ['x','w','y','shape','x_stride','w_stride'])
result['same_addresses_verified']=True
result['same_layer1_addresses']=[v for v in fullmap.values() if '.layers.1.mlp.' in v['name']]
result['working_sets']={order:{
    'calls_per_graph':plan['calls_per_graph'],
    'total_logical_weight_bytes':sum(v['shape'][0]*v['shape'][1]*2 for v in plan['tensors']),
    'distinct_logical_weight_bytes':sum(v['shape'][0]*v['shape'][1]*2
        for v in {v['w']:v for v in plan['tensors']}.values())}
    for order,plan in plans.items()}
result['capture_utc']={order:{k:plan[k] for k in ['started_utc','completed_utc']}
                       for order,plan in plans.items()}

# Backend statistics aggregate all six replays, unlike the stable four above.
# Pass the backend output directory as the optional third argument.
if len(sys.argv)>3:
    backend_root=pathlib.Path(sys.argv[3]);checks=[]
    for kind in ['tlb','l2']:
        for order in ['small','full','sweep2','adjacent']:
            label=kind+'-'+order
            backend=read(backend_root/(label+'-backend-matmuls.json'))
            kernels=csv_rows(next((root/('profile-'+label)).rglob('kernel_details.csv')))
            assert len(backend)==2
            for row in backend:
                selected=[x for x in kernels if x['Input Shapes']==row['inputShape']]
                assert int(row['count'])==len(selected)
                avg=mean(selected,'Duration(us)')
                assert abs(float(row['avgTime'])-avg)<.011,(label,row['inputShape'],avg,row['avgTime'])
                checks.append({'profile':label,'shape':row['inputShape'],'samples':len(selected),
                               'backend_mean_us':float(row['avgTime']),'csv_mean_us':avg})
    result['insight_cross_checks']=checks
    state=backend_root/'final-service-state.json'
    if state.exists():
        result['restored_services']=read(state)

# Raw CANN HBMbwData is MiB/s; unified msprof HBM.bandwidth uses Byte/s.
# The conversion is established in installed msmodel/hardware/hbm_model.py.
receipt=read(root/'hbm-phases.json');assert receipt['msprof_exit_code']==0
file=next((root/'hbm-profile').rglob('hbm.db'))
con=sqlite3.connect('file:'+str(file)+'?mode=ro',uri=True)
range_=con.execute('SELECT MIN(timestamp),MAX(timestamp) FROM HBMbwData').fetchone()
hbm=[]
query='''SELECT hbmid,event_type,COUNT(*),AVG(bandwidth),MIN(bandwidth),MAX(bandwidth)
FROM HBMbwData WHERE timestamp BETWEEN ? AND ?
GROUP BY hbmid,event_type ORDER BY hbmid,event_type LIMIT 12'''
for phase in receipt['phases']:
    start,end=phase['active_start_ns']+100_000_000,phase['active_end_ns']-100_000_000
    assert range_[0]<=start<end<=range_[1],phase['label']
    rows=con.execute(query,(start,end)).fetchall()
    reads=[x for x in rows if x[1]=='read'];assert len(reads)==4
    values=[x[3] for x in reads]
    hbm.append({'label':phase['label'],'order':phase['order'],
      'active_start_ns':phase['active_start_ns'],'active_end_ns':phase['active_end_ns'],
      'trimmed_ns_each_edge':100_000_000,'graphs':phase['graphs'],'active_seconds':phase['active_seconds'],
      'read_TB_per_s':sum(values)*1024**2/1e12,
      'max_unit_imbalance_fraction':(max(values)-min(values))/statistics.mean(values),
      'units':[dict(hbm_id=x[0],mode=x[1],samples=x[2],mean_MiB_per_s=x[3]) for x in rows],
      'hbm_clocks_MHz':sorted(set(int(n) for side in ['telemetry_before','telemetry_after']
           for n in re.findall(r'HBM Clock Speed\(MHz\)\s*:\s*(\d+)',phase[side]['memory']))),
      'memory_telemetry_before':{k.strip():v.strip() for k,v in
         re.findall(r'([^\n:]+):\s*([^\n]+)',phase['telemetry_before']['memory'])
         if any(n in k for n in ['Clock','Temperature'])}})
con.close();result['hbm']=hbm;result['hbm_query']=query
result['hbm_units']='HBMbwData values interpreted as MiB/s from installed CANN HbmModel.SCALE; aggregated result converted to decimal TB/s'
result['platform_l2_size_bytes']=201326592
result['limits']=[
 'Interpret current ND/M8 counters and controls before assigning cause. No per-miss stall-cycle counter is available; controlled differences are not a directly measured TLB duration.',
 'Adjacent second down-projection also hits the L2 data cache nearly completely; its extra speedup is not attributed exclusively to TLB.',
 'HBM counters measure four memory-access units and clock snapshots; they are not per-bank queue-depth or row-conflict counters.',
 'Instrumented kernel timing is distinct from unprofiled graph timing; these experiments do not claim a new model token/s result.',
 'Isolated full-FFN graphs exclude attention and other model work; compare separately against actual model requests.'
]
result['memory_access']={}
for order in ['small','full','adjacent']:
    rows=csv_rows(next((root/('profile-memory-'+order)).rglob('kernel_details.csv')))
    result['memory_access'][order]={}
    for name,shape in shapes.items():
        g=tail_stable([k for k in rows if k['Input Shapes']==shape])
        result['memory_access'][order][name]={key:mean(g,key) for key in g[0] if key.startswith('aic_') and (key.endswith('(KB)') or 'bw' in key.lower()) and all(k[key] not in ['', 'N/A'] for k in g)}
result['extra_metric_groups']={}
for tag in ['conflict','bandwidth','l0']:
    rows=csv_rows(next((root/('profile-'+tag+'-full')).rglob('kernel_details.csv')))
    result['extra_metric_groups'][tag]={}
    for name,shape in shapes.items():
        g=tail_stable([k for k in rows if k['Input Shapes']==shape])
        metrics={}
        for key in g[0]:
            if not key.startswith('aic_'):continue
            try:metrics[key]=mean(g,key)
            except (ValueError,TypeError):pass
        result['extra_metric_groups'][tag][name]=metrics
result['shapes']=shapes
(dest/'hardware-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for order,data in result['profiles'].items():
    print(order,json.dumps(data,ensure_ascii=False))
print('HBM',[(x['order'],x['read_TB_per_s'],x['hbm_clocks_MHz']) for x in hbm])
