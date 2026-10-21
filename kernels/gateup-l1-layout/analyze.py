"""Read-only timing/PMU analysis. HBM counters are physical-device total traffic."""
import argparse,csv,json,pathlib,sqlite3,statistics
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent/'results')
p.add_argument('--output',type=pathlib.Path)
args=p.parse_args();root=args.root
out={'timing':{},'l2':{},'hbm':{}}
for layout in ['native','nz','packed']:
    reps=[json.loads((root/(layout+'-'+s+'.json')).read_text()) for s in ['a','b']]
    out['timing'][layout]={}
    for scope in ['gate','interleaved']:
        values=[v for r in reps for v in r['timing'][scope]['samples_ms']]
        med=statistics.median(values)
        out['timing'][layout][scope]={'graph_median_ms':med,'per_layer_us':med*1000/64,
            'run_medians_us':[r['timing'][scope]['per_layer_us'] for r in reps],
            'samples':len(values),'min_ms':min(values),'max_ms':max(values)}
    paths=list((root/(layout+'-l2-profile')).glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
    if not paths and (root/(layout+'-l2-kernel_details.csv')).exists():paths=[root/(layout+'-l2-kernel_details.csv')]
    if len(paths)==1:
        rows=list(csv.DictReader(paths[0].open()))
        rows=[r for r in rows if r['Accelerator Core']=='AI_CORE']
        assert len(rows)==256,(layout,len(rows))
        assert {int(r['Device_id']) for r in rows}=={7}
        gate=[r for r in rows if '34816' in r.get('Input Shapes','') or 'gateup_' in r['Name']]
        assert len(gate)==128,(layout,len(gate))
        out['l2'][layout]={}
        for label,group in [('gate',gate),('all_ffn',rows)]:
            hit=sum(float(r[f'aic_r{i}_read_cache_hit']) for r in group for i in [0,1])
            miss=sum(float(r[f'aic_r{i}_read_cache_miss_allocate']) for r in group for i in [0,1])
            out['l2'][layout][label]={'calls':len(group),'read_hits':hit,'read_misses':miss,
                'read_hit_rate':hit/(hit+miss),'mean_kernel_us':statistics.mean(float(r['Duration(us)']) for r in group)}
        out['l2'][layout]['csv']=str(paths[0])
    result=root/(layout+'-hbm.json')
    if result.exists():
        row=json.loads(result.read_text());phase=row['phase']
        dbs=list((root/(layout+'-hbm-sys')).glob('PROF*/msprof*.db'))
        if not dbs and (root/(layout+'-hbm.db')).exists():dbs=[root/(layout+'-hbm.db')]
        assert len(dbs)==1,dbs
        c=sqlite3.connect('file:'+str(dbs[0])+'?mode=ro',uri=True)
        ids=c.execute('SELECT deviceId,COUNT(*) FROM HBM GROUP BY deviceId').fetchall()
        assert [a[0] for a in ids]==[7],ids
        lo=phase['start_ns']+500_000_000;hi=phase['end_ns']-500_000_000
        rate=phase['layers']/phase['elapsed_s'];data={}
        for typ in ['read','write']:
            rows=c.execute('SELECT h.timestampNs,SUM(h.bandwidth),COUNT(DISTINCT h.hbmId) '
                'FROM HBM h JOIN STRING_IDS s ON s.id=h.type '
                'WHERE h.deviceId=7 AND s.value=? AND h.timestampNs BETWEEN ? AND ? '
                'GROUP BY h.timestampNs ORDER BY h.timestampNs',(typ,lo,hi)).fetchall()
            assert len(rows)>100 and all(r[2]==4 for r in rows),(typ,len(rows))
            def bandwidth(items):
                return sum((items[i][0]-items[i-1][0])*items[i][1] for i in range(1,len(items)))/(items[-1][0]-items[0][0])
            bps=bandwidth(rows)
            data[typ]={'GB_s':bps/1e9,'bytes_per_ffn_layer':bps/rate,'samples':len(rows),
                'half_window_bytes_per_layer':[bandwidth(rows[:len(rows)//2])/rate,bandwidth(rows[len(rows)//2:])/rate]}
        data['phase']=phase;data['db']=str(dbs[0]);out['hbm'][layout]=data;c.close()
base=out['timing']['native']['gate']['per_layer_us'];nz=out['timing']['nz']['gate']['per_layer_us'];packed=out['timing']['packed']['gate']['per_layer_us']
out['comparisons']={'gate_time_reduction_vs_native':1-packed/base,'gate_time_reduction_vs_matched_nz':1-packed/nz,
    'gate_speedup_vs_native':base/packed,'gate_speedup_vs_matched_nz':nz/packed,
    'ffn_time_reduction_vs_native':1-out['timing']['packed']['interleaved']['per_layer_us']/out['timing']['native']['interleaved']['per_layer_us'],
    'useful_gate_weight_delivery_TB_s':{k:356515840/(out['timing'][k]['gate']['per_layer_us']*1e-6)/1e12 for k in out['timing']}}
(args.output or root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
