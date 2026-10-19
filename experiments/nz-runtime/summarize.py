"""Build small, path-free evidence from actual captures and Insight responses."""
import collections, csv, hashlib, json, pathlib, statistics, struct, sys

raw=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
shapes={'up':'2,5120;320,2176,16,16','down':'2,17408;1088,320,16,16'}
result={'dtype':'BF16','M':2,'physical_device':6,'custom_npu_kernel':False,
        'instrumentation':'Temporary process-local launch logging; original math forwarded unchanged',
        'profiled_model_tokens':32,'full_model_benchmark_repeated':False,'profiles':{}}
for label,folder in [('model','model/profiles-mtp'),('same-address','profile-same-address'),
    ('full-weight-set','profile-full-weight-set'),('repeat-small-128','profile-repeat-small-128'),
    ('memory-small','profile-memory-access-small'),('memory-full','profile-memory-access-full')]:
    file=next((raw/folder).rglob('kernel_details.csv'))
    rows=list(csv.DictReader(file.open(encoding='utf-8')))
    data={}
    for name,shape in shapes.items():
        group=[r for r in rows if r['Input Shapes'].strip('"')==shape]
        keys=['Duration(us)','aic_mte2_time(us)','aic_mac_time(us)','aic_mte1_time(us)',
              'aic_read_main_memory_datas(KB)','aic_GM_to_L1_datas(KB)']
        data[name]={'count':len(group),'mean':{k:statistics.mean(float(x[k]) for x in group) for k in keys if k in group[0]}}
    backend=read(out/(label+'-backend-matmuls.json'))
    for name,shape in shapes.items():
        b=next(v for v in backend if v['inputShape'].strip('"')==shape)
        assert int(b['count'])==data[name]['count']
        assert abs(float(b['avgTime'])-data[name]['mean']['Duration(us)'])<.011
        data[name]['insight_backend_avg_us']=float(b['avgTime'])
    result['profiles'][label]=data

micro=[];model=[]
for folder,target in [('micro',micro),('model-trace',model)]:
    for f in (raw/folder).glob('launch-*.jsonl'):
        for line in f.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);b=bytes.fromhex(r['args_hex'])
            if len(b)!=200:continue
            target.append(dict(key=r['key'],blocks=r['blocks'],pointers=[hex(p) for p in struct.unpack('<5Q',b[:40])],
                words=list(struct.unpack('<40I',b[40:])),tiling=b[40:]))
tiling={}
for key in [229571,229585]:
    m=next(r for r in micro if r['key']==key)
    g=[r for r in model if r['key']==key and r['words'][:3]==m['words'][:3]]
    assert g and all(r['blocks']==23 and r['tiling'][:152]==m['tiling'][:152] and r['words'][39]==m['words'][39] for r in g)
    tiling[str(key)]={'model_matching_M2_launches_in_archived_capture':len(g),
      'tiling_size_bytes':160,'matching_payload_bytes':152,'blocks':23,
      'payload152_sha256':hashlib.sha256(m['tiling'][:152]).hexdigest(),
      'micro_words':m['words'],'only_different_u32_index':38,'only_different_byte_offset':152,
      'last_word_hex':hex(m['words'][39]),
      'limitation':'Offset 152 varies across launches; its semantic field name is unverified. Core 0 up-kernel scalar loads do not read bytes 152..159.'}
result['tiling']=tiling
result['native_nz_binary_sha256']='f4bd226d4a56698f7e6de39bbd9731855913f010d7a2aa14fd3eddaf285fae5d'

same=read(raw/'same-address.json');full=read(raw/'full-weight-set.json');small=read(raw/'repeat-small-128.json')
addresses=[]
for case in same['cases']:
    for t in case['tensors']:
        if 'language_model.model.layers.1.' not in t['name']:continue
        f=next(v for c in full['cases'] for v in c['tensors'] if v['name']==t['name'])
        s=next(v for c in small['cases'] for v in c['tensors'] if v['name']==t['name'])
        assert all(t[k]==f[k]==s[k] for k in ['x','w','y','x_stride','w_stride','format'])
        matches=[(i,r) for i,r in enumerate(model) if r['pointers'][:3]==[t['x'],t['w'],t['y']] and r['words'][1]==2]
        assert len(matches)>=2 and all(r['tiling'][:152]==matches[0][1]['tiling'][:152] for _,r in matches)
        addresses.append({**t,'matching_launches_in_archived_capture':len(matches),'earliest_launch_index':matches[0][0],
                          'same_in_original_capture_and_both_isolated_probes':True})
result['same_address_layer1']=addresses
result['same_layer1_results']=read(out/'same-layer1-summary.json')
dma=read(out/'dma-summary.json')
for key,shape in [('229571','up'),('229585','down')]:
    backend=read(out/(shape+'-source-api-instructions-dynamic.json'))['body']['Instructions']
    for group in dma[key]['groups']:
        instruction=next(x for x in backend if x['Address']==group['pc'])
        assert instruction['Instructions Executed']==group['count']
        assert instruction['Process Bytes']==sum(group['bytes_each'])*group['count']
        if group['name']=='MOV_OUT_TO_L1':
            p=group['params'][0]
            assert p['nburst']*p['burst_len_units']*32==group['bytes_each'][0]
            group['fragment_bytes']=p['burst_len_units']*32
            group['fragment_start_stride_bytes']=(p['burst_len_units']+p['src_stride_units'])*32
            group['transfer_block_bytes']=group['bytes_each'][0]
        group['origin']='Native binary + captured M2 tiling simulator, core 0; dynamic instruction counts confirmed via compiled Insight backend'
result['dma']=dma
result['working_set_bytes']={'two_layers':2*(34816*5120+5120*17408)*2,'64_layers':64*(34816*5120+5120*17408)*2}
result['limits']=[
 'Working-set context causally reproduces most of the regression; no SMMU/TLB or HBM-channel counters were collected, so no specific hardware-internal cause is claimed.',
 'Simulator confirms native instruction structure, not device instruction timestamps or device latency.',
 'Device TimelineDetail returned 107001 after saving kernel inputs/binary/tiling; its instruction timing was not used.',
 'Simulator source/details/baseInfo and memoryGraph returned empty data; only valid instructions/dynamic was used.',
 'Graph profile warned about missing ACL-to-NPU flow; results use device kernels and PMUs, not CPU flow associations.',
 'These are targeted profiles and isolated native operator probes, not a new model speedup or an optimization deployment.'
]
(out/'runtime-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'tiling_counts':{k:v['model_matching_M2_launches_in_archived_capture'] for k,v in tiling.items()},'same_addresses':addresses,'working_set_bytes':result['working_set_bytes']},ensure_ascii=False,indent=2))
