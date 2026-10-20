"""Apply profiler-data-check and logical GEMM MFU rules to existing captures."""
import argparse,collections,csv,hashlib,json,math,tarfile
from decimal import Decimal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('profile_evidence',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks=load(a.data/'skill-data-check.json');prior=load(a.profile_evidence)
peak=24*1800e6*16*16*16*2/1e12
assert math.isclose(peak,353.8944,rel_tol=1e-12)
archive_db_hashes={}
for mode in ['old','new']:
 with tarfile.open(a.data/f'{mode}.tar.gz','r:gz') as tar:
  members=sorted((m for m in tar.getmembers() if m.name.endswith('/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db')),key=lambda m:m.name)
  assert len(members)==2
  for i,m in enumerate(members):archive_db_hashes[f'{mode}-p{i}']=hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()
source_checks={}
rows=[];pools=collections.defaultdict(lambda:dict(calls=0,total_us=Decimal(0)))
for mode in ['old','new']:
 for i,folder in enumerate(sorted((a.data/mode/'profiles-mtp').glob('*/ASCEND_PROFILER_OUTPUT'))):
  label=f'{mode}-p{i}';check=checks['captures'][label];d=prior['profiles'][label]
  assert check['status']=='valid' and check['stop_metadata_present'] and check['end_info_present'] and check['parse_output_present']
  common=check['config']['common_config'];exp=check['config']['experimental_config']
  assert common['record_shapes'] and not common['profile_memory'] and exp['_profiler_level']=='Level1' and exp['_export_type']==['text','db']
  for name in ['trace_view.json','kernel_details.csv','ascend_pytorch_profiler_0.db']:
   actual=hashlib.sha256((folder/name).read_bytes()).hexdigest()
   assert actual==prior['source_sha256'][f'{label}/{name}']
   if name.endswith('.db'):
    assert archive_db_hashes[label]==check['files'][name]['sha256']
    source_checks[label]=dict(original_archive_db_matches_server=True,local_analysis_db_matches_previous_evidence=True,local_analysis_db_matches_server=(actual==check['files'][name]['sha256']),note='Local DB used for Insight analysis differs from archived server DB; MFU uses unchanged kernel CSV, not this DB.')
   else:
    assert actual==check['files'][name]['sha256']
  grouped=collections.defaultdict(list)
  with (folder/'kernel_details.csv').open(encoding='utf-8-sig',newline='') as h:
   for k in csv.DictReader(h):
    if k['Model ID']!=d['main_graph_model_id'] or k['Type']!='MatMulV2':continue
    shape=k['Input Shapes'].strip(chr(34));x,w=[list(map(int,s.split(','))) for s in shape.split(';')]
    if w not in [[34816,5120],[5120,17408]]:continue
    assert k['Input Data Types']=='DT_BF16;DT_BF16'
    assert x[1]==w[1]
    grouped[shape].append(Decimal(k['Duration(us)']))
  assert len(grouped)==2
  for shape,times in sorted(grouped.items()):
   x,w=[list(map(int,s.split(','))) for s in shape.split(';')];m,k=x;n=w[0]
   assert m==(2 if mode=='old' else 8) and len(times)==64*d['decode_rounds']
   mean=float(sum(times)/len(times));flops=2*m*n*k;achieved=flops/(mean*1e-6)/1e12;mfu=100*achieved/peak
   op='up_gate' if n==34816 else 'down';weight_bytes=n*k*2
   row=dict(capture=label,op=op,input_shape=shape,dtype='BF16',M=m,N=n,K=k,calls=len(times),mean_us=mean,logical_flops_per_call=flops,achieved_tflops=achieved,peak_tflops=peak,mfu_percent=mfu,logical_weight_bytes_per_call=weight_bytes,logical_flops_per_weight_byte=flops/weight_bytes)
   rows.append(row);pool=pools[(mode,op)];pool['calls']+=len(times);pool['total_us']+=sum(times);pool['flops_per_call']=flops
pooled=[]
for (mode,op),v in sorted(pools.items()):
 mean=float(v['total_us']/v['calls']);ach=v['flops_per_call']/(mean*1e-6)/1e12
 pooled.append(dict(mode=mode,op=op,calls=v['calls'],mean_us=mean,achieved_tflops=ach,mfu_percent=100*ach/peak))
result=dict(source_hash_validation=source_checks,skills=['mindstudio_profiler_data_check','op-mfu-calculator'],data_check=checks,peak=dict(tflops=peak,precision='BF16 dense',source='Installed CANN 9.1.0 Ascend910B2.ini plus op-mfu-calculator approximate 353.89 TFLOP/s reference',platform_config_sha256='307315c7d531f2ed86549a095d525592a641f4e81c2d3404541f6400e9d2471e',cube_core_cnt=24,cube_freq_mhz=1800,cube_m=16,cube_n=16,cube_k=16,formula='24 * 1.8e9 * 2 * 16 * 16 * 16 / 1e12',measured_sustained_peak=False),formula=dict(logical_flops='2*M*N*K',achieved_tflops='logical_flops/(mean_us*1e-6)/1e12',mfu_percent='100*achieved_tflops/peak_tflops'),rows=rows,pooled=pooled,scope='Two target-model FFN MatMulV2 groups, captured decode only; prefill and draft excluded.',limits=['Operator logical-FLOP MFU, not an end-to-end model MFU counting only accepted generated tokens.','M=8 includes computation of rejected candidates; increased MFU is not by itself increased useful accepted output.','Padding and hardware extra work are excluded from logical FLOPs; MFU is not MAC activity time ratio.','CANN nominal peak is a reference denominator, not a measured sustained device peak or a live frequency measurement.','Profiled durations are reused; this analysis does not start or reconfigure an NPU service.','Logical weight bytes/FLOPs are not measured HBM transaction counts or bandwidth.'])
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(all_four_data_checks='valid',pooled=pooled,first_question=[r for r in rows if r['capture'].endswith('p0')]),indent=2))
