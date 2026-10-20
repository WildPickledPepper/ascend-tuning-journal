"""Join native SMMU records to actual model kernels, using trace request markers."""
import collections,csv,json,statistics,struct,sys,re
from pathlib import Path
from decimal import Decimal
root=Path(sys.argv[1]);out=Path(sys.argv[2])
folders=sorted(root.glob('profile-model-pipe/*_ascend_pt'))
valid=[f for f in folders if list(f.glob('profiler_info*.json')) and (f/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').exists()]
assert valid,'No stopped parsed model profile'
p=valid[-1];d=p/'ASCEND_PROFILER_OUTPUT'
def ns(v):return int(Decimal(str(v))*1000)
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=readcsv(d/'kernel_details.csv')
for k in rows:k['_lo']=ns(k['Start Time(us)']);k['_hi']=k['_lo']+ns(k['Duration(us)'])
t=json.loads((d/'trace_view.json').read_text());events=t['traceEvents'] if isinstance(t,dict) else t
bounds={e['name']:ns(e['ts']) for e in events if e.get('name','').startswith('ND_M8_REQUEST_') and e.get('ph')=='X'}
assert len(bounds)==4,bounds
raw=collections.defaultdict(list)
for f in p.rglob('socpmu.data.*'):
 if not re.search(r'\.slice_\d+$',f.name):continue
 b=f.read_bytes();assert len(b)%80==0
 for r in struct.iter_unpack('<8B4H8Q',b):
  if r[11]==2:
   vals=r[12:]
   raw[r[9],r[10]].append(vals)
bykey=collections.defaultdict(list)
for k in rows:bykey[int(k['Stream ID']),int(k['Task ID'])].append(k)
matched=0;unmatched=[]
for key,ks in bykey.items():
 vals=raw.get(key,[])
 if len(vals)!=len(ks):
  if any(k['Type']=='MatMulV2' for k in ks):unmatched.append(dict(key=key,kernels=len(ks),smmu=len(vals)))
  continue
 for k,v in zip(sorted(ks,key=lambda x:x['_lo']),vals):k['_smmu']=v;matched+=1
result={'profile':p.name,'raw_smmu_keys':len(raw),'matched_kernel_records':matched,'unmatched_matmul_keys':unmatched,'requests':{}}
for q in [0,2]:
 lo=bounds[f'ND_M8_REQUEST_{q}_BEGIN'];hi=bounds[f'ND_M8_REQUEST_{q}_END']
 ks=[k for k in rows if lo<=k['_lo']<hi];assert ks
 groups=collections.defaultdict(list)
 for k in ks:groups[k['Type'],k['Input Shapes'],k['Model ID']].append(k)
 stats=[]
 for (typ,shape,mid),g in groups.items():
  if typ!='MatMulV2':continue
  v=[k['_smmu'] for k in g if '_smmu' in k]
  stats.append(dict(shape=shape,model_id=mid,calls=len(g),mean_us=statistics.mean(float(k['Duration(us)']) for k in g),total_ms=sum(float(k['Duration(us)']) for k in g)/1000,mte2_us=statistics.mean(float(k.get('aic_mte2_time(us)') or 0) for k in g),mac_us=statistics.mean(float(k.get('aic_mac_time(us)') or 0) for k in g),smmu_samples=len(v),smmu_identity_failures=sum(x[0]+x[3]+x[4]!=x[1] for x in v),smmu_misses_per_call=statistics.mean(x[0] for x in v) if v else None,smmu_requests_per_call=statistics.mean(x[1] for x in v) if v else None,smmu_miss_ratio=sum(x[0] for x in v)/sum(x[1] for x in v) if v and sum(x[1] for x in v) else None))
 stats.sort(key=lambda x:-x['total_ms'])
 result['requests'][str(q)]={'kernel_records':len(ks),'all_kernel_sum_ms':sum(float(k['Duration(us)']) for k in ks)/1000,'matmuls':stats,'bounds_trace_ns':[lo,hi]}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
