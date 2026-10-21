"""Strict native comparison, queue/graph checks, then interleaved graph timing."""
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import torch_npu

root=Path(__file__).resolve().parent
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='6'
torch.npu.set_device(0)
torch.npu.config.allow_internal_format=True
torch.ops.load_library(str(root/'libdown_torch.so'))
report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dtype':'BF16','M':8,'physical_device':6,'cases':[]}
torch.manual_seed(12345)
w=torch.randn((5120,17408),device='npu',dtype=torch.bfloat16)/17408**0.5
assert torch_npu.get_npu_format(w)==2
for seed in range(24):
    scale=[0.0,0.03125,1.0,32.0][seed%4]
    torch.manual_seed(seed+20261021)
    x=torch.randn((8,17408),device='npu',dtype=torch.bfloat16)*scale
    ref=torch.nn.functional.linear(x,w)
    outputs=[]
    for mode in [0,1]:
        print('VALIDATING_MODE',seed,mode,flush=True)
        outputs.append(torch.ops.down_prefetch_m8.linear(x,w,mode))
        torch.npu.synchronize()
    torch.npu.synchronize()
    row={'seed':seed,'scale':scale,'outputs':ref.numel(),'modes':[]}
    for mode,z in enumerate(outputs):
        delta=(z.float()-ref.float()).abs()
        entry={'mode':mode,'finite':bool(z.isfinite().all()),'different_outputs':int((z!=ref).sum()),'max_abs_diff':float(delta.max())}
        row['modes'].append(entry)
    report['cases'].append(row)
    (root/'validation.json').write_text(json.dumps(report,indent=2))
    print('CASE',json.dumps(row),flush=True)
    assert all(v['finite'] and v['different_outputs']==0 for v in row['modes']), 'Strict native equality gate failed'
    if seed<4:
        graphs=[]
        for mode in [0,1]:
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g): y=torch.ops.down_prefetch_m8.linear(x,w,mode)
            g.replay();torch.npu.synchronize()
            assert torch.equal(y,ref)
            graphs.append((g,y))
        x.zero_()
        for g,y in graphs:g.replay()
        torch.npu.synchronize()
        assert all(bool((y==0).all()) for g,y in graphs)
        row['graph_replay_and_changed_input_equal']=True
        del graphs,g,y
del x,w,ref,outputs
torch.npu.empty_cache()
report['validation_passed']=True
(root/'validation.json').write_text(json.dumps(report,indent=2))
print('VALIDATION_PASSED',flush=True)

torch.manual_seed(444)
layers=[(torch.randn((8,17408),device='npu',dtype=torch.bfloat16),
         torch.randn((5120,17408),device='npu',dtype=torch.bfloat16)/17408**0.5) for _ in range(4)]
graphs={};saved=[]
for mode in ['native',0,1]:
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for x,w in layers:
            y=torch.nn.functional.linear(x,w) if mode=='native' else torch.ops.down_prefetch_m8.linear(x,w,mode)
            saved.append(y)
    graphs[str(mode)]=g
for _ in range(10):
    for g in graphs.values():g.replay()
torch.npu.synchronize()
for i in range(4):
    assert torch.equal(saved[i],saved[4+i]) and torch.equal(saved[i],saved[8+i])
samples=[]
for i in range(40):
    order=list(graphs) if i%2==0 else list(reversed(graphs))
    row={}
    for label in order:
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record();graphs[label].replay();end.record();end.synchronize()
        row[label]=start.elapsed_time(end)*1000/4
    samples.append(row)
bench={'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'profiler':False,'calls_per_graph':4,'distinct_weights':4,'paired_rounds':40,
       'median_us':{label:statistics.median(r[label] for r in samples) for label in graphs},
       'native_over_prefetch_paired_ratio':statistics.median(r['native']/r['1'] for r in samples),
       'control_over_prefetch_paired_ratio':statistics.median(r['0']/r['1'] for r in samples),
       'samples':samples}
(root/'bench.json').write_text(json.dumps(bench,indent=2))
print('BENCH',json.dumps({k:v for k,v in bench.items() if k!='samples'}),flush=True)
