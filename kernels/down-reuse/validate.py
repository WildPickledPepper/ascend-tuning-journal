"""Validate fixed-shape BF16 kernels on physical NPU 7 only."""
import argparse, ctypes, datetime, json, os, pathlib, statistics
import torch
import torch_npu

p=argparse.ArgumentParser();p.add_argument('--profile',type=int);p.add_argument('--bench',action='store_true')
a=p.parse_args()
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
assert os.environ.get('TASK_QUEUE_ENABLE')=='0', 'Direct ACL test launch requires synchronous Torch dispatch.'
torch.npu.set_device(0);torch.set_num_threads(4)
root=pathlib.Path(__file__).resolve().parent
lib=ctypes.CDLL(str(root/'libdown_reuse.so'))
lib.launch_down.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4
lib.launch_down.restype=None

def launch(mode,x,w,out):
    assert x.dtype==w.dtype==out.dtype==torch.bfloat16
    assert x.numel()==17408 and tuple(w.shape)==(5120,17408)
    assert x.is_contiguous() and w.is_contiguous() and out.is_contiguous()
    stream=torch.npu.current_stream().npu_stream
    lib.launch_down(mode,stream,x.data_ptr(),w.data_ptr(),out.data_ptr())

results={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'physical_device':7,'dtype':'bfloat16','shape':{'M':1,'K':17408,'N':5120},
 'modes':{'0':'independent baseline','1':'hoisted zero','2':'input reuse + hoisted zero','3':'reuse + pipelined prefetch'},'cases':[]}

if a.profile is not None:
    torch.manual_seed(32)
    x=torch.randn(1,17408,device='npu',dtype=torch.bfloat16)
    w=torch.randn(5120,17408,device='npu',dtype=torch.bfloat16)
    y=torch.empty(1,5120,device='npu',dtype=torch.bfloat16)
    if a.profile==4:
        y=torch.nn.functional.linear(x,w)
    else:
        launch(a.profile,x,w,y)
    torch.npu.synchronize()
    print('profile_complete',a.profile,flush=True)
    raise SystemExit

for seed,kind in [(32,'normal'),(49,'normal'),(71,'normal'),(0,'zero'),(1,'one')]:
    torch.manual_seed(seed)
    x=torch.randn(1,17408,dtype=torch.bfloat16,device='npu')
    w=torch.randn(5120,17408,dtype=torch.bfloat16,device='npu')
    if kind=='zero':x.zero_()
    if kind=='one':x.fill_(1);w.fill_(0.015625)
    ref=torch.nn.functional.linear(x,w)
    outs=[]
    row={'seed':seed,'kind':kind,'results':[]}
    for mode in range(4):
        buf=torch.full((5152,),float('nan'),device='npu',dtype=torch.bfloat16)
        launch(mode,x,w,buf);torch.npu.synchronize()
        z=buf[:5120].reshape(1,5120)
        err=(z.float()-ref.float()).abs()
        r={'mode':mode,'finite':bool(z.isfinite().all()),
           'guard_intact':bool(buf[5120:].isnan().all()),
           'equal_cann':bool(torch.equal(z,ref)),
           'max_abs_vs_cann':err.max().item(),
           'mismatches_vs_cann':int((z!=ref).sum().item())}
        outs.append(z.clone());row['results'].append(r)
        print(json.dumps(r),flush=True)
        assert r['finite'] and r['guard_intact'],r
    row['baseline_hoist_equal']=bool(torch.equal(outs[0],outs[1]))
    row['baseline_reuse_equal']=bool(torch.equal(outs[0],outs[2]))
    row['baseline_pipeline_equal']=bool(torch.equal(outs[0],outs[3]))
    results['cases'].append(row)
    (root/'validation.json').write_text(json.dumps(results,indent=2))
    assert row['baseline_hoist_equal'] and row['baseline_reuse_equal'] and row['baseline_pipeline_equal'],row
    # Record mismatches rather than hide them with a relaxed tolerance.
    ix=(outs[0]!=ref).flatten().nonzero().flatten()
    if ix.numel():
        oracle=w[ix].cpu().double() @ x.cpu().double().flatten()
        row['mismatch_details']={'indices':ix.cpu().tolist(),
          'cann':ref.flatten()[ix].float().cpu().tolist(),
          'independent':outs[0].flatten()[ix].float().cpu().tolist(),
          'float64_dot':oracle.tolist(),'float64_rounded_bf16':oracle.bfloat16().float().tolist()}
    del w,x,outs,ref

if a.bench:
    torch.manual_seed(32)
    # Alternate weights exceeding L2; all variants see the same data.
    layers=[(torch.randn(1,17408,device='npu',dtype=torch.bfloat16),
             torch.randn(5120,17408,device='npu',dtype=torch.bfloat16)) for _ in range(4)]
    out=torch.empty(1,5120,device='npu',dtype=torch.bfloat16)
    bench={}
    for mode in ['cann',0,1,2,3]:
        def step(i):
            x,w=layers[i%4]
            if mode=='cann':return torch.nn.functional.linear(x,w)
            launch(mode,x,w,out)
        for i in range(12):step(i)
        torch.npu.synchronize()
        times=[]
        for i in range(40):
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record();step(i);end.record();end.synchronize()
            times.append(start.elapsed_time(end)*1000)
        bench[str(mode)]={'median_event_us':statistics.median(times),'event_us':times}
    results['bench']=bench
    graph_bench={}
    for mode in ['cann',0,1,2,3]:
        g=torch.npu.NPUGraph()
        saved=[]
        with torch.npu.graph(g):
            for x,w in layers:
                if mode=='cann':saved.append(torch.nn.functional.linear(x,w))
                else:launch(mode,x,w,out)
        for _ in range(5):g.replay()
        torch.npu.synchronize()
        times=[]
        for _ in range(20):
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record();g.replay();end.record();end.synchronize()
            times.append(start.elapsed_time(end)*1000/4)
        graph_bench[str(mode)]={'median_us_per_kernel':statistics.median(times),'samples_us':times}
    results['graph_bench']=graph_bench
results['cann_bitwise_parity']=all(r['equal_cann'] for row in results['cases'] for r in row['results'])
results['integration_allowed']=results['cann_bitwise_parity']
(root/'validation.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results),flush=True)
