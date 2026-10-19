"""Interleaved graph comparison, same physical device and fixed weights."""
import ctypes,datetime,json,os,pathlib,statistics
import torch,torch_npu
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
assert os.environ.get('TASK_QUEUE_ENABLE')=='0'
torch.npu.set_device(0);torch.manual_seed(93)
root=pathlib.Path(__file__).resolve().parent
lib=ctypes.CDLL(str(root/'libdown_reuse.so'))
lib.launch_down.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4;lib.launch_down.restype=None
layers=[(torch.randn(1,17408,device='npu',dtype=torch.bfloat16),
         torch.randn(5120,17408,device='npu',dtype=torch.bfloat16)) for _ in range(4)]
graphs={};saved=[]
for mode in ['cann',0,2,3]:
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for x,w in layers:
            if mode=='cann':saved.append(torch.nn.functional.linear(x,w))
            else:
                out=torch.empty(1,5120,device='npu',dtype=torch.bfloat16);saved.append(out)
                lib.launch_down(mode,torch.npu.current_stream().npu_stream,x.data_ptr(),w.data_ptr(),out.data_ptr())
    graphs[mode]=g
for _ in range(10):
    for g in graphs.values():g.replay()
torch.npu.synchronize()
samples=[]
for round_no in range(40):
    order=list(graphs) if round_no%2==0 else list(reversed(graphs))
    row={}
    for mode in order:
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record();graphs[mode].replay();end.record();end.synchronize()
        row[str(mode)]=start.elapsed_time(end)*1000/4
    samples.append(row)
med={str(mode):statistics.median(r[str(mode)] for r in samples) for mode in graphs}
r={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'physical_device':7,'dtype':'BF16','profiler':False,'graph_replay':True,
 'weights':4,'weight_bytes_each':178257920,'calls_per_replay':4,'paired_rounds':40,
 'median_us_per_call':med,'paired_median_speedup_cann_over_pipeline':statistics.median(r['cann']/r['3'] for r in samples),
 'samples':samples,'scope':'standalone operator, not model throughput; CANN bitwise differences remain under investigation'}
(root/'paired-bench.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='samples'}))
