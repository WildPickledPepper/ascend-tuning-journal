"""Targeted SMMU/TLB and HBM probes on the stock model's BF16 weights."""
import datetime, json, os, pathlib, statistics, time
import torch, torch_npu
from torch_npu.profiler import (AiCMetrics, ExportType, ProfilerActivity, ProfilerLevel,
    _ExperimentalConfig, profile, tensorboard_trace_handler)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def get_entries(worker):
    from observe_weights import observed
    refs={tuple(w.shape):(x,y) for name,(x,w,y) in observed.items()
          if name.startswith('language_model.model.layers.1.')}
    assert len(refs)==2, list(observed)
    entries=[]
    for _,module in worker.model_runner.model.named_modules():
        name=getattr(module,'prefix','');w=getattr(module,'weight',None)
        if w is not None and name.startswith('language_model.') and name.endswith(('.mlp.gate_up_proj','.mlp.down_proj')):
            assert torch_npu.get_npu_format(w)==29 and w.dtype==torch.bfloat16
            x,y=refs[tuple(w.shape)]
            entries.append((name,x,w,y))
    assert len(entries)==128,len(entries)
    return entries

def order_entries(entries,order):
    if order=='small': return entries[:4]*32
    if order=='full': return entries
    if order=='adjacent': return [item for e in entries for item in (e,e)]
    if order=='sweep2': return entries*2
    raise ValueError(order)

def capture(entries):
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _,x,w,y in entries: torch.mm(x,w.t(),out=y)
    return g

def run(worker, mode='tlb', order='small', label=None, seconds=2.0, repeats=6,
        metric_name='PipeUtilization', l2_enabled=True):
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='6'
    root=pathlib.Path(os.environ['NZ_LAUNCH_TRACE_DIR']).parent
    entries=order_entries(get_entries(worker),order)
    label=label or (mode+'-'+order)
    result={'label':label,'started_utc':now(),'order':order,'calls_per_graph':len(entries),
      'mode':mode,'metric':metric_name,'l2_enabled':l2_enabled,'custom_kernel':False,
      'tensors':[dict(name=n,x=hex(x.data_ptr()),w=hex(w.data_ptr()),y=hex(y.data_ptr()),
                      shape=list(w.shape),x_stride=list(x.stride()),w_stride=list(w.stride()))
                 for n,x,w,y in entries]}
    with torch.inference_mode():
        if mode=='tlb':
            # Start L2/SMMU collection BEFORE graph capture so post-processing flags
            # are recorded in the graph. Enabling only on replay can return no data.
            config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
              aic_metrics=getattr(AiCMetrics,metric_name),l2_cache=l2_enabled,
              data_simplification=False,op_attr=True,export_type=[ExportType.Db,ExportType.Text])
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
              record_shapes=True,experimental_config=config,
              on_trace_ready=tensorboard_trace_handler(str(root/('profile-'+label)))):
                g=capture(entries)
                for _ in range(repeats):g.replay()
                torch.npu.synchronize()
            result['replays']=repeats
        elif mode=='timing':
            g=capture(entries)
            for _ in range(3):g.replay()
            torch.npu.synchronize();samples=[]
            for _ in range(20):
                a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                a.record();g.replay();b.record();b.synchronize()
                samples.append(a.elapsed_time(b)*1000/len(entries))
            result.update(samples_us_per_call=samples,median_us_per_call=statistics.median(samples))
        elif mode=='hbm':
            g=capture(entries)
            for _ in range(3):g.replay()
            torch.npu.synchronize()
            result['active_started_utc']=now();start=time.time_ns()
            stop=time.perf_counter()+seconds;count=0
            while time.perf_counter()<stop:
                for _ in range(4):g.replay()
                torch.npu.synchronize();count+=4
            end=time.time_ns();result['active_ended_utc']=now()
            result.update(active_start_ns=start,active_end_ns=end,graphs=count,
                          active_seconds=(end-start)/1e9)
        else:raise ValueError(mode)
    result['completed_utc']=now()
    (root/(label+'.json')).write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k!='tensors'}
