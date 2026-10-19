"""Targeted probes on the captured model's exact weight and input addresses."""
import datetime, json, os, pathlib, statistics
import torch, torch_npu
from torch_npu.profiler import (AiCMetrics, ExportType, ProfilerActivity, ProfilerLevel,
    _ExperimentalConfig, profile, tensorboard_trace_handler)

def run(worker, label='same-address', do_profile=True, all_layers=False, repeat_small=False,
        metric_name='PipeUtilization'):
    from observe_weights import observed
    root = pathlib.Path(os.environ['NZ_LAUNCH_TRACE_DIR']).parent
    result = dict(label=label,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  observed=list(observed),cases=[])
    if not observed:
        result['worker_type']=str(type(worker))
        result['model_type']=str(type(worker.model_runner.model))
        return result
    groups = {}
    for name, tensors in observed.items():
        x,w,y=tensors
        groups.setdefault(tuple(w.shape),[]).append((name,x,w,y))
    if all_layers:
        refs={tuple(w.shape):(x,y) for name,(x,w,y) in observed.items()
              if name.startswith('language_model.')}
        entries=[]
        for name,module in worker.model_runner.model.named_modules():
            prefix=getattr(module,'prefix','')
            w=getattr(module,'weight',None)
            if (w is not None and prefix.startswith('language_model.')
                and prefix.endswith(('.mlp.gate_up_proj','.mlp.down_proj'))):
                x,y=refs[tuple(w.shape)]
                entries.append((prefix,x,w,y))
        assert len(entries)==128, len(entries)
        if repeat_small:
            entries=entries[:4]*32
        groups={'all_64_ffn_layers_interleaved':entries}
    graphs = []
    with torch.inference_mode():
        for shape, entries in groups.items():
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g):
                for name,x,w,y in entries:
                    torch.mm(x,w.t(),out=y)
            for _ in range(3): g.replay()
            torch.npu.synchronize()
            samples=[]
            for _ in range(20):
                start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                start.record();g.replay();end.record();end.synchronize()
                samples.append(start.elapsed_time(end)*1000/len(entries))
            row=dict(shape=shape,median_us=statistics.median(samples),samples_us=samples,
              tensors=[dict(name=name,x=hex(x.data_ptr()),w=hex(w.data_ptr()),y=hex(y.data_ptr()),
                x_stride=x.stride(),w_stride=w.stride(),format=torch_npu.get_npu_format(w))
                for name,x,w,y in entries])
            result['cases'].append(row)
            graphs.append((shape,g))
        if do_profile:
            config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                aic_metrics=getattr(AiCMetrics,metric_name),op_attr=True,
                export_type=[ExportType.Db,ExportType.Text])
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
                record_shapes=True,experimental_config=config,
                on_trace_ready=tensorboard_trace_handler(str(root/('profile-'+label)))):
                for shape,g in graphs:
                    with torch.profiler.record_function(str(shape)):
                        for _ in range(4):g.replay()
                        torch.npu.synchronize()
    (root/(label+'.json')).write_text(json.dumps(result,indent=2))
    return result
