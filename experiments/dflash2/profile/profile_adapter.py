"""Profiler-only host instrumentation. No NPU math, sync or kernel changes."""
import functools

def install():
    import torch,torch_npu
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
    def create(config,trace_name):
        experimental=torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text,torch_npu.profiler.ExportType.Db],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            msprof_tx=False,l2_cache=False,op_attr=False,data_simplification=True,
            record_op_args=False,gc_detect_threshold=None)
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,with_stack=False,with_modules=False,profile_memory=False,
            experimental_config=experimental,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(config.torch_profiler_dir,worker_name=trace_name))
    TorchNPUProfilerWrapper._create_profiler=staticmethod(create)
    def wrap(owner,name,label):
        original=getattr(owner,name)
        @functools.wraps(original)
        def tagged(*args,**kwargs):
            with torch.profiler.record_function(label):return original(*args,**kwargs)
        setattr(owner,name,tagged)
    from vllm_ascend.models import qwen3_dflash2 as model
    from vllm_ascend.spec_decode import dflash2_proposer as proposer
    wrap(model.DFlash2Qwen3ForCausalLM,'compute_candidates','DFlash2::lm_head_and_topk')
    wrap(model.CandidateSelector,'forward','DFlash2::candidate_selector')
    wrap(proposer,'greedy_select_path','DFlash2::selector_walk')
