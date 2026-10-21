"""Enable text/DB profiler output; no model, kernel or normal-call hooks."""
def install():
    import torch_npu
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

    def create(config, trace_name):
        options = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text, torch_npu.profiler.ExportType.Db],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            msprof_tx=False, l2_cache=False, op_attr=False, data_simplification=True,
            record_op_args=False, gc_detect_threshold=None)
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True, with_stack=False, with_modules=False, profile_memory=False,
            experimental_config=options,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(config.torch_profiler_dir, worker_name=trace_name))

    TorchNPUProfilerWrapper._create_profiler = staticmethod(create)
