"""Worker profiler configuration only; leaves inference kernels and graphs intact."""
import os


def install():
    import torch_npu
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

    def create(config, trace_name):
        metric = os.environ.get("DECODE_PROFILE_METRIC", "PipeUtilization")
        options = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text, torch_npu.profiler.ExportType.Db],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=getattr(torch_npu.profiler.AiCMetrics, metric),
            msprof_tx=False, l2_cache=False, op_attr=True,
            data_simplification=False, record_op_args=False, gc_detect_threshold=None)
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True, with_stack=False, with_modules=False, profile_memory=False,
            experimental_config=options,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                config.torch_profiler_dir, worker_name=trace_name))

    TorchNPUProfilerWrapper._create_profiler = staticmethod(create)


class ProfileControl:
    def configure_decode_profile(self, metric):
        import torch
        from vllm.v1 import utils
        assert metric in ("PipeUtilization", "L2Cache", "Memory")
        if self.profiler is not None:
            assert not self.profiler._active and not self.profiler._running
            self.profiler.shutdown()
            self.profiler = None
        os.environ["DECODE_PROFILE_METRIC"] = metric
        utils._PROFILER_FUNC = torch.profiler.record_function
        install()
        return {"metric": metric, "record_shapes": True,
                "export": ["Text", "Db"], "recapture": False}
