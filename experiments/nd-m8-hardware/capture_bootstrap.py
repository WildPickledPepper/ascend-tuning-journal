import os,pathlib,inspect
import torch,torch_npu
BOOT=None

def install():
    import vllm_ascend.worker.model_runner_v1 as module
    for cls in vars(module).values():
        if not inspect.isclass(cls) or cls.__module__!=module.__name__ or 'capture_model' not in cls.__dict__:continue
        original=cls.capture_model
        def capture(self,*a,**kw):
            global BOOT
            if BOOT is None:
                root=pathlib.Path(os.environ['ND_M8_RUN'])
                cfg=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,l2_cache=True,data_simplification=False,op_attr=True,export_type=[torch_npu.profiler.ExportType.Db,torch_npu.profiler.ExportType.Text])
                BOOT=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,experimental_config=cfg,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root/'profile-model-pipe')))
                BOOT.start()
            return original(self,*a,**kw)
        cls.capture_model=capture
