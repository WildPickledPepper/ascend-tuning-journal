"""Opt-in worker extension. Patches only this process, never package files."""
import os,pathlib,json
import torch,torch_npu
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

root=pathlib.Path(__file__).resolve().parent
enabled=os.environ.get('DOWN_TUNING_ENABLE')=='1'
if enabled:
    torch.ops.load_library(str(root/'libdown_torch.so'))
    original_process=AscendUnquantizedLinearMethod.process_weights_after_loading
    eligible=[]
    def process(self,layer):
        original_process(self,layer)
        prefix=getattr(layer,'prefix','')
        ok=(prefix.endswith('.mlp.down_proj') and tuple(layer.weight.shape)==(5120,17408)
            and layer.weight.dtype==torch.bfloat16 and layer.weight.is_contiguous()
            and torch_npu.get_npu_format(layer.weight) in (0,2))
        layer._down_tuning_eligible=ok
        if ok:
            eligible.append(prefix)
            print('DOWN_TUNING_LAYER '+prefix,flush=True)
            (root/'enabled-layers.json').write_text(json.dumps(eligible,indent=2))
    # Override the actual opaque GEMM dispatch entry, rather than a Python
    # method that a compiled model may bypass. This affects this process only.
    def gemm(x,weight,bias=None):
        if (bias is None and x.ndim==2 and tuple(x.shape)==(1,17408)
            and tuple(weight.shape)==(5120,17408) and x.dtype==torch.bfloat16
            and weight.dtype==torch.bfloat16 and x.is_contiguous()
            and weight.is_contiguous() and torch_npu.get_npu_format(weight) in (0,2)):
            return torch.ops.down_tuning.linear(x,weight)
        return torch.nn.functional.linear(x,weight,bias)
    AscendUnquantizedLinearMethod.process_weights_after_loading=process
    gemm_library=torch.library.Library('vllm','IMPL','PrivateUse1')
    gemm_library.impl('unquantized_gemm',gemm,allow_override=True)
    print('DOWN_TUNING_REGISTERED: fixed-shape BF16 decode only',flush=True)

class WorkerExtension:
    def down_tuning_status(self):
        return {'enabled':enabled,'submitted':torch.ops.down_tuning.submitted() if enabled else 0,
                'layers':eligible if enabled else []}
