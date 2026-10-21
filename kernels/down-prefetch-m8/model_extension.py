"""Worker-local, opt-in replacement for the audited BF16/ND M8 down shape."""
import os
from pathlib import Path
import torch
import torch_npu
from graph_audit_base import GraphAudit as BaseGraphAudit

enabled=os.environ.get('DOWN_PREFETCH_M8_ENABLE')=='1'
eligible=[]
if enabled:
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
    torch.ops.load_library(str(Path(__file__).resolve().parent/'libdown_torch.so'))
    original_process=AscendUnquantizedLinearMethod.process_weights_after_loading
    def process(self,layer):
        original_process(self,layer)
        if tuple(layer.weight.shape)==(5120,17408) and layer.weight.dtype==torch.bfloat16:
            eligible.append(getattr(layer,'prefix',''))
    def gemm(x,weight,bias=None):
        if (bias is None and x.ndim==2 and tuple(x.shape)==(8,17408)
            and tuple(weight.shape)==(5120,17408) and x.dtype==torch.bfloat16
            and weight.dtype==torch.bfloat16 and x.is_contiguous() and weight.is_contiguous()
            and torch_npu.get_npu_format(weight) in (0,2)):
            return torch.ops.down_prefetch_m8.linear(x,weight,1)
        return torch.nn.functional.linear(x,weight,bias)
    AscendUnquantizedLinearMethod.process_weights_after_loading=process
    registration=torch.library.Library('vllm','IMPL','PrivateUse1')
    registration.impl('unquantized_gemm',gemm,allow_override=True)
    print('DOWN_PREFETCH_M8_REGISTERED',flush=True)

class GraphAudit(BaseGraphAudit):
    def graph_audit(self):
        result=super().graph_audit()
        result['prefetch_m8']={'enabled':enabled,'eligible_layers':eligible,
            'submitted_before_graph_replay':torch.ops.down_prefetch_m8.submitted() if enabled else 0,
            'scope':'Any matched M8 down projection, including main and draft if applicable.'}
        return result
