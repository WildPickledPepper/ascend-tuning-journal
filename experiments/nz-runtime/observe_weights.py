"""Observe final weight storage after the stock loading/conversion method."""
import json, os, pathlib
import torch, torch_npu
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

original = AscendUnquantizedLinearMethod.process_weights_after_loading
observed = {}
prefixes = {}

def process(self, layer):
    result = original(self, layer)
    if layer.prefix.endswith(('.mlp.gate_up_proj', '.mlp.down_proj')):
        w = layer.weight
        row = dict(prefix=layer.prefix,shape=list(w.shape),stride=list(w.stride()),
                   pointer=hex(w.data_ptr()),storage_offset=w.storage_offset(),
                   dtype=str(w.dtype),format=torch_npu.get_npu_format(w),
                   contiguous=w.is_contiguous())
        path = pathlib.Path(os.environ['NZ_LAUNCH_TRACE_DIR']) / f'weights-{os.getpid()}.jsonl'
        with path.open('a') as f:
            f.write(json.dumps(row)+'\n')
        prefixes[w.data_ptr()] = layer.prefix
    return result

AscendUnquantizedLinearMethod.process_weights_after_loading = process

def gemm(x, weight, bias=None):
    y = torch.nn.functional.linear(x, weight, bias)
    prefix = prefixes.get(weight.data_ptr(), '')
    # Retain just two layers' actual decode buffers for a same-address read probe.
    if x.ndim == 2 and x.shape[0] == 2:
        if any(f'.layers.{i}.mlp.' in prefix for i in (0, 1)):
            observed[prefix] = (x, weight, y)
    return y

gemm_library = torch.library.Library('vllm', 'IMPL', 'PrivateUse1')
gemm_library.impl('unquantized_gemm', gemm, allow_override=True)

class WorkerExtension:
    def nz_runtime_probe(self, **kwargs):
        import importlib, runtime_probe
        return importlib.reload(runtime_probe).run(self, **kwargs)
