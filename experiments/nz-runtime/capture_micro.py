"""Capture the unmodified native BF16 launches, including host tiling bytes."""
import json, os, pathlib
import torch, torch_npu

out = pathlib.Path(os.environ['NZ_LAUNCH_TRACE_DIR'])
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
rows = []
keep = []
for n,k in [(34816,5120),(5120,17408)]:
    x = torch.randn(2,k,device='npu',dtype=torch.bfloat16)
    w = torch.randn(n,k,device='npu',dtype=torch.bfloat16)
    z = torch_npu.npu_format_cast(w,29)
    keep.extend([x,w,z])
    for fmt,weight in [('ND',w),('NZ',z)]:
        y = torch.nn.functional.linear(x,weight)
        torch.npu.synchronize()
        row = dict(N=n,K=k,M=2,format=fmt,x=hex(x.data_ptr()),w=hex(weight.data_ptr()),
                   y=hex(y.data_ptr()),x_stride=x.stride(),w_stride=weight.stride(),
                   physical_format=torch_npu.get_npu_format(weight))
        rows.append(row)
        print(json.dumps(row),flush=True)
(out/'tensors.json').write_text(json.dumps(rows,indent=2))
