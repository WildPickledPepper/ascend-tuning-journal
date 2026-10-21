import os
from pathlib import Path
import sys
import torch
import torch_npu
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='6'
torch.npu.set_device(0)
torch.npu.config.allow_internal_format=True
root=Path(__file__).resolve().parent
torch.ops.load_library(str(root/'libdown_torch.so'))
torch.manual_seed(17)
x=torch.randn((8,17408),device='npu',dtype=torch.bfloat16)
w=torch.randn((5120,17408),device='npu',dtype=torch.bfloat16)
which=sys.argv[1]
y=torch.nn.functional.linear(x,w) if which=='native' else torch.ops.down_prefetch_m8.linear(x,w,int(which))
torch.npu.synchronize()
assert bool(y.isfinite().all())
print('PROFILE_COMPLETED',which,flush=True)
