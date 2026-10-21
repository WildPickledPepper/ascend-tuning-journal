"""Check packed storage and fallback before launching the whole model."""
import json
import torch
import torch_npu
from model_extension import pack, unpack

torch.manual_seed(314)
w = torch.randn(34816, 5120, dtype=torch.bfloat16)
p = pack(w).npu()
assert torch.equal(unpack(p).cpu(), w)
w = w.npu()
results = []
for m in (8, 19, 512):
    x = torch.randn(m, 5120, dtype=torch.bfloat16, device="npu")
    reference = torch.nn.functional.linear(x, w)
    y = torch.ops.l1_tile_layout.linear(x, p, 2) if m == 8 else torch.nn.functional.linear(x, unpack(p))
    torch.npu.synchronize()
    equal = torch.equal(y, reference)
    results.append({"m": m, "bit_equal": equal})
    assert equal, results
print(json.dumps(results), flush=True)
