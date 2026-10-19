"""Native matmul cache-state intervention, profiled with msOpProf application replay."""
import gc
import json
import os

import torch
import torch_npu

assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "6"
torch.npu.config.allow_internal_format = True
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)
torch.manual_seed(137)
assert torch.npu.mem_get_info()[0] > 4 * 1024**3

# Two disjoint 512 MiB BF16 matrices. Reading them before a target is the intervention;
# cache counters, rather than this size assumption, decide whether it evicts target data.
busters = [torch.randn(32768, 8192, device="npu", dtype=torch.bfloat16) for _ in range(2)]
bx = torch.randn(2, 8192, device="npu", dtype=torch.bfloat16)
kept_output = None
ordinal = 0
for n, k in [(34816, 5120), (5120, 17408)]:
    weights = [torch.randn(n, k, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    nz = [torch_npu.npu_format_cast(w, 29) for w in weights]
    x = torch.randn(2, k, device="npu", dtype=torch.bfloat16)
    for state in ["repeat_two_weights", "after_other_weight_reads"]:
        for layout, ws in [("ND", weights), ("NZ", nz)]:
            for _ in range(3):
                for w in ws:
                    kept_output = torch.nn.functional.linear(x, w)
            if state == "after_other_weight_reads":
                for b in busters:
                    kept_output = torch.nn.functional.linear(bx, b)
            torch.npu.synchronize()
            print("TARGET " + json.dumps(dict(ordinal=ordinal, N=n, K=k, M=2,
                                               layout=layout, state=state,
                                               weight_format=torch_npu.get_npu_format(ws[0]))), flush=True)
            mark = torch_npu.npu.mstx.range_start("probe")
            kept_output = torch.nn.functional.linear(x, ws[0])
            torch.npu.synchronize()
            torch_npu.npu.mstx.range_end(mark)
            ordinal += 1
    del weights, nz, x
    gc.collect()
    torch.npu.empty_cache()
print("CACHE_PROBE_COMPLETE", flush=True)
