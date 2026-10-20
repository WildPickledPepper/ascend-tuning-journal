"""Inspect allocator mapping with <= 1 GiB of live test tensors; no model load.

Run each allocator setting in a separate process, on an explicitly selected NPU.
The ACL counters are device-wide availability, not per-process page ownership.
"""
import ctypes
import gc
import json
import os
from datetime import datetime, timezone

import torch
import torch_npu


def main():
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "6"
    torch.npu.set_device(0)
    torch.npu.init()
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetMemInfo.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
                                  ctypes.POINTER(ctypes.c_size_t)]
    acl.aclrtGetMemInfo.restype = ctypes.c_int

    def counters():
        result = {}
        # Values from installed CANN 9.1 acl_rt.h's aclrtMemAttr enum.
        for name, attr in [("hbm", 1), ("hbm_2m", 4), ("hbm_1g", 10)]:
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            code = acl.aclrtGetMemInfo(attr, ctypes.byref(free), ctypes.byref(total))
            result[name] = dict(return_code=code, free_bytes=free.value,
                                total_bytes=total.value)
        return result

    def snapshot():
        return [dict(total_size=s["total_size"], allocated_size=s["allocated_size"],
                     segment_type=s.get("segment_type"),
                     is_expandable=s.get("is_expandable"))
                for s in torch.npu.memory_snapshot()]

    before = counters()
    assert before["hbm"]["return_code"] == 0
    assert before["hbm"]["free_bytes"] > 2 * 1024**3, "Insufficient probe headroom"
    tensors = []
    phases = []
    try:
        # Representative BF16 FFN weight sizes; no matrix multiplication.
        for mib in [4, 170, 340]:
            tensors.append(torch.empty(mib * 1024**2 // 2, dtype=torch.bfloat16,
                                       device="npu:0"))
            torch.npu.synchronize()
            phases.append(dict(request_mib=mib, segments=snapshot(), counters=counters(),
                               reserved_bytes=torch.npu.memory_reserved(),
                               allocated_bytes=torch.npu.memory_allocated()))
    finally:
        tensors.clear()
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
                  config=os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
                  torch_npu_version=torch_npu.__version__,
                  before=before, phases=phases, after=counters(),
                  reserved_after=torch.npu.memory_reserved(),
                  limitation="Allocator probe only; no model throughput or TLB measurements")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
