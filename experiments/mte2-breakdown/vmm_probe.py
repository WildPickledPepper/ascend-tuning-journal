"""Native ND linear probe with controlled physical weight working set.

The logical weight matrix remains full sized. Repeated read-only VMM mappings
allow checking L2 residency without replacing or recompiling a device kernel.
Only a synthetic constant weight is used; never apply this to model weights.
"""
import ctypes as C
from contextlib import nullcontext
import gc
import json
import math
import os
import sys
import time

import torch
import torch_npu

assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "6"
mode = sys.argv[1]
fmt = sys.argv[3] if len(sys.argv) > 3 else "nd"
assert fmt in ["nd", "nz"]
fmt_id = 2 if fmt == "nd" else 29
which = sys.argv[2] if len(sys.argv) > 2 else "up"
n, k = (34816, 5120) if which == "up" else (5120, 17408)
chunk = 64 * 1024**2
size = 8 * k * 2
weight_size = n * k * 2
mapped_size = math.ceil(weight_size / chunk) * chunk
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

class Location(C.Structure):
    _fields_ = [("id", C.c_uint32), ("type", C.c_int)]

class Prop(C.Structure):
    _fields_ = [("handleType", C.c_int), ("allocationType", C.c_int),
                ("memAttr", C.c_int), ("location", Location), ("reserve", C.c_uint64)]

acl = C.CDLL("libascendcl.so")
acl.aclrtReserveMemAddress.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint64]
acl.aclrtMallocPhysical.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(Prop), C.c_uint64]
acl.aclrtMapMem.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint64]
acl.aclrtUnmapMem.argtypes = [C.c_void_p]
acl.aclrtFreePhysical.argtypes = [C.c_void_p]
acl.aclrtReleaseMemAddress.argtypes = [C.c_void_p]

def check(code, name):
    if code:
        raise RuntimeError(f"{name}: {code}")

ptr = C.c_void_p()
check(acl.aclrtReserveMemAddress(C.byref(ptr), mapped_size, 0, None, 0), "reserve")
handles, maps = [], []
w = storage = None
try:
    # ACL_HBM_MEM_HUGE requests the runtime's ordinary 2 MiB VMM pages.
    prop = Prop(0, 0, 4, Location(0, 1), 0)
    for i in range(mapped_size // chunk):
        if mode == "full" or not handles:
            handle = C.c_void_p()
            check(acl.aclrtMallocPhysical(C.byref(handle), chunk, C.byref(prop), 0), "mallocPhysical")
            handles.append(handle)
        else:
            assert mode == "alias64", mode
            handle = handles[0]
        addr = ptr.value + i * chunk
        check(acl.aclrtMapMem(addr, chunk, 0, handle, 0), "map")
        maps.append(addr)
    storage = torch_npu._C._construct_storage_from_data_pointer(ptr.value, torch.device("npu:0"), weight_size)
    metadata = dict(size=torch.Size([n, k]), stride=[k, 1], nbytes=weight_size,
                    dtype=torch.bfloat16, data_ptr=ptr.value, storage_offset=0,
                    device=torch.device("npu:0"), npu_format=fmt_id, layout=torch.strided,
                    memory_format=torch.contiguous_format, requires_grad=False)
    w = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
    assert w.data_ptr() == ptr.value and torch_npu.get_npu_format(w) == fmt_id
    w.fill_(0.01)
    x = torch.ones((8, k), dtype=torch.bfloat16, device="npu:0")
    torch.npu.synchronize()
    print("VMM_METADATA", json.dumps(dict(mode=mode, format=fmt, shape=[8, n, k], logical_bytes=weight_size,
          physical_bytes=len(handles)*chunk, mapped_bytes=mapped_size, address=hex(ptr.value))), flush=True)
    capture_dir = os.environ.get("PROBE_FRAMEWORK_OUT")
    capture = nullcontext()
    if capture_dir:
        from torch_npu.profiler import (AiCMetrics, ExportType, ProfilerActivity,
            ProfilerLevel, _ExperimentalConfig, profile, tensorboard_trace_handler)
        config = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization, l2_cache=True,
            data_simplification=False, op_attr=True, export_type=[ExportType.Db, ExportType.Text])
        capture = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True, experimental_config=config,
            on_trace_ready=tensorboard_trace_handler(capture_dir))
    with capture:
        for _ in range(int(os.environ.get("PROBE_CALLS", "1"))):
            y = torch.nn.functional.linear(x, w)
        torch.npu.synchronize()
    torch.npu.synchronize()
    reference = float(torch.tensor(k * float(torch.tensor(.01, dtype=torch.bfloat16)), dtype=torch.bfloat16))
    assert bool((y == reference).all().item())
    print("VMM_COMPLETE", json.dumps(dict(first=float(y[0, 0].item()), reference=reference,
          finite=bool(torch.isfinite(y).all().item()))), flush=True)
finally:
    torch.npu.synchronize()
    w = storage = None
    gc.collect()
    for addr in maps:
        check(acl.aclrtUnmapMem(addr), "unmap")
    for handle in handles:
        check(acl.aclrtFreePhysical(handle), "freePhysical")
    check(acl.aclrtReleaseMemAddress(ptr), "release")
