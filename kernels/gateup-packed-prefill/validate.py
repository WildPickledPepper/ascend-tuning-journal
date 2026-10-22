"""Bounded real-weight correctness and latency check on physical card 7."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent
MODEL=Path(os.environ["QWEN_MODEL_PATH"])
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "7"
torch.set_num_threads(4)
torch.npu.set_device(0)
torch.ops.load_library(str(ROOT/"libprefill_torch.so"))
index=json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]


def load(layer,kind):
    key=f"model.language_model.layers.{layer}.mlp.{kind}_proj.weight"
    with safe_open(MODEL/index[key],framework="pt",device="cpu") as f:
        return f.get_tensor(key)


def pack(w):
    return w.reshape(136,16,16,20,16,16).permute(0,3,4,1,2,5).contiguous().reshape(34816,5120)


def unpack(w):
    return w.view(136,20,16,16,16,16).permute(0,3,4,1,2,5).contiguous().view(34816,5120)


def timing(fn):
    for _ in range(3):fn()
    torch.npu.synchronize()
    vals=[]
    for _ in range(12):
        a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        a.record();y=fn();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
    return {"median_us":statistics.median(vals),"samples_us":vals}


result={"device":7,"dtype":"BF16","cases":[],"timings":{},
        "tiling":{"BM":128,"BN":256,"BK":256,"L1_A_bytes":131072,"L1_B_bytes":262144,
                  "L0A_bytes":32768,"L0B_bytes":65536,"L0C_bytes":131072},
        "kernel_source_sha256":hashlib.sha256((ROOT/"kernel.asc").read_bytes()).hexdigest()}
out=ROOT/"results/validation.json"
out.parent.mkdir(exist_ok=True)
with torch.inference_mode():
    for layer in (0,31):
        cpu=torch.cat([load(layer,"gate"),load(layer,"up")],dim=0)
        w=cpu.npu();p=pack(cpu).npu()
        assert torch.equal(unpack(p).cpu(),cpu)
        for m in (1,17,65,83,92,114,121,128,257,512):
            torch.manual_seed(20261021+m+layer)
            x=torch.randn(m,5120,dtype=torch.bfloat16,device="npu")
            ref=torch.nn.functional.linear(x,w).cpu()
            y=torch.ops.packed_prefill_bf16.linear(x,p,0).cpu()
            diff=(y.float()-ref.float()).abs()
            row={"layer":layer,"M":m,"different":int((y.view(torch.int16)!=ref.view(torch.int16)).sum()),
                 "max_abs_diff":float(diff.max()),"reference_max_abs":float(ref.abs().max()),"finite":bool(y.isfinite().all())}
            result["cases"].append(row);out.write_text(json.dumps(result,indent=2))
            print("CHECK",json.dumps(row),flush=True)
            assert row["finite"] and row["max_abs_diff"]<=max(1e-5,row["reference_max_abs"]*.01),row
            if layer==0 and m in (83,121,512):
                result["timings"][str(m)]={"native":timing(lambda:torch.nn.functional.linear(x,w)),
                    "unpack_native":timing(lambda:torch.nn.functional.linear(x,unpack(p))),
                    "packed_direct":timing(lambda:torch.ops.packed_prefill_bf16.linear(x,p,0))}
                print("TIMING",m,json.dumps({k:v["median_us"] for k,v in result["timings"][str(m)].items()}),flush=True)
        del w,p,cpu,x,ref,y
        torch.npu.empty_cache()
result["all_bit_equal"]=all(r["different"]==0 for r in result["cases"])
result["passed"]=True
out.write_text(json.dumps(result,indent=2))
print("VALIDATION_DONE",result["all_bit_equal"],flush=True)
