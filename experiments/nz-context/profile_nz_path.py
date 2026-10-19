"""Profile the original ND/NZ microbenchmark path; existing model services stay idle."""
import argparse
import datetime
import gc
import json
import os
import pathlib
import statistics

import torch
import torch_npu
from torch_npu.profiler import (
    AiCMetrics, ExportType, ProfilerActivity, ProfilerLevel,
    _ExperimentalConfig, profile, tensorboard_trace_handler,
)

p = argparse.ArgumentParser()
p.add_argument("--output", type=pathlib.Path, required=True)
p.add_argument("--device", type=int, required=True)
a = p.parse_args()
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(a.device)
assert os.environ["TASK_QUEUE_ENABLE"] == "1"
a.output.mkdir(parents=True, exist_ok=True)
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              physical_device=a.device, dtype="BF16", M=2, calls_per_replay=2,
              resident_model="Existing model left running but receives no requests during probe",
              custom_kernel=False, cases=[], profiler_used_only_after_timing=True,
              allow_internal_format=True,
              default_jit_compile_false=torch.npu.is_jit_compile_false(),
              grad_enabled=torch.is_grad_enabled(), inference_mode=torch.is_inference_mode_enabled())
print(json.dumps(result), flush=True)


def save():
    (a.output / "micro-path.json").write_text(json.dumps(result, indent=2) + "\n")


def run(policy):
    graphs, outputs = {}, {}
    cases = []
    keepalive = []
    for n, k in [(34816, 5120), (5120, 17408)]:
        torch.manual_seed(137)
        free, _ = torch.npu.mem_get_info()
        assert free > 4 * n * k * 2 + 1024**3, free
        ws = [torch.randn(n, k, device="npu", dtype=torch.bfloat16) for _ in range(2)]
        zs = [torch_npu.npu_format_cast(w, 29) for w in ws]
        x = torch.randn(2, k, device="npu", dtype=torch.bfloat16)
        assert all(torch_npu.get_npu_format(z) == 29 for z in zs)
        keepalive += [x, *ws, *zs]
        for _ in range(3):
            for w, z in zip(ws, zs):
                torch.nn.functional.linear(x, w)
                torch.nn.functional.linear(x, z)
        torch.npu.synchronize()
        for name, weights in [("ND", ws), ("NZ", zs)]:
            key = (n, k, name)
            g = torch.npu.NPUGraph()
            outputs[key] = []
            with torch.npu.graph(g):
                for w in weights:
                    outputs[key].append(torch.nn.functional.linear(x, w))
            graphs[key] = g
        for _ in range(5):
            for name in ["ND", "NZ"]:
                graphs[n, k, name].replay()
        torch.npu.synchronize()
        mismatches = sum(int((y != z).sum()) for y, z in zip(outputs[n, k, "ND"], outputs[n, k, "NZ"]))
        samples = []
        for i in range(40):
            row = {}
            for name in (["ND", "NZ"] if i % 2 == 0 else ["NZ", "ND"]):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[n, k, name].replay()
                end.record()
                end.synchronize()
                row[name] = start.elapsed_time(end) * 1000 / 2
            samples.append(row)
        med = {name: statistics.median(s[name] for s in samples) for name in ["ND", "NZ"]}
        row = dict(policy=policy, N=n, K=k, M=2, paired_rounds=40,
                   jit_compile_false=torch.npu.is_jit_compile_false(),
                   output_mismatches=mismatches, median_us=med,
                   nz_time_change_pct=100 * (med["NZ"] / med["ND"] - 1), samples=samples)
        cases.append(row)
        result["cases"].append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    config = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                 aic_metrics=AiCMetrics.PipeUtilization,
                                 l2_cache=True, op_attr=True,
                                 export_type=[ExportType.Db, ExportType.Text])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                 record_shapes=True, with_stack=False,
                 experimental_config=config,
                 on_trace_ready=tensorboard_trace_handler(str(a.output / ("profile-" + policy)))):
        for n, k in [(34816, 5120), (5120, 17408)]:
            for name in ["ND", "NZ"]:
                with torch.profiler.record_function(f"{policy}_{name}_N{n}_K{k}"):
                    for _ in range(4):
                        graphs[n, k, name].replay()
                    torch.npu.synchronize()
    graphs.clear()
    outputs.clear()
    keepalive.clear()
    gc.collect()
    torch.npu.empty_cache()


run("default")
if not result["default_jit_compile_false"]:
    torch.npu.set_compile_mode(jit_compile=False)
    run("vllm_jit_false")
result["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
