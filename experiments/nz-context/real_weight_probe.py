"""Hold graph/addresses fixed and replace random values with actual model weights."""
import datetime
import gc
import json
import os
import pathlib
import statistics
import sys

import torch
import torch_npu
from safetensors import safe_open
from torch_npu.profiler import (AiCMetrics, ExportType, ProfilerActivity, ProfilerLevel,
                                _ExperimentalConfig, profile, tensorboard_trace_handler)

model, out = map(pathlib.Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "6"
torch.npu.config.allow_internal_format = True
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              physical_device=6, M=2, dtype="BF16", source_layers=[0, 1],
              controlled="Same tensors, data pointers, graph and inputs; only weight values change",
              cases=[])


def cpu_weights(layer, kind):
    prefix = f"model.language_model.layers.{layer}.mlp."
    def get(suffix):
        key = prefix + suffix + ".weight"
        with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
            t = f.get_tensor(key)
        assert t.dtype == torch.bfloat16
        return t
    return torch.cat([get("gate_proj"), get("up_proj")], dim=0) if kind == "up" else get("down_proj")


for kind, n, k in [("up", 34816, 5120), ("down", 5120, 17408)]:
    torch.manual_seed(137)
    ws = [torch.randn(n, k, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    zs = [torch_npu.npu_format_cast(w, 29) for w in ws]
    x = torch.randn(2, k, device="npu", dtype=torch.bfloat16)
    pointers = [w.data_ptr() for w in ws + zs]
    graphs, outputs = {}, {}
    for layout, weights in [("ND", ws), ("NZ", zs)]:
        for _ in range(3):
            for w in weights:
                y = torch.nn.functional.linear(x, w)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        outputs[layout] = []
        with torch.npu.graph(g):
            for w in weights:
                outputs[layout].append(torch.nn.functional.linear(x, w))
        graphs[layout] = g
    for content in ["random", "real"]:
        if content == "real":
            for layer, (w, z) in enumerate(zip(ws, zs)):
                cpu = cpu_weights(layer, kind)
                assert tuple(cpu.shape) == (n, k)
                w.copy_(cpu)
                z.copy_(torch_npu.npu_format_cast(w, 29))
            assert [w.data_ptr() for w in ws + zs] == pointers
        for _ in range(5):
            for g in graphs.values():
                g.replay()
        torch.npu.synchronize()
        samples = []
        for i in range(40):
            row = {}
            for layout in (["ND", "NZ"] if i % 2 == 0 else ["NZ", "ND"]):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[layout].replay()
                end.record()
                end.synchronize()
                row[layout] = start.elapsed_time(end) * 1000 / 2
            samples.append(row)
        row = dict(kind=kind, N=n, K=k, content=content, addresses_unchanged=True,
                   pointer_mod_2MiB=[p % (2 * 1024**2) for p in pointers],
                   median_us={s: statistics.median(r[s] for r in samples) for s in ["ND", "NZ"]},
                   samples=samples)
        result["cases"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
        cfg = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                  aic_metrics=AiCMetrics.PipeUtilization,
                                  export_type=[ExportType.Db, ExportType.Text])
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                     record_shapes=True, experimental_config=cfg,
                     on_trace_ready=tensorboard_trace_handler(str(out / f"{kind}-{content}"))):
            for layout, g in graphs.items():
                for _ in range(4):
                    g.replay()
                torch.npu.synchronize()
    del graphs, outputs, ws, zs, x
    gc.collect()
    torch.npu.empty_cache()
result["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(out / "real-weight.json").write_text(json.dumps(result, indent=2) + "\n")
