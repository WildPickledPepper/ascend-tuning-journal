"""Measure cache PMUs for the same native ND/NZ graphs, with a controlled read intervention."""
import datetime
import json
import os
import pathlib
import statistics
import sys

import torch
import torch_npu
from torch_npu.profiler import (AiCMetrics, ExportType, ProfilerActivity, ProfilerLevel,
                                _ExperimentalConfig, profile, tensorboard_trace_handler)

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "6"
assert os.environ["TASK_QUEUE_ENABLE"] == "1"
torch.npu.config.allow_internal_format = True
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)
torch.manual_seed(137)
assert torch.npu.mem_get_info()[0] > 4 * 1024**3

busters = [torch.randn(32768, 8192, device="npu", dtype=torch.bfloat16) for _ in range(2)]
bx = torch.randn(2, 8192, device="npu", dtype=torch.bfloat16)
kept, outputs, graphs = [*busters, bx], [], {}
for n, k in [(34816, 5120), (5120, 17408)]:
    weights = [torch.randn(n, k, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    nz = [torch_npu.npu_format_cast(w, 29) for w in weights]
    x = torch.randn(2, k, device="npu", dtype=torch.bfloat16)
    kept += [*weights, *nz, x]
    for layout, ws in [("ND", weights), ("NZ", nz)]:
        for _ in range(3):
            for w in ws:
                y = torch.nn.functional.linear(x, w)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for w in ws:
                outputs.append(torch.nn.functional.linear(x, w))
        graphs[n, k, layout] = graph

g_bust = torch.npu.NPUGraph()
with torch.npu.graph(g_bust):
    for b in busters:
        outputs.append(torch.nn.functional.linear(bx, b))
torch.npu.synchronize()

result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              physical_device=6, dtype="BF16", M=2, custom_kernel=False,
              intervention="Read two disjoint 512 MiB weights before each target graph replay",
              calls_per_target_graph=2, timing_profiler_disabled=True, cases=[])
for n, k in [(34816, 5120), (5120, 17408)]:
    for layout in ["ND", "NZ"]:
        g = graphs[n, k, layout]
        for _ in range(5):
            g.replay()
        torch.npu.synchronize()
        samples = []
        for i in range(40):
            row = {}
            # Pair states with alternating order; unmeasured graph reads are on the same stream.
            for state in (["repeat", "other_reads"] if i % 2 == 0 else ["other_reads", "repeat"]):
                if state == "other_reads":
                    g_bust.replay()
                else:
                    for _ in range(3):
                        g.replay()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                g.replay()
                end.record()
                end.synchronize()
                row[state] = start.elapsed_time(end) * 1000 / 2
            samples.append(row)
        item = dict(N=n, K=k, layout=layout, paired_rounds=40,
                    median_us={s: statistics.median(r[s] for r in samples) for s in ["repeat", "other_reads"]},
                    samples=samples)
        result["cases"].append(item)
        print(json.dumps({k: v for k, v in item.items() if k != "samples"}), flush=True)

for state in ["repeat", "other_reads"]:
    cfg = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                              aic_metrics=AiCMetrics.L2Cache,
                              export_type=[ExportType.Db, ExportType.Text])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                 record_shapes=True, experimental_config=cfg,
                 on_trace_ready=tensorboard_trace_handler(str(out / ("l2-" + state)))):
        for key, g in graphs.items():
            n, k, layout = key
            # Start each observation from the declared state, including the first replay.
            for _ in range(4):
                if state == "other_reads":
                    g_bust.replay()
                else:
                    for _ in range(3):
                        g.replay()
                with torch.profiler.record_function(f"measure_{state}_{layout}_N{n}_K{k}"):
                    g.replay()
                torch.npu.synchronize()
result["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(out / "cache-graph.json").write_text(json.dumps(result, indent=2) + "\n")
