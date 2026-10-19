"""Compare official ND/NZ BF16 linear paths; no custom kernels or model patching."""
import datetime
import gc
import json
import os
import pathlib
import statistics
import traceback

import torch
import torch_npu

assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "7"
assert os.environ.get("TASK_QUEUE_ENABLE") == "1"
# Match vllm_ascend/worker/model_runner_v1.py, rather than plain Python defaults.
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
root = pathlib.Path(__file__).resolve().parent
result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              physical_device=7, dtype="BF16", custom_kernel=False,
              scope="isolated standard torch.nn.functional.linear, not full-model speed",
              official_candidate="weight_nz_mode=2 -> npu_format_cast(weight, 29)",
              allow_internal_format=True,
              measurements=[], limitations=[
                  "Existing baseline model stays resident on this card and is idle during this probe.",
                  "ND/NZ use identical logical values but CANN may select different accumulation orders.",
                  "Two weight sets per graph; fixed small-M shapes only.",
                  "Paired graph event timings, profiler disabled, no end-to-end throughput claim."])


def save():
    (root / "nz-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def test_shape(n, k):
    torch.manual_seed(137)
    free, total = torch.npu.mem_get_info()
    required = 4 * n * k * 2 + 1024**3
    if free < required:
        raise RuntimeError(f"Insufficient headroom for isolated probe: {free} < {required}")
    weights = [torch.randn(n, k, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    nz_weights = [torch_npu.npu_format_cast(w, 29) for w in weights]
    torch.npu.synchronize()
    formats = [torch_npu.get_npu_format(w) for w in nz_weights]
    assert formats == [29, 29], formats
    exact_layout_roundtrip = all(torch.equal(w, torch_npu.npu_format_cast(z, 2))
                                 for w, z in zip(weights, nz_weights))
    assert exact_layout_roundtrip
    for m in [1, 2]:
        x = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
        outputs = {"ND": [], "NZ": []}
        for _ in range(3):
            for w, z in zip(weights, nz_weights):
                torch.nn.functional.linear(x, w)
                torch.nn.functional.linear(x, z)
        torch.npu.synchronize()
        graphs = {}
        for name, ws in [("ND", weights), ("NZ", nz_weights)]:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for w in ws:
                    outputs[name].append(torch.nn.functional.linear(x, w))
            graphs[name] = graph
        for _ in range(5):
            for graph in graphs.values():
                graph.replay()
        torch.npu.synchronize()
        mismatches = sum(int((y != z).sum()) for y, z in zip(outputs["ND"], outputs["NZ"]))
        max_abs = max(float((y.float() - z.float()).abs().max())
                      for y, z in zip(outputs["ND"], outputs["NZ"]))
        samples = []
        for i in range(40):
            row = {}
            for name in (["ND", "NZ"] if i % 2 == 0 else ["NZ", "ND"]):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                row[name] = start.elapsed_time(end) * 1000 / 2
            samples.append(row)
        med = {name: statistics.median(s[name] for s in samples) for name in graphs}
        item = dict(M=m, N=n, K=k, weights=2, weight_bytes_each=n*k*2,
                    output_comparisons=2*m*n, layout_roundtrip_bitwise_equal=exact_layout_roundtrip,
                    nd_nz_output_mismatches=mismatches, max_abs_difference=max_abs,
                    output_finite=all(bool(y.isfinite().all()) for values in outputs.values() for y in values),
                    nz_format_ids=formats, paired_rounds=40, profiler=False,
                    median_us=med, nz_time_change_pct=100*(med["NZ"]/med["ND"]-1), samples=samples)
        result["measurements"].append(item)
        save()
        print(json.dumps({k:v for k,v in item.items() if k!="samples"}), flush=True)
        del outputs, graphs, graph
        gc.collect()
        torch.npu.empty_cache()


try:
    for n, k in [(34816, 5120), (5120, 17408)]:
        test_shape(n, k)
        gc.collect()
        torch.npu.empty_cache()
    result["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["all_measured_outputs_equal"] = all(r["nd_nz_output_mismatches"] == 0 for r in result["measurements"])
except Exception as exc:
    result["error"] = str(exc)
    result["traceback"] = traceback.format_exc()
    print(result["traceback"], flush=True)
    raise
finally:
    save()
