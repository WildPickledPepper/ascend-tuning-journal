"""Compare the derived native tiling with stock F.linear, including graph replay."""
import ctypes
import gc
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu

assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '6'
torch.npu.set_device(0)
library = ctypes.CDLL(None)
library.native_m8_set_mode.argtypes = [ctypes.c_int]
library.native_m8_get_count.argtypes = [ctypes.c_int]
library.native_m8_get_count.restype = ctypes.c_uint64
report = {'dtype': 'BF16', 'M': 8, 'device_binary_unchanged': True, 'cases': []}
for shape, n, k, plan in [('down', 5120, 17408, 1), ('up', 34816, 5120, 2)]:
    torch.manual_seed(20261020)
    weight = torch.randn((n, k), device='npu', dtype=torch.bfloat16) / k ** 0.5
    for seed, scale in [(11, 1.0), (12, 0.03125), (13, 32.0), (14, 0.0)]:
        torch.manual_seed(seed)
        x = torch.randn((8, k), device='npu', dtype=torch.bfloat16) * scale
        library.native_m8_set_mode(0)
        reference = torch.nn.functional.linear(x, weight)
        torch.npu.synchronize()
        before = library.native_m8_get_count(0) + library.native_m8_get_count(1)
        library.native_m8_set_mode(plan)
        candidate = torch.nn.functional.linear(x, weight)
        torch.npu.synchronize()
        after = library.native_m8_get_count(0) + library.native_m8_get_count(1)
        assert after > before, 'Target native launch was not intercepted'
        delta = (candidate.float() - reference.float()).abs()
        record = {'shape': shape, 'seed': seed, 'scale': scale,
                  'outputs': candidate.numel(),
                  'finite': bool(torch.isfinite(candidate).all()),
                  'bitwise_equal': bool(torch.equal(candidate, reference)),
                  'different_outputs': int((candidate != reference).sum()),
                  'max_abs_diff': float(delta.max()),
                  'relative_l2': float(torch.linalg.vector_norm(delta) /
                                       torch.linalg.vector_norm(reference.float()).clamp_min(1e-20))}
        assert record['finite'] and record['relative_l2'] < 0.003
        print(json.dumps(record), flush=True)
        report['cases'].append(record)
        # Capturing the patched launch must retain its arguments after hook returns.
        with torch.npu.stream(torch.npu.Stream()):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                replay_output = torch.nn.functional.linear(x, weight)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(replay_output, candidate)
        record['graph_replay_equal_to_candidate'] = True
        del graph, x, reference, candidate, delta, replay_output
    del weight
    gc.collect()
    torch.npu.empty_cache()
report['patch_counts'] = [library.native_m8_get_count(i) for i in range(2)]
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print('VALIDATION_COMPLETE', flush=True)
