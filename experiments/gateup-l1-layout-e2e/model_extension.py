"""Isolated serving experiment; never modifies installed vLLM sources."""
import collections
import os
from pathlib import Path
import re
import time

import torch
import torch_npu
from graph_audit_base import GraphAudit as BaseGraphAudit
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

ROOT = Path(__file__).resolve().parent
ENABLED = os.environ.get("GATE_LAYOUT_ENABLE", "0") == "1"
DOWN_ROOT = ROOT.parents[1] / "kernels/down-prefetch-m8"
torch.ops.load_library(str(DOWN_ROOT / "libdown_torch.so"))
torch.ops.load_library(str(ROOT.parents[1] / "kernels/gateup-l1-layout/liblayout_torch.so"))
PACKED = {}
ELIGIBLE = []
COUNTS = collections.Counter()
PACK_SECONDS = 0.0


@torch.library.register_fake("l1_tile_layout::linear")
def gate_fake(x, w, layout):
    return x.new_empty((8, 34816))


def pack(w):
    return w.reshape(136, 16, 16, 20, 16, 16).permute(0, 3, 4, 1, 2, 5).contiguous().reshape(34816, 5120)


def unpack(w):
    return w.view(136, 20, 16, 16, 16, 16).permute(0, 3, 4, 1, 2, 5).contiguous().view(34816, 5120)


ORIGINAL_PROCESS = AscendUnquantizedLinearMethod.process_weights_after_loading


def process(self, layer):
    global PACK_SECONDS
    ORIGINAL_PROCESS(self, layer)
    prefix = getattr(layer, "prefix", "")
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
    if (match and int(match[1]) < 64 and "mtp" not in prefix.lower()
            and tuple(layer.weight.shape) == (34816, 5120)
            and layer.weight.dtype == torch.bfloat16):
        ELIGIBLE.append(prefix)
        if ENABLED:
            start = time.perf_counter()
            source = layer.weight.detach().cpu()
            packed = pack(source)
            assert torch.equal(unpack(packed), source), prefix
            layer.weight.data = packed.to(layer.weight.device)
            torch.npu.synchronize()
            PACKED[layer.weight.data_ptr()] = prefix
            PACK_SECONDS += time.perf_counter() - start
        print("GATE_LAYOUT_LAYER", len(ELIGIBLE), ENABLED, prefix, flush=True)


def gemm(x, weight, bias=None):
    if weight.data_ptr() in PACKED:
        COUNTS[f"packed_M{x.shape[0]}"] += 1
        if (bias is None and x.ndim == 2 and tuple(x.shape) == (8, 5120)
                and x.dtype == torch.bfloat16 and x.is_contiguous()):
            COUNTS["gate_custom_host_calls"] += 1
            return torch.ops.l1_tile_layout.linear(x, weight, 2)
        COUNTS["gate_unpack_fallback_host_calls"] += 1
        # Temporary one-layer materialization; its time is included in requests.
        return torch.nn.functional.linear(x, unpack(weight), bias)
    if (bias is None and x.ndim == 2 and tuple(x.shape) == (8, 17408)
            and tuple(weight.shape) == (5120, 17408)
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_contiguous() and weight.is_contiguous()
            and torch_npu.get_npu_format(weight) in (0, 2)):
        COUNTS["down_custom_host_calls"] += 1
        return torch.ops.down_prefetch_m8.linear(x, weight, 1)
    return torch.nn.functional.linear(x, weight, bias)


AscendUnquantizedLinearMethod.process_weights_after_loading = process
REGISTRATION = torch.library.Library("vllm", "IMPL", "PrivateUse1")
REGISTRATION.impl("unquantized_gemm", gemm, allow_override=True)


class GraphAudit(BaseGraphAudit):
    def graph_audit(self):
        result = super().graph_audit()
        result["layout_experiment"] = {
            "enabled": ENABLED, "eligible": ELIGIBLE,
            "packed_count": len(PACKED), "pack_seconds_startup": PACK_SECONDS,
            "counts": dict(COUNTS),
            "down_prefetch_in_both_arms": True,
            "counter_note": "Host dispatch counts include capture, not graph replay executions.",
            "npu_allocated_bytes": torch.npu.memory_allocated(),
            "npu_reserved_bytes": torch.npu.memory_reserved(),
        }
        return result
