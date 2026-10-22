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
PREFILL_ENABLED = os.environ.get("PACKED_PREFILL_ENABLE", "0") == "1"
torch.ops.load_library(os.environ["DOWN_LIBRARY"])
torch.ops.load_library(os.environ["GATE_DECODE_LIBRARY"])
torch.ops.load_library(os.environ["PREFILL_LIBRARY"])
PACKED = {}
ELIGIBLE = []
COUNTS = collections.Counter()
PACK_SECONDS = 0.0
PREFILL_CHECKS = {}


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
        if (PREFILL_ENABLED and bias is None and x.ndim == 2
                and 1 <= x.shape[0] <= 512 and x.shape[1] == 5120
                and x.dtype == torch.bfloat16 and x.is_contiguous()):
            COUNTS["prefill_direct_host_calls"] += 1
            y = torch.ops.packed_prefill_bf16.linear(x, weight, 0)
            prefix = PACKED[weight.data_ptr()]
            # The first warmup has 65 real prompt rows. Check every layer,
            # outside measured requests, without changing subsequent results.
            if x.shape[0] == 65 and prefix not in PREFILL_CHECKS:
                reference = torch.nn.functional.linear(x, unpack(weight))
                yc, rc = y.cpu(), reference.cpu()
                error = (yc.float() - rc.float()).abs()
                info = {"M": 65, "different": int((yc.view(torch.int16) != rc.view(torch.int16)).sum()),
                        "max_abs_diff": float(error.max()), "finite": bool(yc.isfinite().all())}
                PREFILL_CHECKS[prefix] = info
                print("PREFILL_REAL_CHECK", prefix, info, flush=True)
                assert info["finite"] and info["different"] == 0, (prefix, info)
            return y
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
            "prefill_direct_enabled": PREFILL_ENABLED,
            "prefill_real_checks": PREFILL_CHECKS,
        }
        return result


import dynamic_vocab
OriginalLayoutAudit = GraphAudit


class GraphAudit(OriginalLayoutAudit):
    def graph_audit(self):
        result = super().graph_audit()
        result["dynamic_vocabulary"] = (dynamic_vocab.STATE.audit()
                                        if dynamic_vocab.STATE is not None
                                        else {"enabled": False, "vocabulary": 248320})
        runner = self.model_runner
        layers = []
        context = runner.vllm_config.compilation_config.static_forward_context
        for name, layer in context.items():
            if "GatedDeltaNetAttention" not in type(layer).__name__:
                continue
            conv, state = layer.kv_cache
            layers.append({"name": name, "conv_dtype": str(conv.dtype),
                           "ssm_dtype": str(state.dtype), "ssm_shape": list(state.shape),
                           "logical_state_view_bytes": state.numel() * state.element_size()})
        result["gdn_state"] = {
            "layers": layers, "layer_count": len(layers),
            "all_ssm_fp32": bool(layers) and all(x["ssm_dtype"] == "torch.float32" for x in layers),
            "all_conv_bf16": bool(layers) and all(x["conv_dtype"] == "torch.bfloat16" for x in layers),
            "allocated_bytes": torch.npu.memory_allocated(),
            "reserved_bytes": torch.npu.memory_reserved(),
        }
        return result


if os.environ.get("STATE_DIAGNOSTIC_ENABLE", "0") == "1":
    import diagnostic
    OriginalStateAudit = GraphAudit

    class GraphAudit(OriginalStateAudit):
        def diag_select(self, mode, label=None):
            return diagnostic.select(self, mode, label)

        def diag_save(self):
            return diagnostic.save(self)


from profile_factory import ProfileControl
import profile_scopes
profile_scopes.install()
OriginalProfileAudit = GraphAudit


class GraphAudit(ProfileControl, OriginalProfileAudit):
    pass

import early_counts
early_counts.install()
OriginalOverlapAudit = GraphAudit

class GraphAudit(OriginalOverlapAudit):
    def configure_overlap(self, enabled, record_candidates=False):
        return early_counts.configure(self, enabled, record_candidates)

    def overlap_audit(self):
        return early_counts.audit(self)
