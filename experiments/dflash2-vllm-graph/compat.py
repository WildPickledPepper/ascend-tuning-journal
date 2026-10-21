"""Process-local V1 graph capture compatibility for unmasked DFlash attention.

The installed eager branch prioritizes causal=False over sliding_window, issuing
sparse_mode=0 with the operator's unrestricted pre/next defaults. Capture must
record the same parameters for replay. No attention kernels are recompiled and
no target-model/causal or masked attention path is changed.
"""
import functools

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

calls = 0
original = AscendAttentionBackendImpl.full_graph_fia


@functools.wraps(original)
def full_graph_fia(self, query, key, value, attn_metadata, output, layer=None):
    global calls
    if (_EXTRA_CTX.is_draft_model and not attn_metadata.causal
            and attn_metadata.attn_mask is None and self.sliding_window):
        previous = self.sliding_window
        # The original function now records sparse=0, pre=INT_MAX,next=INT_MAX,
        # matching its existing non-causal eager branch. Restore the instance's
        # configuration afterward; captured task parameters retain these values.
        self.sliding_window = None
        calls += 1
        try:
            return original(self, query, key, value, attn_metadata, output, layer)
        finally:
            self.sliding_window = previous
    return original(self, query, key, value, attn_metadata, output, layer)


AscendAttentionBackendImpl.full_graph_fia = full_graph_fia
