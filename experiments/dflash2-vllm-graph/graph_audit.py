"""Read-only DFlash2 audit, called outside timed requests."""
from graph_audit_base import GraphAudit as BaseGraphAudit
import compat


class GraphAudit(BaseGraphAudit):
    def graph_audit(self):
        import torch_npu
        result = super().graph_audit()
        drafter = self.model_runner.drafter
        result.update(method=drafter.speculative_config.method,
                      proposer_class=type(drafter).__name__,
                      selector_enabled=bool(getattr(drafter, 'use_dflash2_selector', False)),
                      selector_top_k=getattr(drafter, 'selector_top_k', None),
                      parallel_drafting=drafter.parallel_drafting,
                      noncausal_capture_compat_calls=compat.calls,
                      allocated_bytes=torch_npu.npu.memory_allocated(),
                      reserved_bytes=torch_npu.npu.memory_reserved())
        return result
