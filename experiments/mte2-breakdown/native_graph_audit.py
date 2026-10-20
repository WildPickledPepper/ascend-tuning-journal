"""Inspect experimental native tiling. Select the mode before service startup.

Hot recapture was rejected after attention update handles failed with 107033.
"""
from graph_audit_base import GraphAudit as BaseGraphAudit


class GraphAudit(BaseGraphAudit):
    def graph_audit(self):
        import ctypes
        import os
        result = super().graph_audit()
        lib = ctypes.CDLL(None)
        lib.native_m8_get_count.argtypes = [ctypes.c_int]
        lib.native_m8_get_count.restype = ctypes.c_uint64
        result['native_m8_patch_counts'] = [lib.native_m8_get_count(i) for i in range(2)]
        result['native_m8_mode'] = int(os.environ.get('NATIVE_M8_RETILE', '0'))
        result['native_m8_preserve_l1'] = int(os.environ.get('NATIVE_M8_PRESERVE_L1', '0'))
        return result

    def recapture_native_matmul(self, mode):
        raise RuntimeError('Hot recapture is unsupported: attention task-update handles require a fresh service.')
