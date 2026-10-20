"""Process-local observation, preserving the stock BF16 CANN calculation."""
from observe_weights import WorkerExtension as PreviousExtension

class WorkerExtension(PreviousExtension):
    def nz_hardware_probe(self, **kwargs):
        import importlib, hardware_probe
        return importlib.reload(hardware_probe).run(self, **kwargs)
