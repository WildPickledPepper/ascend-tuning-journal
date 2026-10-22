"""CPU labels around original Python calls; no inference algorithm changes."""
from contextlib import nullcontext
import functools

ENABLED = False


def install():
    import torch
    from dynamic_vocab import Vocabulary
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    def wrap(original, label):
        @functools.wraps(original)
        def call(*args, **kwargs):
            scope = torch.profiler.record_function(label) if ENABLED else nullcontext()
            with scope:
                return original(*args, **kwargs)
        return call

    Vocabulary.update = wrap(Vocabulary.update, "dynamic_vocab.update")
    Vocabulary.repack = wrap(Vocabulary.repack, "dynamic_vocab.repack")
    Vocabulary.reset = wrap(Vocabulary.reset, "dynamic_vocab.reset")
    NPUModelRunner._sample = wrap(NPUModelRunner._sample, "dynamic_decode.sample_and_update")
