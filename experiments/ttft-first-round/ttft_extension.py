"""Bounded first-round telemetry and a prefill-only scheduling control.

Extends count-ready-overlap through COUNT_READY_BASE on PYTHONPATH. All changes
are process-local; no installed source, weight, or NPU kernel is modified.
"""
import functools
import gc
import time

import torch
import model_extension
import early_counts
import dynamic_vocab

TRACE = False
PREFILL_ORIGINAL = False
FIRST = False
EVENTS = []
GC_EVENTS = []
DEVICE_EVENTS = []


def gc_callback(phase, info):
    if TRACE:
        GC_EVENTS.append({"phase": phase, "generation": info["generation"],
                          "wall_ns": time.perf_counter_ns(), "cpu_ns": time.thread_time_ns()})


gc.callbacks.append(gc_callback)


def timed(original, name):
    @functools.wraps(original)
    def call(*args, **kwargs):
        if not TRACE or not FIRST:
            return original(*args, **kwargs)
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            return original(*args, **kwargs)
        finally:
            EVENTS.append({"name": name, "start_ns": start, "end_ns": time.perf_counter_ns(),
                           "cpu_ms": (time.thread_time_ns() - cpu) / 1e6})
    return call


from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

original_execute = NPUModelRunner.execute_model
original_tokens = NPUModelRunner.sample_tokens


@functools.wraps(original_execute)
def execute(self, scheduler_output, *args, **kwargs):
    global FIRST
    FIRST = bool(scheduler_output.scheduled_new_reqs)
    if not TRACE or not FIRST:
        return original_execute(self, scheduler_output, *args, **kwargs)
    a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    a.record()
    try:
        return timed(original_execute, "execute_model")(self, scheduler_output, *args, **kwargs)
    finally:
        b.record()
        DEVICE_EVENTS.append((a, b))


@functools.wraps(original_tokens)
def tokens(self, *args, **kwargs):
    stage = early_counts.STAGE
    if FIRST and PREFILL_ORIGINAL:
        early_counts.STAGE = 0
    try:
        return timed(original_tokens, "sample_tokens")(self, *args, **kwargs)
    finally:
        early_counts.STAGE = stage


NPUModelRunner.execute_model = execute
NPUModelRunner.sample_tokens = tokens
for attr in ("_prepare_inputs", "_sample", "_bookkeeping_sync", "_update_states_after_model_execute",
             "_copy_valid_sampled_token_count", "propose_draft_token_ids"):
    if hasattr(NPUModelRunner, attr):
        setattr(NPUModelRunner, attr, timed(getattr(NPUModelRunner, attr), attr))
for attr in ("update", "reset", "repack"):
    setattr(dynamic_vocab.Vocabulary, attr, timed(getattr(dynamic_vocab.Vocabulary, attr), "vocab." + attr))


class GraphAudit(model_extension.GraphAudit):
    def configure_first_copy(self, enabled=False):
        import first_output_copy
        return first_output_copy.configure(self, enabled)

    def first_copy_audit(self):
        import first_output_copy
        return first_output_copy.audit()

    def reload_first_copy(self):
        import importlib
        import first_output_copy
        assert self.model_runner.execute_model_state is None
        first_output_copy.uninstall()
        importlib.reload(first_output_copy)
        return first_output_copy.configure(self, False)

    def configure_ttft(self, mode, telemetry=False, record_candidates=False):
        global TRACE, PREFILL_ORIGINAL
        assert mode in ("baseline", "ready3", "prefill_original")
        TRACE = False
        result = early_counts.configure(self, 0 if mode == "baseline" else 3, record_candidates)
        PREFILL_ORIGINAL = mode == "prefill_original"
        EVENTS.clear(); GC_EVENTS.clear(); DEVICE_EVENTS.clear()
        TRACE = bool(telemetry)
        return {"mode": mode, "telemetry": TRACE, "prefill_original": PREFILL_ORIGINAL,
                "gc_enabled": gc.isenabled(), "gc_thresholds": gc.get_threshold(), "count_stage": result}

    def first_round_audit(self):
        torch.npu.synchronize()
        return {"events": EVENTS, "gc_events": GC_EVENTS,
                "execute_device_event_ms": [a.elapsed_time(b) for a, b in DEVICE_EVENTS],
                "gc_counts": gc.get_count(), "gc_stats": gc.get_stats(),
                "prefill_original": PREFILL_ORIGINAL, "telemetry": TRACE}
