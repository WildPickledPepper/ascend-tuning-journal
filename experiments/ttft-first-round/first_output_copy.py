"""Copy the first sampled result before draft work; retain original conversion.

Only TP1 greedy first-round outputs without logprobs/routing/fault checks are
eligible. Later rounds use the original output constructor. No new GPU math.
"""
import collections
import functools
import ast
import hashlib
import inspect
import textwrap

import torch

ENABLED = False
INSTALLED = False
PENDING = {}
KEEPALIVE = []
COUNTS = collections.Counter()
SOURCE = {}
ORIGINAL = {}


def eligible_prefill(scheduler_output):
    if len(scheduler_output.scheduled_new_reqs) != 1 or len(scheduler_output.num_scheduled_tokens) != 1:
        return False
    req = scheduler_output.scheduled_new_reqs[0]
    prompt = getattr(req, "prompt_token_ids", None)
    req_id = getattr(req, "req_id", None)
    if prompt is None or req_id is None:
        return False
    remaining = len(prompt) - getattr(req, "num_computed_tokens", 0)
    if scheduler_output.num_scheduled_tokens.get(req_id, 0) < remaining:
        return False
    return True


def flush_prefill(scheduler_output):
    if not ENABLED or not eligible_prefill(scheduler_output):
        return False
    COUNTS["prefill_outputs_flushed"] += 1
    COUNTS["new_prefill_outputs_flushed"] += 1
    return True


def flush_pending_prefill(batch_queue):
    if not ENABLED or not batch_queue or not eligible_prefill(batch_queue[-1][1]):
        return False
    COUNTS["prefill_outputs_flushed"] += 1
    COUNTS["pending_prefill_outputs_flushed"] += 1
    return True


def install():
    global INSTALLED
    if INSTALLED:
        return
    import ttft_extension
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
    from vllm.v1.engine.core import EngineCore
    original_sample = NPUModelRunner._sample
    original_init = AsyncGPUModelRunnerOutput.__init__
    original_get = AsyncGPUModelRunnerOutput.get_output
    ORIGINAL.update(sample=original_sample, init=original_init, get=original_get)
    source = textwrap.dedent(inspect.getsource(original_init))
    tree = ast.parse(source)
    fields = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
              and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ast.Store)}
    expected = {"_model_runner_output", "_invalid_req_indices", "async_copy_ready_event", "_sampled_token_ids",
                "vocab_size", "_logprobs_tensors", "_routed_experts", "_has_fault", "sampled_token_ids_cpu",
                "_logprobs_tensors_cpu", "_routed_experts_cpu"}
    assert fields == expected, ("Async output constructor field set changed", fields ^ expected)
    SOURCE.update({"constructor_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "get_output_sha256": hashlib.sha256(inspect.getsource(AsyncGPUModelRunnerOutput.get_output).encode()).hexdigest(),
                   "constructor_fields_checked": len(fields), "conversion_uses_original_get_output": True})
    core = inspect.unwrap(EngineCore.step_with_batch_queue)
    assert core.__closure__ is None
    core_source = textwrap.dedent(inspect.getsource(core))
    tree = ast.parse(core_source)
    fn = tree.body[0]
    fn.decorator_list = []
    clauses = [n for n in ast.walk(fn) if isinstance(n, ast.If)
               and "len(batch_queue) < self.batch_queue_size" in ast.unparse(n.test)]
    assert len(clauses) == 1
    clauses[0].test = ast.BoolOp(op=ast.And(), values=[clauses[0].test,
        ast.UnaryOp(op=ast.Not(), operand=ast.Call(func=ast.Name(id="_ttft_flush_prefill", ctx=ast.Load()),
                                                args=[ast.Name(id="scheduler_output", ctx=ast.Load())], keywords=[]))])
    schedule_clauses = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "self.scheduler.has_requests()"]
    assert len(schedule_clauses) == 1
    schedule_clauses[0].test = ast.BoolOp(op=ast.And(), values=[schedule_clauses[0].test,
        ast.UnaryOp(op=ast.Not(), operand=ast.Call(func=ast.Name(id="_ttft_flush_pending_prefill", ctx=ast.Load()),
                                                args=[ast.Name(id="batch_queue", ctx=ast.Load())], keywords=[]))])
    ast.fix_missing_locations(tree)
    namespace = dict(core.__globals__)
    namespace["_ttft_flush_prefill"] = flush_prefill
    namespace["_ttft_flush_pending_prefill"] = flush_pending_prefill
    exec(compile(tree, "<first-prefill-delivery>", "exec"), namespace)
    ORIGINAL.update(core=core, core_code=core.__code__)
    core.__globals__["_ttft_flush_prefill"] = flush_prefill
    core.__globals__["_ttft_flush_pending_prefill"] = flush_pending_prefill
    SOURCE.update({"core_original_sha256": hashlib.sha256(core_source.encode()).hexdigest(),
                   "core_modified_sha256": hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
                   "core_changed_clauses": 2})

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        output = original_sample(self, *args, **kwargs)
        if (ENABLED and ttft_extension.FIRST and self.use_async_scheduling
                and self.num_discarded_requests == 0
                and self.input_batch.sampling_metadata.all_greedy
                and not self.routed_experts_initialized and output.logprobs_tensors is None):
            KEEPALIVE[:] = [entry for entry in KEEPALIVE if not (entry[1].query() and entry[2].query())]
            ids = output.sampled_token_ids
            assert self.input_batch.num_reqs == 1 and ids.shape == (1, 1) and id(ids) not in PENDING
            default = torch.npu.current_stream()
            event = torch.npu.Event(blocking=True)
            with torch.npu.stream(self.async_output_copy_stream):
                self.async_output_copy_stream.wait_stream(default)
                cpu = ids.to("cpu", non_blocking=True)
                event.record()
            PENDING[id(ids)] = (ids, cpu, event)
            COUNTS["first_copies_primed"] += 1
        return output

    @functools.wraps(original_init)
    def init(self, model_runner_output, sampled_token_ids, logprobs_tensors,
             invalid_req_indices, async_output_copy_stream, vocab_size,
             routed_experts=None, check_ep_fault=False):
        pending = PENDING.pop(id(sampled_token_ids), None)
        if pending is None:
            return original_init(self, model_runner_output, sampled_token_ids, logprobs_tensors,
                                 invalid_req_indices, async_output_copy_stream, vocab_size,
                                 routed_experts, check_ep_fault)
        ids, cpu, event = pending
        assert ids is sampled_token_ids and logprobs_tensors is None and routed_experts is None and not check_ep_fault
        # Keep the original internal CPU buffer AND event, including the late
        # copy. Some internal input preparation retains those objects. The
        # extra early copy is only one token and serves client delivery alone.
        original_init(self, model_runner_output, sampled_token_ids, logprobs_tensors,
                      invalid_req_indices, async_output_copy_stream, vocab_size,
                      routed_experts, check_ep_fault)
        from vllm_ascend.worker.model_runner_v1 import global_stream
        draft_done, state_done = torch.npu.Event(), torch.npu.Event()
        draft_done.record()
        state_done.record(global_stream())
        KEEPALIVE.append((ids, draft_done, state_done))
        self._first_token_delivery_event = event
        self._first_token_delivery_cpu = cpu
        COUNTS["first_copies_consumed"] += 1

    @functools.wraps(original_get)
    def get(self):
        early = getattr(self, "_first_token_delivery_event", None)
        if early is None:
            return original_get(self)
        internal = self.async_copy_ready_event
        internal_cpu = self.sampled_token_ids_cpu
        self.async_copy_ready_event = early
        self.sampled_token_ids_cpu = self._first_token_delivery_cpu
        try:
            # Conversion, invalid-token filtering, logprobs and return object
            # all remain the installed implementation. input_batch already
            # captured the separate internal event before this call is exposed.
            return original_get(self)
        finally:
            self.async_copy_ready_event = internal
            self.sampled_token_ids_cpu = internal_cpu

    NPUModelRunner._sample = sample
    AsyncGPUModelRunnerOutput.__init__ = init
    AsyncGPUModelRunnerOutput.get_output = get
    core.__code__ = namespace[core.__name__].__code__
    INSTALLED = True


def configure(worker, enabled):
    global ENABLED
    torch.npu.synchronize()
    assert not PENDING
    KEEPALIVE.clear(); COUNTS.clear()
    install()
    ENABLED = bool(enabled)
    return audit()


def audit():
    assert COUNTS["first_copies_primed"] == COUNTS["first_copies_consumed"] and not PENDING
    return {"enabled": ENABLED, "counts": dict(COUNTS), "pending": len(PENDING),
            "retained_first_token_tensors": len(KEEPALIVE),
            "source": SOURCE,
            "scope": "completed fresh prefill only; TP1 greedy; original internal CPU buffer/fence retained; first output before queue refill; later rounds unchanged"}


def uninstall():
    global INSTALLED, ENABLED
    torch.npu.synchronize()
    assert not PENDING
    if INSTALLED:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
        NPUModelRunner._sample = ORIGINAL["sample"]
        AsyncGPUModelRunnerOutput.__init__ = ORIGINAL["init"]
        AsyncGPUModelRunnerOutput.get_output = ORIGINAL["get"]
        ORIGINAL["core"].__code__ = ORIGINAL["core_code"]
    KEEPALIVE.clear()
    ENABLED = INSTALLED = False
