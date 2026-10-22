"""Process-local BF16 vocabulary experiment, TP1/one request/greedy only.

This is a NanoSpec-inspired hybrid, not an exact reproduction: a train-only
frequency seed replaces all-position prefill logits. Target logits and prompt
IDs refresh a recency-ranked device-resident set. Fixed buffer addresses retain
the merged seven-step draft graph. No installed framework file is modified.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

ENABLED = os.environ.get("DYNAMIC_VOCAB_ENABLE", "0") == "1"
CAPACITY = int(os.environ.get("DYNAMIC_VOCAB_CAPACITY", "4096"))
TARGET_TOPK = int(os.environ.get("DYNAMIC_VOCAB_TOPK", "8"))
STATE = None


class Vocabulary:
    def __init__(self, model):
        self.weight = model.lm_head.weight
        self.vocab, self.hidden = self.weight.shape
        assert self.weight.dtype == torch.bfloat16
        assert self.vocab == 248320 and self.hidden == 5120
        assert model.logits_processor.scale == 1.0
        assert model.logits_processor.soft_cap is None
        assert model.lm_head.tp_size == 1
        device = self.weight.device
        seed = json.loads(Path(os.environ["DYNAMIC_VOCAB_SEED"]).read_text())
        self.seed_ids = torch.tensor(seed["token_ids"], dtype=torch.long, device=device)
        assert 0 < len(seed["token_ids"]) < CAPACITY
        self.scores = torch.full((self.vocab,), -1, dtype=torch.int32, device=device)
        self.ids = torch.zeros(CAPACITY, dtype=torch.long, device=device)
        self.valid = torch.zeros(CAPACITY, dtype=torch.bool, device=device)
        self.packed = torch.empty((CAPACITY, self.hidden), dtype=self.weight.dtype, device=device)
        self.request_id = None
        self.step = 0
        self.updates = 0
        self.requests = 0
        self.validation = None
        self.reset("startup", [])
        self.repack()

    def reset(self, request_id, prompt_ids):
        self.request_id = request_id
        self.step = 1
        self.scores.fill_(-1)
        if prompt_ids:
            p = torch.tensor(prompt_ids, device=self.weight.device, dtype=torch.long)
            self.scores.index_fill_(0, p, 1)
        # Protected train-only seed, not populated from any evaluation output.
        self.scores.index_fill_(0, self.seed_ids, 1_000_000_000)
        self.requests += 1

    def repack(self):
        _, indices = torch.topk(self.scores, CAPACITY, sorted=False)
        # Match full-vocabulary argmax's smallest-global-ID tie rule.
        indices = indices.sort().values
        values = self.scores.index_select(0, indices)
        self.ids.copy_(indices)
        self.valid.copy_(values.ge(0))
        torch.index_select(self.weight, 0, self.ids, out=self.packed)

    def update(self, runner, logits):
        assert runner.input_batch.num_reqs == 1, "Only concurrency 1 is supported"
        assert runner.input_batch.sampling_metadata.all_greedy
        req_id = runner.input_batch.req_ids[0]
        fresh_request = req_id != self.request_id
        if fresh_request:
            self.reset(req_id, runner.requests[req_id].prompt_token_ids)
        self.step += 1
        # All available target positions are candidate hints. Rows after the
        # first rejection are conditional on speculative context, not truth.
        top_ids = torch.topk(logits, TARGET_TOPK, dim=-1, sorted=False).indices.flatten()
        self.scores.index_fill_(0, top_ids, self.step)
        draft = getattr(runner, "_draft_token_ids", None)
        if not fresh_request and isinstance(draft, torch.Tensor):
            # Refresh recent proposals too. Clamping only fills a possible
            # placeholder with token zero; it never inserts a negative ID.
            self.scores.index_fill_(0, draft.flatten().long().clamp_min(0), self.step)
        self.scores.index_fill_(0, self.seed_ids, 1_000_000_000)
        self.repack()
        self.updates += 1

    def audit(self):
        torch.npu.synchronize()
        return {"enabled": True, "capacity": CAPACITY, "target_topk": TARGET_TOPK,
                "seed_tokens": self.seed_ids.numel(), "active_tokens": int(self.valid.sum()),
                "updates": self.updates, "requests_including_startup": self.requests,
                "packed_bytes": self.packed.numel() * self.packed.element_size(),
                "ids_pointer": self.ids.data_ptr(), "packed_pointer": self.packed.data_ptr(),
                "validation": self.validation,
                "selection": "protected train-only seed + prompt + recency-ranked target TopK",
                "gather": "once per target round, same stream, included in E2E",
                "limitations": "greedy TP1 concurrency1; no all-position prefill logits; no async overlap"}


def install():
    if not ENABLED:
        return
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_load = AscendSpecDecodeBaseProposer.load_model
    original_sample = AscendSpecDecodeBaseProposer._sample_draft_from_logits
    original_runner_sample = NPUModelRunner._sample

    def load(self, model):
        global STATE
        result = original_load(self, model)
        assert isinstance(self.model, Qwen3_5MTP)
        STATE = Vocabulary(self.model)
        # Selected-row projection validation before any timed requests or capture.
        x = torch.randn((1, STATE.hidden), device=STATE.weight.device, dtype=torch.bfloat16)
        full = self.model.logits_processor(self.model.lm_head, x)
        reduced = F.linear(x, STATE.packed).float()
        reference = full.index_select(-1, STATE.ids)
        error = (reduced - reference).abs()
        STATE.validation = {"max_abs_diff": float(error.max()),
                            "different": int((reduced != reference).sum()),
                            "finite": bool(reduced.isfinite().all())}
        assert STATE.validation["finite"]
        print("DYNAMIC_VOCAB_LOADED", STATE.audit(), flush=True)
        return result

    def logits(self, hidden_states, spec_step_idx=0):
        scores = F.linear(hidden_states, STATE.packed).float()
        return scores.masked_fill(~STATE.valid.unsqueeze(0), -float("inf"))

    def sample(self, logits, sampling_metadata=None):
        if isinstance(self.model, Qwen3_5MTP):
            assert logits.shape[-1] == CAPACITY
            # Greedy only. Do not export reduced-vocabulary probabilities to
            # the target-space rejection sampler.
            local = logits.argmax(dim=-1)
            return STATE.ids.index_select(0, local.reshape(-1)).reshape(local.shape), None
        return original_sample(self, logits, sampling_metadata)

    def runner_sample(self, logits, spec_decode_metadata):
        if STATE is not None and logits is not None:
            STATE.update(self, logits)
        return original_runner_sample(self, logits, spec_decode_metadata)

    AscendSpecDecodeBaseProposer.load_model = load
    Qwen3_5MTP.compute_logits = logits
    AscendSpecDecodeBaseProposer._sample_draft_from_logits = sample
    NPUModelRunner._sample = runner_sample


install()
