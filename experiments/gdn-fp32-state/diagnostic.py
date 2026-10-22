"""Same-process diagnostic only: full or masked full head in ONE draft graph.

No throughput claims. Trace actual target inputs/hidden states/logits before
verification; runtime mask contents change while graph addresses stay fixed.
"""
import array
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from dynamic_vocab import Vocabulary
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

MODE = "full"
LABEL = None
STATE = None
MASK = None
WRAPPERS = {}
RECORDS = []
HISTORY = {}
LAST_FORWARD = None

ORIGINAL_LOAD = AscendSpecDecodeBaseProposer.load_model
ORIGINAL_LOGITS = Qwen3_5MTP.compute_logits
ORIGINAL_DRAFT_SAMPLE = AscendSpecDecodeBaseProposer._sample_draft_from_logits
ORIGINAL_SAMPLE = NPUModelRunner._sample
ORIGINAL_FORWARD = NPUModelRunner._model_forward


def load(self, model):
    global STATE, MASK
    result = ORIGINAL_LOAD(self, model)
    STATE = Vocabulary(self.model)
    MASK = torch.ones(STATE.vocab, dtype=torch.bool, device=STATE.weight.device)
    WRAPPERS["full"] = self._runnable
    return result


def draft_logits(self, hidden_states, spec_step_idx=0):
    scores = ORIGINAL_LOGITS(self, hidden_states, spec_step_idx)
    return scores.masked_fill(~MASK.unsqueeze(0), -float("inf"))


def draft_sample(self, logits, sampling_metadata=None):
    return ORIGINAL_DRAFT_SAMPLE(self, logits, sampling_metadata)


def forward(self, num_tokens_padded, input_ids=None, positions=None,
            intermediate_tensors=None, inputs_embeds=None, **kwargs):
    global LAST_FORWARD
    result = ORIGINAL_FORWARD(self, num_tokens_padded, input_ids, positions,
                              intermediate_tensors, inputs_embeds, **kwargs)
    if LABEL is not None:
        hidden = result[0] if isinstance(result, tuple) else result
        LAST_FORWARD = {"ids":input_ids.detach().clone(),
                        "positions":positions.detach().clone(),
                        "hidden":hidden.detach().clone()}
    return result


def sample(self, logits, metadata):
    record = None
    if LABEL is not None:
        req_id = self.input_batch.req_ids[0]
        prompt = self.requests[req_id].prompt_token_ids
        history = HISTORY.setdefault(req_id, list(prompt))
        positions = LAST_FORWARD["positions"].cpu()
        if positions.ndim == 2:
            positions = positions[0]
        positions = positions.tolist()
        ids = LAST_FORWARD["ids"].cpu().tolist()
        start = positions[0]
        assert start <= len(history), (start, len(history))
        history = history[:start] + ids
        HISTORY[req_id] = history
        indices = (metadata.logits_indices.cpu().tolist() if metadata is not None
                   else [len(prompt)-1])
        assert len(indices) == logits.shape[0], (indices, logits.shape)
        top = torch.topk(logits, 8, dim=-1)
        top_ids, top_values = top.indices.cpu().tolist(), top.values.float().cpu().tolist()
        hidden = LAST_FORWARD["hidden"][indices].cpu()
        rows = []
        full_rows = {}
        for r,i in enumerate(indices):
            p = positions[i]
            j = p + 1 - len(prompt)
            prefix = history[:p+1]
            rows.append({"output_index":j, "position":p,
                         "prefix_sha256":hashlib.sha256(array.array('i',prefix).tobytes()).hexdigest(),
                         "argmax_id":int(logits[r].argmax()),
                         "top8_ids":top_ids[r], "top8_values":top_values[r]})
            if 280 <= j <= 330:
                full_rows[r] = logits[r].detach().cpu().clone()
        record = {"prompt_length":len(prompt), "input_ids":ids, "positions":positions,
                  "rows":rows, "hidden":hidden, "full_logits":full_rows,
                  "draft_ids":metadata.draft_token_ids.cpu().tolist() if metadata is not None else []}
    if MODE == "masked":
        STATE.update(self, logits)
        if MODE == "masked":
            MASK.zero_()
            MASK.scatter_(0, STATE.ids, STATE.valid)
    output = ORIGINAL_SAMPLE(self, logits, metadata)
    if record is not None:
        record["sampled_ids"] = output.sampled_token_ids.cpu().tolist()
        RECORDS.append(record)
    return output


def select(worker, mode, label=None):
    global MODE, LABEL, RECORDS, HISTORY, LAST_FORWARD
    assert mode in ("full", "masked")
    torch.npu.synchronize()
    runner = worker.model_runner
    drafter = runner.drafter
    assert drafter._runnable is WRAPPERS["full"]
    MODE, LABEL = mode, label
    RECORDS, HISTORY, LAST_FORWARD = [], {}, None
    STATE.request_id = None
    MASK.fill_(True)
    main_model = runner.model.get_language_model() if hasattr(runner.model,"get_language_model") else runner.model
    return {"mode":MODE, "label":LABEL, "main_head_shape":list(main_model.lm_head.weight.shape),
            "main_head_ptr":main_model.lm_head.weight.data_ptr(),
            "draft_head_ptr":drafter.model.lm_head.weight.data_ptr(),
            "packed_ptr":STATE.packed.data_ptr(),
            "mask_ptr":MASK.data_ptr(), "draft_graph_wrapper_id":id(drafter._runnable),
            "main_head_version":main_model.lm_head.weight._version}


def save(worker):
    assert LABEL is not None
    torch.npu.synchronize()
    path = Path(__file__).parent / "results" / (LABEL + "-trace.pt")
    torch.save({"mode":MODE, "label":LABEL, "records":RECORDS,
                "vocabulary":STATE.audit()}, path)
    return {"label":LABEL, "rounds_including_prefill":len(RECORDS), "bytes":path.stat().st_size}


AscendSpecDecodeBaseProposer.load_model = load
Qwen3_5MTP.compute_logits = draft_logits
AscendSpecDecodeBaseProposer._sample_draft_from_logits = draft_sample
NPUModelRunner._model_forward = forward
NPUModelRunner._sample = sample
