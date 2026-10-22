"""Process-only schedule experiment. Reuse original sampler and GDN handling.

No installed file, model graph, candidate selection or NPU kernel is changed.
The original async-state-update block is moved immediately after sampling.
Raw logits/previous proposals are snapshotted before sampling so the exact
same-round vocabulary policy observes the original inputs.
"""
import ast
import collections
import hashlib
import inspect
import json
import textwrap
from types import SimpleNamespace

import torch
import dynamic_vocab

ENABLED=False
RECORD=False
COUNTS=collections.Counter()
CANDIDATES=[]
PATCH={}


def snapshot_inputs(runner, logits):
    if logits is None or dynamic_vocab.STATE is None:return None
    draft=getattr(runner,'_draft_token_ids',None)
    if isinstance(draft,torch.Tensor):draft=draft.clone()
    proxy=SimpleNamespace(input_batch=runner.input_batch,requests=runner.requests,_draft_token_ids=draft)
    return proxy,logits.clone()


def update_after_counts(runner, saved):
    if saved is not None:
        with torch.profiler.record_function('early_counts.same_round_vocab'):
            dynamic_vocab.STATE.update(*saved)
        COUNTS['early_vocab_updates']+=1


def install():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    original_tokens=NPUModelRunner.sample_tokens
    original_dynamic=NPUModelRunner._sample
    dynamic_fn=inspect.unwrap(original_dynamic)
    base_sampler=inspect.getclosurevars(dynamic_fn).nonlocals['original_runner_sample']
    source=textwrap.dedent(inspect.getsource(original_tokens))
    tree=ast.parse(source)
    fn=tree.body[0]
    assert isinstance(fn,ast.FunctionDef) and fn.name=='sample_tokens'
    late=[node for node in fn.body if isinstance(node,ast.If) and 'async_state_update' in ast.unparse(node)]
    assert len(late)==1
    late=late[0]
    assert '_update_states_after_model_execute' in ast.unparse(late)
    fn.body.remove(late)
    sample=[node for node in fn.body if isinstance(node,ast.With) and 'sampler_output = self._sample(' in ast.unparse(node)]
    assert len(sample)==1
    index=fn.body.index(sample[0])
    fn.body[index:index]=ast.parse('_early_saved = _overlap_snapshot(self, logits)').body
    ready=[node for node in fn.body if isinstance(node,ast.If) and 'self.sampling_done_event.record()' in ast.unparse(node)]
    assert len(ready)==1
    index=fn.body.index(ready[0])+1
    fn.body[index:index]=[late]+ast.parse('_overlap_vocab_after_counts(self, _early_saved)').body
    ast.fix_missing_locations(tree)
    # Use a private globals copy; installed framework module remains untouched.
    original_fn=inspect.unwrap(original_tokens)
    namespace=dict(original_fn.__globals__)
    namespace.update(_overlap_snapshot=snapshot_inputs,_overlap_vocab_after_counts=update_after_counts)
    exec(compile(tree,'<accepted-count-overlap>','exec'),namespace)
    early_tokens=namespace['sample_tokens']
    PATCH.update(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 transformed_source_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
                 moved_block_sha256=hashlib.sha256(ast.unparse(late).encode()).hexdigest(),
                 preserved_original_state_update=True)

    def sample(self,logits,spec_decode_metadata):
        if ENABLED:
            assert self.use_async_scheduling and self.need_accepted_tokens
            assert self.input_batch.num_reqs==1
            COUNTS['early_sample_calls']+=1
            return base_sampler(self,logits,spec_decode_metadata)
        COUNTS['baseline_sample_calls']+=1
        return original_dynamic(self,logits,spec_decode_metadata)

    def tokens(self,*args,**kwargs):
        return (early_tokens if ENABLED else original_tokens)(self,*args,**kwargs)

    original_update=dynamic_vocab.Vocabulary.update
    def update(self,*args,**kwargs):
        result=original_update(self,*args,**kwargs)
        if RECORD:CANDIDATES.append(self.ids.clone())
        return result

    NPUModelRunner._sample=sample
    NPUModelRunner.sample_tokens=tokens
    dynamic_vocab.Vocabulary.update=update
    print('EARLY_COUNTS_INSTALLED',PATCH,flush=True)


def configure(worker,enabled,record_candidates=False):
    global ENABLED,RECORD
    torch.npu.synchronize()
    runner=worker.model_runner
    assert runner.use_async_scheduling and runner.need_accepted_tokens
    assert runner.execute_model_state is None,'Only switch after sampling has completed'
    assert runner.cache_config.mamba_ssm_cache_dtype=='float32'
    ENABLED=bool(enabled);RECORD=bool(record_candidates)
    COUNTS.clear();CANDIDATES.clear()
    return {'enabled':ENABLED,'record_candidates':RECORD,'patch':PATCH,
            'mamba_cache_mode':runner.cache_config.mamba_cache_mode,
            'scope':'same graph and dynamic policy; process-only scheduling'}


def audit(worker):
    torch.npu.synchronize()
    ids=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in CANDIDATES]
    return {'enabled':ENABLED,'record_candidates':RECORD,'counts':dict(COUNTS),
            'candidate_set_sequence_sha256':ids,'patch':PATCH,
            'vocabulary':dynamic_vocab.STATE.audit() if dynamic_vocab.STATE else None}
