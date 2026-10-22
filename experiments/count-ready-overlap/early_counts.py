"""Process-only scheduling stages; preserve both counters' original semantics."""
import ast
import collections
import hashlib
import inspect
import textwrap
from types import SimpleNamespace
import torch
import dynamic_vocab

STAGE=0
RECORD=False
COUNTS=collections.Counter()
CANDIDATES=[]
SAMPLED=[]
DRAFTED=[]
PATCH={}
PENDING={}
ORIGINAL_COPY=None


def snapshot_inputs(runner,logits):
    if logits is None or dynamic_vocab.STATE is None:return None
    draft=getattr(runner,'_draft_token_ids',None)
    if isinstance(draft,torch.Tensor):draft=draft.clone()
    return SimpleNamespace(input_batch=runner.input_batch,requests=runner.requests,_draft_token_ids=draft),logits.clone()


def update_after_counts(runner,saved):
    if saved is not None:
        import profile_scopes
        from contextlib import nullcontext
        with (torch.profiler.record_function('count_ready.same_round_vocab') if profile_scopes.ENABLED else nullcontext()):
            dynamic_vocab.STATE.update(*saved)
        COUNTS['early_vocab_updates']+=1


def prime_valid_counts(runner,sampler_output):
    assert STAGE>=2 and runner.cache_config.mamba_cache_mode=='none'
    assert runner.use_async_scheduling and runner.valid_sampled_token_count_event is not None
    assert id(runner.drafter) not in PENDING
    # Run after ORIGINAL bookkeeping: fallback token lookup observes precisely
    # the same request/token state as the original proposer path.
    args=(sampler_output.sampled_token_ids,runner.requests,runner.input_batch,
          runner.discard_request_indices.gpu,runner.num_discarded_requests)
    pair=runner.drafter.prepare_next_token_ids_padded(*args)
    ORIGINAL_COPY(runner,*pair)
    PENDING[id(runner.drafter)]={'args':args,'pair':pair,'consumed':False}
    COUNTS['valid_count_primed']+=1


def install():
    global ORIGINAL_COPY
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    original_tokens=NPUModelRunner.sample_tokens
    original_dynamic=NPUModelRunner._sample
    base_sampler=inspect.getclosurevars(inspect.unwrap(original_dynamic)).nonlocals['original_runner_sample']
    original_propose=NPUModelRunner.propose_draft_token_ids
    ORIGINAL_COPY=NPUModelRunner._copy_valid_sampled_token_count
    original_prepare=AscendSpecDecodeBaseProposer.prepare_next_token_ids_padded
    source=textwrap.dedent(inspect.getsource(original_tokens))
    variants={0:original_tokens}
    PATCH['original_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    for stage in (1,2):
        tree=ast.parse(source);fn=tree.body[0]
        late=[n for n in fn.body if isinstance(n,ast.If) and 'async_state_update' in ast.unparse(n)]
        assert len(late)==1
        late=late[0];fn.body.remove(late)
        sample=[n for n in fn.body if isinstance(n,ast.With) and 'sampler_output = self._sample(' in ast.unparse(n)]
        assert len(sample)==1
        i=fn.body.index(sample[0]);fn.body[i:i]=ast.parse('_saved = _overlap_snapshot(self, logits)').body
        ready=[n for n in fn.body if isinstance(n,ast.If) and 'self.sampling_done_event.record()' in ast.unparse(n)]
        assert len(ready)==1
        i=fn.body.index(ready[0])+1;fn.body[i:i]=[late]
        if stage==1:
            fn.body[i+1:i+1]=ast.parse('_overlap_vocab(self, _saved)').body
        else:
            bookkeeping=[n for n in fn.body if isinstance(n,ast.Assign) and 'self._bookkeeping_sync(' in ast.unparse(n)]
            assert len(bookkeeping)==1
            i=fn.body.index(bookkeeping[0])+1
            fn.body[i:i]=ast.parse('_overlap_prime(self, sampler_output)\n_overlap_vocab(self, _saved)').body
        ast.fix_missing_locations(tree)
        namespace=dict(inspect.unwrap(original_tokens).__globals__)
        namespace.update(_overlap_snapshot=snapshot_inputs,_overlap_vocab=update_after_counts,_overlap_prime=prime_valid_counts)
        exec(compile(tree,f'<count-ready-stage-{stage}>','exec'),namespace)
        variants[stage]=namespace['sample_tokens']
        PATCH[str(stage)]={'source_sha256':hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
                           'state_block_sha256':hashlib.sha256(ast.unparse(late).encode()).hexdigest()}
    variants[3]=variants[2]

    def prepare(self,*args,**kwargs):
        pending=PENDING.get(id(self))
        if STAGE>=2 and pending is not None:
            assert not kwargs and len(args)==5 and not pending['consumed']
            assert all(a is b for a,b in zip(args[:4],pending['args'][:4]))
            assert args[4]==pending['args'][4]
            pending['consumed']=True
            COUNTS['valid_count_prepare_reused']+=1
            return pending['pair']
        return original_prepare(self,*args,**kwargs)

    def copy_valid(self,next_ids,counts):
        pending=PENDING.get(id(self.drafter))
        if STAGE>=2 and pending is not None:
            assert pending['consumed'] and next_ids is pending['pair'][0] and counts is pending['pair'][1]
            del PENDING[id(self.drafter)]
            COUNTS['valid_count_copy_reused']+=1
            return None
        return ORIGINAL_COPY(self,next_ids,counts)

    def sample(self,logits,metadata):
        result=(base_sampler if STAGE else original_dynamic)(self,logits,metadata)
        COUNTS['sample_calls']+=1
        if RECORD:SAMPLED.append(result.sampled_token_ids.clone())
        return result

    def tokens(self,*args,**kwargs):
        result=variants[STAGE](self,*args,**kwargs)
        assert id(self.drafter) not in PENDING,'Early valid-count pair was not consumed'
        return result

    def propose(self,*args,**kwargs):
        result=original_propose(self,*args,**kwargs)
        if RECORD and isinstance(result,torch.Tensor):DRAFTED.append(result.clone())
        return result

    original_update=dynamic_vocab.Vocabulary.update
    def update(self,*args,**kwargs):
        result=original_update(self,*args,**kwargs)
        if RECORD:CANDIDATES.append(self.ids.clone())
        return result
    AscendSpecDecodeBaseProposer.prepare_next_token_ids_padded=prepare
    NPUModelRunner._copy_valid_sampled_token_count=copy_valid
    NPUModelRunner._sample=sample
    NPUModelRunner.sample_tokens=tokens
    NPUModelRunner.propose_draft_token_ids=propose
    dynamic_vocab.Vocabulary.update=update
    print('COUNT_READY_INSTALLED',PATCH,flush=True)


def configure(worker,stage,record=False):
    global STAGE,RECORD
    torch.npu.synchronize();r=worker.model_runner
    assert stage in (0,1,2,3) and not PENDING and r.execute_model_state is None
    assert r.use_async_scheduling and r.need_accepted_tokens
    assert r.cache_config.mamba_ssm_cache_dtype=='float32'
    if stage>=2:assert r.cache_config.mamba_cache_mode=='none'
    STAGE=int(stage);RECORD=bool(record)
    import fast_metadata
    fast_metadata.configure(STAGE==3)
    COUNTS.clear();CANDIDATES.clear();SAMPLED.clear();DRAFTED.clear()
    return {'stage':STAGE,'record':RECORD,'patch':PATCH,'mamba_cache_mode':r.cache_config.mamba_cache_mode}


def audit(worker):
    torch.npu.synchronize()
    def digest(items):return [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in items]
    return {'stage':STAGE,'counts':dict(COUNTS),'candidate_set_sequence_sha256':digest(CANDIDATES),
            'sampled_sequence_sha256':digest(SAMPLED),'draft_sequence_sha256':digest(DRAFTED),'patch':PATCH,
            'vocabulary':dynamic_vocab.STATE.audit()}
