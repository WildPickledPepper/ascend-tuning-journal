"""Guarded TP1/MTP7 pure-decode GDN metadata path; no math/kernel changes.

Only the all-speculative single-request [0,8] query is specialized. The two
dynamic state-index/accepted-count buffers are still copied every invocation.
All other request shapes use the original builder.
"""
import collections
from dataclasses import is_dataclass
import torch

ENABLED=False
VALIDATE=False
COUNTS=collections.Counter()
CHECKS={}
ORIGINAL=None


def eligible(builder,m,accepted,draft):
    if not (builder.use_full_cuda_graph and builder.use_spec_decode and builder.num_spec==7
            and builder.decode_cudagraph_max_bs>=8 and m.num_reqs==1
            and m.num_actual_tokens==8 and m.max_query_len==8
            and builder.vllm_config.cache_config.mamba_cache_mode=='none'
            and accepted is not None and draft is not None):return False
    q=m.query_start_loc_cpu.numpy();d=draft.numpy()
    if m.is_prefilling is not None and bool(m.is_prefilling.numpy()[0]):return False
    return len(q)==2 and int(q[0])==0 and int(q[1])==8 and len(d)>=1 and int(d[0])==7


def fast(builder,m,accepted):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm_ascend.ops.gdn_attn_builder import GDNSpecDecodeMetadata,GDNSpecCausalConv1dMetadata,NULL_BLOCK_ID
    assert m.block_table_tensor.shape[0]>=1 and m.block_table_tensor.shape[1]>=8
    if not getattr(builder,'_count_ready_constants',False):
        builder.spec_sequence_masks[:1].fill_(True)
        builder.spec_sequence_masks_cpu[:1].fill_(True)
        builder.spec_sequence_indices_cpu[:1].zero_()
        builder.spec_sequence_indices[:1].zero_()
        builder.spec_state_indices_tensor[1:].fill_(NULL_BLOCK_ID)
        builder.spec_token_indx[:8].copy_(torch.arange(8,dtype=torch.int32,device=m.query_start_loc.device))
        builder.spec_query_start_loc[:2].copy_(m.query_start_loc[:2],non_blocking=True)
        builder.spec_actual_seq_lengths[:2].copy_(m.query_start_loc[:2],non_blocking=True)
        builder._count_ready_constants=True
        COUNTS['constant_initializations']+=1
    # In mode=none mamba_get_block_table_tensor returns this block table.
    # With one all-spec request the original boolean selection selects row 0.
    builder.spec_state_indices_tensor[:1].copy_(m.block_table_tensor[:1,:8],non_blocking=True)
    builder.num_accepted_tokens[:1].copy_(accepted[:1],non_blocking=True)
    COUNTS['fast_calls']+=1
    result=GDNAttentionMetadata(
        num_prefills=0,num_prefill_tokens=0,num_decodes=0,num_decode_tokens=0,
        num_spec_decodes=1,num_spec_decode_tokens=8,num_actual_tokens=8,
        spec_query_start_loc=builder.spec_query_start_loc[:2],
        spec_state_indices_tensor=builder.spec_state_indices_tensor[:1],
        spec_sequence_masks=builder.spec_sequence_masks[:1],
        spec_token_indx=builder.spec_token_indx[:8],non_spec_token_indx=builder.non_spec_token_indx[:0],
        num_accepted_tokens=builder.num_accepted_tokens[:1])
    result.non_spec_prefill_metadata=None
    result.non_spec_decode_metadata=None
    result.spec_decode_metadata=GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=result.spec_query_start_loc,cache_indices=result.spec_state_indices_tensor,
            num_accepted_tokens=result.num_accepted_tokens),
        actual_seq_lengths=builder.spec_actual_seq_lengths[:2])
    return result


def install():
    global ORIGINAL
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
    if ORIGINAL is not None:return
    ORIGINAL=AscendGDNAttentionMetadataBuilder.build
    def build(self,common_prefix_len,common_attn_metadata,num_accepted_tokens=None,num_decode_draft_tokens_cpu=None,fast_build=False):
        args=(common_prefix_len,common_attn_metadata,num_accepted_tokens,num_decode_draft_tokens_cpu,fast_build)
        if not ENABLED or common_prefix_len!=0 or not eligible(self,common_attn_metadata,num_accepted_tokens,num_decode_draft_tokens_cpu):
            self._count_ready_constants=False
            COUNTS['fallback_calls']+=1
            return ORIGINAL(self,*args)
        if VALIDATE and id(self) not in CHECKS:
            original=ORIGINAL(self,*args)
            def flatten(value,prefix='',out=None):
                out={} if out is None else out
                if is_dataclass(value):
                    # vars also includes dynamically attached Ascend metadata.
                    for name,item in vars(value).items():flatten(item,prefix+'.'+name,out)
                else:out[prefix]=value
                return out
            saved=flatten(original)
            layouts={k:(tuple(v.shape),tuple(v.stride()),v.dtype,v.device) for k,v in saved.items() if isinstance(v,torch.Tensor)}
            saved={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in saved.items()}
            self._count_ready_constants=False
            result=fast(self,common_attn_metadata,num_accepted_tokens)
            actuals=flatten(result)
            assert set(actuals)==set(saved),('metadata field set differs',set(actuals)^set(saved))
            for k,v in saved.items():
                actual=actuals[k]
                if isinstance(v,torch.Tensor):
                    assert isinstance(actual,torch.Tensor) and (tuple(actual.shape),tuple(actual.stride()),actual.dtype,actual.device)==layouts[k],k
                    assert torch.equal(actual,v),(k,'metadata mismatch')
                else:assert actual==v,(k,actual,v)
            CHECKS[id(self)]={'fields_checked':len(saved),'tensor_fields_checked':len(layouts),'all_equal':True}
            return result
        return fast(self,common_attn_metadata,num_accepted_tokens)
    AscendGDNAttentionMetadataBuilder.build=build


def configure(enabled,validate=False):
    global ENABLED,VALIDATE
    torch.npu.synchronize()
    ENABLED=bool(enabled);VALIDATE=bool(validate)
    COUNTS.clear()
    if validate:CHECKS.clear()
    return audit()


def audit():
    return {'enabled':ENABLED,'validate':VALIDATE,'counts':dict(COUNTS),'validated_builders':list(CHECKS.values()),
            'scope':'TP1 one-request MTP7 full graph, [0,8] query, mamba cache none; other shapes fall back'}
