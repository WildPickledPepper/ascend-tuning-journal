"""Check specialized fields against the actual Ascend builder on CPU tensors.

No device initialization or model execution. Exercise all accepted lengths.
Real NPU metadata and full-model checks are still required separately.
"""
import os
os.environ['ASCEND_RT_VISIBLE_DEVICES']=''
from types import SimpleNamespace
import torch
import fast_metadata as fm
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

fm.install();fm.ENABLED=True;fm.VALIDATE=True
b=AscendGDNAttentionMetadataBuilder.__new__(AscendGDNAttentionMetadataBuilder)
b.use_full_cuda_graph=True;b.use_spec_decode=True;b.num_spec=7;b.decode_cudagraph_max_bs=8
b.vllm_config=SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode='none'),parallel_config=SimpleNamespace(decode_context_parallel_size=1,prefill_context_parallel_size=1))
b.kv_cache_spec=None
for name,shape,dtype in (
    ('spec_state_indices_tensor',(8,8),torch.int32),('num_accepted_tokens',(8,),torch.int32),
    ('spec_sequence_masks',(8,),torch.bool),('spec_sequence_masks_cpu',(8,),torch.bool),
    ('spec_sequence_indices_cpu',(8,),torch.int64),('non_spec_sequence_indices_cpu',(8,),torch.int64),
    ('spec_sequence_indices',(8,),torch.int64),('non_spec_sequence_indices',(8,),torch.int64),
    ('spec_actual_seq_lengths',(9,),torch.int32),('spec_token_indx',(64,),torch.int32),
    ('non_spec_token_indx',(64,),torch.int32),('spec_query_start_loc',(9,),torch.int32)):
    setattr(b,name,torch.zeros(shape,dtype=dtype))
for accepted in range(1,9):
    q=torch.tensor([0,8],dtype=torch.int32)
    m=SimpleNamespace(num_reqs=1,num_actual_tokens=8,max_query_len=8,query_start_loc=q,
        query_start_loc_cpu=q,seq_lens=torch.tensor([100]),seq_lens_cpu_upper_bound=torch.tensor([100]),
        is_prefilling=torch.tensor([False]),block_table_tensor=torch.arange(8,dtype=torch.int32).reshape(1,8)+accepted*8)
    m.compute_num_computed_tokens=lambda:torch.tensor([92])
    fm.CHECKS.clear()
    b.build(0,m,torch.tensor([accepted],dtype=torch.int32),torch.tensor([7],dtype=torch.int32))
    assert fm.CHECKS[id(b)]['all_equal']
print({'accepted_lengths_checked':8,'field_comparison':fm.CHECKS[id(b)],'device':'CPU only'})
