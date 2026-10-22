"""Existing GDN operator only: identical inputs, different call boundaries.

Small diagnostic, not a performance benchmark or a new kernel. Run only on
physical card 7, after the model service exits. FP32 state is a counterfactual
control; query/key/value and outputs remain BF16.
"""
import importlib.util
import json
import os
from pathlib import Path

# Must precede torch/CANN initialization. Otherwise the CANN built-in with
# the same name can be selected; its ABI differs from the model's custom op.
vendor = Path(importlib.util.find_spec('vllm_ascend').origin).parent / '_cann_ops_custom/vendors/custom_transformer'
assert str(vendor) in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':'), 'Use run_state_probe.sh before importing CANN'

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
torch.manual_seed(20261022)
T, NK, NV, D = 8, 2, 4, 128
q = torch.nn.functional.normalize(torch.randn(T,NK,D),dim=-1).to(torch.bfloat16).npu()
k = torch.nn.functional.normalize(torch.randn(T,NK,D),dim=-1).to(torch.bfloat16).npu()
v = torch.randn(T,NV,D,dtype=torch.bfloat16).npu()
g = (-torch.rand(T,NV)*0.2).npu()
beta = torch.rand(T,NV,dtype=torch.bfloat16).npu()
initial = (torch.randn(NV,D,D)*0.1).to(torch.bfloat16).npu()


def run(lengths, dtype):
    current = initial.to(dtype)
    outputs = []
    offset = 0
    for length in lengths:
        bank = current.unsqueeze(0).expand(length,-1,-1,-1).clone()
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=q[offset:offset+length].contiguous(),
            key=k[offset:offset+length].contiguous(),
            value=v[offset:offset+length].contiguous(), state=bank,
            beta=beta[offset:offset+length].contiguous(), scale=D**-0.5,
            # This deployed interface expects [start_offset, sequence_length],
            # NOT a one-element [length] array (that would describe zero batches).
            actual_seq_lengths=torch.tensor([0,length],device='npu',dtype=torch.int32),
            ssm_state_indices=torch.arange(length,device='npu',dtype=torch.int32),
            num_accepted_tokens=torch.ones(1,device='npu',dtype=torch.int32),
            g=g[offset:offset+length].contiguous())
        outputs.append(out.clone())
        assert bool(out.isfinite().all())
        assert int((bank[length-1]!=current).count_nonzero()) > 0, 'Final state was not updated'
        current = bank[length-1].clone()
        offset += length
    assert offset==T
    return torch.cat(outputs).cpu(),current.cpu()


def diff(a,b):
    e=(a.float()-b.float()).abs()
    return {'different':int((a!=b).sum()),'elements':a.numel(),'max_abs':float(e.max())}


report = {}
for dtype in (torch.float32,torch.bfloat16):
    try:
        whole, state = run([8],dtype)
        repeated, repeated_state = run([8],dtype)
        assert torch.equal(whole,repeated) and torch.equal(state,repeated_state), 'Nonrepeatable probe'
        entry = {}
        for chunks in ([4,4],[1]*8):
            out, endstate = run(chunks,dtype)
            assert torch.equal(whole[:chunks[0]],out[:chunks[0]]), 'First chunk differs despite identical inputs/state'
            entry[str(chunks)] = {'output':diff(whole,out), 'last_state':diff(state,endstate),
                                  'per_token_different':[(whole[i]!=out[i]).sum().item() for i in range(T)]}
    except RuntimeError as exc:
        # FP32 first verifies that the intended custom kernel, not a CANN
        # built-in of the same name with a different ABI, has been selected.
        raise RuntimeError('Custom GDN ABI validation failed; stop probe') from exc
    report[str(dtype)] = entry
    Path(__file__).with_name('results').joinpath('state-segmentation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
