"""Single-NPU native-size MTP audit using synthetic data only. Never saves weights."""
import argparse,json,sys,time,datetime,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('model');p.add_argument('output');a=p.parse_args()
source,model,output=map(Path,[a.source,a.model,a.output]);sys.path.insert(0,str(source))
import torch
import torch_npu
from safetensors import safe_open
from transformers import Qwen3Config
from specforge.modeling.draft.mtp.qwen3_5 import Qwen3_5MTPDraftModel,PartialRotaryEmbedding
from specforge.core.mtp import OnlineMTPModel
from specforge.optimizer import BF16Optimizer
r={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'preflight','data':'Seeded synthetic ids/hidden states only; no corpus or evaluation questions loaded.','target_main_model_loaded':False,'weights_saved':False}
def save():output.write_text(json.dumps(r,indent=2))
def memory():
 torch.npu.synchronize();return {k:int(getattr(torch.npu,k)()) for k in ['memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved']}
try:
 torch.set_num_threads(4);torch.manual_seed(7);torch.npu.set_device(0);torch.npu.reset_peak_memory_stats()
 cfg=json.loads((model/'config.json').read_text());t=cfg.get('text_config',cfg)
 payload={k:t[k] for k in ['hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','vocab_size','rms_norm_eps','attention_bias','hidden_act'] if k in t}
 rope=t.get('rope_parameters',{})
 payload.update(num_hidden_layers=1,max_position_embeddings=4096,pad_token_id=cfg.get('pad_token_id',t.get('pad_token_id',248044)),tie_word_embeddings=False,attn_output_gate=True,partial_rotary_factor=t.get('partial_rotary_factor',rope.get('partial_rotary_factor',.25)),mtp_config={'share_lm_head':True})
 c=Qwen3Config(**payload);c.rope_theta=rope.get('rope_theta',10000000);c.partial_rotary_factor=payload['partial_rotary_factor'];c._attn_implementation='eager'
 index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];names=[k for k in index if k.startswith('mtp.')];assert len(names)==15
 embed='model.language_model.embed_tokens.weight';head='lm_head.weight'
 selected=names+[embed,head];shards={model/index[k] for k in selected};before_stat={str(f):(f.stat().st_size,f.stat().st_mtime_ns) for f in shards}
 state={}
 for f in shards:
  with safe_open(f,framework='pt',device='cpu') as handle:
   for k in selected:
    if model/index[k]==f:state[k]=handle.get_tensor(k)
 frozen_cpu={'embed_tokens.weight':state.pop(embed),'mtp.lm_head.weight':state.pop(head)};state.update(frozen_cpu)
 assert all(v.dtype==torch.bfloat16 for v in state.values())
 with torch.device('meta'):draft=Qwen3_5MTPDraftModel(c)
 draft.load_state_dict(state,strict=True,assign=True)
 draft.mtp.rotary_emb=PartialRotaryEmbedding(c,c.head_dim)
 draft.embed_tokens.requires_grad_(False);draft.mtp.lm_head.requires_grad_(False)
 assert all(k.startswith('mtp.') and k!='mtp.lm_head.weight' for k,v in draft.named_parameters() if v.requires_grad)
 trainable=sum(v.numel() for v in draft.parameters() if v.requires_grad);assert trainable==424699392
 r.update(phase='native_weights_loaded',trainable_parameters=trainable,frozen_parameters=sum(v.numel() for v in draft.parameters() if not v.requires_grad),dtype=str(next(draft.parameters()).dtype),config=payload,rope_theta=c.rope_theta);save();print('NATIVE_WEIGHTS_MATCH',flush=True)
 draft=draft.to('npu:0');wrapper=OnlineMTPModel(draft,objective_chunk_size=16).train();r['loaded_memory']=memory()
 optimizer=BF16Optimizer(draft,lr=1e-4,warmup_ratio=0,total_steps=2,lr_scheduler='constant')
 ids=torch.randint(0,c.vocab_size,(1,32),device='npu:0');hidden=torch.randn(1,32,c.hidden_size,device='npu:0',dtype=torch.bfloat16);mask=torch.ones(1,32,device='npu:0')
 before=draft.mtp.fc.weight.detach().clone();r['phase']='forward';save()
 loss,correct,denom=wrapper(input_ids=ids,hidden_states=hidden,loss_mask=mask);assert torch.isfinite(loss).item();r['loss']=loss.item();print('LOSS_FINITE',r['loss'],flush=True)
 loss.backward();r['phase']='backward';save()
 r['trainable_gradients_finite']=all(v.grad is not None and torch.isfinite(v.grad).all().item() for v in draft.parameters() if v.requires_grad)
 assert r['trainable_gradients_finite'];assert all(v.grad is None for v in draft.parameters() if not v.requires_grad)
 optimizer.step();r['phase']='optimizer_step';save();print('OPTIMIZER_STEP_OK',flush=True)
 r['fc_weight_changed']=not torch.equal(before,draft.mtp.fc.weight);assert r['fc_weight_changed']
 r['optimizer_parameter_count']=sum(v.numel() for v in optimizer.model_params);assert r['optimizer_parameter_count']==trainable
 r['final_memory']=memory()
 r['frozen_exactly_unchanged']={k:torch.equal(dict(draft.named_parameters())[k].detach().cpu(),v) for k,v in frozen_cpu.items()};assert all(r['frozen_exactly_unchanged'].values())
 r['checkpoint_files_unchanged']=all((f.stat().st_size,f.stat().st_mtime_ns)==before_stat[str(f)] for f in shards);assert r['checkpoint_files_unchanged']
 r['phase']='complete'
except BaseException as error:
 r['error']=type(error).__name__+': '+str(error);save();raise
finally:
 r['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(r,indent=2),flush=True)
