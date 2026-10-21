"""Small native MTP fine-tune; offline train-only teacher features, frozen tables."""
import argparse,datetime,hashlib,json,math,random,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('source');p.add_argument('model');p.add_argument('draft_config');a=p.parse_args()
root,source,model,config_file=map(Path,[a.run,a.source,a.model,a.draft_config]);sys.path.insert(0,str(source))
import torch,torch_npu
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config
from specforge.modeling.draft.mtp.qwen3_5 import Qwen3_5MTPDraftModel,PartialRotaryEmbedding
from specforge.core.mtp import OnlineMTPModel
from specforge.optimizer import BF16Optimizer
progress={'phase':'loading','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epochs':[]}
def save(): (root/'training-progress.json').write_text(json.dumps(progress,indent=2))
def memory():
 torch.npu.synchronize();return {k:int(getattr(torch.npu,k)()) for k in ['memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved']}
def hashed(t):return hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
def read_feature(i,split):
 x=torch.load(root/'features'/split/(str(i)+'.ckpt'),map_location='cpu',weights_only=True)
 assert x['input_ids'].ndim==1 and x['target_last_hidden_states'].shape==(len(x['input_ids']),5120)
 assert x['target_last_hidden_states'].dtype==torch.bfloat16
 ids=x['input_ids'].unsqueeze(0).to('npu:0');hidden=x['target_last_hidden_states'].unsqueeze(0).to('npu:0');mask=x['loss_mask'].unsqueeze(0).to('npu:0')
 return ids,hidden,mask
try:
 torch.set_num_threads(4);torch.manual_seed(20261021);torch.npu.set_device(0);torch.npu.reset_peak_memory_stats();save()
 assert json.loads((root/'teacher-progress.json').read_text())['phase']=='complete'
 manifest=json.loads((root/'data-manifest.json').read_text());assert manifest['selected_test_overlap']==0 and manifest['excluded_all_test']
 c=Qwen3Config.from_dict(json.loads(config_file.read_text()));c.rope_theta=c.rope_parameters['rope_theta'];c._attn_implementation='eager'
 index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];native=[k for k in index if k.startswith('mtp.')];assert len(native)==15
 names=native+['model.language_model.embed_tokens.weight','lm_head.weight'];state={};shards={model/index[k] for k in names}
 for shard in shards:
  with safe_open(shard,framework='pt',device='cpu') as f:
   for k in names:
    if model/index[k]==shard:state[k]=f.get_tensor(k)
 frozen={'embed_tokens.weight':state.pop('model.language_model.embed_tokens.weight'),'mtp.lm_head.weight':state.pop('lm_head.weight')};state.update(frozen)
 with torch.device('meta'):draft=Qwen3_5MTPDraftModel(c)
 draft.load_state_dict(state,strict=True,assign=True);draft.mtp.rotary_emb=PartialRotaryEmbedding(c,c.head_dim)
 draft.embed_tokens.requires_grad_(False);draft.mtp.lm_head.requires_grad_(False)
 trainable={n:p for n,p in draft.named_parameters() if p.requires_grad};assert set(trainable)==set(native) and sum(p.numel() for p in trainable.values())==424699392
 native_hash={n:hashed(p) for n,p in trainable.items()};draft.to('npu:0');trainable={n:p for n,p in draft.named_parameters() if p.requires_grad};wrapper=OnlineMTPModel(draft,objective_chunk_size=16)
 train_ids=list(range(64));validation_ids=list(range(64,72));assert len(list((root/'features/train').glob('*.ckpt')))==64 and len(list((root/'features/validation').glob('*.ckpt')))==8
 @torch.no_grad()
 def evaluate():
  wrapper.eval();loss_sum=correct_sum=denom_sum=0.
  for i in validation_ids:
   ids,h,mask=read_feature(i,'validation');loss,correct,denom=wrapper(input_ids=ids,hidden_states=h,loss_mask=mask);n=denom[0].sum().item();assert torch.isfinite(loss).item();loss_sum+=loss.item()*n;correct_sum+=correct[0].sum().item();denom_sum+=n
  return {'ce':loss_sum/denom_sum,'teacher_forced_token_accuracy':correct_sum/denom_sum,'supervised_positions':denom_sum}
 progress.update(phase='initial_validation',initial_validation=evaluate());save();print('INITIAL_VALIDATION',json.dumps(progress['initial_validation']),flush=True)
 assert progress['initial_validation']['teacher_forced_token_accuracy']>.5,'Native-head feature alignment sanity check failed'
 optimizer=BF16Optimizer(draft,lr=2e-5,max_grad_norm=1.,warmup_ratio=.1,total_steps=32,lr_scheduler='cosine')
 assert sum(p.numel() for p in optimizer.model_params)==424699392
 checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True);best=None;updates=0
 for epoch in range(1,3):
  wrapper.train();order=train_ids.copy();random.Random(20261021+epoch).shuffle(order);numerator=denominator=0.;window_denom=0.
  for step,i in enumerate(order):
   ids,h,mask=read_feature(i,'train');loss,correct,denom=wrapper(input_ids=ids,hidden_states=h,loss_mask=mask);n=denom[0].sum().item();assert torch.isfinite(loss).item();(loss*n).backward();window_denom+=n;numerator+=loss.item()*n;denominator+=n
   if (step+1)%4==0:
    for p in optimizer.model_params:
     assert p.grad is not None;p.grad.mul_(1/window_denom)
    optimizer.step();updates+=1;window_denom=0.
    progress.update(phase='training',epoch=epoch,microbatches=step+1,optimizer_updates=updates,train_ce_so_far=numerator/denominator);save();print('TRAIN_UPDATE',epoch,updates,round(numerator/denominator,5),flush=True)
  val=evaluate();path=checkpoints/f'epoch-{epoch}.safetensors';save_file({n:p.detach().cpu().contiguous() for n,p in trainable.items()},str(path),metadata={'format':'pt','training_split':'GSM8K train only','epoch':str(epoch)})
  record={'epoch':epoch,'training_ce':numerator/denominator,'validation':val,'checkpoint':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};progress['epochs'].append(record)
  if best is None or val['ce']<best['validation']['ce']:best=record
  save();print('EPOCH_COMPLETE',json.dumps(record),flush=True)
 final={n:hashed(p) for n,p in draft.named_parameters() if n in frozen};assert all(final[n]==hashed(t) for n,t in frozen.items())
 assert all(p.grad is None for n,p in draft.named_parameters() if n in frozen)
 progress.update(phase='complete',selected_checkpoint=best['checkpoint'],selected_epoch=best['epoch'],selection='lowest independent validation CE among trained epochs only',frozen_tables_exactly_unchanged=True,native_mtp_changed=any(hashed(trainable[n])!=native_hash[n] for n in native),optimizer_updates=updates,memory=memory(),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());assert progress['native_mtp_changed'];save();print('TRAINING_COMPLETE',json.dumps(progress),flush=True)
except BaseException as e:
 progress.update(phase='failed',error=type(e).__name__+': '+str(e));save();raise
