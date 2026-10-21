"""Build isolated inference checkpoint; verify every non-MTP tensor in rewritten shards."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('original');p.add_argument('destination');a=p.parse_args();root,original,dest=map(Path,[a.run,a.original,a.destination])
progress=json.loads((root/'training-progress.json').read_text());assert progress['phase']=='complete' and progress['frozen_tables_exactly_unchanged']
head_path=root/'checkpoints'/progress['selected_checkpoint'];head=load_file(str(head_path));assert len(head)==15 and all(k.startswith('mtp.') and k!='mtp.lm_head.weight' for k in head)
index=json.loads((original/'model.safetensors.index.json').read_text());weight_map=index['weight_map'];native={k for k in weight_map if k.startswith('mtp.')};assert native==set(head)
changed_shards={weight_map[k] for k in head};assert not dest.exists(),'Candidate path already exists; preserve partial artifacts for review';dest.mkdir()
for f in original.iterdir():
 if f.name not in changed_shards:(dest/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
def digest(t):return hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
all_before={};all_after={};changed=0
for name in sorted(changed_shards):
 with safe_open(original/name,framework='pt',device='cpu') as f:state={k:f.get_tensor(k) for k in f.keys()};meta=f.metadata()
 for k,t in state.items():
  if k in head:
   assert t.dtype==head[k].dtype==torch.bfloat16 and t.shape==head[k].shape
   changed+=not torch.equal(t,head[k]);state[k]=head[k]
  else:all_before[k]=digest(t)
 save_file(state,str(dest/name),metadata=meta)
 with safe_open(dest/name,framework='pt',device='cpu') as f:
  for k in f.keys():
   if k not in head:all_after[k]=digest(f.get_tensor(k))
 del state
assert all_before==all_after and changed>0
result={'selected_epoch':progress['selected_epoch'],'selected_head_sha256':hashlib.sha256(head_path.read_bytes()).hexdigest(),'head_tensors':len(head),'changed_head_tensors':changed,'patched_shard_count':len(changed_shards),'other_weight_shards_symlinked':len(set(weight_map.values())-changed_shards),'non_head_tensors_in_patched_shards_verified':len(all_before),'non_head_tensors_exactly_unchanged':True,'unchanged_tensor_hash_digest':hashlib.sha256(json.dumps(all_before,sort_keys=True).encode()).hexdigest(),'original_directory_written':False,'candidate_index_unchanged':(dest/'model.safetensors.index.json').read_bytes()==(original/'model.safetensors.index.json').read_bytes()}
(root/'candidate-integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
