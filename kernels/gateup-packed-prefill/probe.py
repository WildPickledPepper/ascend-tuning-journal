import json,torch,torch_npu,pathlib,os
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
r=pathlib.Path(__file__).resolve().parent
torch.npu.set_device(0);torch.ops.load_library(str(r/'libprefill_torch.so'))
torch.manual_seed(123)
for kind in ('ones','random_x','random_w'):
 w=torch.ones(34816,5120,dtype=torch.bfloat16) if kind!='random_w' else torch.randn(34816,5120,dtype=torch.bfloat16)*.01
 p=w.reshape(136,16,16,20,16,16).permute(0,3,4,1,2,5).contiguous().reshape(34816,5120).npu()
 w=w.npu()
 for m in (1,8,16,32,65,121,128):
  x=torch.randn(m,5120,dtype=torch.bfloat16,device='npu') if kind=='random_x' else torch.ones(m,5120,dtype=torch.bfloat16,device='npu')
  y=torch.ops.packed_prefill_bf16.linear(x,p,0).cpu();ref=torch.nn.functional.linear(x,w).cpu()
  print(json.dumps({'kind':kind,'M':m,'y_min':float(y.min()),'y_max':float(y.max()),'ref_min':float(ref.min()),'ref_max':float(ref.max()),'different':int((y!=ref).sum()),'sample_y':y[0,:32].tolist(),'sample_ref':ref[0,:32].tolist()}),flush=True)
 del w,p
 torch.npu.empty_cache()
