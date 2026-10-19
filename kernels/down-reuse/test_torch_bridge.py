import os,pathlib,json,torch,torch_npu
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='7'
assert os.environ['TASK_QUEUE_ENABLE']=='1'
torch.npu.set_device(0)
root=pathlib.Path(__file__).resolve().parent
torch.ops.load_library(str(root/'libdown_torch.so'))
torch.manual_seed(32)
x=torch.randn(1,17408,device='npu',dtype=torch.bfloat16)
w=torch.randn(5120,17408,device='npu',dtype=torch.bfloat16)
y=torch.ops.down_tuning.linear(x,w)
ref=torch.nn.functional.linear(x,w)
torch.npu.synchronize()
assert bool(y.isfinite().all())
g=torch.npu.NPUGraph()
with torch.npu.graph(g):z=torch.ops.down_tuning.linear(x,w)
g.replay();torch.npu.synchronize();assert torch.equal(y,z)
# Change data on the same stream to check queued dependencies under graph replay.
x.zero_();g.replay();torch.npu.synchronize();assert bool((z==0).all())
print(json.dumps({'queued_launch':True,'graph_capture_replay':True,'changed_input_dependency':True,
                 'submitted':torch.ops.down_tuning.submitted(),'reference_mismatches':int((y!=ref).sum())}),flush=True)
