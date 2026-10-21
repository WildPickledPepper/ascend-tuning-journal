"""Card-7-only real-weight layout checks and complete FFN working-set timing."""
import argparse, datetime, hashlib, json, os, pathlib, statistics, time
import torch, torch_npu
from safetensors import safe_open

ROOT=pathlib.Path(__file__).resolve().parent
MODEL=pathlib.Path(os.environ['QWEN_MODEL_PATH'])
INDEX=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
N,K=34816,5120
MODES={'nd':0,'nz':1,'packed':2}

def load(layer,kind):
    key=f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
    with safe_open(MODEL/INDEX[key],framework='pt',device='cpu') as f:return f.get_tensor(key)

def weight(layer):return torch.cat([load(layer,'gate'),load(layer,'up')],dim=0)

def pack(w,layout):
    if layout in ('native','nd'):return w
    if layout=='nz':return w.reshape(N//16,16,K//16,16).permute(2,0,1,3).contiguous()
    return w.reshape(N//256,16,16,K//256,16,16).permute(0,3,4,1,2,5).contiguous()

def unpack(w,layout):
    if layout=='nz':return w.permute(1,2,0,3).contiguous().reshape(N,K)
    return w.permute(0,3,4,1,2,5).contiguous().reshape(N,K)

def apply(x,w,layout):
    if layout=='native':return torch.nn.functional.linear(x,w)
    return torch.ops.l1_tile_layout.linear(x,w,MODES[layout])

def save(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(row,indent=2)+'\n')

def check(args,row):
    cpu=weight(0)
    ws={'native':cpu.to('npu')}
    for layout in ['nz','packed']:
        p=pack(cpu,layout)
        assert torch.equal(unpack(p,layout).view(torch.int16),cpu.view(torch.int16))
        ws[layout]=p.to('npu')
    ws['nd']=ws['native']
    row.update(weight_restore_bit_exact=True,cases=[])
    for seed in range(8):
        torch.manual_seed(20261021+seed)
        x=(torch.randn(8,K,dtype=torch.bfloat16)*[.03125,1.,32.,0.][seed%4]).to('npu')
        out={}
        for layout in ws:
            out[layout]=apply(x,ws[layout],layout);torch.npu.synchronize()
        z={}
        for layout,y in out.items():
            yc=y.cpu();ref=out['native'].cpu()
            z[layout]={'different_from_native':int((yc.view(torch.int16)!=ref.view(torch.int16)).sum()),
                       'max_abs_diff':float((yc.float()-ref.float()).abs().max()),
                       'finite':bool(yc.isfinite().all())}
        same=torch.equal(out['nz'].cpu().view(torch.int16),out['packed'].cpu().view(torch.int16))
        same_nd=torch.equal(out['nd'].cpu().view(torch.int16),out['packed'].cpu().view(torch.int16))
        row['cases'].append({'seed':seed,'layout_outputs_bit_equal':same and same_nd,'outputs':z})
        save(args.output,row);print('CHECK',seed,json.dumps(row['cases'][-1]),flush=True)
        if not same_nd:
            bad=(out['nd'].cpu()!=out['packed'].cpu()).sum(0)
            print('ND_BAD_TILES',[int(bad[i*256:(i+1)*256].sum()) for i in range(136)],flush=True)
        assert same and same_nd and all(a['finite'] for a in z.values())
    graphs={};ys={}
    x=(torch.randn(8,K,dtype=torch.bfloat16)*.1).to('npu')
    for layout in ws:
        g=torch.npu.NPUGraph()
        with torch.npu.graph(g):y=apply(x,ws[layout],layout)
        graphs[layout]=g;ys[layout]=y
    for trial in range(12):
        x.copy_(torch.randn(8,K,dtype=torch.bfloat16)*.1)
        for _ in range(10):
            for g in graphs.values():g.replay()
        torch.npu.synchronize()
        assert torch.equal(ys['nz'].cpu().view(torch.int16),ys['packed'].cpu().view(torch.int16))
        assert torch.equal(ys['nd'].cpu().view(torch.int16),ys['packed'].cpu().view(torch.int16))
        assert torch.equal(ys['native'].cpu().view(torch.int16),ys['packed'].cpu().view(torch.int16))
    row.update(validation_passed=True,graph_replays_per_layout=120,
               native_bit_equal=all(v['outputs']['packed']['different_from_native']==0 for v in row['cases']))
    save(args.output,row);print('VALIDATION_DONE',row['native_bit_equal'],flush=True)

def bench(args,row):
    entries=[];addresses=[];checks=[]
    torch.manual_seed(20261021)
    x=torch.randn(8,K,dtype=torch.bfloat16).to('npu')
    xd=torch.randn(8,17408,dtype=torch.bfloat16).to('npu')
    for i in range(args.layers):
        cpu=weight(i);p=pack(cpu,args.layout)
        restored=True
        if args.layout in ('nz','packed'):
            restored=torch.equal(unpack(p,args.layout).view(torch.int16),cpu.view(torch.int16))
            assert restored, ('weight_restore',i)
        w=p.to('npu');down=load(i,'down').to('npu') if args.include_down else None
        torch.npu.synchronize()
        if args.layout!='native':
            rw=cpu.to('npu');ref=torch.nn.functional.linear(x,rw);got=apply(x,w,args.layout)
            torch.npu.synchronize()
            different=int((ref.cpu().view(torch.int16)!=got.cpu().view(torch.int16)).sum())
            checks.append({'layer':i,'different_from_native':different,'weight_restore_bit_exact':restored})
            assert different==0, checks[-1]
            del rw,ref,got
        entries.append((w,down));addresses.append({'layer':i,'gate':hex(w.data_ptr()),'down':hex(down.data_ptr()) if down is not None else None})
        if i%8==0:print('LOADED',i,flush=True)
    del cpu,p
    row.update(layers=args.layers,layout=args.layout,include_down=args.include_down,
               weight_bytes=args.layers*(N*K*2+(5120*17408*2 if args.include_down else 0)),
               addresses=addresses,layer_checks=checks,memory_allocated=torch.npu.memory_allocated())
    scopes=['interleaved'] if args.include_down else ['gate']
    # The gate-only graph retains all 64 real gate/up weights.
    if args.include_down and not args.seconds and not args.profile:scopes=['gate','interleaved']
    graphs={};keep=[]
    for scope in scopes:
        g=torch.npu.NPUGraph()
        with torch.npu.graph(g):
            for w,down in entries:
                keep.append(apply(x,w,args.layout))
                if scope=='interleaved':keep.append(torch.nn.functional.linear(xd,down))
        graphs[scope]=g
    for _ in range(6):
        for g in graphs.values():g.replay()
    torch.npu.synchronize()
    row['timing']={}
    for scope,g in graphs.items():
        vals=[]
        for _ in range(args.rounds):
            a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b))
        row['timing'][scope]={'median_ms':statistics.median(vals),'samples_ms':vals,'per_layer_us':statistics.median(vals)*1000/args.layers}
        print('TIMING',scope,json.dumps({k:v for k,v in row['timing'][scope].items() if k!='samples_ms'}),flush=True)
    if args.profile:
        from torch_npu.profiler import AiCMetrics, ExportType, ProfilerActivity, ProfilerLevel, _ExperimentalConfig, profile, tensorboard_trace_handler
        cfg=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,aic_metrics=getattr(AiCMetrics,args.metric),op_attr=True,export_type=[ExportType.Db,ExportType.Text])
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],record_shapes=True,
                     experimental_config=cfg,on_trace_ready=tensorboard_trace_handler(str(args.profile))):
            for _ in range(2):
                for g in graphs.values():g.replay()
            torch.npu.synchronize()
    if args.seconds:
        g=next(iter(graphs.values()));time.sleep(1)
        start=time.time_ns();deadline=time.monotonic()+args.seconds;count=0
        while time.monotonic()<deadline:
            for _ in range(4):g.replay()
            torch.npu.synchronize();count+=4
        end=time.time_ns()
        row['phase']={'start_ns':start,'end_ns':end,'graphs':count,'layers':count*args.layers,'elapsed_s':(end-start)/1e9}
        print('PHASE',json.dumps(row['phase']),flush=True);time.sleep(1)
    row['all_outputs_finite']=all(bool(y.isfinite().all()) for y in keep)
    assert row['all_outputs_finite']
    save(args.output,row)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','bench'])
    p.add_argument('--layout',choices=['native','nd','nz','packed'],default='packed')
    p.add_argument('--layers',type=int,default=64);p.add_argument('--include-down',action='store_true')
    p.add_argument('--rounds',type=int,default=30);p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--profile',type=pathlib.Path);p.add_argument('--metric',default='PipeUtilization')
    p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    assert 1<=args.layers<=64
    torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    torch.ops.load_library(str(ROOT/'liblayout_torch.so'))
    row={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_device':7,
         'dtype':'BF16','shape':[8,N,K],'model':str(MODEL),'torch':torch.__version__,
         'kernel_sha256':hashlib.sha256((ROOT/'liblayout_kernel.so').read_bytes()).hexdigest(),
         'source_sha256':hashlib.sha256((ROOT/'kernel.asc').read_bytes()).hexdigest()}
    with torch.inference_mode():
        if args.mode=='check':check(args,row)
        else:bench(args,row)
    print('DONE',args.output,flush=True)
if __name__=='__main__':main()
