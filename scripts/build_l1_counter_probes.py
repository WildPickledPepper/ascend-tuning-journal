import os,pathlib,json,argparse
from tbe import tik,tvm
from tbe.common.platform import set_current_compile_soc_info
from tbe.tik.tik_lib.tik_params import PIPE_MTE2
from tbe.tik.tik_lib.tik_source_info import source_info_decorator
@source_info_decorator()
def emit_zero(t,l):
 with t.new_scope():
  t.scope_attr(tvm.thread_axis('cce'),'coproc_scope',PIPE_MTE2)
  t.emit(tvm.call_extern('float16','set_l1_2d',l.access_ptr('w'),tvm.const(0x100020,'uint64')))
set_current_compile_soc_info('Ascend910B2')
parser=argparse.ArgumentParser(description='Build two B16 movement counter calibration kernels; compilation only.')
parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('generated/counter-probes'))
out=parser.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=True);os.chdir(out)
for kind in ['zero','nd2nz']:
 t=tik.Tik();x=t.Tensor('float16',(512,),name='x',scope=tik.scope_gm);l=t.Tensor('float16',(8192,),name='l',scope=tik.scope_cbuf)
 if kind=='zero':
  emit_zero(t,l)
 else:
  t.data_move_multi_nd2nz(l,x,1,1,512,0,512,16,1,16)
 t.BuildCCE(kernel_name='probe_'+kind,inputs=[x],outputs=[])
 print('BUILT',kind,flush=True)
