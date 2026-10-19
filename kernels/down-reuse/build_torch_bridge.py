import pathlib,os,subprocess,torch,torch_npu
root=pathlib.Path(__file__).resolve().parent
t=pathlib.Path(torch.__file__).parent;n=pathlib.Path(torch_npu.__file__).parent
cann=pathlib.Path(os.environ['ASCEND_HOME_PATH'])
cmd=['g++','-O2','-std=c++17','-fPIC','-shared',str(root/'torch_bridge.cpp'),'-o',str(root/'libdown_torch.so'),
 '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
 '-I'+str(t/'include'),'-I'+str(t/'include/torch/csrc/api/include'),'-I'+str(n/'include'),'-I'+str(cann/'include'),
 '-L'+str(t/'lib'),'-L'+str(n/'lib'),'-L'+str(cann/'lib64'),'-L'+str(root),
 '-Wl,-rpath,$ORIGIN','-Wl,-rpath,'+str(t/'lib'),'-Wl,-rpath,'+str(n/'lib'),
 '-ldown_reuse','-ltorch_npu','-ltorch_cpu','-ltorch','-lc10','-lascendcl','-lruntime']
subprocess.run(cmd,check=True)
print('TORCH_BRIDGE_BUILT',flush=True)
