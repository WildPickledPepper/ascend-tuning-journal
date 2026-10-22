import os, pathlib, subprocess, torch, torch_npu
root=pathlib.Path(__file__).resolve().parent
cann=pathlib.Path(os.environ['ASCEND_HOME_PATH'])
cmd=['bisheng','-xcce','-O2','-g','--cce-aicore-arch=dav-c220-cube','-fPIC','-shared','-std=c++17']
cmd+=['-I'+str(cann/p) for p in ['include','aarch64-linux/asc/include','aarch64-linux/asc/include/basic_api','aarch64-linux/asc/impl','aarch64-linux/asc']]
for p in ['/usr/include/c++/11','/usr/include/aarch64-linux-gnu/c++/11','/usr/include/aarch64-linux-gnu','/usr/include']:cmd+=['-isystem',p]
cmd+=['-L'+str(cann/'lib64'),'-lascendcl','-lruntime',str(root/'kernel.asc'),'-o',str(root/'libprefill_kernel.so')]
subprocess.run(cmd,check=True)
t=pathlib.Path(torch.__file__).parent;n=pathlib.Path(torch_npu.__file__).parent
cmd=['g++','-O2','-std=c++17','-fPIC','-shared',str(root/'bridge.cpp'),'-o',str(root/'libprefill_torch.so'),'-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
cmd+=['-I'+str(p) for p in [t/'include',t/'include/torch/csrc/api/include',n/'include',cann/'include']]
cmd+=['-L'+str(p) for p in [t/'lib',n/'lib',cann/'lib64',root]]
cmd+=['-Wl,-rpath,$ORIGIN','-Wl,-rpath,'+str(t/'lib'),'-Wl,-rpath,'+str(n/'lib'),'-lprefill_kernel','-ltorch_npu','-ltorch_cpu','-ltorch','-lc10','-lascendcl','-lruntime']
subprocess.run(cmd,check=True)
print('BUILD_OK',torch.__version__,torch_npu.__version__,flush=True)
