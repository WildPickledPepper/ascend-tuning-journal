"""Torch-free launcher, used only with the explicit CANN simulator runtime."""
import ctypes
import os
from pathlib import Path
import sys
assert 'tools/simulator/Ascend910B2/lib' in os.environ['LD_LIBRARY_PATH']
acl=ctypes.CDLL('libascendcl.so')
def bind(name,args):
    f=getattr(acl,name);f.argtypes=args;f.restype=ctypes.c_int;return f
def check(code):
    if code:raise RuntimeError('ACL '+str(code))
init=bind('aclInit',[ctypes.c_void_p]);dev=bind('aclrtSetDevice',[ctypes.c_int])
create=bind('aclrtCreateStream',[ctypes.POINTER(ctypes.c_void_p)])
alloc=bind('aclrtMalloc',[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_int])
zero=bind('aclrtMemset',[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_size_t])
sync=bind('aclrtSynchronizeStream',[ctypes.c_void_p])
check(init(None));check(dev(0));stream=ctypes.c_void_p();check(create(ctypes.byref(stream)))
buffers=[]
for size in [8*17408*2,5120*17408*2,8*5120*2]:
    p=ctypes.c_void_p();check(alloc(ctypes.byref(p),size,2));check(zero(p,size,0,size));buffers.append(p)
lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'libdown_prefetch_m8.so'))
lib.launch_down_m8.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4
lib.launch_down_m8.restype=None
lib.launch_down_m8(int(sys.argv[1]),stream,*buffers)
check(sync(stream));print('SIMULATOR_KERNEL_COMPLETED',flush=True)
