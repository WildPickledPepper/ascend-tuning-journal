"""Torch-free launcher for a captured input set under the CANN simulator."""
import ctypes,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parent
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text())
desc=cfg['test_cases'][0]['param_desc'][:3]
acl=ctypes.CDLL('libascendcl.so')
def bind(name,args):
    f=getattr(acl,name);f.argtypes=args;f.restype=ctypes.c_int
    return f
init=bind('aclInit',[ctypes.c_void_p]);setdev=bind('aclrtSetDevice',[ctypes.c_int])
create=bind('aclrtCreateStream',[ctypes.POINTER(ctypes.c_void_p)])
malloc=bind('aclrtMalloc',[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_int])
copy=bind('aclrtMemcpy',[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int])
sync=bind('aclrtSynchronizeStream',[ctypes.c_void_p]);free=bind('aclrtFree',[ctypes.c_void_p])
destroy=bind('aclrtDestroyStream',[ctypes.c_void_p])
def check(code):
    if code:raise RuntimeError('ACL error '+str(code))
check(init(None));check(setdev(0));stream=ctypes.c_void_p();check(create(ctypes.byref(stream)))
dev=[]
try:
    for d in desc:
        data=pathlib.Path(d['data_path']).read_bytes();p=ctypes.c_void_p()
        check(malloc(ctypes.byref(p),len(data),2));dev.append(p)
        check(copy(p,len(data),ctypes.cast(ctypes.c_char_p(data),ctypes.c_void_p),len(data),1))
    lib=ctypes.CDLL(str(root/'libdown_reuse.so'))
    lib.launch_down.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4;lib.launch_down.restype=None
    lib.launch_down(3,stream,*dev);check(sync(stream));print('ACL launch completed',flush=True)
finally:
    for p in dev:check(free(p))
    check(destroy(stream))
