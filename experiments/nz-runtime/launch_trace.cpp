// Host-side observation only: all launch arguments are forwarded unchanged.
// No NPU kernels, device copies, synchronization, or tensor contents are added.
#include <runtime/kernel.h>
#include <rt_external_kernel.h>
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <set>
#include <vector>
#include <string>

namespace {
struct Binary { const void* data; uint64_t length; };
std::mutex& lock() { static std::mutex m; return m; }
std::map<void*, Binary>& binaries() { static std::map<void*, Binary> m; return m; }
std::map<uint64_t, unsigned>& counts() { static std::map<uint64_t, unsigned> m; return m; }
std::map<void*, uint64_t>& keys() { static std::map<void*, uint64_t> m; return m; }
std::set<void*>& saved() { static std::set<void*> s; return s; }
bool selected(uint64_t k) { return k==98499 || k==98513 || k==229571 || k==229585; }
void observe(const char* api, void* handle, uint64_t key, uint32_t blocks,
             const rtArgsEx_t* a, rtStream_t stream) {
    const char* dir = getenv("NZ_LAUNCH_TRACE_DIR");
    if (!dir || !selected(key) || !a || !a->args || a->argsSize>65536) return;
    std::lock_guard<std::mutex> g(lock());
    if (counts()[key]++ >= 1200) return;
    char path[4096];
    snprintf(path,sizeof(path),"%s/launch-%d.jsonl",dir,getpid());
    FILE* f=fopen(path,"a"); if (!f) return;
    auto p=static_cast<const unsigned char*>(a->args);
    fprintf(f,"{\"api\":\"%s\",\"pid\":%d,\"key\":%llu,\"blocks\":%u,\"handle\":\"%p\",\"stream\":\"%p\",\"args_size\":%u,\"tiling_addr_offset\":%u,\"tiling_data_offset\":%u,\"has_tiling\":%u,\"no_h2d_copy\":%u,\"host_inputs\":[",
      api,getpid(),(unsigned long long)key,blocks,handle,stream,a->argsSize,
      a->tilingAddrOffset,a->tilingDataOffset,a->hasTiling,a->isNoNeedH2DCopy);
    for(unsigned i=0;a->hostInputInfoPtr && i<a->hostInputInfoNum && i<128;i++)
      fprintf(f,"%s[%u,%u]",i?",":"",a->hostInputInfoPtr[i].addrOffset,a->hostInputInfoPtr[i].dataOffset);
    fprintf(f,"],\"args_hex\":\"");
    for(unsigned i=0;i<a->argsSize;i++) fprintf(f,"%02x",p[i]);
    fprintf(f,"\"}\n"); fclose(f);
    auto b=binaries().find(handle);
    if(b!=binaries().end() && !saved().count(handle) && b->second.length<128*1024*1024) {
      snprintf(path,sizeof(path),"%s/binary-%d-%p.o",dir,getpid(),handle);
      f=fopen(path,"wb");
      if(f) { fwrite(b->second.data,1,b->second.length,f); fclose(f); saved().insert(handle); }
    }
}
}

extern "C" rtError_t rtRegisterAllKernel(const rtDevBinary_t* bin, void** hdl) {
    using Fn=rtError_t(*)(const rtDevBinary_t*,void**);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtRegisterAllKernel");
    auto ret=real(bin,hdl);
    if(ret==0 && bin && hdl) { std::lock_guard<std::mutex> g(lock()); binaries()[*hdl]={bin->data,bin->length}; }
    return ret;
}
extern "C" rtError_t rtKernelLaunchWithHandle(void* h,uint64_t k,uint32_t b,
    rtArgsEx_t* a,rtSmDesc_t* sm,rtStream_t s,const void* info) {
    using Fn=rtError_t(*)(void*,uint64_t,uint32_t,rtArgsEx_t*,rtSmDesc_t*,rtStream_t,const void*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtKernelLaunchWithHandle");
    observe("rtKernelLaunchWithHandle",h,k,b,a,s);
    return real(h,k,b,a,sm,s,info);
}
extern "C" rtError_t rtKernelLaunchWithHandleV2(void* h,uint64_t k,uint32_t b,
    rtArgsEx_t* a,rtSmDesc_t* sm,rtStream_t s,const rtTaskCfgInfo_t* cfg) {
    using Fn=rtError_t(*)(void*,uint64_t,uint32_t,rtArgsEx_t*,rtSmDesc_t*,rtStream_t,const rtTaskCfgInfo_t*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtKernelLaunchWithHandleV2");
    observe("rtKernelLaunchWithHandleV2",h,k,b,a,s);
    return real(h,k,b,a,sm,s,cfg);
}
extern "C" rtError_t rtBinaryGetFunction(rtBinHandle h,uint64_t k,rtFuncHandle* f) {
    using Fn=rtError_t(*)(rtBinHandle,uint64_t,rtFuncHandle*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtBinaryGetFunction");
    auto ret=real(h,k,f);
    if(!ret && f) { std::lock_guard<std::mutex> g(lock()); keys()[*f]=k; }
    return ret;
}
extern "C" rtError_t rtFunctionRegister(void* h,const void* f,const char_t* name,
    const void* ext,uint32_t mode) {
    using Fn=rtError_t(*)(void*,const void*,const char_t*,const void*,uint32_t);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtFunctionRegister");
    auto ret=real(h,f,name,ext,mode);
    if(!ret && name && strstr(name,"MatMul")) {
      const char* last=strrchr(name,'_');
      uint64_t k=last?strtoull(last+1,nullptr,10):0;
      std::lock_guard<std::mutex> g(lock()); keys()[const_cast<void*>(f)]=k;
      auto bi=binaries().find(h);
      if(bi!=binaries().end()) binaries()[const_cast<void*>(f)]=bi->second;
    }
    return ret;
}
extern "C" rtError_t rtLaunchKernelByFuncHandleV3(rtFuncHandle f,uint32_t b,
    const rtArgsEx_t* const a,rtStream_t s,const rtTaskCfgInfo_t* const cfg) {
    using Fn=rtError_t(*)(rtFuncHandle,uint32_t,const rtArgsEx_t*,rtStream_t,const rtTaskCfgInfo_t*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtLaunchKernelByFuncHandleV3");
    uint64_t k=0; { std::lock_guard<std::mutex> g(lock()); k=keys()[f]; }
    observe("rtLaunchKernelByFuncHandleV3",f,k,b,a,s);
    return real(f,b,a,s,cfg);
}
extern "C" rtError_t rtKernelLaunchWithFlagV2(const void* f,uint32_t b,
    rtArgsEx_t* a,rtSmDesc_t* sm,rtStream_t s,uint32_t flags,const rtTaskCfgInfo_t* cfg) {
    using Fn=rtError_t(*)(const void*,uint32_t,rtArgsEx_t*,rtSmDesc_t*,rtStream_t,uint32_t,const rtTaskCfgInfo_t*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rtKernelLaunchWithFlagV2");
    // The stub is the kernel name for CANN's registered native MatMul path.
    uint64_t k=0;
    { std::lock_guard<std::mutex> g(lock()); k=keys()[const_cast<void*>(f)]; }
    observe("rtKernelLaunchWithFlagV2",const_cast<void*>(f),k,b,a,s);
    return real(f,b,a,sm,s,flags,cfg);
}
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle h,uint64_t k,aclrtFuncHandle* f) {
    using Fn=aclError(*)(aclrtBinHandle,uint64_t,aclrtFuncHandle*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"aclrtBinaryGetFunctionByEntry");
    auto ret=real(h,k,f);
    if(!ret && f) { std::lock_guard<std::mutex> g(lock()); keys()[*f]=k;
      auto bi=binaries().find(h); if(bi!=binaries().end()) binaries()[*f]=bi->second; }
    return ret;
}
extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle h,const char* name,aclrtFuncHandle* f) {
    using Fn=aclError(*)(aclrtBinHandle,const char*,aclrtFuncHandle*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"aclrtBinaryGetFunction");
    auto ret=real(h,name,f);
    if(!ret && f && name && strstr(name,"MatMul")) {
      const char* last=strrchr(name,'_');
      std::lock_guard<std::mutex> g(lock()); keys()[*f]=last?strtoull(last+1,nullptr,10):0;
      auto bi=binaries().find(h); if(bi!=binaries().end()) binaries()[*f]=bi->second;
    }
    return ret;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle f,uint32_t b,
    aclrtStream s,aclrtLaunchKernelCfg* cfg,void* a,size_t size,
    aclrtPlaceHolderInfo* placeholders,size_t num) {
    using Fn=aclError(*)(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"aclrtLaunchKernelWithHostArgs");
    uint64_t k=0; { std::lock_guard<std::mutex> g(lock()); k=keys()[f]; }
    rtArgsEx_t args{}; args.args=a; args.argsSize=size;
    args.hostInputInfoPtr=reinterpret_cast<rtHostInputInfo_t*>(placeholders);
    args.hostInputInfoNum=num;
    if(num) { args.hasTiling=1; args.tilingAddrOffset=placeholders[num-1].addrOffset;
      args.tilingDataOffset=placeholders[num-1].dataOffset; }
    observe("aclrtLaunchKernelWithHostArgs",f,k,b,&args,s);
    return real(f,b,s,cfg,a,size,placeholders,num);
}
extern "C" aclError aclrtBinaryLoadFromData(const void* data,size_t len,
    const aclrtBinaryLoadOptions* opt,aclrtBinHandle* h) {
    using Fn=aclError(*)(const void*,size_t,const aclrtBinaryLoadOptions*,aclrtBinHandle*);
    static Fn real=(Fn)dlsym(RTLD_NEXT,"aclrtBinaryLoadFromData");
    auto ret=real(data,len,opt,h);
    if(!ret && h) { std::lock_guard<std::mutex> g(lock()); binaries()[*h]={data,len}; }
    return ret;
}
