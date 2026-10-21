#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <atomic>
extern "C" void launch_down_m8(int,void*,void*,void*,void*);
static std::atomic<int64_t> submitted{0};
at::Tensor down_linear(const at::Tensor& x,const at::Tensor& w,int64_t mode) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && w.device()==x.device(),"NPU required");
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16,"BF16 required");
    TORCH_CHECK(x.dim()==2 && x.size(0)==8 && x.size(1)==17408,"fixed M8 input required");
    TORCH_CHECK(w.dim()==2 && w.size(0)==5120 && w.size(1)==17408,"fixed weight required");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),"contiguous ND required");
    TORCH_CHECK(mode==0 || mode==1,"mode 0 or 1 required");
    auto y=at::empty({8,5120},x.options());
    auto stream=c10_npu::getCurrentNPUStream().stream();
    auto run=[x,w,y,stream,mode]() -> int {
        launch_down_m8(int(mode),stream,x.data_ptr(),w.data_ptr(),y.data_ptr()); return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("DownPrefetchM8BF16",run);
    ++submitted;
    return y;
}
at::Tensor down_meta(const at::Tensor& x,const at::Tensor& w,int64_t mode) {
    return at::empty({8,5120},x.options().device(at::kMeta));
}
TORCH_LIBRARY(down_prefetch_m8,m) {
    m.def("linear(Tensor x, Tensor w, int mode=1) -> Tensor");
    m.def("submitted() -> int",[]() -> int64_t{return submitted.load();});
}
TORCH_LIBRARY_IMPL(down_prefetch_m8,PrivateUse1,m){m.impl("linear",&down_linear);}
TORCH_LIBRARY_IMPL(down_prefetch_m8,Meta,m){m.impl("linear",&down_meta);}
