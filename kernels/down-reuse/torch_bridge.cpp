#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <atomic>
extern "C" void launch_down(int,void*,void*,void*,void*);
static std::atomic<int64_t> submitted{0};

at::Tensor down_linear(const at::Tensor& x,const at::Tensor& w) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && w.device()==x.device(),"NPU tensors required");
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16,"BF16 required");
    TORCH_CHECK(x.dim()==2 && x.size(0)==1 && x.size(1)==17408,"fixed input shape required");
    TORCH_CHECK(w.dim()==2 && w.size(0)==5120 && w.size(1)==17408,"fixed weight shape required");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),"ND contiguous storage required");
    auto y=at::empty({1,5120},x.options());
    auto stream=c10_npu::getCurrentNPUStream().stream();
    auto run=[x,w,y,stream]() -> int {
        launch_down(3,stream,x.data_ptr(),w.data_ptr(),y.data_ptr());
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("DownReuseBF16",run);
    ++submitted;
    return y;
}
at::Tensor down_meta(const at::Tensor& x,const at::Tensor& w) {
    return at::empty({1,5120},x.options().device(at::kMeta));
}
TORCH_LIBRARY(down_tuning,m) {
    m.def("linear(Tensor x, Tensor w) -> Tensor");
    m.def("submitted() -> int",[]() -> int64_t { return submitted.load(); });
}
TORCH_LIBRARY_IMPL(down_tuning,PrivateUse1,m) {m.impl("linear",&down_linear);}
TORCH_LIBRARY_IMPL(down_tuning,Meta,m) {m.impl("linear",&down_meta);}
