#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void launch_packed_prefill(void*,void*,void*,void*,int,int);
at::Tensor prefill(const at::Tensor& x,const at::Tensor& w,int64_t order) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && w.device()==x.device(),"same NPU required");
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16,"BF16 required");
    TORCH_CHECK(x.dim()==2 && x.size(0)>0 && x.size(0)<=512 && x.size(1)==5120,"M1..512 K5120 required");
    TORCH_CHECK(w.numel()==34816LL*5120 && x.is_contiguous() && w.is_contiguous(),"weight size and contiguous inputs required");
    TORCH_CHECK(order==0 || order==1,"order 0 or 1");
    auto y=at::empty({x.size(0),34816},x.options());
    auto stream=c10_npu::getCurrentNPUStream().stream();
    auto run=[x,w,y,stream,order]() -> int {
        launch_packed_prefill(stream,x.data_ptr(),w.data_ptr(),y.data_ptr(),int(x.size(0)),int(order));return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("PackedPrefillBF16",run);return y;
}
at::Tensor meta(const at::Tensor& x,const at::Tensor& w,int64_t order) {
    return at::empty({x.size(0),34816},x.options().device(at::kMeta));
}
TORCH_LIBRARY(packed_prefill_bf16,m){m.def("linear(Tensor x, Tensor w, int order=0) -> Tensor");}
TORCH_LIBRARY_IMPL(packed_prefill_bf16,PrivateUse1,m){m.impl("linear",&prefill);}
TORCH_LIBRARY_IMPL(packed_prefill_bf16,Meta,m){m.impl("linear",&meta);}
