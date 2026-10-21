#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void launch_gateup(int,void*,void*,void*,void*);
at::Tensor linear(const at::Tensor& x,const at::Tensor& w,int64_t layout) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && w.device()==x.device(),"NPU required");
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16,"BF16 required");
    TORCH_CHECK(x.dim()==2 && x.size(0)==8 && x.size(1)==5120,"M8 K5120 required");
    TORCH_CHECK(w.numel()==34816LL*5120 && x.is_contiguous() && w.is_contiguous(),"weight size/contiguity");
    TORCH_CHECK(layout>=0 && layout<=2,"layout 0,1,2 required");
    auto y=at::empty({8,34816},x.options());
    auto stream=c10_npu::getCurrentNPUStream().stream();
    auto run=[x,w,y,stream,layout]() -> int {
        launch_gateup(int(layout),stream,x.data_ptr(),w.data_ptr(),y.data_ptr()); return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("GateUpLayoutBF16",run);return y;
}
TORCH_LIBRARY(l1_tile_layout,m){m.def("linear(Tensor x, Tensor w, int layout) -> Tensor");}
TORCH_LIBRARY_IMPL(l1_tile_layout,PrivateUse1,m){m.impl("linear",&linear);}
