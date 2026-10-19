# 社区方案：BF16 权重预排 NZ

本轮参考现有 vLLM-Ascend 能力处理小 M 矩阵乘，不继续自写内核。目标是检查候选是否适用于 910B2 上的两组 FFN 形状，而不是直接采用其他芯片的速度结论。

## 已核对的官方路径

- [v0.26.0rc1 配置文档](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/user_guide/configuration/additional_config.html)：`weight_nz_mode=2` 可对支持的浮点权重使用 NZ。
- 安装源码 `vllm_ascend/utils.py`：BF16/FP16 仅模式 2 执行 NZ 转换；`maybe_trans_nz` 使用 `npu_format_cast(..., 29)`。
- `vllm_ascend/ops/linear.py`：未量化 linear 在加载权重后调用上述转换，运行时仍调用原生 `torch.nn.functional.linear`。
- `vllm_ascend/worker/model_runner_v1.py`：启用 `torch.npu.config.allow_internal_format=True`。独立 Python 探针必须匹配这个前提，且应检查实际格式 ID。
- [社区 issue #1563](https://github.com/vllm-project/vllm-ascend/issues/1563)：310P 上观察到大量运行时格式转换，提出加载阶段预排。该芯片、dtype 和负载与本实验不同，其速度数字不用于预测本机收益。

NZ 是权重在存储中的排列方式，BF16 数值位宽不变。但存储路径可能改变 CANN 的内核选择和浮点累加顺序，仍须检查输出。

## 本次测量

时间：2026-10-19 16:31:58—16:32:16，Asia/Shanghai。

独立进程、物理 7 号卡，现有基线模型驻留且没有并发推理请求。每形状使用两份随机 BF16 权重，分别保存 ND 与 NZ，逻辑值相同。预热后捕获含两个 linear 的设备图，每格式 40 轮计时并交替先后顺序，用 NPU 事件时间除以 2 得到每次调用时间。profiler 关闭。

| FFN | M | ND（μs） | NZ（μs） | 耗时变化 |
|---|---:|---:|---:|---:|
| 门控＋升维，N=34816、K=5120 | 1 | 290.745 | 251.875 | −13.37% |
| 门控＋升维，N=34816、K=5120 | 2 | 282.120 | 241.505 | −14.40% |
| 降维，N=5120、K=17408 | 1 | 163.225 | 143.200 | −12.27% |
| 降维，N=5120、K=17408 | 2 | 155.625 | 136.850 | −12.06% |

NZ 的实际格式 ID 均为 29；转回 ND 后原始权重逐位一致。4 个形状、共 239616 个输出比较均无差异且有限。完整成对样本保存在[结果 JSON](../../evidence/2026-10-19/community-matmul-nz.json)。

第一次因默认禁止内部格式而得到 ND，格式断言阻止了测速；另一次读取只写配置属性失败。两次均没有纳入上述有效结果。

## 使用与限制

[check_nz.py](check_nz.py) 使用当前已配置的私有 Torch NPU/CANN 环境运行。要求设备可见性指向物理 7 号卡、`TASK_QUEUE_ENABLE=1`，并有足够空闲显存；脚本会在自身目录写结果。它只调用现有原生算子，不修改模型服务或安装包。

这不是完整模型实验，也没有采集 NZ 路径的 PMU，因此尚未确认具体减少了多少内部搬运。事件时间含图内调用间隔，不直接等于 kernel 活动时间。模型驻留、有限随机样本、缓存及调度状态均可能影响结果。

目前仅将 NZ 列为有实测支持的后续候选。真实权重、Prefill、MTP 草稿头、逐项输出和整体速度均未验证，不宣称已取得整模型收益。另一个已由此前采样定位的方向是官方 MTP 草稿/采样图外流程；本轮没有修改该配置。
