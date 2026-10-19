# BF16 单用户 MTP 对照实验

Qwen3.8-27B 的配置架构为 `Qwen3_5ForConditionalGeneration`，64 个主模型层，另有 1 层 MTP 预测头，权重索引包含 15 个 `mtp.*` 张量。使用已经安装的 vLLM 0.26.0 与 vLLM-Ascend 0.26.0rc1；不接入先前的自写内核。

## 配置

| 项目 | 基线 | MTP |
|---|---|---|
| 物理卡 | 7，Ascend 910B2 | 6，Ascend 910B2 |
| localhost 端口 | 8038 | 8039 |
| dtype / TP / 并发 | BF16 / 1 / 1 | BF16 / 1 / 1 |
| 主模型图模式 | FULL_DECODE_ONLY，[1] | FULL_DECODE_ONLY，[2] |
| 草稿配置 | 无 | qwen3_5_mtp，1 个候选，enforce_eager=true |
| 上下文 / 批 token 上限 | 4096 / 512 | 4096 / 512 |
| 显存预算 / 前缀缓存 | 0.90 / 关闭 | 0.90 / 关闭 |

`enforce_eager` 放在 speculative config 内，仅用于草稿；主模型保持图模式。配置方法名按安装版本解析，日志会将旧别名 `qwen3_5_mtp` 归一化为 `mtp`。

两个服务可同时保留，正式请求错开执行。不同物理卡是比较限制，应结合之前约 20.3 token/s 的单卡测量判断，不能把卡间波动写成精确收益保证。

## 文件与依赖

- `start.sh`：依赖外部的 `QWEN_MODEL_ROOT/env-npu.sh` 和已准备好的模型；`MTP_RUN` 指向独立结果目录。模型、环境和二进制不包含在仓库中。
- `client.py`：等待健康检查，预热，两个提示各 4 次、每次 128 输出 token，另做 3 个输出检查。保存 token ID、SSE 时间和 Prometheus 指标；profiler 单独运行 32 token 请求。
- `analyze.py`：分析两组已保存的回执及接受率，逐项报告输出差异。
- `backend-query.cjs`：将本地 `ASCEND_PROFILER_OUTPUT` 导入运行中的编译版 Insight 后端 9099，读取按输入形状聚合的算子耗时；参数依次为输入目录、结果目录、标签、物理设备号。

脚本仅归档本次配置；不要与已有服务抢用端口或设备。运行前应检查指定租用卡的状态。

## 指标口径

速度采用 `(输出 token 数−1)/(末 token 时间−首 token 时间)`，排除首 token 等待；同时保存 TTFT 与整请求时间。MTP 可以每次流式返回多个 token，因此并非所有 token 都有独立的物理执行时间点。

接受率为 `accepted_tokens / draft_tokens`；每轮平均产出近似为 `1 + accepted_tokens / drafts`，包括主模型补充 token。计数取预热、速度测试和输出检查后的总差值，等待统计刷新；速度摘要仅使用 8 次正式请求。性能请求时不运行 profiler。

完整精度评测超出这些短提示的覆盖范围；是否逐 token 一致由实际回执决定。权重读取摊销估算必须明确标为推导，不冒充 HBM PMU 实测。
