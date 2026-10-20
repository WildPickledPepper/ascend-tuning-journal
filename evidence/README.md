# 证据摘要

这里保存从原始 profiler 数据和编译版 Insight 后端响应提取的小型摘要，日志通过相对链接引用。

- `2026-10-16/`：上板小例子、9B 采样、27B 离线准备。
- `2026-10-19/eager-vs-graph.json`：同请求 3 次无 profiler 基准的中位数。
- `2026-10-19/graph-phases.json`：Prefill/Decode 分阶段统计。
- `2026-10-19/hbm.json`：独立 HBM 采样与单位核对。
- `2026-10-19/two-matmuls.json`：两组前馈矩阵乘与补采摘要。
- `2026-10-19/down-movement.json`：21.5625% 分解、23 核核验及结构推演。
- `2026-10-19/backend-instruction-focus.json`：编译版后端返回的关键指令记录。
- `2026-10-19/calibration.json`：两个上板探针的后端内存计数。
- `2026-10-19/down-reuse-experiment.json`：独立 BF16 内核的修改、23 核计数、交替图回放基准及正确性差异。
- `2026-10-19/down-reuse-instructions.json`：修改版仿真指令的编译版后端返回字段，保留 PC、次数与处理字节。

`manifest.json` 记录原始摘要的产物名称和 SHA256；仓库文件经过字段筛选，不应与原始文件逐字节相同。

上板、仿真、推导分别标注；不上传模型权重、完整 trace/DB、CANN 内核二进制和服务器登录信息。完整原始产物仍保存在原采集环境。

- `2026-10-19/down-model-ab.json`：整模型 A/B/A2/C，包含未命中路由、输出失败、D 阶段中止与基线筛选限制。
- `2026-10-19/down-model-backend.json`：编译版 Insight 返回的原 CANN 与自写 decode 统计。
- `2026-10-19/down-original-k-orders.json`：原 CANN 全部 23 核的 K 块顺序。
- `2026-10-19/down-korder-validation.json`：修正 K 顺序后的 25 案例、4 版本逐位检查字段。
- `2026-10-19/down-final-state.json`：异步提交检查及停止后的资源状态。
- `2026-10-19/down-model-source-manifest.json`：本轮保留的小型源摘要 SHA256。

- `2026-10-19/mtp-comparison.json`：单用户 BF16 MTP 的速度、接受率、逐项输出对比与限制。
- `2026-10-19/mtp-backend.json`：编译版 Insight 的 M=1/2、FFN 调用次数与累计耗时。
- `2026-10-19/mtp-service-state.json`：基线与 MTP 两服务的健康状态。
- `2026-10-19/mtp-source-manifest.json`：本轮摘要来源 SHA256 与运行时代码版本。

- `2026-10-19/mtp-bottleneck.json`：MTP 后连续解码占比、逐轮图内/图后窗口、PMU 与只读 SQL 返回。
- `2026-10-19/mtp-gap-evidence.json`：三段最大设备计算空档与 CPU 小操作的重叠样本及限制。
- `2026-10-19/community-matmul-nz.json`：官方 BF16 ND/NZ 路径的独立算子对照，包含全部 40 轮成对计时；不是整模型成绩。
- `2026-10-19/community-matmul-source-manifest.json`：本次原生探针与结果摘要的来源哈希、版本和社区文档。

- `2026-10-19/nz-*.json`：官方 NZ 整模型对照、编译版后端、逻辑形状匹配、恢复与按用户要求停止复测的状态。

- `2026-10-19/nz-context-*.json`：同卡 kernel/PMU、L2 干预和同地址真实权重对照，保留未支持的解释及当前根因边界。

- `2026-10-19/nz-runtime-comparison.json`：真实 ELF/TilingData、同层 X/W/Y 地址、原生 DMA 仿真参数、两层反复读与 64 层工作集的上板 PMU/MemoryAccess、Insight 返回交叉验证；没有声称已取得 TLB 或 HBM 通道根因。
- `2026-10-19/nz-runtime-source-manifest.json`：本次归档主机观察器、隔离探针、参数比较与后端查询脚本的 SHA256。

- `2026-10-20/nz-hardware-comparison.json`：原始 SMMU/TLB 事件、官方 CSV 校验、稳定区间、L2 数据命中、同权重相邻重读干预、完整覆盖的四阶段 HBM、16 项 Insight 均值/次数核对和原服务恢复状态。
- `2026-10-20/nz-hardware-source-manifest.json`：本次采集/分析脚本、过滤摘要、原始 SMMU 数据 SHA256 与安装源码依据。
- `2026-10-20/dflash2-comparison.json`：单卡 BF16、同版本 MTP/DFlash2 四次正式请求、投机接受计数、原始 token 与 EOS 前缀比较，含强制续写边界。
- `2026-10-20/dflash2-runtime-evidence.json`：过滤启动失败、显存预算和就绪进程占用、原服务最终恢复状态。
- `2026-10-20/dflash2-source-manifest.json`：官方镜像/框架 commit、草稿固定版本与 SHA256、归档脚本哈希。
- `2026-10-20/dflash2-memory-capacity.json`：测速结束后运行时可见容量与 95% 预算换算，不是峰值采样。
- `2026-10-20/dflash2-restored-configs.json`：原基线/MTP 的卡号、BF16、90% 预算、图/投机配置及调试/NZ 环境清理状态。

- `2026-10-20/vllm-mtp7-vs-sglang.json`：单卡七候选、草稿图，vLLM59.389与SGLang42.167四题回执、版本、实际捕获图与显存快照；包含输出长度和首题限制。
- `2026-10-20/vllm-mtp1-vs-mtp7-gsm.json`：同四题旧配置30.443与新59.389，无profiler，各844token、4/4，正式接受计数和最终恢复。
- `2026-10-20/vllm-mtp1-vs-mtp7-profiles.json`：四份128-token配对采样，720个Insight分组与SQL TOP20全核验、主/草稿图区分、FFN/周期时间和原始文件SHA256。

- `2026-10-20/vllm-mtp-mfu.json`：新增data-check/MFU两个skill的四份Stop/解析配置校验、服务器/本地来源哈希口径、8组BF16矩阵乘MFU及调用加权汇总，理论峰值来自本机CANN平台配置。

- `2026-10-20/vllm-mtp7-gsm128-result.json`：单卡BF16 MTP7草稿图完整128题，57.198token/s、服务端TTFT341.543ms、原评分121/128；逐题过滤数据、4个长度截断、3个解析误判、接受位置计数与恢复回执。
