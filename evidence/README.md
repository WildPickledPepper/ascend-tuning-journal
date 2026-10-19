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
