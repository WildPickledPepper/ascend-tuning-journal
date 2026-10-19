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

`manifest.json` 记录原始摘要的产物名称和 SHA256；仓库文件经过字段筛选，不应与原始文件逐字节相同。

上板、仿真、推导分别标注；不上传模型权重、完整 trace/DB、CANN 内核二进制和服务器登录信息。完整原始产物仍保存在原采集环境。
