# 优化后整网 decode 定位

**主模型 FFN 占 decode 约 36.5%，词表投影 21.3%，主模型其他线性投影 17.5%；无计算空隙约 0.9%。** FFN 与词表投影的 MTE2 活动接近持续，MAC 活动相对低，L2 读命中低，证据支持小批量矩阵乘的权重供给限制。目前不能把整条供数链进一步唯一归因于 HBM 控制器或 L2→L1 的物理带宽上限。

2026-10-22，用户授权在物理 7 卡重新采集当前优化版本，只定位、不改推理实现。沿用 BF16/TP1/并发 1/MTP7，主模型和草稿图均捕获 M8；保留 gate/up packed、down 预取、prefill 直接读取。没有重跑优化前基线，没有扫参或新增内核。

完整解释、指标和边界见 [结果报告](results.md)。[过滤证据](../../evidence/2026-10-22/optimized-decode-profile.json)、[来源校验](../../evidence/2026-10-22/optimized-decode-profile-sources.json)、[释放确认](../../evidence/2026-10-22/optimized-decode-profile-cleanup.json)。

随后按用户要求只读比较已有三份采样：[FFN前后同口径对照](history-comparison.md)。主模型FFN从“已down预取、尚未重排”的758.812→684.598ms（−9.78%），占比38.978%→36.503%（−2.475个百分点）；旧6卡/当前7卡，保留跨卡跨日限制，没有重新采样。[复算代码](compare_history.py)、[证据](../../evidence/2026-10-22/ffn-profile-history-comparison.json)。

## 代码与复算

- [prepare.py](prepare.py)：从上一版私有运行目录复制启动器、客户端和扩展，只增加 profiler 入口和 RPC。此参数化整理器在采集后归档；实际采集使用同样的复制、追加扩展和启动参数方式。
- [profile_factory.py](profile_factory.py)：Level1，CPU/NPU，record_shapes，Text/DB；依次 PipeUtilization、L2Cache、Memory。指标切换只重建已停止的 profiler，不重新捕获模型图。
- [capture.py](capture.py)：相同输入的三个 128-token 采样，输出与无 profiler 参考一致；正常 Stop，并在 finally 中关闭自己的服务进程组。开始前拒绝替换已有端口服务。
- [analyze.py](analyze.py)：完整性检查、按图边界识别 decode、重建自定义 kernel 的归属、互斥耗时、PMU、只读 SQLite 聚合与 CSV 交叉核对。
- [export_evidence.py](export_evidence.py)：校验输出、图、64 层真实激活、路径计数、195 个 Insight 分组、压缩包哈希及释放回执，导出不含私有路径的小型证据。

私有数据准备好后，离线执行：

```text
python analyze.py PRIVATE_CAPTURE_ROOT PRIVATE_CAPTURE_ROOT/summary-local.json
python export_evidence.py PRIVATE_CAPTURE_ROOT FILTERED_EVIDENCE_DIRECTORY
```

复算需要原始采集目录、`results`、服务器分析副本 `summary-server.json`、Insight 的 `optimized-pipe-backend-operators.json` 和原始归档包。原始 trace/DB、完整请求、模型路径和 SSH 信息不入库。重新采集需要先确认物理 7 卡空闲，并显式启动；本目录不会自动恢复服务。
