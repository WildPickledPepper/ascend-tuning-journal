# 动态4096＋FP32状态版本的decode采样

结果见[报告](results.md)。主FFN34.47%，动态候选维护10.87%，小head投影0.37%；无计算区间14.35%，主要位于模型图之外，尚未唯一归因于Host或实际HBM带宽。另一题前256token的类别分摊一致。

后续只用已有数据继续查输入准备：[重叠与同步分析](prepare-input-analysis.md)。约96%的间隙位于草稿图结束与下一主图开始之间；旧输入准备比草稿结束早3.935ms，当前晚11.053ms。准备操作顺序相同，第二个同步等待增加，较短草稿图无法覆盖后续准备。仅离线定位，没有改推理实现或新增NPU请求。

2026-10-22用户授权“再采当前优化版本整网profiling，先定位别急着改”。仅物理7卡BF16权重/FP32 SSM/动态4096/既有FFN优化/MTP7；没有修改推理实现、改默认、跑其他卡或提交Git。

本轮服务已于15:25:36.905停止。后续回执确认本轮进程退出、8047关闭，但7卡已有另一进程；没有处理其他进程，不声称整卡空闲。

## 代码用途

- `start.sh`、`serve.py`、`model_extension.py`：从已验证FP32状态版本复制，只增加profiler入口/RPC/标签；通过环境变量提供私有路径。
- `dynamic_vocab.py`、`graph_audit_base.py`：沿用现有动态词表和只读图审计。
- `profile_factory.py`、`profile_scopes.py`：四次Level1采集和异步Host标签，不改原函数数学操作。
- `capture.py`、`client.py`：预热、三组首题128token＋另一题256token，输出前缀对已有32题，正常Stop并在finally中退出自己的服务。
- `analyze.py`：decode分段、主/草稿/图外归属、互斥时间、PMU、原样SQL宏与CSV核对。
- `analyze_gaps.py`：无计算区间的图内/图外位置与相邻任务，不能代替因果分析。
- `compare_gaps.py`、`attribute_gap_launch.py`：后续离线核对旧/新空隙的同一统计口径，通过trace flow识别后续任务的Host调用归属。已将主要新增空隙缩小至`prepare input`链路，不将关联当成完整根因。
- `prepare_input_breakdown.py`：进一步拆CPU嵌套区间、三个同步、提交/launch时序和草稿至下一主图的衔接；比较相同CPU操作序列，核对另一题58个稳态衔接。
- `backend-query.cjs`：向本地编译版Insight导入两份Pipe，读取结构化算子统计，结果写仓库外。
- `collect_receipts.py`：只读记录自身进程/端口退出及后续卡状态；不终止其他任务。
- `export_evidence.py`：校验四份采样、输出前缀、440个Insight分组、原始归档哈希，导出不含私有路径的数字摘要。

私有数据准备好后可离线复算，不启动NPU：

```text
python analyze.py PRIVATE_CAPTURE_ROOT PRIVATE_CAPTURE_ROOT/summary-local.json
python analyze_gaps.py PRIVATE_CAPTURE_ROOT
python export_evidence.py PRIVATE_CAPTURE_ROOT FILTERED_EVIDENCE_DIRECTORY
```

复算还需要已有服务器分析副本、参考输出前缀、前后图审计、退出回执、来源清单和Insight两份结构化结果。完整请求/回答、trace/DB、checkpoint、seed、服务器路径与凭证不入库。重新启动采集须先确认卡空闲并有相应授权；本目录不会自动恢复服务。
