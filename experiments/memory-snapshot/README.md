# 27B 单卡显存快照：实际占用、预留、峰值和碎片

本次用msMemScope采集12个阶段的计数快照，用torch_npu记录分配历史，再将最终pickle导入编译版Insight。结论：显存主要由BF16权重与预分配缓存占据；三次短请求未见持续增长；普通池空闲部分大多连续，不能把预留空闲全部称作碎片或泄漏。

## 配置与采集范围

2026-10-21北京时间11:17:46启动，11:21:11就绪，11:21:27采集完成并清理。Qwen3.8-27B、单卡6/TP1、BF16、vLLM0.27.1/vLLM-Ascend0.27.1rc1、MTP7、主/草稿图模式capture8、maxlen4096、并发1、max_batched_tokens512、显存预算0.97、prefix cache关闭，使用原生MatMul。本次用户授权显存分析，不重跑速度基线、不改变权重或内核。

在工作进程加载模型前启用allocator历史：`enabled=all, context=alloc, stacks=python, max_entries=500000`。进程内包装load_model/profile_run/initialize_kv_cache/capture_model的前后边界；服务就绪及三次相同的128token请求结束后再取快照，总计12份。最终106842事件，未达到截断上限。msMemScope只取阶段计数，细粒度分配历史来自torch_npu；没有将前者说成完整的alloc/free追踪。

工具依据：[官方Insight内存调优说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/visualization_tool/MindStudioInsight/docs/en/user_guide/memory_tuning.md)、[官方msMemScope快照说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msMemScope/docs/en/user_guide/memory_profile.md)。实际参数以本机安装源码核实，未安装或升级共享环境。

## 占用拆解

最终请求结束时运行时可见总显存60.957GiB，按分配调用栈与快照计数分解如下。GiB=2³⁰字节，不与GB混用。

| 项目 | GiB | 依据 |
|---|---:|---|
| 权重创建的常驻分配 | 53.256 | create_weights调用栈，360个活动块 |
| 缓存常驻分配 | 5.623 | _allocate_kv_cache_tensors调用栈，17块 |
| 其他框架活动张量/缓冲 | 0.145 | 剩余491个活动块，含RoPE与运行缓冲 |
| 框架预留但未活动 | 0.978 | reserved−allocated |
| 框架预留之外的设备已用量 | 0.567 | total−free−reserved，含运行时等，未逐组件拆分 |
| 设备可用 | 0.388 | 设备级free计数 |

前三项合计框架实际占用59.024GiB；加框架预留空闲后为60.002GiB。框架外的差额是设备级残差，不等同torch张量，也不是某个CANN组件的精确归属。

调用栈前三组常驻块中，65组升维权重21.583GiB、65组降维权重10.791GiB，合计约32.374GiB；包含主模型64层及草稿层。另有三块词表相关权重，每块约2.368GiB，不能仅凭形状相同就判为可删除的重复权重。

## 阶段与峰值

| 阶段结束 | 框架实际占用GiB | 框架预留GiB |
|---|---:|---:|
| 加载模型前 | 0.015 | 0.021 |
| 模型加载后 | 53.397 | 55.773 |
| 预热profile_run后 | 53.397 | 53.508 |
| 初始化缓存后 | 59.020 | 59.055 |
| 图捕获后/就绪 | 59.024 | 59.568 |
| 第一次请求后 | 59.024 | 60.002 |
| 第二、第三次请求后 | 59.024 | 60.002 |

模型加载净增加53.382GiB包含权重及该阶段创建的常驻缓冲，因此与按create_weights栈分类的纯权重组53.256GiB不是同一口径。缓存初始化净增加5.623GiB，17个大块各申请355155968字节，分配后各占355156480字节；该缓存是预分配容量，不代表当前对话已写满。

Insight分配事件历史给出：全程框架实际占用峰值约59.495GiB，出现在图捕获阶段；首个短请求的峰值约59.462GiB。首请求峰值点的一笔分配453050880字节，即432.0625MiB，栈尾为`qwen_gdn_attention_core → _forward_core → _chunk_gated_delta_rule_fused`，定位在GDN线性注意力。请求结束后释放回池，不是常驻权重；后续相同请求复用相同规模空间。

第一次请求后实际活动量仅增加1536字节，但预留量增加约0.434GiB；第二、第三次请求结束后，实际占用与预留量净增均为0。这符合临时工作区回池复用，本次没有持续增长迹象。**仅三次相同短请求，不能据此证明长期运行绝无泄漏。**

这些峰值是框架分配器事件峰值，不是连续采样的整个NPU进程或全设备峰值；0.388GiB是请求结束时的可用量，不称为推理过程最低剩余值。配置中的0.97是框架预算，最终设备使用还包括框架外分配，不能按“剩余3%”代替快照实数。

## 碎片与预留空间

0.978GiB预留空闲由约0.492GiB图私有池与0.486GiB普通池构成。普通池最大连续空闲块456850944字节，约435.69MiB，占该池全部空闲量87.58%；并非大量小碎片占据主要空间。其余普通池空闲合计约61.76MiB，不能将全部0.978GiB都写成“碎片”。图私有池的预留与图回放生命周期相关，本次没有证明可以安全清空。

保持BF16时，若后续专门做显存优化，应先核对需要的上下文/并发与缓存容量，考虑显式控制KV预算，而非删除权重或盲目empty_cache。本次只分析，不改缓存预算、不清图池、不做量化；没有新增速度优化结论。

## 工具核对与边界

- 编译版Insight导入完成：106842条事件、36076次分配、8个末态segment。`Memory/snapshot/allocations`、`allocationLines`、`blocks`、`state`均成功；末态实际/预留字节与原始pickle和采样计数精确一致。
- 12份msMemScope的allocated/reserved/total与同阶段torch_npu计数均精确一致。ms快照在pickle导出前采，补充metadata在导出后采，free计数有最多10MiB时间差，最终差48KiB，不伪称同时采样。
- Insight历史回放在部分中间端点存在最多4024字节舍入/重建差额，末态精确一致；报告以GiB展示，不把中间端点称为逐字节吻合。图上曲线经过转折点压缩，阶段峰值从完整事件表提取，未直接对压缩曲线做阶段最大值。
- 初次查询修正了编译版的`leaks`路由及顶层fileId，未更改分析器实现。torch_npu生成pickle初始权限为0200，完成导出后改为本用户0600才能分析；公开采集脚本加入同样修正，不需要重采。

代码：[采集扩展](snapshot_extension.py)、[后端查询](query_snapshot.cjs)、[只读复算](analyze_snapshot.py)。扩展依赖已有[只读GraphAudit](../../kernels/down-prefetch-m8/model/graph_audit_base.py)，私有运行目录和库通过环境设置，不覆盖已安装包。复算命令：`python analyze_snapshot.py PRIVATE_ROOT FILTERED_JSON`。

原始pickle、msMemScope DB及完整调用栈只留私有采集目录，Git只提交[过滤证据](../../evidence/2026-10-21/memory-snapshot.json)与必要代码。11:21:27控制器确认6/7均无活动进程、服务全部关闭，未恢复旧服务，模型文件保留。
