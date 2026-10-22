# 两份计数提前与固定MTP7元数据准备

在[首次接受数量提前实验](../accepted-count-overlap/results.md)之后，继续定位为什么等待转移、收益有限，并尽量降低同轮动态4096词表的图间空隙。

已完成：[实际结果与限制](results.md)。同服务采样空隙13.821%→0.940%，输入准备从晚10.573ms变为早6.423ms；同服务四题无profiler吞吐仅+0.805%，32题实验版63.990 token/s。保留采样扰动、共享Host波动和接受轨迹差异，不设默认。

后续[TTFT追查](ttft-analysis.md)与`analyze_ttft.py`只复用已有数据：核对客户端流式记录、服务端指标、后续同题记录和首轮采样；首轮阶段变慢已定位，原499ms请求的内部根因尚未确定，没有新增NPU请求。

本轮仍使用物理6卡、Qwen3.8-27B、BF16权重/conv、FP32 SSM、TP1/并发1/MTP7，保留既有FFN重排、prefill直读及down预取。所有改动仅当前服务进程，不改安装源码、权重、数学计算或NPU内核。同轮候选策略代码保持不变。

`explain_waits.py`只读上一轮trace，以CPU memcpy scope → torch_to_npu flow → 设备MEMCPY_ASYNC建立链路。第二份计数的实际设备拷贝约1.9µs、就绪时间约60.19ms未变；CPU到达第三同步从62.44ms提前到54.10ms，因此暴露约6ms等待。长同步不能直接解释为拷贝慢。

`early_counts.py`保留四种可切换阶段：0原顺序；1只提前原状态/接受数处理；2同时提前原来的有效token数计算及回传，草稿准备复用同一结果；3在阶段2上启用固定MTP7元数据快路径。阶段2在原始bookkeeping之后调用原始prepare_next_token_ids_padded，保留discard/vocab/fallback语义，随后本轮词表维护；原草稿入口消费缓存结果，避免重复计算或再次晚回传。两份计数没有直接互相替代。

`fast_metadata.py`仅对单请求、查询范围[0,8]、7草稿、mamba_cache_mode=none的完整图decode生效。固定掩码、token索引和query范围初始化后复用；状态索引和接受数量仍每轮更新，其余形状回退原GDN builder。验证模式先运行原builder、克隆全部Tensor字段，再比较快路径的每个字段、形状、stride、dtype和设备。验证在正式测速前执行。

`hold.py`检查6卡和端口空闲，只管理自己的进程组，等待STOP或最长一小时，失败时释放自己的服务。`phase.py`执行有界正确性、自然结束测速与单独profiling；候选/采样/草稿哈希记录仅用于未计时预检。`analyze.py`和`prepare_input_breakdown.py`从真实decode边界复算，排除prefill，不把CPU等待相加为NPU空隙。

阶段2使用独立同服务32题原顺序/优化对照；阶段3复用其近期32题原顺序参考，并补同服务四题对照与双profiling，优化版测完整32题。两种比较分别报告，不能把跨服务32题差值当成严格同进程A/B。第二步原顺序采样出现明显Host时间波动，所有原始结果保留，不把异常偏高的空隙全部归因于调度改动。

`validate_metadata_cpu.py`以实际Ascend builder作参考，覆盖接受数1～8；真实NPU预检另核对3个实际builder的所有30个字段。`export_results.py`校验SQL/CSV/Insight、两端复算、图/精度/源码/释放后导出数字。`capture_lite.py`的动态附加在当前环境被工具拒绝，没有轻量设备空隙成绩；仅用同128请求开/关采样E2E作扰动核对，并保留一轮执行数量差异。

本轮启动环境设置`OVERLAP_STARTUP_COLLECT=1`，在词表初始化前回收加载阶段临时对象，位于计时之外。原始回答、候选ID、seed、trace/DB、运行路径和启动失败记录留私有数据区；自己的服务已关闭。
