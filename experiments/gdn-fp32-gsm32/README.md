# FP32状态与动态词表：32道自然题吞吐对照

用户要求“测试32道题看看平均吞吐”。本轮将此前四题两遍扩为相同32道GSM8K test题、每组各一次；正常EOS、上限2048、greedy/temperature=0、单并发。不拼接或截断输入来制造1K/3K档位，不强制凑满输出。

保持物理7号910B2、Qwen3.8-27B、BF16权重/conv、FP32 SSM状态、TP1、MTP7、主/草稿capture8、ctx4096及既有FFN优化。只改变测试样本数和客户端记录，不修改候选策略、服务源码、checkpoint或内核。

## 实验口径

- 原GSM8K test文件前32题，原顺序、不重复；记录数据源和每题哈希，不用test答案拟合或选词。
- 完整词表和动态4096各一次独立服务，均两个预热请求，不计入正式32题。
- 通过私有`EXPERIMENT_START_SCRIPT`调用此前FP32版本的原服务入口，服务源码与各自full/dynamic缓存路径均不改变；客户端及输出保存在本实验独立目录，避免覆盖四题结果。
- 主指标：总输出token/32个流式请求E2E之和。另记完整客户端计时窗口吞吐，其中含请求之间的JSON整理/写盘间隙；不把请求吞吐的算术平均混为同一指标。
- 比较32对完整token序列，记录`finish_reason`，明确有无触及2048输出上限；候选更新、权重整理、验证均在E2E内，profiler与诊断关闭。
- `snapshot`核验48层实际FP32状态、BF16 conv、64层packed FFN和主/草稿图；前后Prometheus计数核验接受长度。

首次启动脚本因Windows CRLF被bash拒绝，未启动模型或产生性能样本；统一LF后重启。失败日志保留私有区，不作为性能数据。

虽然服务路径和缓存目录不变，含Triton的AOT产物未能直接跨进程加载，框架自动重新编译后继续启动；编译与预热不进入请求计时。没有手工修改或清除历史缓存。

## 入口

用户在测试进行中追加要求详细展示TTFT、TPOT等指标。`latency_statistics.py`从本轮已有流式记录计算E2E、TTFT、TPOT、输入输出长度、请求级吞吐的均值/min/max/P50/P75/P90/P95/P99。保留沿用前轮的流结束TPOT，以及以最后token到达为终点的TPOT。客户端逐token交付间隔按同一SSE事件中的token共享到达时刻计算，因此含零值；另列非空SSE事件间隔。两者都不伪称设备内部的逐token生成ITL，不追加NPU采集。

`benchmark_controller.py`只关闭自己启动的进程组；需要既有私有部署环境和`EXPERIMENT_START_SCRIPT`，不提交服务器地址、路径、seed或原始回答。

- `bench.py` / `client.py`：32题顺序发送、自然结束、流式计时和原始记录。
- `analyze.py` / `export_evidence.py`：统计、配对输出检查及过滤摘要。
- `collect_receipts.py`：完成后只读检查7卡与端口，并记录源码哈希。

已完成两组32题：[详细结果](results.md)包含整批吞吐、TTFT/TPOT各分位数、客户端交付间隔和32对逐题记录。此前[四题结果](../gdn-fp32-state/results.md)保留，不用本轮成绩覆盖。32题点估计吞吐+3.55%，不能继续将此前四题+13.19%视为稳定收益；32/32输出逐token一致、全部正常EOS。
