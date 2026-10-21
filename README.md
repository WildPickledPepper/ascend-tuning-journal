# Ascend 调优日志

按时间记录我们在 Ascend 910B2 上部署、采集和调优 Qwen 推理的过程。

每条日志回答：**什么时候，做了什么，分析出什么，实际改了什么，验证结果怎样。**

## 从这里看

| 日期 | 当天的工作 |
|---|---|
| [早期记录](journal/undated-history.md) | MindStudio/Insight、示例仓库和分析环境准备，部分操作未保留准确时间 |
| [2026-10-16](journal/2026-10-16.md) | 服务器环境、6 号卡小例子、9B 推理采样、27B 离线准备 |
| [2026-10-19](journal/2026-10-19.md) | 27B 部署；图模式；HBM/矩阵乘；内核实验停止；MTP；NZ 回退与差异补采 |
| [2026-10-20](journal/2026-10-20.md) | 实采 TLB/HBM 定位 NZ 退化；DFlash2/MTP 采样归因；SGLang 昇腾双卡/单卡；vLLM MTP7同题对照及Insight归因 |
| [2026-10-21](journal/2026-10-21.md) | 完整调优复盘；复算最终M8采样总耗时，解释局部4%与整模型0.47%的关系 |

**完整过程：[Qwen3.8-27B在910B2上的定位与优化复盘](notes/qwen27b-tuning-retrospective.md)。** 按16步串联现象、假设、受控实验、Insight证据、实际改动与采用判断，保留失败和结论更正。最后附当前完整采样耗时分摊及复算代码。

搬运技术说明：[HBM/L2/L1/L0 数据流、ND/NZ 地址、分块与同步](notes/matmul-data-movement.md)。

## 最近一次记录的状态

- **原生跨分块预取已核对**：[CCU、地址、SET/WAIT及Insight逐项证据](experiments/wait-timeline/prefetch-audit.md)。检查原23核仿真的core0/1/22：升维13个输出块交界均有预取，降维3个均无；K包切换升维301/304提前、降维0/198提前。降维目标L1槽已释放且复用WAIT不阻塞，却在旧块最后MMAD后才下发新权重，存在单包前视调度候选。**这些是原生二进制仿真结构证据，不是上板WAIT时间或已实现的加速**。未改内核、未启动模型；10月21日10:37:28确认6/7无活动进程。[过滤证据](evidence/2026-10-21/native-prefetch-audit.json)。

- **上板WAIT时序核验完成，尚未定位具体WAIT**：[实际采集与工具能力边界](experiments/wait-timeline/README.md)。910B2拒绝真实指令时序`instrTimeLine`；两份Default和一份Source已导入编译版Insight，基础计数有效但动态指令为空。核实`TimelineDetail`的指令时间来自仿真，不能当成上板WAIT实测；没有证明双缓冲的软件串行堵塞。10月21日10:08:08确认6/7卡均无活动进程，本次未启动模型服务。[过滤证据](evidence/2026-10-21/device-wait-timeline-capability.json)。

- **原生M8三版优化已完成整模型验证**：[参数、指令与完整结果](experiments/mte2-breakdown/native-m8-audit.md)。大包、小M、错峰原生已有；仅放大L1反而变慢。最终仅调整降维L0分块，独立算子150.436→143.190μs，模型内153.430→147.308μs；完整27B同四题 **59.420→59.697 token/s（+0.466%，基本持平）**，4/4正确且全部token ID相同。BF16不变，保留实验方案、未设为默认；没有证明稳定的整网提速。三份Insight共579分组已核对。**先推送实验记录，再于19:42:23清理6号卡；6/7均无活动进程，服务全部关闭，模型文件保留。** [实验过滤证据](evidence/2026-10-20/native-m8-tiling-optimization.json)、[清理回执](evidence/2026-10-20/native-m8-cleanup.json)。

- **BF16权重读取资料研究**：[论文、官方案例与本机适用性](notes/bf16-weight-bandwidth-research.md)。重点核对大包搬运、小M分块、错峰读取及当前M8的NZ路径；预加载有独立适用条件。无损压缩的公开实现未证明适合本机小批量提速。本次没有新上板实验或速度结果。

- **HBM→L1链已取得受控定位证据**：[缓存对照与带宽解析修复](experiments/mte2-breakdown/cache-controls.md)。同一ND升维地址/分块/搬运字节，L2读命中0.505%→81.253%，MTE2活动262.652→113.853μs；主要成本在L2未命中后的HBM供数。保留转换仍大幅加快；TLB miss未改善。修正原始DBI核号错配后23核带宽均有效，与PMU字节完全一致。原装msprof未替换，无新整模型速度结果。**7号卡已按用户要求关闭，保持释放；6号卡原MTP服务已恢复**。[过滤证据](evidence/2026-10-20/mte2-memoryfix-cache-controls.json)。

- **MTE2内部拆分尚未完成**：[原生ND/NZ补采](experiments/mte2-breakdown/README.md)。同形状升维279.046→240.965μs、主要分块参数相同，但该差额不是纯转换耗时。MemoryDetail排除内存不足失败轮后，仍有16/23核活跃带宽字段为零的覆盖或映射异常。不能把“供数通路主导”直接写成“HBM物理带宽已打满”。17:38:39确认原6/7服务均健康，7号卡未重启。[证据](evidence/2026-10-20/mte2-breakdown-native.json)。

- **最新搬运实采已完成**：[当前ND/M8诊断](experiments/nd-m8-hardware/README.md)。15份采样、35个Insight分组核验通过。small/full的TLB miss虽增加，矩阵乘均约215.5μs/次、HBM约1.243TB/s，未复现旧NZ/M2的TLB退化；当前优先关注GM→L1权重读取。真实主FFN约28.82ms/轮，草稿词表投影另约13.34ms/轮。17:08:04已恢复原服务。[证据](evidence/2026-10-20/nd-m8-memory-diagnosis.json)。

- 新核验：[1GB显存大页](experiments/hugepage-audit/README.md)软件支持，但两种1g子进程配置均因无可用预分配资源回退2MB；1GiB内存池块不代表物理大页。原服务未重启，未测加速。[证据](evidence/2026-10-20/hugepage-allocation-audit.json)。

- 当前目标：**保持 BF16，依据 ms 工具证据调优推理**；新增按公开昇腾配方的双卡小样本验证。
- **128题已完成**：[最佳单卡BF16 MTP7草稿图](experiments/vllm-mtp7/gsm128/README.md)，GSM8K前128题、单并发、无profiler：**57.198 token/s**、40767token/712.739秒，平均服务端TTFT **341.543ms**，就绪显存59.83GiB（非峰值）。原评测器 **121/128=94.53%**；7项未得分含4个2048-token截断及3个答案末尾数字误提取，原得分未改。16:13:44已恢复原服务，卡7未重启。[最终证据](evidence/2026-10-20/vllm-mtp7-gsm128-result.json)。
- 最新同题单卡对照：[vLLM MTP1→MTP7草稿图](experiments/vllm-mtp7/README.md)，同四题均844 token、4/4正确，**30.443→59.389 token/s，1.951倍**；验证448→171轮。Insight四份128-token采样共720个形状分组核对通过：第1题主模型70→26轮，两组FFN累计1.966→0.753秒，单次矩阵乘没有更快，收益来自更少重复验证与权重读取。SSM状态FP32→BF16等因素同时改变，未做独立消融。15:38:46确认恢复原服务，卡7未重启；MTP7未设默认。[正常测速](evidence/2026-10-20/vllm-mtp1-vs-mtp7-gsm.json)、[采样](evidence/2026-10-20/vllm-mtp1-vs-mtp7-profiles.json)。
- 新增两个skill复核：[数据校验与算子MFU](experiments/vllm-mtp7/mfu.md)。四份数据均valid；主模型升维MFU **0.702%→2.701%**，降维 **0.663%→2.615%**，按CANN平台约353.89TFLOP/s理论分母估算。M2→8提高权重计算复用，仍未消除搬运主导；这是算子逻辑FLOP口径，不是整模型有效输出MFU。本次复用旧采样，没有新增NPU请求。
- 同四题vLLM新配置59.389相对此前SGLang单卡42.167高40.84%，但第三题输出575/692、首题耗时差异，不能视作普遍框架排名。[独立回执](evidence/2026-10-20/vllm-mtp7-vs-sglang.json)。
- 此前SGLang单卡验证：[BF16、MTP7与草稿图](experiments/sglang-ascend/single-card.md)已在物理6跑通，同四题端到端**42.167 token/s**、4/4正确；长回答第三题**53.254 token/s**，未达到此前双卡69.748。静态预算0.97、上下文4096、缓存8192 token；就绪进程合计约55.90 GiB，非峰值。14:47:00已确认恢复原服务，卡7原进程未重启；没有新做profiler采样，不猜测剩余差额的算子归因。[回执](evidence/2026-10-20/sglang-single-mtp7.json)。
- 最新实验：[SGLang 昇腾方案，双卡 BF16、batch1、同四道 GSM8K](experiments/sglang-ascend/README.md)：AR **28.668**、MTP7 **69.748**、DFlash2 **57.560 token/s**，各 4/4 正确。DFlash2 为 AR 的 **2.008 倍**，吞吐相对 MTP 低 **17.475%**；其首题耗时偏长，第三题输出长度也不同。结果只覆盖四题，包含 prefill，不当作稳态解码上限，也不与旧单卡速度直接相除。[回执](evidence/2026-10-20/sglang-ascend-four-questions.json)。
- 接受率复核：本次MTP7日志窗口 **39%～61%**、DFlash2 **42%～60%**，每轮均约四五个token；不是全程平均，未显示明显接受率优势。旧MTP为单候选、草稿eager、单卡，约59毫秒返回1.8个token；新测试改为七候选、草稿图、双卡且题目/计时口径不同，不能把30→70全部归给框架，亦不能将旧30视为硬件上限。相同长度的第2、4题DFlash2更快，首题多3.492秒影响四题总吞吐，尚未采样其内部原因。[配置与结论复核](experiments/sglang-ascend/README.md#接受率与旧约-30-tokens-配置的复核)。
- DFlash2 已补完 [四份采样与 Insight 后端分析](experiments/dflash2/profiling.md)：每次返回更多 token，但 eager 草稿/准备流程也变贵，抵消多数收益；候选路径选择设备内核约 5.18 μs/轮，不是主热点。全部 778 个形状分组的调用数和总耗时与 CSV 一致，SQL TOP20 也核对通过。随后按用户要求完成双卡 SGLang 昇腾配方小样本复现，与此前单卡结果分别记录。
- 模型：Qwen3.8-27B，Dense，每个服务单卡 TP=1；7 号卡保留基线，6 号卡启用 MTP。
- 此前单卡实验：[DFlash2 单卡 BF16 验证](experiments/dflash2/README.md) 已跑通；95% 框架预算下就绪时进程约 **58.64 GiB**。同一 0.27.1rc1 框架、4 次 256-token 请求，MTP **30.673**、DFlash2 **33.614 token/s（+9.59%）**，首 token **282→372 ms**；提示间收益不同。7 项正常响应前缀一致，固定长度测试中有 2 项在 EOS 之后的强制续写不同；新环境未替换默认服务。[过滤比较](evidence/2026-10-20/dflash2-comparison.json)。
- 已实施并验证：切换为 `FULL_DECODE_ONLY` 图模式，3 次无 profiler 基准的生成速度中位数 **5.55 → 20.28 token/s，约 3.66 倍**。
- 已定位：门控＋升维、降维两组前馈矩阵乘合计占连续解码时间约 **55.9%**，主要负担是权重读取。
- 已澄清：降维算子“多 21.6%”的计数由 **14.375% L1 清零＋7.1875% 输入 ND→NZ 写请求**构成，不能当作显存重复读量或提速比例。
- 自写 BF16 降维内核路线已按用户要求停止，代码和测量结果保留；没有部署为默认模型实现。
- 已完成一次真正接入模型的实验 C：约 **20.298 → 20.443 token/s（+0.713%）**，但生成 token 存在差异，**不计为通过的模型优化**。更早的实验 B 实际没有调用自写内核，已排除。
- 随后确认原 CANN 各核 K 遍历顺序不同并修正；25 个输入案例×4 版本，共 **512000 个输出比较与 CANN 逐位一致**，异步提交及图回放检查通过。修正后的整模型实验 D 被停止，未完成验证；旧版单算子 7.83% 降幅不代表修正后的性能。
- 新增 MTP 对照：保持 BF16 与原 CANN，8 次正式请求速度中位数 **20.249 → 31.136 token/s（+53.77%）**；候选接受率 **85.89%**，8 次性能请求＋3 次输出检查 token ID 全部一致，仅覆盖本次样本。
- Insight 后端确认：矩阵乘 M=1→2、单次耗时接近，相同 32-token 采样的主模型 decode 轮数 **31→17**；含草稿层的两组 FFN 累计内核时间 **861.437→485.547 ms**。
- 最近一次补采后已核对：基线 7 号卡 localhost:8038、MTP ND 6 号卡 localhost:8039，两个服务健康且保留，准确检查秒数未单独记录；未修改共享 CANN 和驱动。[补采范围](experiments/nz-context/README.md)。
- MTP 后瓶颈已复核：矩阵乘合计约 **73.1%** 的连续解码窗口，其中两组 FFN **46.3%**；另有 **12.7%** 无设备计算任务区间，主要新增在主模型图之后的草稿/采样流程。[分析与口径](experiments/mtp/bottleneck.md)。
- [MTP 配置与脚本](experiments/mtp/README.md)、[MTP 结果与边界](experiments/mtp/results.md)、[MTP 证据](evidence/2026-10-19/mtp-comparison.json)。
- 社区方案核对：官方 `weight_nz_mode=2` 可对 BF16 权重预排 NZ。相同 FFN 形状的独立原生算子对照中，单次耗时下降 **12.1%～14.4%**，239616 个所测输出逐位一致；随后完成同卡整模型测试，**31.207→29.053 token/s（−6.90%）**，11 项输出一致；NZ 未采用，6 号卡已恢复 MTP ND，恢复后的完整复测按用户要求停止。[整模型证据](experiments/nz-model/README.md)。
- [实验代码与当前限制](kernels/down-reuse/README.md)、[整模型实验摘要](evidence/2026-10-19/down-model-ab.json)。

NZ 跨层读取退化现已取得 **SMMU/TLB 原始计数与复用顺序干预证据**：升维从两层反复读到 64 层不同权重，TLB miss **33.47→3926.23 次/调用**，kernel **236.044→293.089 μs**，L2 数据命中基本不变。同一升维权重立即重读，miss 降到 **176.84**、耗时降到 **241.558 μs**，L2 仍仅约 0.15% 命中；**地址翻译映射的复用变差是这段搬运退化的主要原因**。HBM 实测读吞吐 **1.436→1.167 TB/s**，四个统计单元均衡、时钟均 1600 MHz。没有每次 miss 的等待周期，未声称解释全模型全部差额或获得新加速成绩。09:26:51 已核对 6 号卡恢复原 MTP ND、7 号卡基线保留，两服务健康；未重复整模型测速。[本次硬件计数与对照](experiments/nz-hardware/README.md)、[此前 tiling、地址与 DMA](experiments/nz-runtime/README.md)。

这里的状态是历史记录，不是实时监控。详细配置、测量条件和限制见当天日志。

## 记录规则

时间统一使用北京时间 `Asia/Shanghai (UTC+8)`。采集目录和服务器日志中的 UTC 时间换算后记录；无法确认的时间明确标注，不用文件修改时间补造操作时间。

本仓库于 2026-10-19 首次建立，之前的条目依据已有采样和日志回填。回填日期与事件发生日期分开。

日志区分 **已执行、已验证、分析结论、待实施**。每次改变结论时保留更正原因；测量结果链接到 [证据摘要](evidence/README.md)。后续新增条目沿用 [日志模板](templates/entry.md)。
