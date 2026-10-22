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
| [2026-10-21](journal/2026-10-21.md) | MTP/预取/显存分析与复盘；BF16按L1块重排，算子与完整请求验证 |
| [2026-10-22](journal/2026-10-22.md) | 优化版本采样、FFN历史对照；动态草稿词表、分叉诊断及FP32状态修复复测 |

**完整过程：[Qwen3.8-27B在910B2上的定位与优化复盘](notes/qwen27b-tuning-retrospective.md)。** 按16步串联现象、假设、受控实验、Insight证据、实际改动与采用判断，保留失败和结论更正。最后附当前完整采样耗时分摊及复算代码。

搬运技术说明：[HBM/L2/L1/L0 数据流、ND/NZ 地址、分块与同步](notes/matmul-data-movement.md)。

## 最近一次记录的状态

- **两份计数提前＋固定MTP7元数据准备，采样空隙已恢复到低水平**：[报告](experiments/count-ready-overlap/results.md)。flow证明首次第三同步增加来自CPU提前到达，拷贝仍约1.9µs、就绪时间约60.19ms未变。6卡新实现保留两份计数语义，复用固定GDN元数据；当前同服务28轮采样空隙13.821%→0.940%，准备从晚10.573ms变早6.423ms，27个衔接全部提前就绪。32/32输出一致、实验版63.990 token/s；同服务四题仅+0.805%、只有1/4题更快且TTFT变差，不证明显著稳定加速。开/关采样同128请求支持采样放大原Host瓶颈，保留一轮执行差异；不把跨服务32题+5.255%当严格A/B。实际3个GDN builder各30字段、CPU接受数1～8、882个Insight组及两端复算通过。18:07:53确认自己的服务退出、8047关闭、6卡无进程；不设默认；本轮代码、报告和过滤证据按用户要求归档。[证据](evidence/2026-10-22/count-ready-overlap.json)。

- **6卡接受数量提前对照已完成，只有部分重叠改善**：[结果](experiments/accepted-count-overlap/results.md)。同服务BF16/FP32 SSM/动态4096/MTP7，32/32自然回答逐token一致；两道预检150轮候选集合一致，正式32题接受轮数1868→1873，未证明全部草稿轨迹等价。无profiler吞吐63.586→64.401 token/s（+1.28%），E2E总和137.830→136.086s。独立采样空隙13.074%→10.896%，接受数量等待22.235→12.849ms，但第三个有效token数同步0.077→6.138ms，准备仍比草稿结束晚7.915ms。442个Insight分组及SQL/CSV核验、本地/服务器复算通过；17:05:06确认自己的服务退出、8047关闭、6卡无进程。不设默认，未Git提交或推送。[证据](evidence/2026-10-22/accepted-count-overlap.json)。

- **按用户要求离线追查prepare input，未新增采样或改实现**：[分析](experiments/dynamic-decode-profile/prepare-input-analysis.md)。旧0.922%/当前14.346%确为间隙增长；当前约96%位于草稿图结束至下一主图启动之间。稳态准备CPU操作顺序相同，第二个同步均值11.631→19.640ms，扣同步后准备18.897→20.613ms；草稿图19.83→6.58ms，输入准备从提前3.935ms完成变为晚11.053ms，支持动态维护推迟关键同步、短草稿不再覆盖准备的流水机制。另一题58个稳态衔接一致。仅描述采样中的重叠变化，不将14%当可回收收益；没有启动NPU、改代码或Git提交/推送。

- **当前动态4096＋FP32状态版本已重新采整网profiling，只定位未改**：[报告](experiments/dynamic-decode-profile/results.md)。首题Pipe/L2/Memory各27轮、另一题前256token共59轮，四组输出前缀均对已有32题一致。主FFN约34.47%、主模型其他线性16.07%，草稿小head仅0.37%；动态TopK/IndexFill等候选维护10.87%，权重gather仅0.068%。无计算区间约13%～15%，主要在模型图外，尚未唯一归因于Host/DMA/采集扰动。两份Insight共440组与CSV一致，四份SQL及服务器/本地复算通过。15:25:36.905本轮服务停止，15:34:51.967确认自身进程退出、8047关闭；后续7卡已有另一进程，未处理，不声称整卡空闲。未改内核、默认配置或提交Git。[证据](evidence/2026-10-22/dynamic-decode-profile.json)。

- **32题自然回答扩样本对照已完成**：[全部指标与逐题记录](experiments/gdn-fp32-gsm32/results.md)。同一FP32状态/BF16权重版本、单卡7/单并发/MTP7，完整与动态4096各32题，输出8764token/组，32/32输出逐token一致、全部正常EOS。吞吐60.583→62.736（+3.55%），E2E总和144.662→139.697s，28/32题更快；TTFT均值338.146→351.424ms、每请求TPOT均值15.436→14.805ms/token，P99 TPOT未改善。四题13.19%不再作为稳定收益；完整分位数及MTP返回块/交付间隔定义已归档。15:03:03确认7卡释放、8047关闭，未Git提交/推送或设默认。

- **GDN状态缓存改FP32后，动态词表所测输出一致，吞吐本轮提升**：[结果与限制](experiments/gdn-fp32-state/results.md)。48个GDN层实际SSM为FP32，权重/conv仍BF16；此前分叉题同图full→masked→full均575token、隐藏向量逐位一致。关闭诊断，真实小head与完整词表四题各两遍，8/8对应输出token一致；各1688token，E2E30.011→26.513s，吞吐56.246→63.667（+13.19%），接受草稿/轮4.006→3.920（−2.15%）。完整组两遍波动较大，第二遍收益+6.97%，不宣称稳定13%或全局无损。12:00:29确认7卡释放、8047关闭，未设默认；后续按用户要求纳入本次Git归档。[此前根因](experiments/dynamic-vocab-consistency/README.md)。

- **动态草稿词表已完成首轮7卡对照，未证明整体加速**：[结果与边界](experiments/dynamic-draft-vocab/results.md)。保留既有BF16 FFN与主/草稿图，动态4096行、约1k独立train保底＋prompt＋target Top8。固定512吞吐46.075→42.641（−7.45%）；自然四题61.421→59.456（−3.20%），一道输出575→447，不把更短E2E当加速。三个输出完全一致的自然题子集+5.86%，长输入单样本+3.46%；接受草稿数/轮在固定组2.431→2.000。存在输出差异及少量投影舍入差异，不宣称逐token无损，不设默认。10:40:03确认7卡释放、8047关闭，未Git提交/推送。[证据](evidence/2026-10-22/dynamic-draft-vocab.json)。

- **已有profiling的FFN前后口径已统一**：[对照表](experiments/optimized-decode-profile/history-comparison.md)。没有重新上机。相同121输入/128输出、26轮：主模型FFN原生771.514ms/39.277%，已down预取758.812ms/38.978%，当前684.598ms/36.503%。最近阶段绝对时间−9.78%、占比−2.475个百分点，gate/up单次297.990→254.834μs（−14.48%）；若从原生算，FFN−11.27%、占比−2.775个百分点。旧6卡/当前7卡，跨卡跨日，不将全部差值唯一归因于重排。581个Insight分组与SQL核验；本次NPU请求0。[证据](evidence/2026-10-22/ffn-profile-history-comparison.json)。

- **优化后 decode 已重新定位，只采集、未改推理实现**：[分析报告](experiments/optimized-decode-profile/results.md)。7号910B2、BF16/TP1/并发1/MTP7，保留packed gate/up、down预取和prefill直读。三个同输入128输出采样各26轮，主模型FFN约36.50%、词表投影21.28%（草稿18.58%）、主模型其他线性投影17.50%，无计算空隙0.92%。MTE2活动高、MAC活动较低、gate/up L2读命中0.53%，支持权重供给限制，不等于证明HBM峰值已满；没有发现大量host下发空等。195个Insight分组与SQL/CSV核验，服务器/本地复算一致；09:22:50服务退出，09:27:17确认7卡释放、8047关闭。保留通信端口告警记录；未改内核、未设默认、未推送Git。[过滤证据](evidence/2026-10-22/optimized-decode-profile.json)。

- **Prefill gate/up直接读packed权重已验证**：[实现与完整请求结果](experiments/gateup-packed-prefill/results.md)。相对上一版“临时还原＋原生linear”，正常四题TTFT362.37→336.38ms（−7.17%），输出61.688→62.212 token/s（+0.85%）；固定512输出仅+0.11%；1513输入单样本TTFT1040.85→947.36ms（−8.98%）。22条输出token一致、64层65-token真实激活逐位一致；计时窗口832次直接读、还原0次。大M独立测试仍有少量BF16舍入差异；MTP总量相同但长输入位置计数各差1，未设默认、不宣称稳定通用收益。18:18:28确认7卡释放。[过滤证据](evidence/2026-10-21/packed-prefill-e2e.json)、[离线复核](experiments/gateup-packed-prefill/verify_evidence.py)。

- **BF16按L1块重排已完成整网验证**：[算子原理与结果](kernels/gateup-l1-layout/README.md)、[完整请求结果](experiments/gateup-l1-layout-e2e/results.md)。7号910B2、64层真实FFN工作集，M8 gate/up原生279.176→239.276μs（耗时−14.29%），同模板跨步NZ295.871→239.276μs（−19.13%）；L2约0.5%、HBM读取字节接近。整网单卡BF16/MTP7/主草稿图，保留相同down预取，正常四题输出吞吐60.056→61.758 token/s（+2.834%），固定512输出44.785→46.244（+3.259%），E2E分别降低2.756%/3.156%；TTFT约多8–9ms。对应输出token及MTP计数一致，prefill临时还原成本已计入；未设默认、未测长输入/多并发。17:17归档确认7卡释放。压缩研究按用户要求等待950卡，当前成果归本调优仓库。[过滤证据](evidence/2026-10-21/gateup-l1-layout-e2e.json)、[只读复算](experiments/gateup-l1-layout-e2e/verify_evidence.py)。

- **原生MTP K6/K8同四题已测，仍保留K7**：[方法与结果](experiments/mtp-k6-k8/README.md)。K6/K7/K8＝58.460/59.386/56.986token/s，均4/4正确；token/轮4.703/4.982/4.994，decode/轮71.151/75.089/77.903ms。K6接受率高但轮数增，K8推进量几乎不增却更耗时；K8第三题多18token，不能声称全部输出相同。单卡BF16、主/草稿图均核验，只新测6/8、复用7，不跑128、不改内核。16:54:51确认6/7释放。[证据](evidence/2026-10-21/mtp-k6-k8.json)。

- **跨层预取时序核验**：[1638个边界与Insight时间线](experiments/cross-layer-prefetch/README.md)。原生MTP7的26轮×63边界，降维结束→下层首投影中位6.14μs，其中无计算仅0.06μs，其余Add/归一化；仅此边界可重叠总量最多占采样0.514%，非实测提速。更早在降维内部预取仍未证实：MTE2中位88.5%不能反推HBM空闲，B2缺真实指令时序。未启动模型、未改内核。[证据](evidence/2026-10-21/cross-layer-prefetch.json)。

- **单卡MTP小规模训练完成，未采用**：[训练与四题结果](experiments/mtp-small-train/README.md)。独立64题由27B生成答案与特征，8题验证；主干BF16冻结，仅头两轮32次更新。验证单步准确率92.606%→91.805%；同四题接受率56.892%→47.313%、59.386→52.861token/s，均4/4对。按用户要求未再跑128题、未重测基线；生成长度844→867，响应token-ID元数据有差异。16:17:49确认6/7释放，保留原生MTP7方案。[过滤证据](evidence/2026-10-21/mtp-small-train.json)。

- **单卡原生MTP头训练组件验证通过**：[入口/兼容修复/实板](experiments/mtp-training-audit/README.md)。真实4.247亿BF16头＋冻结词表，在6号B2完成B1/S32合成前向、反向和AdamW更新，allocated峰值14.327GiB、reserved14.430GiB；只有头更新，冻结词表逐位不变、原checkpoint未写。独立SpecForge副本修复注意力布局和RoPE字段兼容；完整CLI语料训练与接受率收益尚未验证。单卡建议冻结target采特征后退出，再离线训练；未读取128题。15:41:54确认6/7释放。[过滤证据](evidence/2026-10-21/mtp-single-card-training-audit.json)。

- **继续MTP，已完成接受率与社区路线核对**：[分析与资料](notes/mtp-acceptance-research.md)。完整128题七候选接受率51.298%，第七位置条件通过率仍81.510%；低的全轮比例主要含前缀截断效应，不能据此认定后几步失效。当前原生MTP一层/4.247亿参数/810.05MiB，递归七步。优先研究冻结主干、任务匹配MTP微调；社区已有vLLM/SpecForge路线，27B两卡B2适配和收益尚未验证。未试K、未训练、未启动模型；15:25:40确认6/7释放。[过滤证据](evidence/2026-10-21/mtp-acceptance-research.json)。

- **vLLM DFlash2 K7草稿图复测完成**：[配置、失败修复、接受率与Insight对照](experiments/dflash2-vllm-graph/README.md)。单卡BF16同四题，复用原生MTP7基线，59.386→51.368token/s（−13.502%）；接受率56.892%→47.546%，验证171→195轮，均844token/4题全对。128token采样主图单轮约52ms不变，草稿图计算更便宜，但验证轮数增加；额外Host标签使新旧profile间隙不能直接归正常运行。未实现接受率提升、不采用替换。15:09:23确认6/7均释放、不恢复旧服务。[过滤证据](evidence/2026-10-21/dflash2-vllm-graph.json)。

- **显存快照已分析**：[msMemScope＋编译版Insight结果](experiments/memory-snapshot/README.md)。12个阶段、106842事件：权重53.256GiB、缓存5.623GiB、其他活动张量0.145GiB，框架实际59.024GiB、预留60.002GiB；设备可用约0.388GiB。短请求框架峰值59.462GiB，峰值点432MiB分配定位在GDN注意力。三次请求后实际/预留不再增长；空闲普通池87.58%集中在一个436MiB连续块，未见碎片主导证据。未修改配置、未跑速度基线；11:21:27确认6/7释放。[过滤证据](evidence/2026-10-21/memory-snapshot.json)。

- **降维单包前视已完成算子与完整27B验证**：[实现、对照和结果](kernels/down-prefetch-m8/results.md)。用户新授权的BF16固定M8独立内核保持原K/MMAD顺序，所测输出逐位一致；独立154.680→145.222μs，模型内154.463→146.231μs。三轮同四题吞吐中位数59.386→59.774 token/s（+0.653%），全部token相同。Insight确认1846次设备替换、386个分组与CSV一致；目标只占窗口12.178%，局部节约折合总窗口0.649%。**未证明HBM带宽提高，未设为默认**。用户指出重复基线时三轮已结束，之后不再追加。11:10:13确认6/7均无活动进程、服务全部关闭。[过滤证据](evidence/2026-10-21/down-m8-prefetch.json)。

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
