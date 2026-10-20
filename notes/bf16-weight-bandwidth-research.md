# BF16 小批量解码权重读取：论文与开源资料核对

2026-10-20，Asia/Shanghai。本页为资料研究及其对已有测量的适用性判断，**没有执行新上板实验、升级环境或取得新整模型速度**。保持单卡 BF16，不重新开展自写 NPU 内核。

## 研究出发点

[现有缓存来源干预](../experiments/mte2-breakdown/cache-controls.md)表明：同一 ND 升维路径的物理工作集从完整大小改为合成别名后，L2 读命中提高，MTE2 显著缩短；TLB miss 没有改善。由此将当前研究重点放在 L2 未命中后的 HBM 供数效率。别名方法仅作诊断，不能套用到不同的真实权重。

当前真实模型形状 M=8；升维 W=34816×5120 BF16（340MiB），降维 W=5120×17408（170MiB）。L2 为 192MiB。当前 ND/M8 与旧 NZ/M2 的跨层 TLB 退化分开讨论。

## 资料与适用性

### 1. 昇腾 MDL 大包搬运：先检查实际事务粒度

[官方 asc-devkit 文档：MDL 模板](https://asc.gitcode.com/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_mdl_template.html)

一次 MTE2 搬入多个计算基本块，降低频繁搬运和同步开销；代价是第一次搬入耗时更长、MTE1 启动较晚。官方 FP16、M128/K1024/N30720 案例中核任务最大耗时 83.68→53.4μs，MTE2 平均 75.64→46.24μs。它说明方法有效的条件，不是本模型的预期收益。

**对本项目的判断**：值得核对当前原生实现的块大小、连续段长度、stride、L1 容量及已采用模板。不能仅凭 MTE2 高就宣布缺少 MDL，也不能假定扩大缓冲区越大越好。当前升维源读取量已接近权重必需量，主要目标是读同样字节更高效，而非声称大量重复读取可消除。上述开关属于算子模板，不是 vLLM 通用命令行选项。

### 2. 昇腾多核 K 轴错峰：减少地址访问冲突

[官方 asc-devkit 文档：K 轴错峰](https://asc.gitcode.com/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_multi_core_k_stagger.html)

多核共享输入时从不同 K 起点开始读取，减少同时读相同 GM 地址的冲突。要求 K 非全载、MDL、多核、输入位于 GM；模板参数为 `enableKdimReorderLoad`。官方 FP16 M768/K6144/N2048 案例：MTE2 平均 90→69.87μs，算子 98.72→85.68μs。

**对本项目的判断**：K5120/17408 较大，只满足部分筛选条件。M8 场景共享输入较小，不能把官方共享大输入案例直接等同于权重 HBM 冲突。需要先核对已有分核/K 遍历顺序，以及是否已使用相似机制。四个 HBM 统计单元均衡并不能证明不存在细粒度地址冲突；现有数据也没有证明存在这种冲突。改变 K 累加顺序可能改变浮点结果，即使仍为 BF16，也需精度验证。

### 3. FlashDecoding++：适配小 M 的分块与双缓冲

[论文（2023），重点第4、5节](https://arxiv.org/html/2311.01282v3)

论文研究扁矩阵乘，强调 N 分块需在并行度和数据复用间取舍，并用双缓冲重叠下一块读取与当前块计算。GPU 的具体线程和 Tensor Core 设计不能照搬成 Ascend 参数。

**对本项目的判断**：M8 与问题类别匹配，但不能采用论文其他模型/后端的整体加速倍数预测本机收益。检查当前原生小 M 路径及预取条件，比照抄大 M 的 tiling 更合理。双缓冲主要隐藏延迟，不改变 HBM 带宽上限，也不会消除必须读取的权重。

[昇腾对应方案：MTE2 Preload](https://asc.gitcode.com/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_mte2_preload.html)：针对 MTE2 间隙较大场景；要求 MDL/SpecialMDL、K 全载及 M/N DoubleBuffer，官方案例 30.88→28.50μs。它与上面的 K 非全载错峰条件不同，不能不检查条件便同时推荐。当前未取得真实板上逐条搬运空隙时间，不认定本机缺少预取。

### 4. 权重预排 NZ：框架已有入口，但本项目必须看整模型

[vLLM-Ascend 官方配置文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/configuration/additional_config.md)：`weight_nz_mode=2` 在支持时将 BF16/FP16 权重也转 NZ。main 文档为研究参考，不能认为本机固定版本支持其所有配置。

本项目 [当前隔离三次重复](../experiments/mte2-breakdown/cache-controls.md)为 ND 276.279μs、NZ239.518μs，任务耗时降低约13.3%，不是纯格式转换计时。旧 [NZ/M2 整模型实验](../experiments/nz-model/README.md)为31.207→29.053token/s；[后续计数](../experiments/nz-hardware/README.md)解释了相应跨层地址翻译退化。当前 M8 整模型 NZ 尚未完成对应验证。

**对本项目的判断**：这是较直接的现成框架入口，但不能把隔离优势当成上线收益。若继续验证，应固定当前 MTP7/图模式/同题口径，保留64层真实权重地址工作集，同时看吞吐、HBM读速率、SMMU和正确性；不回到只反复读取一块权重的测速。

### 5. L2 切分：依赖复用，不能靠切小就省掉首次读取

[官方 asc-devkit 文档：L2 Cache 切分](https://asc.gitcode.com/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_l2_cache_split.html)

方法将工作块组织进缓存，并调整分核以改善复用和同地址冲突。对当前权重基本流式读取的 M8 场景，仍需先从 HBM 读取每个不同权重；仅将340MiB分成几份，不能复制64MiB合成别名诊断中相同物理数据的缓存命中。输入重复读和调度局部性仍可审计，但不据此宣称能把全权重长期放入L2。

### 6. 无损压缩研究：保留 BF16 不代表推理更快

[DFloat11 论文（NeurIPS 2025）](https://arxiv.org/abs/2504.11651)、[作者代码及限制](https://github.com/LeanModels/DFloat11)：压缩 BF16 指数字段，推理时恢复原始权重；约30%存储节省。代码要求 CUDA12，作者 README 明确写 batch=1 约比原 BF16 慢2倍。因此不能当成910B2可直接安装的提速方案。

[Cloudflare Unweight 技术报告（2026-04-17）](https://research.cloudflare.com/papers/unweight-2026.pdf)、[开源内核](https://github.com/cloudflareresearch/unweight-kernels)：直接在 Hopper shared memory 恢复 BF16 小块后供 WGMMA，避免将完整解压权重写回 HBM。报告自述为阶段性工程结果，非最终论文。其表9 Llama-3.1-8B/H100 batch1 为原始168、压缩100token/s，仍减速40.7%；小批量路径仍有固定开销。该结果不能用作我们的模型速度预测。

**对本项目的判断**：这是值得研究的无损减字节方向，但目前检查到的实现不能直接用于 Ascend，且不满足立即提速的证据要求。解压若先完整写回 HBM，再由 MatMul 读取，可能增加流量；需要与消费计算融合才能有理想收益。遵守停止自写内核要求，当前不实施移植。

## 下一步优先级（待执行）

1. 对齐当前原生 MatMul 实际 tiling/搬运描述与上述官方策略，确认哪些机制已存在，重点连续段长度、stride、事务大小和小 M 分核；不先修改内核或乱扫参数。
2. 已有测量支撑的可用入口是权重布局。若开展新性能验证，优先核实当前 M8 的官方 NZ 路径能否在真实64层工作集保留优势，同时监测旧退化机制。
3. 大包、错峰、预取若需改变算子模板而当前库没有现成入口，整理形状和结构化计数向上游提出优化需求；不伪装成改一个 vLLM 参数即可生效。
4. 缓存切分、1GB页和无损压缩均不列为已证实的本机修复。缓存复用机会、翻译等待和解压代价必须分别证明。

asc.gitcode.com 引用来自华为 asc-devkit master 自动构建的预览文档，页面列有构建提交与开发版提示；本机 CANN9.1 的可用性须按安装源码核实。本次只阅读公开资料与已有本地报告，未连接或改变服务器。
