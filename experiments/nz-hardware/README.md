# 1. 算子基本信息

2026-10-20 继续定位“同一 NZ 矩阵乘，跨 64 层读取不同权重后变慢”。本次取得了 **SMMU/TLB 原始计数**，并做复用顺序干预：结论从工作集现象推进到 **地址翻译映射的复用变差、TLB 未命中增多，是这段权重搬运退化的主要原因**。

| 项目 | 内容 |
|---|---|
| 模式 | device，上板；application/system profiling；没有仿真测速 |
| 芯片、工具 | Ascend 910B2；私有 CANN 9.1.0 / msprof / PyTorch Profiler；编译版 Insight 后端交叉核对 |
| 形状 | BF16，M=2；升维 N=34816,K=5120；降维 N=5120,K=17408 |
| 计算实现 | 原 CANN；没有写、替换或编译 NPU 内核 |
| 保持条件 | 单卡物理 6；相同模型真实权重、X/W/Y 地址、stride、NZ 格式和计算；7 号卡基线保留 |
| 主要指标 | `--l2=on` 对应的 SMMU 事件；PipeUtilization；AI Core L2Cache；`--sys-hardware-mem=on` 的 HBM 数据 |
| 产物 | `socpmu.data`、官方 `soc_pmu.csv`、kernel_details、HBM DB、阶段时间标记；[过滤摘要](../../evidence/2026-10-20/nz-hardware-comparison.json) |

四种读取顺序：

- **small**：第 0、1 层的四次升维/降维反复执行 32 遍，共 128 次矩阵乘。
- **full**：按 64 层顺序执行升维/降维，共 128 次；与 small 总逻辑权重字节数相同，只有不同权重的工作集扩大。
- **sweep2**：完整扫 64 层两遍，共 256 次。
- **adjacent**：每份权重紧接着执行两次，再读下一份，共 256 次。与 sweep2 使用完全相同的 64 层权重及总逻辑读取量，只改变复用间隔。

每项设备计数采 6 次图回放，稳定统计去掉前两次。small/full 每形状保留 256 个样本；sweep2 每形状 512 个；adjacent 每形状、每位置各 256 个。无 profiler 的图事件计时另做，不能混为 kernel PMU 时间。

# 2. 关键数据 TOP5

## ① 原始 SMMU 事件确认 TLB 未命中大幅增加

稳定区间，每次 kernel 的计数与耗时：

| 指标 | 两层反复读：升维 | 64 层不同权重：升维 | 两层反复读：降维 | 64 层不同权重：降维 |
|---|---:|---:|---:|---:|
| TLB 请求数 | 697600.77 | 697600.75 | 354733.69 | 354733.68 |
| TLB miss 事件数 | 33.47 | 3926.23 | 734.80 | 1967.39 |
| 加权 TLB miss 比例 | 0.004798% | 0.562819% | 0.207140% | 0.554612% |
| kernel 均值，μs | 236.044 | 293.089 | 137.592 | 165.820 |
| MTE2 活动时间，μs | 227.725 | 270.366 | 113.705 | 140.971 |
| MAC 活动时间，μs | 31.799 | 31.799 | 22.838 | 22.772 |

升维 miss 数约为小工作集的 **117 倍**，降维约 **2.68 倍**。请求数几乎一样；不能将 miss 百分比直接当成耗时百分比，也不能把 miss 次数当成串行页表查询的时间总和。

## ② 排除 L2 数据缓存变化解释原来的 small/full 差异

另采 `aic_metrics=L2Cache`，按 `(r0_hit+r1_hit)/(r0_hit+r1_hit+r0_miss_allocate+r1_miss_allocate)` 汇总：

| 读数据命中比例 | small | full |
|---|---:|---:|
| 升维 L2 | 约 0.1293% | 约 0.1293% |
| 降维 L2 | 约 1.7329% | 约 1.7329% |

数据缓存命中没有产生能解释这次退化的变化，变化明显的是 **地址映射的 TLB 命中情况**。这里的数据来自 AI Core L2Cache PMU，不使用本次全局 HA 报表中返回 `N/A` 的 Hit Rate。

## ③ 保留全部 64 层权重，仅缩短复用间隔，翻译 miss 与耗时一起回落

adjacent 顺序中，同一升维权重第一次与紧接着的第二次：

| 指标 | 第一次读 | 立即再读同一份权重 |
|---|---:|---:|
| TLB miss 事件/次 | 3926.13 | 176.84 |
| TLB miss 比例 | 0.562805% | 0.025350% |
| kernel 均值，μs | 293.326 | 241.558 |
| MTE2，μs | 270.173 | 227.825 |
| MAC，μs | 31.799 | 31.799 |
| L2 数据读命中比例 | 约 0.1293% | 约 0.1547% |

L2 数据读命中只变化约 **0.025 个百分点**，但 kernel 时间下降约 **17.65%**，TLB miss 数下降约 **95.50%**。这条复用干预是把主要原因定位到地址翻译的关键证据，区别于只观察一次相关性。

**降维的立即重读另有明显的数据缓存收益，不能只归给 TLB**：其第二次 L2 命中接近 100%，耗时约 100.305 μs。安装的 910B2 平台配置给出 L2=201326592 B（192 MiB）；一份降维权重为 170 MiB，升维为 340 MiB。这里用升维作为主要的“翻译变热、数据仍几乎全未命中”的对照。

无 profiler、同为 256 次矩阵乘的图事件时间，sweep2 中位数约 **229.269 μs/次**，adjacent 约 **199.997 μs/次**。这只是隔离探针；真实自回归网络不能据此直接重排层或重复执行层，**没有新增整模型 token/s 成绩**。

## ④ HBM 实测吞吐降低，四个统计单元仍均衡，时钟不变

只采物理 6 号卡，100 Hz；每个阶段连续回放约 2 秒。以阶段绝对时间对齐 HBM 采样，前后各裁掉 100 ms：

| 阶段 | HBM 总读带宽，十进制 TB/s | HBM 时钟 |
|---|---:|---:|
| small 第一次 | 1.436916 | 1600 MHz |
| full 第一次 | 1.166834 | 1600 MHz |
| small 第二次 | 1.435562 | 1600 MHz |
| full 第二次 | 1.166691 | 1600 MHz |

四个 HBM 访问统计单元的平均读带宽彼此非常接近，最大相对差异小于 0.02%；没有观察到单个统计单元失衡，也没有 HBM 时钟下降。此卡在同样时钟的小工作集下实际能提供约 1.436 TB/s，跨层时请求送达效率下降后，实际吞吐约 1.167 TB/s。

这些是带宽计数，**不是每个 DRAM bank 的队列深度、行冲突或内存控制器等待周期**。未把未采集的 bank 排队机制写成已排除的事实。

原始 CANN `HBMbwData.bandwidth` 单位由安装源码 `msmodel/hardware/hbm_model.py` 的 `SCALE` 核实为 MiB/s；转换为十进制 TB/s。不要把它和统一 DB 文档中 `HBM.bandwidth` 的 Byte/s 混淆。

## ⑤ 原始记录、官方导出与 Insight 对得上

- CANN platform_version=5，`socpmu.data` 每条为 `<8B4H8Q`、80 B；数据类型 2 是 SMMU，类型 1 是 HA。
- 采集配置的 SMMU 事件依次为 `0x2,0x8a,0x8b,0x8c,0x8d`。按安装的 `SocPmuConfig`：请求=`0x8a`，miss=`0x2`，hit=`0x8c+0x8d`。
- 原始数据满足 `hit+miss=request`。逐 stream/task、逐条事件算出的 miss 比例与官方 `soc_pmu.csv` 对齐至其 6 位小数精度；不是只读取一个无法核实的“命中率”字段。
- 编译版 Insight `operator/statistic` 的 6 次回放次数、均值与 kernel_details 交叉验证。表中稳定区间来自排除前两次后的设备记录，和后端全 6 次均值口径分开。
- 当前 Insight 统一 DB 未暴露 TLB 字段，**TLB 数据明确来自 msprof 的原始记录和官方 CSV**，未伪称由 Insight operator 接口返回。

# 3. 核心瓶颈 TOP5

本次只定位到一项主要原因，不凑满五项：

**大工作集使相同权重地址映射的复用间隔拉长，SMMU/TLB miss 增加，限制了这两组算子的权重搬运效率。**

执行矩阵乘需要把设备虚拟地址翻译到实际显存地址。两层循环时，很多映射可以在 TLB 中复用；64 层不断换权重，重新遇到同一映射前隔了更大的访问集合，实际 miss 增多。数据请求等待翻译会拖长 MTE2，并使 HBM 不能持续得到同样密度的有效请求；乘加工作本身没有增加。

证据链是：相同地址和参数 → 翻译请求数不变 → miss 数增加 → L2 数据命中不变 → MTE2 增时；再缩短同一权重的复用间隔，升维 miss 和增时同时明显回落，L2 数据命中仍很低。

边界：没有取得每次 miss 的 stall-cycle 计数，不能把每一微秒按事件精确分摊；也没有测出 TLB 的精确容量或哪一级翻译缓存承担了全部 miss。此前完整模型升维相对 64 层 FFN 隔离图还约有 12.3 μs 差额，本次不声称已把整个模型所有差额解释完。

# 4. 优化建议 TOP5

本次目标是定位，未继续部署新优化：

1. 当前仍保留原 MTP ND；NZ 在单算子小工作集的优势不代表真实跨层部署更快。
2. 后续候选应围绕进程内地址映射、受支持的页表/分配粒度和实际可行的权重复用；需要再次用 SMMU miss 和端到端数据验证。**本次没有更改全局大页、驱动、页表配置，也没有承诺某个开关能直接解决。**
3. 不再把这段退化归因于未经测量的 tiling 改变、数据缓存变冷或泛指“HBM 已到峰值”。保留各个指标的原始口径。

## 采集排障、源码与恢复

- 短 application 探针先验证 `msprof --l2=on --sys-hardware-mem=on` 能在本机产生非空 SMMU/HBM 数据。
- 对隔离图，先启动 `l2_cache=True`，再捕获新图，最后回放。官方文档说明，在 aclgraph 已执行后才开启该采集项可能不生效。[官方参数说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html)。
- PyTorch 解析后未保留中间 `soc_pmu.db`，最初按 DB 名查找没有结果；随后找到非空官方 CSV 和 `socpmu.data`，完成逐条校验。没有把“没找到中间 DB”误报成“不支持 TLB”。
- HBM 首次采集 18 秒未覆盖第 4 阶段，时间覆盖检查识别了这一问题；完整四阶段比较只用后续 36 秒、全部覆盖的采集，原失败范围保留。
- 全局 HA 的 L2 Hit Rate 返回 N/A，改用独立的 AI Core L2Cache 事件解释数据缓存。QoS 只提供配置过的带宽类别，未把未配置字段或 bandwidth 字段当成队列等待计数。
- 首次只读 `npu-smi sensors` 缺少 chip id，补上 `-c 0` 后获取遥测；没有修改设备配置。
- 临时服务只有主机观察 extension 与 RPC 探针，原 `F.linear` 计算照常执行；没有 LD_PRELOAD 或优化内核。采集后恢复物理 6 原 MTP ND；7 号卡基线保留。09:26:51 核对两服务 health=200，6 号卡仍为 BF16、TP=1、主模型图模式、单候选 MTP；两服务均无 NZ 参数、观察 extension、LD_PRELOAD 或 dev mode。详见过滤摘要中的 `restored_services`。
- 仓库只存脚本、过滤后的计数和来源哈希；原始 DB/trace/PMU 二进制、权重、服务器路径留在私有采集环境。

## 脚本和复核入口

`hardware_observe.py` 继承上一轮主机观察器 [observe_weights.py](../nz-runtime/observe_weights.py)，保留实际模型张量；`start_capture.sh` 通过 `QWEN_MODEL_ROOT` 和 `NZ_HARDWARE_RUN` 指定私有目录，加载该观察器与本轮模块。它只用于自己的临时采集进程。

`hardware_probe.py` 提供 small/full/sweep2/adjacent 顺序和 timing/tlb/hbm 三种探针；`capture_hbm.py` 调用该 RPC，并将系统采集限定到物理 6。TLB 探针也用于单独的 L2Cache 采集，通过 `metric_name='L2Cache'` 选择指标。

离线分析和校验不需要访问 NPU：

```bash
python analyze_hardware.py RAW_CAPTURE_ROOT OUTPUT_ROOT BACKEND_OUTPUT_ROOT
```

`RAW_CAPTURE_ROOT` 保留原始 profile 目录、探针 receipt 与完整 HBM receipt；第三个参数是可选的已保存 Insight 响应目录。后端响应由既有 [backend-query.cjs](../nz-model/backend-query.cjs) 导入 8 份 `ASCEND_PROFILER_OUTPUT` 后取得，文件名为 `{tlb,l2}-{small,full,sweep2,adjacent}-backend-matmuls.json`。分析器验证原始/官方 SMMU、地址、HBM 时间覆盖、次数和均值，不重新运行模型。

[脚本与摘要 SHA256](../../evidence/2026-10-20/nz-hardware-source-manifest.json)。
