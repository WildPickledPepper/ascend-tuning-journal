# 1GB 显存大页核验

2026-10-20 16:40:07–16:40:35（Asia/Shanghai），物理6号910B2；三个独立子进程依次分配4/170/340MiB BF16张量，不加载模型、不运行矩阵乘。16:42:03确认原服务仍健康且未重启。

**结果：软件支持配置，但本次未取得物理1GB大页；两个1g配置均明确回退2MB页。**

| 分配器配置 | 分配514MiB张量后保留量 | expandable | 1GB物理大页 |
|---|---:|---|---|
| expandable_segments:True | 520MiB | 是 | 未启用 |
| page_size:1g | 1024MiB | 否 | 失败，回退2MB |
| page_size:1g,expandable_segments:True | 1024MiB | 是 | 失败，回退2MB |

驱动25.3.rc1.2、CANN9.1、torch_npu2.10.post4具备相关版本/代码支持。`aclrtGetMemInfo(ACL_HBM_MEM_HUGE1G)`在三个探针各阶段均返回成功但free=total=0；两份stderr均警告1GB大页预分配不足或未启用，并回退2MB。不能把内存池1GiB块当成物理1GiB页生效。当前两模型服务只有expandable_segments:True；最快128题实验也使用相同环境脚本，没有1g配置。

[探针](probe.py)只在显式物理6号运行，每次显存池最多保留1GiB；结束empty_cache并退出子进程。原6/7服务HTTP200、原进程仍在；没有修改服务配置、全局大页、驱动或重启设备。[过滤证据](../../evidence/2026-10-20/hugepage-allocation-audit.json)。ACL可用内存计数是设备级，不能当成每个模型张量的逐页所有权。

后续若验证大页收益，须先确认6号设备实际提供1GB HBM页资源；确切管理员配置步骤尚未核实，不能套用Host Linux HugeTLB配置。仅添加进程环境变量目前会回退，不应据此做“1GB对2MB”的性能结论。

## 地址翻译开销已有的证据

[此前NZ/M2硬件采样](../nz-hardware/README.md)已经通过msprof取得SMMU请求、命中和未命中计数，源于socpmu.data及官方soc_pmu.csv；Insight统一DB未暴露TLB字段，并非ms工具不能采。升维立即重读对照的miss为3926.125→176.836、kernel293.326→241.558μs，差51.768μs；MAC约31.799μs不变、L2数据命中仍极低。该受控对照定位翻译局部性引起的主要退化，但没有每次miss的stall-cycle，不能称51.768μs是直接读取的纯TLB耗时字段。

诊断不依赖1GB对照；1GB对照用于验证一种优化是否有效。当前最好ND/M8仍需另外采SMMU/HBM，不能把NZ/M2结论直接套用。本轮只核验分配器，未做新的整模型或TLB采样。
