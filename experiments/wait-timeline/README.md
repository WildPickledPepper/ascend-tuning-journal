# 910B2 上板 WAIT 时序采集核验

**本次没有取得逐条 WAIT 的真实上板起止时间，也没有证明原生双缓冲存在软件串行堵塞。** 已实际执行时序参数请求、两组 Default 上板采集及一次 Source 采集，并将三份 bin 导入编译版 Insight 核对。不能把本次结果写成“已经定位 Cube 在等某次搬运”。

## 1. 算子基本信息

2026-10-21 10:05:55～10:06:37（Asia/Shanghai）完成时序参数验证和两组 Default；随后追加 Source 采集，10:08:08核对6/7卡均无活动进程。时间来自服务器UTC记录。本次未启动27B服务。

物理6号卡，Ascend910B2，原生 `F.linear`，BF16、ND、M=8、23核；升维N=34816/K=5120，降维N=5120/K=17408。输入为1、权重为0.01的合成张量。每组Default选中一次MatMul，重放预热5次；不是64层真实模型运行，不作新的模型速度基准。结果均为有限值，这不是全面精度验证。未加载tiling适配器或修改共享CANN/驱动、原生NPU二进制。

安装工具26.1.0，对应源码提交 `56632dc4a944410c791538451dc2156a151bd105`。先访问官方GitCode仓库，指定版本文档网页读取失败后，检查本地同提交源码与文档，而非凭参数名称判断。

实际命令要点（`PROBE`为私有探针路径，`OUT`为独立输出目录，激活环境并仅暴露物理6）：

```bash
msprof op --aic-metrics=instrTimeLine --kernel-name=MatMulV2 \
  --launch-count=1 --output="$OUT/instruction-up" python3 "$PROBE" up
msprof op --aic-metrics=Default --kernel-name=MatMulV2 \
  --launch-count=1 --warm-up=5 --output="$OUT/default-up" python3 "$PROBE" up
msprof op --aic-metrics=Default --kernel-name=MatMulV2 \
  --launch-count=1 --warm-up=5 --output="$OUT/default-down" python3 "$PROBE" down
msprof op --aic-metrics=Source,Default --kernel-name=MatMulV2 \
  --launch-count=1 --output="$OUT/source-down" python3 "$PROBE" down
```

## 2. 关键数据 TOP5

1. **真实指令时序请求被拒绝。** `instrTimeLine`返回`Unexpected argument in --aic-metrics, maybe in wrong soc platform`。命令退出码却为0，不能仅凭退出码标记采集成功。
2. **Default上板数据有效。** 升维完整kernel277.585541μs，降维148.942978μs；两组bin的Insight `source/details/baseInfo`耗时与CSV一致，`memoryGraph`的core0 Cube占比与CSV一致。
3. **实采core0数据如下。** 各流水活动和阻塞来自不同计数事件，不相加构造总耗时，也不以相减结果冒充Cube WAIT时长。

   | 指标，μs | 升维core0 | 降维core0 |
   |---|---:|---:|
   | 该核总时长 | 261.796112 | 145.793335 |
   | Cube活动 | 28.533333 | 18.879999 |
   | MTE1活动 | 80.102776 | 45.518890 |
   | MTE2活动 | 259.203339 | 129.852219 |
   | Scalar因CUBE IQ满而阻塞 | 168.669449 | 46.078335 |
   | Scalar核内wait阻塞计数 | 0 | 0 |

4. **Source的PMU有效，指令热点不可用。** Source降维kernel151.043015μs，工具报告缺少`debug_line`、热点生成失败。Insight基础信息有效，动态指令接口`source/api/instructions/dynamic`返回`Instructions: []`。Default两份也为空；空数组不是“没有WAIT”。
5. **能力边界已从实现核实。** 同版本`docs/zh/user_guide/msopprof_user_guide.md`的`--aic-metrics`行明确：`TimelineDetail`在采上板数据的同时生成**仿真**指令流水；`instrTimeLine`才是逐指令实际上板时长，限制为950。`csrc/op_profiling/argparser/arg_checker.cpp`的产品映射也把`INSTR_TIMELINE`、`PIPE_TIMELINE`、`PCSAMPLING`限制为950；`interface/ms_op_prof.cpp::ProfilingRun`将`TimelineDetail`送入`OpSimProf`解析。Source热点表还明确A2/A3不支持Stall Sampling。

## 3. 核心瓶颈 TOP5

本次没有新增足够证据定位具体WAIT根因，不凑五项瓶颈。

- 仍能观测搬运活动多、Cube活动少，但不能由这些汇总计数指认某条`WAIT_FLAG(MTE1→CUBE)`，更不能声称其依赖的某次GM→L1是关键路径。
- `aic_scalar_cube_stall_time`的官方定义是**Scalar下发因CUBE IQ满而阻塞**。168.669/46.078μs不是“Cube等待数据”的时长；队列积压也不能单独证明是搬运供数、同步缺陷或计算本身造成。
- `aic_scalar_wait_time=0`仅代表对应Scalar事件，不证明MTE/Cube队列内所有WAIT_FLAG都没有阻塞。基础PMU没有逐PC、逐次依赖时间戳。

此前缓存来源受控实验支持“L2未命中后的HBM供数路径是主要成本”，本次没有推翻它；但这是路径定位，与“已证明双缓冲某处出现可消除的串行等待”是两个证据层次。后者仍未证实。

## 4. 优化建议 TOP5

仅列有依据的后续选择，本次不修改优化配置。

- 不把`TimelineDetail`修到成功当成获得真实WAIT时序；即使成功，其细粒度指令时间仍来自仿真。此前107001确为失败，但不是拿不到真实指令时序的唯一原因。
- 若继续在910B2上取得具体生产/消费阶段的真实时间，需要可插桩的等价内核，在搬运、同步、MMAD边界打设备时间戳并评估插桩扰动；属于新的内核实验，受用户“停止重写内核”限制，本次没有开展。只添加`-g`也不会让910B2获得950的硬件指令追踪能力。
- 950上支持的`instrTimeLine`也不能替代本机证据：换芯片测得的等待时间不直接代表910B2。

## 证据与更正

[过滤证据](../../evidence/2026-10-21/device-wait-timeline-capability.json)包含实测core0计数、bin哈希、接口空列表验证、错误文本和清理回执。[只读复算脚本](summarize.py)逐项核对CSV及Insight响应。原始bin/CSV、日志和接口响应保留在私有数据目录，不入Git。

官方依据：[msOpProf用户指南](https://gitcode.com/Ascend/msopprof/blob/26.1.0/docs/zh/user_guide/msopprof_user_guide.md)、[芯片能力检查源码](https://gitcode.com/Ascend/msopprof/blob/26.1.0/csrc/op_profiling/argparser/arg_checker.cpp)。本次核对依据为同提交本地内容，网页未成功读取，未将网页抓取失败说成在线核验成功。

**更正历史口径：** 旧记录称“上板TimelineDetail未取得有效指令时序”，容易暗示只要修复107001就能获取真实上板WAIT时间。准确说法是：该路径的细粒度时序是仿真，之前原生二进制simulator数据也始终属于仿真；至今没有本机逐条WAIT真实时间的证据。
