# BF16 M8 降维：下一包 GM→L1 前视实验

这是用户在[原生预取审计](../../experiments/wait-timeline/prefetch-audit.md)后授权的窄范围实验。它是**独立编译的固定形状替代内核**，不是给未改动的 CANN ELF 加一个开关；不修改共享 CANN、驱动或已安装的框架包。

完整测量结果见[本次报告](results.md)，可复算数字见[过滤证据](../../evidence/2026-10-21/down-m8-prefetch.json)。原始 ELF、CSV、DB、trace 和地址留在私有实验目录。

## 改了什么，保持什么

适用形状：`X=[8,17408]`、`W=[5120,17408]`，ND/BF16；输出 `[8,5120]`。23 核，每核两组 N112，最后一核的第二组 N80；K17408 切为34包K512，每包8次K64 MMAD。

- A 的 L1 为两个16KiB槽，B 为两个112KiB槽，总计256KiB；没有放大缓冲区。
- `mode=0`：取当前包、等搬完、计算当前包，作为同代码无预取对照。
- `mode=1`：先取第0包；计算第t包前，把第t+1包下发到空闲的另一个槽。线性68包遍历同时覆盖K包交界和两组N块交界。
- 保留 MTE1→MTE2 的槽释放依赖与 MTE2→MTE1 的数据就绪依赖。没有删除必需的 WAIT，没有覆盖尚被使用的 L1 槽。
- 保持原生各核的 K 顺序：core21/22顺序遍历；其余偶数核从 `core/2*3` 正向绕行，奇数核从 `core/2*3-1` 反向绕行。每包内部仍按K64顺序MMAD，FP32累加后转BF16。
- L0读写同步采用保守的逐MMAD完成栅栏。因标量实现、L0调度与原生不完全相同，**原生与新版本的全部差额不能全部归给预取**。mode0/1才用于隔离本实现中前视的增量效果。

## 构建、验证与模型接入

在独立的 aarch64/CANN 9.1/torch2.10/torch_npu2.10 环境中，先激活环境、设置 `ASCEND_HOME_PATH`，再执行 `bash build.sh`。编译产物 `libdown_prefetch_m8.so` 与 `libdown_torch.so` 不入Git。

`validate_bench.py` 在物理6卡运行，先验证24个输入案例、两个模式、图回放与更新输入，再做40轮正反交替的图计时。每图4组不同权重，总工作集680MiB，大于192MiB L2；仍不等同真实64层的地址工作集。`profile_probe.py` 用于 `msprof op --aic-metrics=Default` 的三份独立上板采样。

`model_extension.py` 由工作进程加载，只有 `DOWN_PREFETCH_M8_ENABLE=1` 时替换 `vllm::unquantized_gemm` 的匹配形状；其他形状走原生。范围包括主模型64层及草稿的匹配降维，不局限主模型。`submitted` 是主机提交数，不能当作图回放的设备调用数；实际调用须检查框架采样。

`model/` 保存本次测量用的启动、客户端、图检查与独立采样脚本。启动器仅将私有 profiler 绝对路径改为由 `DFLASH_RUN` 推导。私有运行目录需提供 runtime、client-runtime、模型路径、`env-npu.sh`、上游评测源码和数据，并将扩展/动态库放到可导入位置。本次运行用软链接指向独立构建目录，未替换系统文件。

同题条件：单卡 TP1、BF16、MTP7、主模型和草稿均为图模式、capture8、maxlen4096、并发1、禁用prefix cache、greedy；沿用公开SGLang四题GSM8K评测与计时。**服务器运行的框架仍为vLLM-Ascend**。测速不启用profiler；另采首题128token，用编译版Insight `operator/statistic` 与CSV逐组核对。

新实验应先复用已有可比基线，确需检查环境变化时只补一次同卡对照；不默认多轮重跑基线。本轮在用户指出重复时，已排定的三轮基线均已结束，之后没有补跑或再启动基线服务。

## 失败与证据边界

最初两次验证出现L0A读写冲突，未计入正确性通过或性能结论；补齐L0消费完成依赖后重新通过验证。

附加的CPU仿真因超时只完成core0的176/544条MMAD，缺少END，生成的bin也未成功导入Insight。该轮为**不完整仿真**，不用于完整时序、加速比或上板WAIT归因。上板Default采样有效，但910B2仍不提供本次所需的真实逐条WAIT时序；MTE2活动时间不可与其他流水活动时间直接相加。

`analyze_results.py PRIVATE_ROOT FILTERED_JSON` 验证采集Stop、配置、产物、Insight/CSV全分组一致性、设备替换次数及同题token一致性，并输出过滤统计，不会启动模型。原始输入不上传。
