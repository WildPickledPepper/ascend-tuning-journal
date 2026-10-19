# 固定形状 BF16 降维内核实验

适用条件：Ascend 910B2、CANN 9.1.0，连续 BF16 输入 `[1,17408]`、权重 `[5120,17408]`，输出 `[1,5120]`。这是独立实验实现，未安装到 CANN、未接入 vLLM。

`down_reuse.asc` 提供四个版本：

| 模式 | 内核 | 修改 |
|---|---|---|
| 0 | down_baseline | 独立实现，按输出子块分别遍历 K；68 次清零、68 次输入搬运/核 |
| 1 | down_hoist | 两个输入缓冲各清零一次；输入仍搬 68 次 |
| 2 | down_reuse | 每个输入块供两个输出子块使用；输入搬运降至 34 次 |
| 3 | down_pipeline | 在模式 2 上双缓冲预取下个 K 块，与当前计算重叠 |

23 核独立写各自的输出区间，没有跨核通信。前 22 核各负责 224 个输出，最后一核负责 192 个输出；两个子块分别为 112+112 或 112+80。每个输出以递增 K 顺序执行 272 次 K=64 的 BF16→FP32 MMAD，再写出 BF16。

模式 3 的 L1 占用为 480 KiB：两个 16 KiB 输入缓冲＋四个 112 KiB 权重缓冲；L0A 4 KiB、L0B 28 KiB、L0C 14 KiB。片内预取保留输入复用，不减少必要权重读取量。

## 构建与验证

先在自己的进程中加载 CANN 9.1.0、PyTorch/torch_npu 环境，然后执行：

```bash
bash build.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export TASK_QUEUE_ENABLE=0
python3 validate.py --bench
python3 paired_bench.py
```

测试只接受物理 7 号卡。`TASK_QUEUE_ENABLE=0` 是这个直接 ACL/ctypes 测试启动器的要求：它必须与 Torch 的输入生成、输出填充和事件保持正确提交顺序。该设置只作用于当前测试进程。将来接入 vLLM 必须封装正式的 Torch NPU 自定义算子，不能直接套用这段 ctypes 启动器。

`validate.py` 对 3 个随机输入、全零输入、固定值输入检查四个自写版本逐位相同，检查输出有限和尾部保护区未改动，并保留与 CANN 及 FP64 点积的差异。`integration_allowed` 当前为 false：5 个案例、每模式共 25600 个输出中，有 4 个输出与 CANN 相差一个 BF16 表示档位，原因尚未完全确认。没有通过放宽容差将其标为一致。

`paired_bench.py` 交替测量 CANN 和自写版本的图回放；4 份不同权重总量超过 L2，40 轮交替正反顺序。避免把 Python 启动开销混成内核收益。

## 工具取证

上板采集使用 `msprof op` application 路径，指标 `Default,MemoryDetail`，通过 `validate.py --profile 3` 运行模式 3；`--profile 4` 运行 CANN 参考算子。显式指定对应 `--kernel-name`。

`backend-query.cjs <本地证据目录>` 直接访问已运行的编译版 Insight 后端 `ws://127.0.0.1:9099`，导入各 `visualize_data.bin`，读取 23 核 `source/details/memoryGraph`；`backend-instructions.cjs` 导入仿真数据并读取指令执行次数。目录布局与本次证据产物一致，需要自行准备运行产物。

`launch_acl.py <仿真config.json>` 是不依赖 Torch 的仿真应用启动器。config 的前三个输入描述依次指向输入、权重、输出缓冲文件。仅在配置好 Ascend910B2 仿真库的环境使用。本次 config 直接启动失败；application 路径内核运行完成并生成有效 trace，但子进程退出阶段出现 signal 11，已在日志中记录。

实测和限制见[当天日志](../../journal/2026-10-19.md)及[证据摘要](../../evidence/2026-10-19/down-reuse-experiment.json)。二进制、原始权重和全量采样不提交。
