# DFlash2 与内置 MTP：单卡 BF16 验证

目标是在 Ascend 910B2 64 GiB 上运行 Qwen3.8-27B + DFlash2，并与同版本框架的内置 MTP 对照。结论与服务恢复状态以 [当天日志](../../journal/2026-10-20.md) 和最终证据为准。

## 软件与草稿

- 原服务使用 vLLM 0.26.0，没有当前 `DFlash2DraftModel` 注册；不能直接把草稿路径填进旧启动命令就视为可用。
- 官方 `quay.io/ascend/vllm-ascend:v0.27.1rc1` 已包含 DFlash2 实现。复用镜像中的预编译框架，创建独立 `venv --system-site-packages`，使用原先正常工作的私有 CANN 9.1.0、PyTorch 2.10.0、torch_npu 2.10.0.post4。原环境不改动。
- 草稿 `incoai/Qwen3.8-27B-DFlash2`：5 层，1,924,404,480 个参数，全部 BF16；权重文件 3,848,817,896 B。HF commit、镜像 ID、框架 commit 和 SHA256 见 [来源摘要](../../evidence/2026-10-20/dflash2-source-manifest.json)。
- `method` 仍写 `dflash`，框架按草稿架构选择 DFlash2；本次 7 个候选、草稿 eager、主模型 `FULL_DECODE_ONLY`，单请求图大小为 8。内置 MTP 为 1 个候选，草稿 eager，图大小为 2。

## 复现入口

脚本只包含可移植配置，不包含服务器登录信息、权重或全量日志。操作者需先检查自己的卡和服务占用，并安排原服务恢复。

```bash
export QWEN_MODEL_ROOT=/absolute/path/to/private/model-workspace
export DFLASH_RUN=/absolute/path/to/private/dflash2-experiment
# 该工作区的 env-npu.sh 应设置已验证的 CANN / PyTorch / Python 路径。
mkdir -p "$DFLASH_RUN/models/Qwen3.8-27B-DFlash2" "$DFLASH_RUN/results"
# 将本目录脚本复制到 DFLASH_RUN。下载脚本自行校验固定版本和 SHA256。
python3 "$DFLASH_RUN/download_draft.py" "$DFLASH_RUN/models/Qwen3.8-27B-DFlash2"
bash "$DFLASH_RUN/prepare_native.sh"
bash "$DFLASH_RUN/start_native.sh" dflash2
# 另一个终端等待 localhost:8040/health 返回 200 后执行：
python3 "$DFLASH_RUN/benchmark.py" --mode dflash2 --run "$DFLASH_RUN/results"
# 停止实验服务后以同样步骤运行 start_native.sh mtp 和 --mode mtp。
python3 "$DFLASH_RUN/analyze.py" "$DFLASH_RUN/results" "$DFLASH_RUN/summary" \
  --generation-config "$QWEN_MODEL_ROOT/models/Qwen3.8-27B/generation_config.json"
```

`start_native.sh` 固定使用实验物理卡 6；移植时应先根据卡授权调整。`serve.py` 在 API 前端校验前注册 Ascend 模型插件，并以 `__main__` 保护多进程入口。所有框架缓存、权重和输出放在独立实验目录。

## 比较口径

每种方法先预热 16 / 128 token，再做两条提示各两次、每次固定 256 token 的正式请求；另有三条最长 96 token 的输出检查。单并发、temperature=0、seed=42、关闭思考、关闭前缀缓存、BF16、TP=1；测速时不运行 profiler。主模型上下文上限 4096，调度 token 上限 512。

连续生成速度排除第一个流式返回块中的全部 token，用剩余 token 数除以首末块时间。这样不会因 DFlash2 一次返回多个 token 而多算分子。保留请求、token ID、文本、分块时间和 vLLM 投机计数原始回执。逐项报告 token 差异，不将有限样本相同称为全面精度证明；HTTP 计时也不是内核耗时。

初始 90% 显存预算下，主模型与草稿权重加载成功，加载阶段的框架分配量为 53.9677 GiB（源码虽将单位打印为 GB，实际除数为 2³⁰）。完成预热后框架报告缓存可用 0.77 GiB，4096 上下文至少需要 3.54 GiB，因缓存不足启动失败。随后将两组的显存预算同改为 95%；DFlash2 缓存可用量实测增至 3.82 GiB。没有改变权重精度。

## 实测结果

两组均在同一物理卡 6、同一官方 0.27.1rc1 框架代码上完成。主模型图、草稿 eager、BF16、单并发、4096 上下文上限相同。

| 指标 | 内置 MTP，1 个候选 | DFlash2，7 个候选 |
|---|---:|---:|
| 四次 256-token 请求连续生成速度中位数 | 30.673 token/s | 33.614 token/s |
| 首 token 延迟中位数 | 282 ms | 372 ms |
| 请求总耗时中位数 | 8.596 s | 8.023 s |
| 服务就绪时 NPU 进程显存 | 59,942 MiB（58.54 GiB） | 60,046 MiB（58.64 GiB） |
| 框架可用缓存预算 | 4.42 GiB | 3.82 GiB |
| 候选 token 接受率，含预热及输出检查 | 84.00% | 25.39% |
| 按投机计数计算的每轮 token 数，1+accepted/drafts | 1.840 | 2.777 |

测试结束后查询到 NPU 运行时可见总显存 **60.9570 GiB**，因此框架 95% 的预算实际约 **57.9092 GiB**。标称 64 GiB 还包含运行时预留区域；图等预算外分配也会影响最终进程占用。最初口头按 64×95%=60.8 GiB 的计算只是标称容量换算，不能代替实占。见 [容量查询](../../evidence/2026-10-20/dflash2-memory-capacity.json)。表中就绪显存不代表持续采样的峰值。

固定长度口径中位数提高 **9.589%**，但两条提示表现不同：矩阵乘解释的两次变化为 **+0.24% / −1.46%**，HBM 解释为 **+21.40% / +18.86%**。不能把该均值差写成所有请求都快 10%。DFlash2 首 token 慢约 90 ms；短回答可能没有总耗时收益。当前证据表明它减少验证轮数，但没有测量完整草稿/验证时间分解，不能把总差额全部归给某一个内核。

输出检查保留两种口径：7 项原始整段 token ID 中 **5 项相同，2 项不同**。不同项都是同一矩阵乘提示的重复请求：首次结束符在零基索引 **246**，首次差异在 **253**，差异属于 `ignore_eos=True` 强制越过结束符后的续写。截到首次结束符（含）或本次输出长度上限后，**7 项 token 前缀全部相同**。三条专门输出检查中，计算和代码最长 96 token、可能截断，翻译自然结束；这不是全面精度评测。

为核对强制续写是否影响结论，另从已存流式回执按首次结束符所在返回块截断计时：MTP **30.673**、DFlash2 **34.094 token/s**，中位数差 **+11.154%**。这是同一回执的补充分析，没有新增请求；不能把块内 token 到达时间细分成逐 token 内核时间。

完整数字及逐请求结果：[比较摘要](../../evidence/2026-10-20/dflash2-comparison.json)。服务恢复与过滤启动日志：[运行证据](../../evidence/2026-10-20/dflash2-runtime-evidence.json)。本轮实验服务结束后恢复原 0.26.0 MTP；独立新环境保留，不替换现有默认环境。

**10:33:43** 恢复回执确认 localhost:8039 与 :8038 均健康；**10:34:41** 只读进程检查确认卡 6 为原 90% 预算、单候选 MTP、主模型图模式，卡 7 为原无投机基线；均无额外 NZ 配置、LD_PRELOAD 或观察 extension。新实验端口 8040 已停止。[恢复配置](../../evidence/2026-10-20/dflash2-restored-configs.json)。

历史 vLLM 0.26.0 MTP 的 31.136 token/s 使用 128-token 请求，与本次口径不同，不能直接用两者的差额归因于 DFlash2。

## 已遇到的启动问题

- 显式映射卡 6 的普通 runc 容器无法取得设备数量，模型尚未加载；未扩展为 privileged 或暴露其他卡。后改用上述原生独立环境，设备识别恢复正常。
- 初版 API 包装脚本漏掉 `__main__` 保护，spawn 子进程递归启动 API，在加载前失败；已修正。该次没有生成性能数据。

官方参考：[Ascend 投机解码指南](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/speculative_decoding.md)、[草稿模型](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)、[模型注册时序问题](https://github.com/vllm-project/vllm-ascend/issues/18240)。
