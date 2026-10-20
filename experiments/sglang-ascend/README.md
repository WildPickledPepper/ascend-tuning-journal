# SGLang 昇腾方案：四题复现记录

用户要求直接采用 [SGLang 昇腾 DFlash2 适配 PR #35629](https://github.com/sgl-project/sglang/pull/35629) 的公开测试条件，只跑少量题目。由原先单卡方案改为物理卡 6、7 双卡 TP2；每种方法使用同样四道 GSM8K，单并发。

## 固定条件

| 项目 | 本次配置 |
|---|---|
| 硬件 | 两张 Ascend 910B2；公开 PR 为 910C，硬件未声称相同 |
| 主模型 / 草稿 | 既有 Qwen3.8-27B / incoai Qwen3.8-27B-DFlash2，均 BF16 |
| SGLang 源码 | PR 合入 commit `30d0eb2ca9dfcfcfcec5eb86c2bb5a83132a766d` |
| 昇腾软件 | 私有 CANN 9.1.0、torch 2.10.0、torch_npu 2.10.0.post4、Triton-Ascend 3.2.2 |
| 官方 NPU 包 | `sgl-kernel-npu`、custom-ops、ops-transformer 的 `2026.9.1` 发布包，910B / CANN9.1 / Python3.12 构建 |
| 模型并行 | `--tp-size 2`，每组两个 rank，最大同时运行请求 1 |
| 注意力 / 状态 | `--attention-backend ascend --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer` |
| 缓存 / 内存 | 禁用 RadixCache；内存静态比例 0.90；上下文上限 32768 |
| 生成 | greedy：temperature=0、top_p=1、top_k=1；最多新增 2048 token |
| 题目 | GSM8K test 前 4 题，zero-shot chat，每组题目相同 |
| profiler | 正式基准关闭；先有一次 64-token 小预热 |
| AR 对照 | 不启用投机解码；用于对齐公开 PR 的速度分母 |
| DFlash2 | `DFLASH`，verify block 8，即七个候选 |
| MTP 补充对照 | 内置 MTP / `EAGLE`、7 steps、topk1、8 draft tokens；公开昇腾 PR 未提供此组成绩 |

PR 没有提供完整依赖 lockfile，也未列出所有启动参数，不能声称逐字节还原作者环境。本次固定其合入源码、同期 NPU 发布包，并采用其公开列出的硬件并行、BF16、采样和评测条件；上下文与内存参数来自该 PR 中公开的启动示例。Python3.12 下旧 `memfabric-zbal==1.1.2` 无匹配 wheel，独立环境使用 1.1.3；本次 Dense 推理不使用 MoE 负载均衡。

## 评测器与脚本

- [启动脚本](start_sglang.sh) 使用环境变量指定模型和独立环境路径，没有改共享 CANN 或驱动。
- [评测观察脚本](ascend_client.py) 加载该 PR 引用的原 DFlash PR #16818 所用公开 SGLang `GSM8KEval` 与 `ChatCompletionSampler`。两份上游文件固定到 `bea986b2f91432e0bdf2c283527427b0e63fcda1`，保持原文件内容；指定 `num_shots=0`、`num_examples=4`、`num_threads=1`。只增加内存中的请求回执收集。
- 数据来自 [GSM8K 官方 test.jsonl](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)，SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
- 汇总吞吐按正式四题输出 token 总量除以评测墙钟时间计算，含输入处理、解码与少量评测开销；不与此前 SSE 连续生成 token/s 直接混算。
- 回执保留每题生成长度、finish reason、返回 meta 信息、耗时、答案与四题准确率。四题只作为小样本检查，不是完整精度评估。

## 安装与失败边界

最初尝试拉取官方 SGLang 910B 镜像，压缩总大小约 6.4 GB；网络在少量数据处停滞。用户要求改用原生环境后停止拉取，没有使用该镜像启动模型。

随后独立环境先安装 SGLang 0.5.20 与较新的 NPU `2026.9.0.post9` 发布包。模型权重和图捕获成功，但服务预热的 GDN prefill 在 beta 形状检查失败：新包 gating 输出二维 `[T,H]`，该调用链的 chunk 接口要求三维 `[B,T,H]`。该轮没有有效测速结果，控制器恢复原服务。根据源码核对，改为上述 PR 合入版本及同期 `2026.9.1` 官方包，其 gating 输出恢复三维；未自行重写 NPU 内核。

固定版本首次已就绪，但评测子进程缺少私有 Python 的动态库路径，未产生有效评测结果；修正为先 source 环境再启动评测。后续重启与自动恢复发生交叠，原基线占着卡 7，触发双卡空闲内存不平衡检查；中止本实验控制器，按 UID、原 API 端口、模型路径和进程父子关系完整退出自己的服务树，确认两卡均无进程后重新执行。该两次启动也不计入性能结果。

## 四题实测结果

| 方法 | 总输出 token | 四题评测耗时 | 端到端吞吐 | 四题准确率 |
|---|---:|---:|---:|---:|
| AR，无投机 | 1173 | 40.916 s | 28.668 token/s | 4/4 |
| 内置 MTP，7 候选 | 1152 | 16.517 s | 69.748 token/s | 4/4 |
| DFlash2，7 候选 | 872 | 15.149 s | 57.560 token/s | 4/4 |

**按本次四题的端到端 token 吞吐，MTP 领先；DFlash2 约为 AR 的 2.008 倍，相对 MTP 为 0.825 倍（低 17.475%）。** MTP 约为 AR 的 2.433 倍。三组请求 payload 逐项相同，所有回答均自然结束，没有触及 2048 上限。

DFlash2 四题总用时更短，但总输出量也更少，不能把“更早答完”直接当成“每秒 token 更多”。第三题的长度为 AR 904、MTP 883、DFlash2 603 token，最终数值答案都正确；这不是逐 token 一致性验证。

| 题号 | AR：token 数 / 耗时（s） | MTP：token 数 / 耗时（s） | DFlash2：token 数 / 耗时（s） |
|---|---|---|---|
| 1 | 137 / 4.534 | 137 / 2.074 | 137 / 5.566 |
| 2 | 44 / 1.734 | 44 / 1.208 | 44 / 1.004 |
| 3 | 904 / 27.178 | 883 / 11.540 | 603 / 7.011 |
| 4 | 88 / 7.433 | 88 / 1.659 | 88 / 1.535 |

同为 137 token，DFlash2 首题比 MTP 多用约 3.492 s，占 DFlash2 四题总时间的约 36.7%，显著影响这个小样本。其第三题的请求 token/秒约 86.0，MTP 约 76.5；不能据总吞吐断言它每题、每段都慢。这里是请求总耗时，包含 prefill，不是纯 decode 速度。本轮只有一次 64-token 小预热，没有每题预热和多轮重复；首题内部多出的时间尚未采样归因，不直接断言为冷编译或某个内核问题。

运行日志另确认：DFlash2 主模型验证图与草稿图均捕获成功，selector decode 进入草稿图，block 为 8、请求 batch 为 1；MTP 解析后的 steps 为 7、验证窗口为 8。DFlash2 较后两个 decode 日志窗口约 89.0 / 90.1 token/s；MTP 后几个窗口约 65.9～92.6。它们是调度器窗口统计，含边界效应，不能替代四题总吞吐。OpenAI 响应没有返回逐题 `spec_accept_length` / `spec_verify_ct`，因此没有伪造四题平均接受长度。

四题仅为 smoke benchmark，不是完整性能或精度评测。公开 PR 的 3.05 倍相对 AR、使用 910C 和 128 题；本次四题、910B2 的速度与它分开记录。此前单卡 vLLM 的 33.614 token/s 也不能与本次双卡值直接相除来声称框架提速。

过滤证据：[四题请求摘要、来源、解析后参数、图日志与最终恢复状态](../../evidence/2026-10-20/sglang-ascend-four-questions.json)。完整回答与带服务器路径的原始日志留在私有数据区。

## 最终服务状态

服务器回执 **2026-10-20 12:09:22（Asia/Shanghai）** 确认两个原服务恢复：卡7为原AR、卡6为原单候选MTP，均单卡BF16、HTTP 200；实验端口8040已关闭。额外配置检查确认没有残留实验LD_PRELOAD或SGLang私有custom-opp路径，新环境没有替换默认服务。
