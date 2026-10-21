# vLLM Ascend：DFlash2 K7 图模式对照原生 MTP7

## 实测结果（2026-10-21）

**本次没有提高接受率；保留MTP7配置方案，不采用DFlash2替换。** 六号卡单卡BF16、七候选、主/草稿图、同四道GSM8K、并发1、无profiler正常测速：

| 指标 | 原生MTP7（复用基线） | DFlash2 K7图 |
|---|---:|---:|
| 生成速度 token/s | 59.386（既有三轮中位数） | 51.368（本次一次） |
| 接受候选 / 提出候选 | 681 / 1197 | 649 / 1365 |
| 接受率 | 56.892% | 47.546% |
| 验证轮数 | 171 | 195 |
| 每轮token（计数器口径） | 4.982 | 4.328 |
| 输出token | 844 | 844 |
| 正确题数 | 4/4 | 4/4 |
| 服务端平均TTFT | 313.717ms（匹配最终基线轮） | 364.145ms |

正常吞吐下降13.502%，验证轮数增加14.035%。MTP基线在当天约11点，DFlash2约15点，非同步重测A/B；复用而不再补基线。第一、二、四题token ID完全相同；第三题长度575及最终content相同，但从0基索引311处token不同，不声称逐token等价或通用精度保证。速度包含prefill、请求及生成，不称纯decode速度。MTP三轮TTFT约313.7～334.2ms，表中只列与计数快照匹配的一轮。

每个候选位置按所有验证轮为分母的通过比例：

| 候选位置（1基） | MTP7 | DFlash2 |
|---|---:|---:|
| 1 | 85.965% | 84.103% |
| 2 | 74.854% | 69.231% |
| 3 | 64.912% | 54.359% |
| 4 | 57.310% | 42.051% |
| 5 | 44.444% | 34.359% |
| 6 | 38.596% | 28.205% |
| 7 | 32.164% | 20.513% |

这些不是“已到这个位置之后”的条件接受率，计数已扣除64token预热及独立profile。越后的候选差距越明显：DFlash2本次没有通过提高接受率减少主模型权重重读。不能靠缩短候选长度提高表面百分比就宣称更快，也不放宽验证阈值。

## Insight定位：草稿图更便宜，但验证次数更多

两份框架采样正常Stop，Pipe指标及Text/DB齐全；编译版Insight返回MTP193、DFlash2228个形状分组，逐组次数/累计耗时与CSV一致；技能SQL TOP20也核对通过。以下是**同第一题128token采样**，请求和全部token ID相同，prefill已排除：

| 设备指标 | MTP7 | DFlash2 |
|---|---:|---:|
| 主模型图验证轮数 | 26 | 33 |
| 每轮主图覆盖窗口中位数 | 52.455ms | 52.161ms |
| 主图kernel累计时间 | 1315.133ms | 1657.437ms |
| 主FFN升/降维各调用次数 | 1664 | 2112 |
| 两组主FFN累计时间 | 752.515ms | 951.903ms |
| 草稿图kernel累计 / 验证轮数 | 19.851ms | 7.475ms |

主FFN升维每次298.185→297.273μs、降维154.047→153.439μs，基本不变；由于128token用了更多轮，累计主FFN多199.388ms。DFlash2并行草稿图的设备执行确实更便宜，但不能只看草稿快：接受少了，昂贵的64层主模型反复验证更多次。正常四题的195/171轮计数独立支持这一解释，不依赖profile时长猜测。

**采样额外开销的限制**：DFlash2 decode窗口3094.696ms、无计算kernel区间33.486%，对照1964.282ms/0.858%；新profile在正式测速结束后才增加`record_function`标签，旧profile没有这些标签。新trace大量间隙重叠`prepare input`及CPU tensor操作，其inclusive时间2489.759ms与设备执行重叠，不能相加；缺少专门selector标签，也不能把它全部归为selector。该差额受不同采样标签和框架追踪开销影响，**不能当作无profiler真实Host瓶颈，不能用来解释正常吞吐下降的精确毫秒分摊**。本次确定的是接受率降低、验证次数增加、主图单轮基本不变、草稿图设备计算减少；正常运行中额外Host成本未隔离。

## 资源和采用判断

14:51首次启动；两次失败保留，修复后15:04:06启动、15:07:40就绪、15:09:23采集清理完成（Asia/Shanghai）。就绪进程显存61244MiB≈59.809GiB，非运行峰值；审计实际allocated约59.016、reserved约60.010GiB，二者与npu-smi计数口径不同，不能直接相加。6/7均无活动进程，旧服务未恢复。后续保持MTP7作为已验证的较快方案；本次DFlash2服务不保留。

结果只覆盖这四题和当前版本/兼容适配，不推及所有任务或DFlash2所有框架实现，也不能与早期MTP1/eager的30token/s直接归因比较。真正继续提高接受率需要更匹配目标模型及任务的草稿预测，当前没有通过调参或训练实现该收益。

[过滤证据](../../evidence/2026-10-21/dflash2-vllm-graph.json)；原始请求、模型、DB/trace、日志及地址均只在私有数据目录。

## 目的与边界

用户要求核对能否提高投机接受率，或将 MTP 换为 DFlash2 再测试。复用10月21日已有原生MTP7同四题基线，单卡910B2、BF16、并发1，不复测基线、不量化、不改验证规则、不使用独立MatMul实验补丁。正常测速不启用profiler，另采第一题128token前缀。

## 可复现配置

- 主模型Qwen3.8-27B，DFlash2 checkpoint为incoai/Qwen3.8-27B-DFlash2；5层、block_size=8、selector_top_k=16。框架配置method=dflash，按模型架构实际实例化AscendDflash2Proposer。
- vLLM 0.27.1 / vLLM-Ascend 0.27.1rc1，torch2.10.0 / torch_npu2.10.0.post4，CANN9.1；V1 runner、K7、主与草稿图、capture8；TP1、4096上下文、max_seqs1、max_batched512、memory0.97、prefix cache关闭、SSM/KV保持原BF16配置。
- 保留公开SGLang四题评测器作为HTTP客户端，服务端是vLLM，评测器源码校验和及请求与既有基线核对；greedy T0/top_p1/top_k1、2048上限、自然EOS。
- [start.sh](start.sh)由私有环境变量指定模型和runtime，不含权重或登录信息；[ascend_client.py](ascend_client.py)进行正常测速及前后计数器快照，[compare_model.py](compare_model.py)随后采样。
- 客户端需提供独立Python3.12的lib共享库路径，并设置TORCH_DEVICE_BACKEND_AUTOLOAD=0。客户端仅发送HTTP，不初始化NPU。

## 兼容问题与修复

首次启动在图捕获失败：aclnnFusedInferAttentionScoreV3错误561002明确要求无mask时sparseMode=0。安装版full_graph_fia优先按sliding_window选择sparse4，而DFlash2草稿非因果、attn_mask=None；同文件已有eager分支对这种情况先选择sparse0及无限制pre/next。

[compat.py](compat.py)仅在非因果、无mask、滑窗草稿的图调用期间暂设sliding_window=None，让原函数记录sparse0和INT_MAX的pre/next，finally恢复实例配置。它对齐已有eager语义；主模型因果/带mask路径不变，没有编译新NPU内核、修改共享安装或放宽接受标准。该适配不是上游正式修复或长期兼容性保证。

第二次图捕获成功，但客户端缺libpython3.12共享库路径，在请求前退出；补齐并单独验证评测依赖后重新启动。两次失败原始日志保留私有目录，不当成性能结果。

图审计在测速前后核对实际captured graph、BF16、K7、TP1、DFlash2 proposer及selector配置；compat调用数仅为捕获路径调用计数，不代表设备回放次数。

## 分析方法

使用mindstudio_profiler_data_check检查Stop、配置及Text/DB产物；使用ascend-profiler-db-explorer的原样compute_view宏只读聚合TOP20，并对照CSV。编译版Insight通过operator/statistic按形状返回全部分组，与CSV逐组核对次数和总时间。

[analyze.py](analyze.py)读取既有MTP和本次DFlash2原始回执，扣除warmup后的计数，核对请求和评测器SHA；记录每题token、答案、结束原因是否相同，不强制把不同输出伪装成同一生成工作量。设备主图按每轮64层FFN核对，与草稿图、其他任务分开；CPU标签为inclusive，不与设备时间相加。

官方配置说明：[vLLM Ascend speculative decoding](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html)。
