# 首token路径定位与调度优化

2026-10-22完成。7号910B2、同服务32题：平均TTFT **331.020→304.672ms（−7.96%）**，输出吞吐 **64.704→64.600 token/s（−0.16%）**，32/32输出逐token一致。收益是降低首token延迟，未证明吞吐提升。[完整结果与限制](results.md)、[过滤证据](../../evidence/2026-10-22/ttft-first-round.json)。实验服务已关闭，不设为默认部署。

`ttft_extension.py`在首次调度的新请求上记录少量CPU scope墙钟/线程CPU时间、GC回调和异步NPU Event。设备Event覆盖流时间区间，包含下发间隙，不等于纯计算累计；墙钟减线程CPU时间可能包含阻塞和线程调度，不能独立当成OS抢占时间。完整profiler关闭。

对照模式：`baseline`为原顺序；`ready3`为两份计数提前和固定MTP7元数据；`prefill_original`仅新请求首轮使用原顺序、后续继续ready3。计数前移也会经过prefill，因此单独比较首轮，而不是假定prefill路径完全未变。模型数学、权重、词表策略、安装源码与NPU内核不改。

`driver.py diagnose`先预热，再对qid0/qid2按正反顺序做有界诊断，校验所有输出；记录的请求、回答、token及原始遥测仅留私有区。`benchmark`关闭遥测后自然结束并记录客户端TTFT/TPOT/E2E及服务端指标；`validate`核对两道题的候选/草稿轨迹。`summarize_diagnosis.py`只导出阶段数字。

启动时设置私有`COUNT_READY_BASE`指向已归档版本的count-ready源码，另提供原启动脚本所需的环境路径。复用旧编译缓存的首次启动在模型预热阶段因Triton运行时参数不兼容失败，未进入请求测试；改用独立缓存，旧缓存保留。首轮计时用于定位，正式测速关闭遥测与完整profiler。

## 最终实现

`first_output_copy.py`在首token采样后额外D2H复制这一个token供客户端读取；原构造函数、内部CPU缓冲和晚就绪Event原样保留，下一轮准备继续用原依赖。交付时仍调用原`get_output`做转换/过滤，GPU张量保留到草稿流与状态流均完成。初版替换内部就绪Event造成轨迹分叉，已被拦截并弃用。

Core队列增加两处首prefill优先级：新入队和已在队列内的完整新prefill均先处理首输出，再填下一批。普通decode保留原行为。构造函数11个赋值字段集合与Core两处AST目标有检查，这是schema保护，不等于运行值逐项比较或跨版本兼容。所测范围限TP1 greedy、单新请求完整prefill、无logprobs/路由/EP故障检查。其他场景未验证。

| 文件 | 用途 |
|---|---|
| `ttft_extension.py` | 首轮有界计时、GraphAudit及首交付补丁开关 |
| `first_output_copy.py` | 独立提前交付副本、保留内部依赖、Core队列优先级 |
| `driver.py` | 诊断、预检、自然结束测速、输出对参考校验 |
| `start.sh` | 单卡服务，最终原allocator、TASK_QUEUE_ENABLE=1 |
| `summarize_diagnosis.py` | 初始诊断的阶段数字汇总 |
| `export_results.py` | 离线复算并校验审计/归档/来源/释放，过滤私有数据 |
| `build_allocator.sh` | jemalloc私有构建的失败候选复现；最终未采用 |

## 操作与复算

外部需提供`DEPLOY_ENV`、`SERVE_PYTHON`、`QWEN_MODEL_PATH`、`COUNT_READY_BASE`；客户端另需`REFERENCE_RESULTS`及旧client的私有数据配置。服务器地址、原题、回答和token数组不放仓库。`TTFT_PHYSICAL_CARD`最终显式为7；脚本历史默认6不代表可自动启动/任意占卡，实际物理卡以自己的worker PID与设备查询匹配。

客户端动作：`warmup`；`diagnose`；`chain`；`validate early_output`；`benchmark ready3 control32 32`；`benchmark early_output optimized32 32`。`early_output`保留stage3并启用首交付改动。注意`validate`中的baseline是stage0，**不是正式32题的ready3对照**；它仅在前后baseline轨迹重复时才断言候选轨迹等价。本轮baseline自身未重复，不能说全轨迹校验通过，见报告。

每组预热一次，正常EOS、2048输出上限，不强制写满。离线复算：

```text
python experiments/ttft-first-round/export_results.py <初始诊断私有目录> <最终同服务私有目录> evidence/2026-10-22/ttft-first-round.json
```

本地与服务器离线复算相同，33次首轮含预热全部命中，32题输出一致。没有重采整网profiling，不声称空隙本轮重新测为0.94%。服务、自有unit、托管SSH均已结束；失败日志和原始证据只留私有归档。
