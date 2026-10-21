# BF16 L1 权重布局的完整模型请求验证

单卡910B2、BF16、TP1、并发1、MTP7、主/草稿图模式。候选仅把主模型64层 gate/up 改为分块连续布局；两组都启用已完成的 down-prefetch。正常四题输出吞吐 **60.06→61.76 token/s（+2.83%）**，固定512输出 **44.78→46.24 token/s（+3.26%）**。完整口径、代价和限制见[结果报告](results.md)。

| 协议 | 基线 | 布局版 | 请求耗时变化 |
|---|---:|---:|---:|
| 固定512 token，6请求平均E2E | 11.4325 s | 11.0716 s | −3.156% |
| 正常四题，E2E之和 | 14.0536 s | 13.6662 s | −2.756% |

生成阶段TPOT改善约3.3%，TTFT平均多8–9ms。三组对应token序列和MTP计数一致。基线值合并A/B两次；不是跨并发吞吐上限或完整GSM8K精度成绩。

## 实现与输入

`model_extension.py` 只给64层主模型 gate/up 打包，排除MTP头；M8执行布局内核，其他M临时还原单层ND后使用原生linear。解包成本包含在请求计时内，避免常驻双份完整权重。依赖[gate/up内核](../../kernels/gateup-l1-layout/README.md)与[既有down内核](../../kernels/down-prefetch-m8/README.md)。

`graph_audit_base.py`/扩展审计主图、草稿图、实际eligible层、打包数量、主机分发和内存。主机分发计数含图捕获，不代表每次设备重放计数。

`client.py` 通过流式HTTP保留token和计时；两次预热排除。前两道GSM8K题每题固定512输出各重复三次，再对前四题正常停止，共10条测量请求/组。返回第一个token即TTFT，可能是reasoning token。

`controller.py` 是原实验的基线A→布局版→基线B流程，仅停止它自己创建的服务进程组；8047端口已占用则拒绝替换。它是历史复现入口，后续新实验仍应按仓库约定优先复用已有基线，必要时只补一次同卡检查，不自动反复测基线。

需要在私有环境显式设置：

| 环境变量 | 内容 |
|---|---|
| `DEPLOY_ENV` | 已有CANN/torch_npu部署环境脚本 |
| `SERVE_PYTHON` | 对应模型服务虚拟环境Python |
| `QWEN_MODEL_PATH` | Qwen3.8-27B checkpoint目录 |
| `GSM8K_TEST_PATH` | GSM8K test JSONL |

先构建两个内核，在匹配环境中执行 `smoke.py`；控制器使用有requests的客户端环境。归档仅将私有绝对路径改成环境输入/仓库相对依赖，未重新上板测试这些移植包装。真实运行源码哈希与归档源码哈希均记录在[来源清单](../../evidence/2026-10-21/gateup-l1-layout-sources.json)。

## 证据与只读复算

- [三组统计、图审计、输出哈希与MTP计数](../../evidence/2026-10-21/gateup-l1-layout-e2e.json)。
- [逐请求过滤计时](../../evidence/2026-10-21/gateup-l1-layout-e2e-requests.json)：仅usage、耗时及token哈希，不含回答或token原数组。
- [清理回执](../../evidence/2026-10-21/gateup-l1-layout-cleanup.json)：17:17归档确认7卡无进程；没有自动恢复服务。

```bash
python verify_evidence.py
```

该命令只读取Git内的小型JSON，复算全部性能数字并核对输出哈希/推测计数，不启动模型。`analyze.py` 则供私有完整原始请求复算；其输入不入库。启动、打包和图初始化不计入稳态请求，运行时解包计入。输入仅83～121 token，未验证长prefill、多并发或其他形状；未设为默认服务实现。
