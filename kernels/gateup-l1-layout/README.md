# BF16 gate/up 按 L1 块连续存储

物理 7 卡、910B2、M8/N34816/K5120、64 层真实权重。原生 gate/up 每层 279.176 μs，跨步 NZ 对照 295.871 μs，按块布局 239.276 μs；相对原生耗时减少 14.29%，相对相同计算模板跨步 NZ 减少 19.13%。[完整算子报告](results.md)、[过滤计数与统计](../../evidence/2026-10-21/gateup-l1-layout.json)。

## 布局原理

全局 NZ `[K/16,N/16,16,16]` 中，一个 256×256 的 L1 权重块分为 16 段 8 KiB，段起点间隔 1,114,112 字节。按块布局把这 16 段集中为连续 128 KiB：

```text
P[i,j,u,v,a,b] = W[256*i+16*v+a, 256*j+16*u+b]
```

`i=N/256` 分块索引，`j=K/256` 分块索引，u/v/a/b 取 0..15。BF16 元素没有量化、补零或丢失。L1 A=160 KiB、双缓冲 B=256 KiB；23 核；L0 MMAD=8×256×64。两种自定义布局保持相同计算、预取、同步和 K 累加顺序。

完整工作集为 34.2255 GB FFN 权重，gate/up 22.8170 GB。L2 命中率约 0.5%，HBM 每层读取量接近；有效供数变快。未拆出 HBM→L2 与 L2→L1 的独立收益。

## 源码与测量边界

- `kernel.asc`、`bridge.cpp` 与原始上板源码逐字节相同；只把采集脚本私有模型路径改成环境变量。
- `run.py` 使用同一命令入口检查正确性、执行无 profiler 图计时或另外采集 L2/HBM。
- `suite.py` 保留本轮历史 native/nz/packed/packed/nz/native 顺序，每种两次独立进程、各40轮。
- `collect.py` 是独立的 PMU 采集流程；计时数据来自无 profiler 轮。
- gate 和 down 回放使用固定随机输入，没有串接激活/残差/attention，不是整网 E2E。
- 最终8组输入、每布局120次图重放、64层权重/输出检查通过；初始化/首次搬入同步不足的开发失败不纳入成绩。

后续已接入完整模型：[端到端结果](../../experiments/gateup-l1-layout-e2e/results.md)，当前短输入单并发 MTP7 请求耗时减少 2.76%～3.16%。

## 复现输入

先设置私有 `DEPLOY_ENV` 和 `QWEN_MODEL_PATH`，使用匹配 torch/torch_npu 的解释器。`env.sh` 限定物理7卡，不停止其他服务。运行前确认设备可用；这些是历史复现命令，本次仓库维护没有启动 NPU。

```bash
bash env.sh /path/to/runtime/venv/bin/python build.py
bash env.sh /path/to/runtime/venv/bin/python run.py check --output results/validation.json
bash env.sh /path/to/runtime/venv/bin/python suite.py
bash env.sh /path/to/runtime/venv/bin/python collect.py
bash env.sh /path/to/runtime/venv/bin/python analyze.py
```

原始权重、trace/DB、指针地址及 `.so` 留在私有采集环境，不入库。过滤后的80轮计时样本见[计时证据](../../evidence/2026-10-21/gateup-l1-layout-timings.json)。从原始私有采样重跑 `analyze.py` 时建议指定外部 `--root` 和不同的 `--output`，保留原始文件。
