# GDN FP32状态缓存：整网一致性与动态词表复测

用户在确认BF16跨轮舍入机制后，明确要求“改成fp32调好再测试”。仅在物理7号910B2运行；主模型与草稿权重保持BF16，保留既有FFN优化，单独将`--mamba-ssm-cache-dtype float32`，卷积缓存仍`--mamba-cache-dtype bfloat16`。未修改checkpoint、安装的框架或NPU内核。

## 验证顺序

1. 独立编译缓存启动，保持主/草稿capture8、MTP7、TP1、并发1、ctx4096和batch tokens512。
2. `graph_audit`检查实际48个GDN层缓存：SSM全为FP32，conv全为BF16；检查模型dtype、图捕获和既有packed FFN配置。
3. `controller.py`只跑此前分叉题，顺序full→masked→full。始终完整草稿投影，仅改同一图中固定地址mask的内容；`analyze_traces.py`按精确前缀对齐，不用于测速。
4. 诊断关闭后，`benchmark_controller.py`分别启动完整head和真实4096行小head服务；各自两个预热，再将既有四题正常EOS结束、上限2048，各测两遍。没有强制忽略EOS的固定512续写，也没有扩大到128题或扫参。
5. 输出吞吐为总输出token/HTTP流式E2E之和；候选选词、整理、映射及验证均计入。接受长度用前后Prometheus计数差；分别核验两遍重复性与完整/动态逐token一致性。

诊断已通过：三次均575token、逐token一致，对齐主模型隐藏向量逐位相同。草稿轮数含prefill分别117/120/117，说明分轮确实改变而输出未变。这里只证明所测请求，不能外推所有输入。真实性能与释放见[结果](results.md)。

FP32状态每个元素由2字节变4字节，但整卡显存和总缓存预算不会简单翻倍；缓存块数、混合attention页对齐和显存预算会影响实际容量。本轮分别保留实际形状与分配量，没有把所有BF16计算改为FP32。尤其不能据本轮宣称不同prefill分块方式也完全数值一致。

## 复现入口

部署环境通过私有`DEPLOY_ENV`、`SERVE_PYTHON`、`QWEN_MODEL_PATH`、`GSM8K_TEST_PATH`、三个既有内核库路径及`DYNAMIC_VOCAB_SEED`提供，不把路径与原始数据写入仓库。启动前核实7卡与8047空闲；控制器只关闭自己的进程组。

- `controller.py`：诊断，自动启用`STATE_DIAGNOSTIC_ENABLE=1`，动态小head禁用。
- `benchmark_controller.py`：性能，强制诊断关闭，完整/动态两个独立服务；可按原控制器的label/enable参数单独运行一个阶段。
- `start.sh`：FP32状态启动配置和按诊断/动态分开的缓存目录。
- `analyze.py`、`export_evidence.py`：私有原始数据离线统计与过滤导出。

[此前根因诊断](../dynamic-vocab-consistency/README.md)；[首轮BF16性能历史](../dynamic-draft-vocab/results.md)。
