# Prefill gate/up 直接读取重排权重的完整请求对照

沿用前次decode布局版：主模型64层gate/up只常驻packed权重，M8仍执行原布局decode内核，down仍用同一预取实现。仅把其他M的gate/up从“临时还原ND＋原生linear”替换为[直接读取内核](../../kernels/gateup-packed-prefill/README.md)。不修改原checkpoint或共享安装。

已完成：[完整结果](results.md)。正常四题TTFT362.37→336.38ms，输出61.688→62.212 token/s（+0.85%）；1513输入单样本TTFT1040.85→947.36ms。固定512输出吞吐仅+0.11%。22条对应输出token一致，较大M仍有舍入差异，未设默认。

控制器只运行一次packed-control与一次prefill-direct；未重跑最早的原生BF16基线。物理7单卡、BF16/TP1/并发1、MTP7、主/草稿图capture8、ctx4096、max-num-batched-tokens512、无prefix cache，无profiler正式测速。

每组两次预热不计时，前两道GSM8K问题各三次固定512输出，再测前四道自然停止，共10条既有协议请求；另加一条长背景输入、固定128输出，触发多次512-token prefill。长输入仅为路径/延迟检查，不是准确率评测，单个样本不代表稳定吞吐。

候选在第一条65-token预热中逐层与“还原＋原生”核验64层真实激活，要求逐位一致；该验证发生在计时窗口外。每次正式请求保留全部输出token和usage以配对检查。M128等形状的独立测试有舍入差异，需单独核对长输入输出，不宣称全形状逐位等价。

`analyze.py --root PRIVATE_EXPERIMENT_ROOT --output PRIVATE_OUTPUT`只读复算私有原始记录。`model_extension.py`审计新增prefill直接调用数与还原回退数，可确认计时窗口是否仍触发解包。正式对照不混入启动、重排、编译、预热。

## 私有环境输入

启动前提供`DEPLOY_ENV`、`SERVE_PYTHON`、`QWEN_MODEL_PATH`、`GSM8K_TEST_PATH`，以及`DOWN_LIBRARY`、`GATE_DECODE_LIBRARY`、`PREFILL_LIBRARY`三个已构建动态库路径。服务只监听localhost:8047，控制器只终止本次创建的进程组，端口已占用则拒绝替换。

18:18:28归档确认7卡释放、8047关闭。[只读复核](verify_evidence.py)可复算过滤数据并核对64层真实激活、832次直接路径和源码哈希。原始权重、回答、token数组、trace和登录路径保持私有。当前源码仍属实验实现，未设默认服务路径。
