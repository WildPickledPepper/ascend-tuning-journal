# 计数校准代码

`build_l1_counter_probes.py` 是 2026-10-19 计数校准使用的构建逻辑，输出目录改为参数以便复核。依赖原环境的 CANN 9.1.0 TBE/TIK，目标为 Ascend910B2。脚本只编译，不启动 NPU，也不修改模型服务。

```bash
python build_l1_counter_probes.py --output ./generated/counter-probes
```

生成 `probe_zero` 和 `probe_nd2nz`：前者执行一次指定大小的 L1 清零，后者搬运一行 512 个 B16 元素到 NZ 布局。两者用于区分清零和数据搬运如何进入 PMU 字段，不是性能优化内核。

校准载体为 float16；验证的是与目标 BF16 内核相同的 B16 搬运指令和字节计数，不涉及模型量化。
