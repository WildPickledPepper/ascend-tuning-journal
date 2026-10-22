# 直接读取 packed 权重的 BF16 prefill gate/up

用户在既有M8布局实验后授权适配prefill并测TTFT/E2E。本目录仅处理gate/up；decode沿用原M8内核，down和其他模块不变。布局仍是[256×256连续块](../gateup-l1-layout/README.md)，不重新打包另一份权重，也不量化。

## Tiling与数据流

| 资源 | 配置 |
|---|---|
| 单核输出块 | M≤128，N=256；K包=256；总K=5120 |
| 核间分工 | 23核按N块分配，每核依次处理所分N块的M分块；无卡间通信或跨核同步 |
| L1 A | 两槽，每槽128×256×2B，共128KiB |
| L1 B | 两槽，每槽256×256×2B，共256KiB |
| L0A | 双槽128×64×2B，共32KiB |
| L0B | 双槽256×64×2B，共64KiB |
| L0C | 128×256 FP32，共128KiB |
| 计算 | 20个K包，每包4条K64 MMAD；FP32累加后BF16输出 |

权重从GM按128KiB连续块直接搬到L1，L1→L0B→Cube；不先把整份权重还原ND。A分包从输入ND搬到L1并做ND2NZ。提前搬下一K包，与当前包计算重叠。一个N块对应的M分块相邻处理，为权重的L2复用提供机会；本轮尚无硬件命中率测量，不能将该调度直接称为已验证的HBM流量降低。

尾M按16行对齐进行内部矩阵计算，Fixpipe只写有效行。原型仅支持BF16、连续输入、K5120/N34816、M1～512，不是通用GEMM。L1分块满足384KiB容量，L0各缓冲分别满足910B2约束。

## 正确性边界

`validate.py` 使用实际第0/31层权重和固定随机输入，覆盖M1/17/65/83/92/114/121/128/257/512。当前短请求实际M65/83/92/114/121检查逐位一致；M1/128/257/512有少量BF16差异，不能宣称所有形状与原生逐位一致。直接版使用顺序K64 MMAD，尚未证明所有形状的原生累加顺序一致；需以原始验证JSON为准。

开发失败保留在私有记录：初版尾M写回错误地固定srcStride=128；第二版单行尾块采用m=1的MMAD读法而输出错误；最终采用16行对齐的矩阵模式并修正写回stride，之后重测。本目录源码是修复版本，失败版本不计入性能。

## 复现与测量

在匹配的CANN9.1、torch2.10/torch_npu2.10环境中构建`build.py`。`validate.py`需要`QWEN_MODEL_PATH`，限定`ASCEND_RT_VISIBLE_DEVICES=7`。事件计时比较native、临时unpack后native与packed直接读取；包含调用/提交间隙，不作为纯设备kernel耗时或完整模型收益。

完整请求接入与结果在[实验目录](../../experiments/gateup-packed-prefill/README.md)。本次按`ascendc-perf-optimize`执行建模及单核流水检查；普通单卡算子不涉及卡间或跨核同步步骤。尚未用新增PMU数据证明具体bound或各流水时长。

接口依据：[官方Load2D](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/Load2D.md)。实际性能和数值结论取本机上板结果，不套用文档示例加速比。
