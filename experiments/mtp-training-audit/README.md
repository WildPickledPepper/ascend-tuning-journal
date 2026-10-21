# 单张910B2训练原生MTP头：入口核对与实板组件验证

2026-10-21，Asia/Shanghai。**结论：真实尺寸原生BF16 MTP头已在单张B2完成一次前向、反向与AdamW更新。采用同一卡先采冻结主模型特征、退出后再训练小头的离线分阶段方案。没有正式训练，没有测接受率收益，没有使用那128道评测题。**

## 1. 实际核对的入口

固定SpecForge源码`03f3682ea852b73488d62dab4cd539033cc60e3e`，单独源码副本；没有修改共享CANN、驱动、torch、vLLM或模型安装。

- 官方入口：`specforge train --config ...`；`specforge.cli`与真实MTP模型均在本机现有runtime导入成功。
- 路径：`algorithms/mtp/providers.py`严格加载原生`mtp.*`、加载并冻结target embedding/lm_head → `core/mtp.py::OnlineMTPModel`构造损失 → `training/strategies/base.py::MTPTrainStrategy` → `optimizer.py::BF16Optimizer`仅收集`requires_grad=True`参数。
- `deployment.mode=local_colocated`、`data.hidden_states_path`选择offline，`nproc_per_node=1`。Schema已校验：MTP/BF16/offline/单进程。`target_backend=sglang`是框架配置字段，不表示训练时会常驻完整target；离线训练读取已落盘特征。
- 输入三类tensor：`input_ids`、`loss_mask`、`target_last_hidden_states`；最后一项是目标模型最终post-norm表示，不是任意中间层或DFlash多层拼接。训练内部将token移为x[t+1]、融合h[t]、监督x[t+2]，位置也需随token移动。
- BF16Optimizer使用FP32 master和Adam状态，是小头优化器内部精度；主模型权重及MTP前向仍保持BF16，不是量化或对主干做全参训练。

[正式训练前审阅配置](offline-review.json)只有占位私有路径、max_steps1，用于入口审阅，未执行。其256长度没有做实板显存验证；本轮实板仅32token。没有安装新的全局pip包，`specforge`命令名本身未安装进现有runtime；本轮通过源码导入验证。

## 2. 权重和结构严格对齐

从本机真实checkpoint读17个tensor：15个原生MTP tensor、一个embedding和一个独立lm_head。原生MTP 424699392参数；冻结两张词表共2542796800参数。

Qwen27B参数：hidden5120、FFN17408、24个Q头/4个KV头、head_dim256、vocab248320、attention output gate开启、RoPE theta10000000、partial0.25（旋转64维）、Gemma式RMSNorm乘`1+weight`。共享词表是**加载同一目标模型对应权重并冻结**；该目标embedding与lm_head彼此不绑定，不能照4B配方把两个key都指向embedding。

模型构造在meta后严格加载全部key，避免随机初始权重混入；不可持久化RoPE buffer按原配置重建。优化器只含424699392个MTP参数；不包含embedding、lm_head或27B主干。

## 3. 查到并修正的两处训练副本兼容问题

该补丁使用零上下文格式，在固定源码上使用`git apply --unidiff-zero training-compat.patch`应用；应用前先用`--check`核对。

[training-compat.patch](training-compat.patch)只影响独立SpecForge源码，不改任何NPU内核或主模型安装：

1. **重复交换注意力布局**：当前Transformers eager attention已经返回[B,S,H,D]，社区训练层又transpose(S,H)后reshape，使后续投影看到错排数据。独立CPU probe使用B1/S3/H2/D8捕获HF输出与o_proj输入：原版最大差0.6622037888、逐位不等；去掉多余transpose后最大差0、逐位相同。不能只做shape正确或loss有限检查就认为训练语义正确。
2. **RoPE新字段迁移**：本机Transformers5.14.1将旧`rope_theta`移入`rope_parameters`。上游训练层只读旧字段时回退10000；本模型应为10000000。副本改为兼容读取新字段。对实际JSON解析后的配置，CPU计算的32个inv_freq与原生theta1e7、64旋转维逐位一致。

实板步骤显式设定正确theta1e7，因此第二项修复不改变已测步骤的RoPE数值；后续补的是JSON入口兼容。HF完整解码与vLLM部署逐token等价、真实feature对齐尚未验证，不能由上述两项局部检查代替。

服务器无法解析github.com，首次直接clone失败；从本机拉取固定源码后传入隔离目录。没有通过更改服务器DNS或升级共享软件绕过。

## 4. 单卡实板结果

15:38:23～15:38:30，在物理6单卡可见、无其他本用户服务的条件下，使用真实BF16 MTP和词表权重，B1/S32**合成随机token与hidden state**；不读取语料、答案、既有请求回执或128题。主64层模型完全没有载入NPU。

| 检查/显存口径 | 结果 |
|---|---:|
| 仅模型加载后的框架allocated | 5.527GiB |
| 前向/反向/FP32 master＋Adam更新全程allocated峰值 | 14.327GiB |
| 框架reserved峰值 | 14.430GiB |
| 步骤结束时allocated | 10.372GiB |
| Loss | 15.193750，有限；合成数据无质量意义 |
| MTP梯度 | 所有可训练参数均有有限梯度 |
| optimizer更新 | 成功，fc权重实际改变 |
| 冻结embedding/lm_head | 与CPU原始tensor逐位完全相同，无梯度 |
| checkpoint | 只读加载，没有保存；相关shard大小/mtime未变 |

测量含一份约100MiB的fc前值克隆用于核对更新；峰值是框架allocator口径，不是npu-smi设备总占用。该32token结果不承诺32K长序列也能装下。未做整个checkpoint前后密码学hash扫描，未把mtime检查伪装成全文件内容哈希。

**完整CLI训练循环没有运行。** 此次实际执行的是经局部修正的真实MTP组件、官方OnlineMTPModel损失和BF16Optimizer的一次更新，不是只拿小随机模型估内存，也不是完整离线训练任务已跑通。入口Schema与导入通过是另一层证据。

## 5. 为什么单卡建议离线分阶段

1. 同一张卡先运行冻结BF16主模型，仅无梯度生成独立训练集响应与post-norm特征，按短块保存。此前已验证27B单卡BF16推理能运行；本轮没有重测teacher采集。
2. target进程退出并确认显存释放后，启动MTP小头训练，使用上述已验证的约14GiB级短序列组件路径，不再载入64层主干。
3. 保存独立小头checkpoint，之后才建立独立推理副本验证加载与MTP7接受率；保持原始27B可回退，主干BF16权重不覆盖。

普通两个进程同时常驻会重复冻结词表并叠加优化器内存：旧整网权重约53GiB，小头训练还需约12GiB静态状态及激活，超过本卡约60.96GiB可见总量。没有证明所有在线方案都不可行；共享权重或CPU offload可能改变预算，但不属于此次已验证路径。

BF16最终hidden为每token5120×2B=10KiB，100万token约9.54GiB，另有token/mask和文件开销。离线将显存需求转为磁盘需求，因此先定小规模训练特征，不能直接生成无限量数据。

## 6. 数据和训练目标边界

- 本轮只合成随机输入，不访问任何训练或评测数据；128题及其答案完全未用于更新。未生成正式训练特征，示例feature路径尚不存在。
- 后续拟只选独立训练split，例如GSM8K train；禁止整个GSM8K test进入训练，不能将128题旧回执/答案目录当训练路径。训练、验证、最终128题评测分开；准备数据时再做规范化问题去重和来源manifest检查，本轮没有声称已完成真实数据去重。
- 当前SpecForge OnlineMTPModel做**单步teacher-forced CE**，`ploss_decay`单层无效，`ttt_length=7`没有在其forward中变成七步递归。它能微调原生头，但不证明一定提升MTP7多步接受长度。需要后续审计多步训练或在保留评测上验证实际接受收益。
- 正式训练前仍需：验证冻结teacher的特征采集、原生/训练层数值对齐与导出加载；准备隔离训练集；小规模真实离线CLI任务验证。不能把本轮合成一步当成这些步骤已完成。

## 7. 资源、版本和产物

15:41:54确认6/7均无活动进程；没有模型服务保留、没有训练checkpoint保存。仅保留独立源码与审阅文件，不自动训练。

- [过滤证据](../../evidence/2026-10-21/mtp-single-card-training-audit.json)
- [实板组件核对脚本](smoke_native_mtp.py)：由参数指定源码、模型和回执路径，只写JSON证据，不保存权重。需使用已修正训练源码和正确NPU环境。
- [27B头配置](qwen27b-mtp.json)、[离线审阅配置](offline-review.json)、[兼容补丁](training-compat.patch)。
- 本机torch2.10.0+torch_npu2.10.0.post4、Transformers5.14.1；上游pyproject声明torch2.13/Transformers5.12.1，未按元数据升级。组件、导入与Schema的实测支持本次窄范围兼容，不代表全项目全部功能兼容。
- [固定版本MTP训练provider](https://github.com/sgl-project/SpecForge/blob/03f3682ea852b73488d62dab4cd539033cc60e3e/specforge/algorithms/mtp/providers.py)、[MTP损失](https://github.com/sgl-project/SpecForge/blob/03f3682ea852b73488d62dab4cd539033cc60e3e/specforge/core/mtp.py)、[优化器](https://github.com/sgl-project/SpecForge/blob/03f3682ea852b73488d62dab4cd539033cc60e3e/specforge/optimizer.py)。
