"""Render measured aggregates only, after validation/export."""
import json
from pathlib import Path
import sys


def main(evidence,destination):
    e=json.loads(evidence.read_text(encoding='utf8'))
    b=e['benchmark'];a=b['arms']['baseline'];c=b['arms']['early']
    p=e['profiling']['captures'];h=e['prepare_input']
    lines=['# 接受数量提前：6卡同服务对照结果','',
           'Qwen3.8-27B，910B2物理6卡，BF16权重/conv、FP32 SSM，TP1、单并发、MTP7、动态4096词表。保留既有FFN优化、prefill直读及主/草稿图；只改变接受状态处理与回传的先后顺序。','',
           '## 输出和策略','',
           '32/32道自然结束题输出逐token一致。两道预检题（137/575输出token，共150轮含prefill）逐轮4096候选集合哈希序列完全一致；候选记录在正式测速时关闭。主模型仍使用完整词表，动态选词规则代码未改，未改成上一轮候选。','',
           '## 无profiler端到端','',
           '| 指标 | 原顺序 | 接受数量提前 |','|---|---:|---:|']
    for title,key in [('输出token总数','output_tokens'),('请求E2E总和（秒）','request_e2e_sum_s'),('输出吞吐：输出/E2E总和（token/s）','output_throughput_by_request_e2e'),('测试墙钟时间（秒）','benchmark_window_s'),('墙钟输出吞吐（token/s）','output_throughput_by_window')]:
        lines.append(f'| {title} | {a["common"][key]:.6f} | {c["common"][key]:.6f} |')
    for metric in ('E2E','TTFT','TPOT_stream_end','TPOT_last_token'):
        for stat in ('mean','p50','p90','p99'):
            x=a['distributions'][metric];y=c['distributions'][metric]
            lines.append(f'| {metric} {stat}（{x["unit"]}） | {x[stat]:.3f} | {y[stat]:.3f} |')
    for title,key in [('草稿轮数','rounds'),('接受草稿token数','accepted'),('每轮接受草稿数（不含主模型补词）','accepted_per_round'),('草稿接受比例','acceptance_fraction')]:
        lines.append(f'| {title} | {a["acceptance"][key]:.6f} | {c["acceptance"][key]:.6f} |')
    d=b['comparison']
    lines+=['',f'吞吐变化 **{d["throughput_change_pct"]:+.3f}%**，E2E总和变化 **{d["e2e_change_pct"]:+.3f}%**；{d["faster_requests"]}/32题耗时降低。两组各一次，先原顺序后提前顺序，保留运行顺序和同机负载波动限制，不宣称长期稳定收益。','',
            '**接受轨迹限制**：正式32题接受草稿轮数1868→1873、接受token数6962→6957。输出完全相同不代表草稿过程逐轮相同；两个预检通过只证明这两个请求的候选序列一致，不能外推成32题候选全过程已证明等价。正式测速未记录全部候选ID；本轮尚未唯一定位这5轮差异的原因，也不能把1.28%全部归因于调度。未以缩短输出或改变选词策略来换取速度。','',
            '## 单独采样的decode窗口','',
            '| 指标 | 原顺序 | 接受数量提前 |','|---|---:|---:|']
    for title,fn in [('主模型验证轮数',lambda x:p[x]['verification_rounds']),('decode窗口（ms）',lambda x:p[x]['decode_window_ms']),('无计算区间（ms）',lambda x:p[x]['exclusive']['no_compute']['exclusive_ms']),('无计算占比（%）',lambda x:p[x]['exclusive']['no_compute']['window_percent']),('每轮输入准备CPU墙钟（ms）',lambda x:h[x]['steady_prepare_cpu_wall_ms']['mean']),('输入准备第二个Event等待（ms）',lambda x:h[x]['steady_sync_duration_ms_by_order']['2']['mean']),('准备完成−草稿图结束（ms，负数表示提前完成）',lambda x:h[x]['host_prep_end_minus_previous_draft_end_ms']['mean'])]:
        lines.append(f'| {title} | {fn("baseline"):.3f} | {fn("early"):.3f} |')
    for order in ('1','3'):
        lines.append(f'| 输入准备第{order}个Event等待（ms） | {h["baseline"]["steady_sync_duration_ms_by_order"][order]["mean"]:.3f} | {h["early"]["steady_sync_duration_ms_by_order"][order]["mean"]:.3f} |')
    lines.append(f'| 准备CPU墙钟扣显式Event等待（ms） | {h["baseline"]["steady_prepare_minus_event_wait_ms"]["mean"]:.3f} | {h["early"]["steady_prepare_minus_event_wait_ms"]["mean"]:.3f} |')
    for phase in ('main','draft'):
        lines.append(f'| {phase}图窗口均值（ms） | {p["baseline"]["graph_windows_ms"][phase]["mean"]:.3f} | {p["early"]["graph_windows_ms"][phase]["mean"]:.3f} |')
    lines+=['','这一步只部分改善重叠。第二个Event等待是接受数量回传；第三个等待对应乐观序列长度校正使用的valid_sampled_token_count事件。第二项提前后，第三项等待暴露出来：原先约0.077ms，现在约6.138ms。源码中该计数由drafter.prepare_next_token_ids_padded产生，经_copy_valid_sampled_token_count的复制流等待当前流后回传；本轮没有把这条依赖一并前移。','',
            '输入准备仍在草稿结束后约7.915ms才完成，尚未恢复此前“草稿结束前准备完”的状态。主图均值49.730→49.716ms、草稿图6.604→6.606ms，基本没变，改善主要表现为图间空隙缩短。两份采样都为28轮、128输出，窗口计算任务并集1878.144→1878.067ms，decode窗口减少52.889ms，其中无计算区间减少52.811ms。','',
            '无计算区间由计算任务时间并集的补集得到，可能包含DMA、同步或Host工作；不等于整卡完全空闲。第二/第三个同步的对象归属来自源码执行顺序核对，trace本身没有提供Event对象ID或完整Python栈。采样值不替代上面的无profiler吞吐，不能把间隙占比直接当可回收加速。','',
            '后续若继续，应先针对已暴露的有效token数校正依赖讨论是否可复用同一份采样计数、提前其必要计算与回传，并先复核正式题的草稿轨迹差异；不能简单删掉第三个同步。本轮没有实施这一步，也不把当前小幅收益设为默认。','',
            '## 校验与实施边界','',
            '两组正常Stop，Text/DB与trace/CSV完整；Compute/Communication原样CTE、CSV及编译版Insight后端交叉核验，服务器和本地复算一致。48个SSM状态实际为FP32，conv/权重BF16，64层packed与主/草稿图保持，动态缓冲地址保持。框架源码运行前后哈希及Git工作区一致。','',
            '提前版本保留原始整个async_state_update块、状态处理及事件语义，并保存原始logits/上轮草稿供同轮选词。本轮实际mamba_cache_mode为none，不能声称已上板验证align模式。新增快照和调度开销计入端到端，不把“提前返回接受数”等同于跳过状态处理。没有新NPU内核、训练、默认部署或Git提交。','',
            '此前2卡模型加载OOM未进入测试，后续2卡被其他量化任务占用，用户改为6卡；失败尝试不混入本表。完成后仅关闭本轮自己的服务，释放核验见过滤证据。','',
            '完整分位数、逐题耗时、接受计数和采样类别见[过滤证据](../../evidence/2026-10-22/accepted-count-overlap.json)。原始回答、候选ID、seed、trace/DB及运行路径保留在私有区。','']
    destination.write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
