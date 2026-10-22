"""Render the detailed report from filtered JSON evidence only."""
import json
from pathlib import Path
import sys


def render(evidence, detailed, cleanup, target):
    x, stats, receipt = (json.loads(p.read_text(encoding="utf-8")) for p in (evidence, detailed, cleanup))
    base, cand = x["baseline"]["natural"], x["candidate"]["natural"]
    change = x["comparison"]["natural"]["throughput_change_pct"]
    labels = ("full-vocab", "dynamic4096")
    names = ("完整词表", "动态4096")
    out = ["# 32道GSM8K自然回答：完整词表与动态词表", "",
           f"相同32道题各测一次，输出逐token一致：**{sum(p['tokens_equal'] for p in x['paired_output_check'])}/32**。"
           f"按总输出token/请求E2E总和，吞吐{base['output_tok_s']:.4f}→{cand['output_tok_s']:.4f} token/s（{change:+.2f}%）。", "",
           "单卡910B2物理7、Qwen3.8-27B、BF16权重/conv、FP32 SSM状态、TP1、并发1、MTP7、主/草稿capture8；保留已有FFN优化。"
           "原GSM8K test前32题、原顺序，正常EOS、输出上限2048，不拼输入长度、不强制续写。各两个预热请求，启动/编译/预热/审计不进入正式计时，profiler与诊断均关闭。", "",
           "## 整批性能", "", "| 指标 | 完整词表 | 动态4096 |", "|---|---:|---:|"]
    common_keys = (("成功请求数", "successful_requests", ""), ("失败请求数", "failed_requests", ""),
                   ("并发数", "concurrency", ""), ("总输入token", "input_tokens", ""),
                   ("总输出token", "output_tokens", ""), ("请求E2E总和", "request_e2e_sum_s", "s"),
                   ("正式客户端窗口", "benchmark_window_s", "s"),
                   ("请求间客户端整理/写盘间隙", "client_inter_request_gap_s", "s"),
                   ("输出吞吐：请求E2E加总口径", "output_throughput_by_request_e2e", "token/s"),
                   ("输出吞吐：完整窗口口径", "output_throughput_by_window", "token/s"),
                   ("请求吞吐：窗口口径", "request_throughput_by_window", "req/s"),
                   ("输入吞吐：窗口口径", "input_throughput_by_window", "token/s"),
                   ("输入+输出吞吐：窗口口径", "total_token_throughput_by_window", "token/s"),
                   ("按输出间隔数加权TPOT，流结束", "weighted_TPOT_stream_end_ms", "ms/token"),
                   ("按输出间隔数加权TPOT，末token到达", "weighted_TPOT_last_token_ms", "ms/token"))
    for name, key, unit in common_keys:
        vals = [stats["arms"][label]["common"][key] for label in labels]
        formatted = [str(int(v)) if not unit else f"{v:.4f} {unit}" for v in vals]
        out.append(f"| {name} | {formatted[0]} | {formatted[1]} |")
    out += ["", f"逐题E2E动态更快：{x['paired_timing']['candidate_faster_requests']}/32。"
            "两组顺序运行、未交错，32题仅一遍；不能将此点估计视为稳定通用收益。", "",
            "## 每请求延迟、长度与吞吐分布", "",
            "均值按请求等权。P50是中位数，分位数按(n−1)×p线性插值；每请求指标N=32，P99容易受少数长请求影响。"
            "请求吞吐的算术平均与整批总token/总时间不同，主结论使用后者。", ""]
    metric_names = {"E2E": "E2E", "TTFT": "TTFT", "TPOT_stream_end": "TPOT：流结束",
                    "TPOT_last_token": "TPOT：末token到达", "InputTokens": "输入token",
                    "OutputTokens": "输出token", "request_output_throughput": "每请求输出吞吐",
                    "client_token_delivery_interval": "客户端token交付间隔",
                    "nonempty_stream_event_interval": "非空SSE返回块间隔",
                    "tokens_per_nonempty_stream_event": "每个非空返回块的token数"}
    for label, name in zip(labels, names):
        out += [f"### {name}", "", "| 指标 | 单位 | N | 均值 | 最小 | P50 | P75 | P90 | P95 | P99 | 最大 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, display in metric_names.items():
            row = stats["arms"][label]["distributions"][key]
            numbers = " | ".join(f"{row[k]:.4f}" for k in ("mean", "min", "p50", "p75", "p90", "p95", "p99", "max"))
            out.append(f"| {display} | {row['unit']} | {row['n']} | {numbers} |")
        out.append("")
    out += ["### 时间定义与MTP的ITL边界", "",
            "- E2E：客户端发出请求到HTTP流结束；TTFT：发出请求到首次收到token ID。",
            "- TPOT（沿用前轮）：(E2E−TTFT)/(输出token数−1)，包含最后token到流关闭的短尾部。另列末token版本，排除该尾部。",
            "- 同一SSE返回块内多个token只有一个可观测到达时间，因此客户端逐token交付间隔中包含0。它衡量交付节奏，不是设备内部逐token生成ITL。",
            "- 非空SSE返回块间隔表示相邻两块的等待，不除以块内token数，也不能直接当单token ITL。未采设备逐token生成时间，不伪造该项。",
            "- token交付间隔和返回块间隔按全部间隔汇总，N与32个请求不同。没有把TTFT当纯prefill内核时间或宣称纯设备prefill带宽。", "",
            "## 接受长度与结束情况", "", "| 指标 | 完整词表 | 动态4096 |", "|---|---:|---:|"]
    for name, key in (("草稿轮数", "rounds"), ("提出草稿token", "drafted"), ("接受草稿token", "accepted"),
                      ("每轮接受草稿数，不含主模型补词", "accepted_per_round"),
                      ("加一个主模型补词后的理论推进长度", "one_plus_accepted_per_round"),
                      ("接受/提出比例", "acceptance_fraction")):
        out.append(f"| {name} | {base['acceptance'][key]:.6f} | {cand['acceptance'][key]:.6f} |")
    for label, name in zip(labels, names):
        r = x["request_statistics"][label]
        out += ["", f"{name}结束原因：`{json.dumps(r['finish_reasons'],ensure_ascii=False)}`；触及长度上限的题号（零基）：`{r['capped_question_ids']}`。"]
    out += ["", "接受计数核对drafted=7×rounds及七个位置之和；EOS附近未送达的额外草稿工作仍可能计数。输出一致性不是完整数学题正确率评估。", "",
            "## 逐题记录", "", "题号为数据文件零基行号。E2E单位秒、TTFT/TPOT单位毫秒；TPOT沿用流结束定义。", "",
            "| 题号 | 输入token | 完整/动态输出token | 输出一致 | 完整E2E | 动态E2E | 完整TTFT | 动态TTFT | 完整TPOT | 动态TPOT |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    br, cr = (x["request_statistics"][label]["per_request"] for label in labels)
    for b, c, pair in zip(br, cr, x["paired_output_check"]):
        bt = (b["e2e_s"] - b["ttft_s"]) * 1000 / (b["output_tokens"] - 1)
        ct = (c["e2e_s"] - c["ttft_s"]) * 1000 / (c["output_tokens"] - 1)
        out.append(f"| {b['question_id']} | {b['input_tokens']} | {b['output_tokens']}/{c['output_tokens']} | {pair['tokens_equal']} | {b['e2e_s']:.4f} | {c['e2e_s']:.4f} | {b['ttft_s']*1000:.3f} | {c['ttft_s']*1000:.3f} | {bt:.3f} | {ct:.3f} |")
    out += ["", "## 证据与释放", "",
            "[过滤结果](../../evidence/2026-10-22/gdn-fp32-gsm32.json)、[完整分布统计](../../evidence/2026-10-22/gdn-fp32-gsm32-statistics.json)、[来源哈希](../../evidence/2026-10-22/gdn-fp32-gsm32-sources.json)、[释放回执](../../evidence/2026-10-22/gdn-fp32-gsm32-cleanup.json)。", "",
            f"{receipt['recorded_beijing']}确认7卡无进程、8047关闭。原始请求、回答、token数组、流式日志、seed和部署路径仅留私有区。未Git提交/推送，未设默认或追加扫参。", ""]
    target.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    render(*(Path(p) for p in sys.argv[1:6]))
