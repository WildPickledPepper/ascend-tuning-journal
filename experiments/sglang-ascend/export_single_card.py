"""Validate private receipts and export a small, path-free experiment summary."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('data', type=Path)
p.add_argument('previous', type=Path)
p.add_argument('repo', type=Path)
a = p.parse_args()
load = lambda path: json.loads(path.read_text(encoding='utf-8'))
receipt = load(a.data / 'mtp-sglang-requests.json')
previous = load(a.previous / 'mtp-sglang-requests.json')
state = load(a.data / 'state.json')
assert 'error' not in state and state['phase'] == 'finished', state
assert state['baseline_api_unchanged'] and state['restored_health'] == {'8038': 200, '8039': 200}
assert state['experiment_health'] == 0
assert len(receipt['records']) == len(previous['records']) == 4
assert [r['request'] for r in receipt['records']] == [r['request'] for r in previous['records']]
assert receipt['evaluator_source_sha256'] == previous['evaluator_source_sha256']
total = sum(r['response']['usage']['completion_tokens'] for r in receipt['records'])
assert total == receipt['output_tokens']
assert math.isclose(total / receipt['latency_s'], receipt['throughput_tok_s'], rel_tol=1e-12)

log = (a.data / 'mtp-sglang-server.log').read_text(encoding='utf-8')
args_line = next(line for line in log.splitlines() if 'server_args=' in line)
args = ast.literal_eval(args_line.split('server_args=', 1)[1])
expected = dict(tp_size=1, dtype='bfloat16', speculative_algorithm='EAGLE',
                speculative_num_steps=7, speculative_eagle_topk=1,
                speculative_num_draft_tokens=8, max_running_requests=1,
                context_length=4096, max_total_tokens=8192, mem_fraction_static=0.97,
                mamba_ssm_dtype='bfloat16', disable_radix_cache=True)
for key, value in expected.items():
    assert args[key] == value, (key, args.get(key), value)
fields = list(expected) + ['device', 'attention_backend', 'disable_cuda_graph', 'cuda_graph_bs']
facts = {key: args[key] for key in fields if key in args}
selected_lines = []
windows = []
for line in log.splitlines():
    if 'Capture ' in line and 'graph' in line and '/home/' not in line:
        selected_lines.append(line)
    if ('KV Cache is allocated' in line or 'Mamba Cache is allocated' in line
            or 'Memory pool end.' in line or 'Load weight end.' in line):
        if '/home/' not in line:
            selected_lines.append(line)
    if 'Decode batch,' in line:
        row = {'log': line}
        for key, pattern in [('accept_length', r'accept len: ([\d.]+)'),
                             ('accept_rate', r'accept rate: ([\d.]+)'),
                             ('throughput_tok_s', r'gen throughput \(token/s\): ([\d.]+)')]:
            match = re.search(pattern, line)
            if match:
                row[key] = float(match.group(1))
        windows.append(row)
assert any('Capture target verify NPU graph end' in line for line in selected_lines)
assert any('Capture draft decode CUDA graph end' in line for line in selected_lines)
assert any('Capture draft extend CUDA graph end' in line for line in selected_lines)

def spec_fields(obj, path=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            sub = f'{path}.{key}' if path else key
            if key in {'avg_spec_accept_length', 'spec_total_num_accept_tokens', 'spec_total_num_forward_ct'}:
                yield {'field': sub, 'value': value}
            else:
                yield from spec_fields(value, sub)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from spec_fields(value, f'{path}[{i}]')

rows = []
for i, (r, old) in enumerate(zip(receipt['records'], previous['records'])):
    response = r['response']
    choice = response['choices'][0]
    assert choice['finish_reason'] == 'stop'
    assert r['request']['max_tokens'] == 2048
    assert response['usage']['prompt_tokens'] + r['request']['max_tokens'] <= 4096
    count = response['usage']['completion_tokens']
    rows.append(dict(question_index=i, started_utc=r['started_utc'],
                     prompt_sha256=hashlib.sha256(json.dumps(r['request']['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                     output_tokens=count, elapsed_s=r['elapsed_s'],
                     request_throughput_tok_s=count / r['elapsed_s'],
                     finish_reason=choice['finish_reason'], usage=response['usage'],
                     message_sha256=hashlib.sha256(json.dumps(choice['message'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                     same_message_as_previous_tp2=choice['message'] == old['response']['choices'][0]['message'],
                     previous_tp2_output_tokens=old['response']['usage']['completion_tokens'],
                     previous_tp2_elapsed_s=old['elapsed_s']))

failure = load(a.data / 'attempt-90-state.json')
snapshot = (a.data / 'card6-ready.txt').read_text(encoding='utf-8')
memory_mib = [int(v) for v in re.findall(r'Process memory\(MB\):(\d+)', snapshot)]
assert len(memory_mib) == 2
restored = load(a.data / 'restored-checks.json')
restored = [{key: value for key, value in item.items() if key != 'pid'} for item in restored]
result = dict(model='Qwen3.8-27B', hardware='Ascend 910B2', physical_card_count=1, physical_device=6,
              target_and_draft_dtype='BF16', profiler=False, same_request_payloads_as_tp2=True,
              resolved_runtime_fields=facts, started_utc=state['started_utc'], ready_utc=state['ready_utc'],
              restored_utc=state['restored_utc'], baseline_api_unchanged=state['baseline_api_unchanged'],
              restored_health=state['restored_health'], experiment_health=state['experiment_health'],
              restored_configuration=restored,
              ready_process_memory_snapshot=dict(process_count=len(memory_mib), total_mib=sum(memory_mib),
                                                 total_gib=sum(memory_mib) / 1024, peak_measured=False),
              benchmark={key: receipt[key] for key in ['protocol', 'example_count', 'warmup_requests',
                         'warmup_max_tokens', 'evaluator_source_commit', 'evaluator_source_sha256',
                         'latency_s', 'output_tokens', 'throughput_tok_s', 'accuracy']},
              requests=rows, graph_and_memory_log=selected_lines, decode_windows=windows,
              server_info_spec_statistics={phase: list(spec_fields(load(a.data / f'server-info-{phase}.json')))
                                           for phase in ['before', 'after']},
              server_info_scope='Cumulative flushed log-window counters only; includes any warmup rounds in those windows and omits the unflushed tail. Not a four-question-only acceptance rate. Confirmed in scheduler_components/metrics_reporter.py:922-925.',
              previous_tp2_summary={key: previous[key] for key in ['latency_s', 'output_tokens', 'throughput_tok_s', 'accuracy']},
              throughput_ratio_to_previous_tp2=receipt['throughput_tok_s'] / previous['throughput_tok_s'],
              failed_attempt=dict(mem_fraction_static=0.90, started_utc=failure['started_utc'],
                                  restored_utc=failure['restored_utc'], benchmark_completed=False,
                                  reason='Loaded target and draft weights plus state reservation exceed static budget; no performance result.'),
              source_manifest=load(a.data / 'source-manifest.json'),
              limitations=['Four math questions, one run; not a full accuracy benchmark or a production speed guarantee.',
                           'Cache cap/context length/static budget differ from the previous TP2 setup; generation requests match exactly.',
                           'Natural output lengths may differ; totals include prefill and evaluator overhead.',
                           'Decode-window metrics are not the full-run mean and may include idle/request-boundary time.',
                           'No new profiler capture; do not infer a precise kernel or bandwidth contribution from this benchmark alone.'])
out = a.repo / 'evidence/2026-10-20/sglang-single-mtp7.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'benchmark': result['benchmark'], 'requests': rows,
                  'windows': [{k: v for k, v in w.items() if k != 'log'} for w in windows],
                  'server_info_spec_statistics': result['server_info_spec_statistics'],
                  'ratio_to_tp2': result['throughput_ratio_to_previous_tp2']}, ensure_ascii=False, indent=2))
