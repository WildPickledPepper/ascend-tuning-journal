"""Validate VLLM MTP7 receipts against the prior single-NPU SGLang run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('data', type=Path)
p.add_argument('sglang_data', type=Path)
p.add_argument('repo', type=Path)
a = p.parse_args()
load = lambda path: json.loads(path.read_text(encoding='utf-8'))
raw = load(a.data / 'requests.json')
sg = load(a.sglang_data / 'mtp-sglang-requests.json')
state = load(a.data / 'state.json')
assert state['phase'] == 'finished' and 'error' not in state
assert state['restored_health'] == {'8038': 200, '8039': 200}
assert state['baseline_api_unchanged'] and state['experiment_health'] == 0
assert len(raw['records']) == len(sg['records']) == 4
assert [r['request'] for r in raw['records']] == [r['request'] for r in sg['records']]
assert raw['evaluator_source_sha256'] == sg['evaluator_source_sha256']
total = sum(r['response']['usage']['completion_tokens'] for r in raw['records'])
assert total == raw['output_tokens']
assert math.isclose(total / raw['latency_s'], raw['throughput_tok_s'], rel_tol=1e-12)

audits = {}
for phase in ['before', 'after']:
    rows = load(a.data / f'graph-audit-{phase}.json')['results']
    assert len(rows) == 1
    item = rows[0]
    for key, expected in [('num_speculative_tokens', 7), ('draft_use_cuda_graph', True),
                          ('draft_enforce_eager', False), ('tp_size', 1), ('context_length', 4096),
                          ('dtype', 'torch.bfloat16'), ('mamba_cache_dtype', 'bfloat16'),
                          ('mamba_ssm_cache_dtype', 'bfloat16'), ('prefix_caching', False),
                          ('gpu_memory_utilization', .97), ('max_num_seqs', 1)]:
        assert item[key] == expected, (phase, key, item.get(key))
    draft = [w for w in item['wrappers'] if w['is_draft_runnable']]
    assert len(draft) == 1 and any(e['captured'] for e in draft[0]['entries'])
    assert item['main_capture_sizes'] == [8]
    audits[phase] = item

def metrics(path):
    totals = {}
    positions = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('vllm:spec_decode_') or '_total' not in line:
            continue
        match = re.match(r'([^\s{]+)(?:\{([^}]*)\})?\s+([-+\deE.]+)', line)
        if not match:
            continue
        name, labels, val = match.groups()
        val = float(val)
        totals[name] = totals.get(name, 0) + val
        if name == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            position = re.search(r'(?:position|pos)="(\d+)"', labels or '')
            if position:
                key = position.group(1)
                positions[key] = positions.get(key, 0) + val
    return totals, positions

before, pb = metrics(a.data / 'metrics-before.txt')
after, pa = metrics(a.data / 'metrics-after.txt')
delta = {name: after.get(name, 0) - before.get(name, 0) for name in before.keys() | after.keys()}
rounds = delta['vllm:spec_decode_num_drafts_total']
proposed = delta['vllm:spec_decode_num_draft_tokens_total']
accepted = delta['vllm:spec_decode_num_accepted_tokens_total']
assert rounds > 0 and proposed == 7 * rounds and 0 <= accepted <= proposed
positions = {pos: pa.get(pos, 0) - pb.get(pos, 0) for pos in pb.keys() | pa.keys()}
if positions:
    assert sum(positions.values()) == accepted

requests = []
for i, (r, s) in enumerate(zip(raw['records'], sg['records'])):
    response = r['response']
    assert r['request']['max_tokens'] == 2048
    assert response['usage']['prompt_tokens'] + 2048 <= 4096
    choice = response['choices'][0]
    assert choice['finish_reason'] == 'stop'
    n = response['usage']['completion_tokens']
    requests.append(dict(question_index=i, started_utc=r['started_utc'],
                         prompt_sha256=hashlib.sha256(json.dumps(r['request']['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                         usage=response['usage'], elapsed_s=r['elapsed_s'],
                         request_throughput_tok_s=n / r['elapsed_s'], finish_reason=choice['finish_reason'],
                         message_sha256=hashlib.sha256(json.dumps(choice['message'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                         prompt_tokens_equal_to_sglang=response['usage']['prompt_tokens'] == s['response']['usage']['prompt_tokens'],
                         sglang_output_tokens=s['response']['usage']['completion_tokens'],
                         sglang_elapsed_s=s['elapsed_s']))
assert all(r['prompt_tokens_equal_to_sglang'] for r in requests)
log = (a.data / 'vllm-server.log').read_text(encoding='utf-8')
selected = []
needles = ['[spec_decode/base]', 'Graph capturing finished', 'Graph memory usage',
           'Replaying aclgraph', 'GPU KV cache size:', 'Available KV cache memory:',
           'Loading model weights took', 'Avg generation throughput:', 'SpecDecoding metrics:']
for line in log.splitlines():
    if any(key in line for key in needles) and '/home/' not in line:
        selected.append(re.sub(r'pid=\d+', 'pid=REDACTED', line))
memory = [int(v) for v in re.findall(r'Process memory\(MB\):(\d+)', (a.data / 'card6-ready.txt').read_text())]
assert memory
restore = [{key: value for key, value in row.items() if key != 'pid'} for row in load(a.data / 'restored-checks.json')]
failure = load(a.data / 'attempt-path-state.json')
result = dict(model='Qwen3.8-27B', hardware='Ascend 910B2', physical_devices=[6], tp_size=1,
              target_and_draft_dtype='BF16', profiler=False, same_request_payloads_as_sglang=True,
              runtime=load(a.data / 'runtime-facts.json'), source_manifest=load(a.data / 'source-manifest.json'),
              benchmark={key: raw[key] for key in ['protocol','example_count','warmup_requests','warmup_max_tokens',
                         'evaluator_source_commit','evaluator_source_sha256','latency_s','output_tokens','throughput_tok_s','accuracy']},
              sglang_reference={key: sg[key] for key in ['latency_s','output_tokens','throughput_tok_s','accuracy']},
              throughput_ratio_to_sglang=raw['throughput_tok_s'] / sg['throughput_tok_s'],
              requests=requests, graph_audit=audits, log_evidence=selected,
              speculative_counter_delta=delta, accepted_per_position_delta=positions,
              candidate_acceptance_rate=accepted / proposed, counter_tokens_per_round=1 + accepted / rounds,
              speculative_counter_scope='Prometheus snapshots immediately after the 64-token warmup and after the four questions; audit and snapshot calls are outside timed evaluation.',
              ready_process_memory_snapshot=dict(process_count=len(memory), total_mib=sum(memory), total_gib=sum(memory)/1024, peak_measured=False),
              started_utc=state['started_utc'],ready_utc=state['ready_utc'],restored_utc=state['restored_utc'],
              restored_health=state['restored_health'],baseline_api_unchanged=state['baseline_api_unchanged'],
              experiment_health=state['experiment_health'],restored_configuration=restore,
              failed_attempt=dict(started_utc=failure['started_utc'], restored_utc=failure['restored_utc'],
                                  reason='Observer module path replaced CANN PYTHONPATH; acl import failed before weight loading. Preserve the inherited path when appending the observer.', benchmark_completed=False),
              limitations=['Four math questions, one run per framework; not a universal framework ranking.',
                           'Natural generation lengths can differ; per-request and total timings include prefill.',
                           'SSM and model dtype, speculation count, graphs, context, concurrency, warmup and client protocol are aligned; cache layouts, dependencies and backend implementations differ.',
                           'vLLM uses the qwen3 response reasoning parser while the SGLang reference used no parser; generated-token totals include the whole generation.',
                           'Worker graph inspection is read-only and outside timed evaluation; no profiler or kernel rewrite was used.'])
out = a.repo / 'evidence/2026-10-20/vllm-mtp7-vs-sglang.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'vllm': result['benchmark'],'sglang':result['sglang_reference'],
                  'ratio':result['throughput_ratio_to_sglang'],'acceptance_rate':result['candidate_acceptance_rate'],
                  'tokens_per_round':result['counter_tokens_per_round'],'requests':requests},ensure_ascii=False,indent=2))
