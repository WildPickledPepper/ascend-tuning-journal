"""Export filtered native tiling/whole-model evidence from private receipts."""
import hashlib
import json
from pathlib import Path
import re
import sys

root, destination = map(Path, sys.argv[1:3])
load = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
result = {'dtype': 'BF16', 'hardware': 'Ascend910B2', 'physical_devices': [6],
          'device_kernel_recompiled': False,
          'audit': load('audit-dbi.json'),
          'simulation': load('simulation-summary.json'),
          'device_profiles': load('device-profiles-summary.json'),
          'validation_v1': load('validation.json'),
          'validation_v2': load('validation-v2.json'),
          'validation_v3': load('validation-v3.json')}
assert all(row['bitwise_equal'] for row in result['validation_v3']['cases'])
fields = list(result['audit']['operators']['up']['tiling'])
assert len(fields) == 38
result['candidate_tilings'] = {}
for label, capture in load('candidate-tilings.json').items():
    shape = label.split('-')[0]
    original = result['audit']['operators'][shape]['tiling']
    t = dict(zip(fields, capture.pop('known_tiling_words')))
    assert len(t) == 38 and capture['launches_checked'] == 3
    assert t['kal0_factor'] * t['k_l0'] == t['kal1_16']
    assert t['kbl0_factor'] * t['k_l0'] == t['kbl1_16']
    m, n, k = t['m_l0'] * 16, t['cub_n1'] * t['n_ub_l0_time'] * 16, t['k_l0'] * 16
    ak, bk = t['kal1_16'] * 16, t['kbl1_16'] * 16
    l1_bytes = (m * t['m_al1'] * ak * 2 * (t['al1_db'] + 1)
                + n * t['n_bl1'] * bk * 2 * (t['bl1_db'] + 1))
    l0b_bytes = n * k * 2 * 2
    assert l1_bytes <= 512 * 1024 and l0b_bytes <= 64 * 1024
    result['candidate_tilings'][label] = dict(
        capture, tiling=t, changes={f: [original[f], t[f]] for f in fields if original[f] != t[f]},
        L1_AB_reserved_bytes=l1_bytes, L0B_double_buffer_bytes=l0b_bytes)
insight = {}
for shape in ['up', 'down']:
    data = load(f'native-m8-{shape}-source-api-instructions-dynamic.json')
    assert data['result']
    rows = [r for r in data['body']['Instructions']
            if r['Source'].startswith(('MOV_OUT_TO_L1_MULTI_ND2NZ', 'MMAD'))]
    sim = result['simulation'][shape + '-base']['cores']['0']
    copies = [r for r in rows if r['Source'].startswith('MOV_OUT_TO_L1')]
    assert sum(int(r['Instructions Executed']) for r in copies) == sum(c['count'] for c in sim['copies'])
    assert sum(int(r['Process Bytes']) for r in copies) == sum(
        c['rows'] * c['columns'] * 2 * c['count'] for c in sim['copies'])
    mmads = [r for r in rows if r['Source'].startswith('MMAD')]
    assert sum(int(r['Instructions Executed']) for r in mmads) == sum(sim['MMAD_MNK_counts'].values())
    insight[shape] = {'instruction_counts_and_bytes_match': True,
                      'rows': [{'instruction': r['Source'].split()[0],
                                'calls': r['Instructions Executed'],
                                'source_bytes': r['Process Bytes']} for r in rows]}
result['insight_instruction_validation'] = insight


def metrics(name):
    values = {}
    for line in (root / name).read_text().splitlines():
        match = re.match(r'([^\s{]+)(?:\{[^}]*\})?\s+([-+\deE.]+)', line)
        if match and (match[1].startswith('vllm:spec_decode_') or
                      match[1].startswith('vllm:time_to_first_token_seconds_')):
            key = match[1]
            values[key] = values.get(key, 0) + float(match[2])
    return values


model = root / 'model'
if (model / 'comparison.json').exists():
    result['whole_model'] = load('model/comparison.json')
    checks = {}
    sources = {'baseline': ('model-v2', 'baseline', 0),
               'optimized_v2': ('model-v2', 'optimized', 3),
               'optimized_v3': ('model', 'optimized', 1)}
    receipts = {}
    for label in result['whole_model']['runs']:
        directory, source_label, mode = sources[label]
        receipt = load(f'{directory}/results/{source_label}-sglang-requests.json')
        receipts[label] = receipt
        records = receipt['records']
        assert len(records) == 4
        count = sum(r['response']['usage']['completion_tokens'] for r in records)
        assert count == receipt['output_tokens']
        assert abs(count / receipt['latency_s'] - receipt['throughput_tok_s']) < 1e-10
        before = metrics(f'{directory}/{source_label}-metrics-before.txt')
        after = metrics(f'{directory}/{source_label}-metrics-after.txt')
        delta = {key: after.get(key, 0) - before.get(key, 0) for key in after.keys() | before.keys()}
        assert delta['vllm:time_to_first_token_seconds_count'] == 4
        rounds = delta['vllm:spec_decode_num_drafts_total']
        proposed = delta['vllm:spec_decode_num_draft_tokens_total']
        accepted = delta['vllm:spec_decode_num_accepted_tokens_total']
        assert proposed == 7 * rounds
        audit = load(f'{directory}/{source_label}-graph-audit-after.json')['results'][0]
        assert audit['native_m8_mode'] == mode
        if label == 'optimized_v3':
            assert audit['native_m8_preserve_l1'] == 1
            assert audit['native_m8_patch_counts'][0] == 0
            assert audit['native_m8_patch_counts'][1] > 0
        for key, expected in [('num_speculative_tokens', 7), ('draft_use_cuda_graph', True),
                              ('tp_size', 1), ('dtype', 'torch.bfloat16'),
                              ('mamba_ssm_cache_dtype', 'bfloat16'), ('main_capture_sizes', [8])]:
            assert audit[key] == expected, (label, key)
        checks[label] = {'ttft_mean_ms': 1000 * delta['vllm:time_to_first_token_seconds_sum'] / 4,
                         'draft_rounds': rounds, 'draft_tokens': proposed,
                         'accepted_tokens': accepted, 'candidate_acceptance': accepted / proposed,
                         'native_mode': audit['native_m8_mode'],
                         'patch_counts': audit['native_m8_patch_counts'],
                         'requests': [{'completion_tokens': r['response']['usage']['completion_tokens'],
                                       'elapsed_s': r['elapsed_s'],
                                       'sequence_sha256': (hashlib.sha256(json.dumps(
                                           r['response']['choices'][0]['token_ids']).encode()).hexdigest()
                                           if r['response']['choices'][0].get('token_ids') is not None else None)}
                                      for r in records]}
    for label in ['optimized_v2', 'optimized_v3']:
        checks[label]['token_ids_equal_to_baseline'] = []
        for a, b in zip(receipts['baseline']['records'], receipts[label]['records']):
            ids_a = a['response']['choices'][0].get('token_ids')
            ids_b = b['response']['choices'][0].get('token_ids')
            assert ids_a is not None and ids_b is not None
            checks[label]['token_ids_equal_to_baseline'].append(ids_a == ids_b)
    result['whole_model_checks'] = checks
    result['whole_model_profiles'] = load('model-profile-comparison.json')
result['limitations'] = [
    'Device probe uses synthetic BF16 weights and a full distinct physical working set.',
    'Simulator timestamps are not actual device latency; v2 single-core runs verify instructions only.',
    'V1/V2 change arithmetic order; V3 preserves bitwise output in the tested operator cases only.',
    'Only matched whole-model performance and output checks determine adoption.',
]
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'whole_model': result.get('whole_model'),
                  'insight': result['insight_instruction_validation']}, indent=2))
