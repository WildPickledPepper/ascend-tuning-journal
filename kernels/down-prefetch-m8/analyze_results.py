"""Read private receipts/Insight exports; emit aggregate evidence without addresses.

Usage: python analyze_results.py PRIVATE_ROOT FILTERED_JSON
PRIVATE_ROOT holds the three operator exports, two framework captures, and
model-run receipts. No model or profiler is launched by this script.
"""
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import sys

root, destination = map(Path, sys.argv[1:3])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def choice(record):
    value = record['response']['choices'][0]
    return value['token_ids'], value['message'], value['finish_reason']


def category(row):
    if 'down_m8_prefetch' in row['Name']:
        return 'target_down'
    if row['Type'] != 'MatMulV2':
        return 'other_compute'
    shape = row['Input Shapes'].strip('"')
    if shape == '8,17408;5120,17408':
        return 'target_down'
    if shape == '8,5120;34816,5120':
        return 'ffn_up'
    if shape in ['1,5120;248320,5120', '8,5120;248320,5120']:
        return 'vocabulary_projection'
    return 'other_matmul'


result = {'operator': {}, 'model': {}, 'profiles': {}}
validation, bench = read(root / 'validation.json'), read(root / 'bench.json')
assert validation['validation_passed']
assert all(m['finite'] and m['different_outputs'] == 0
           for row in validation['cases'] for m in row['modes'])
result['operator']['validation'] = {
    'started_utc': validation['started_utc'], 'dtype': validation['dtype'],
    'input_cases': len(validation['cases']),
    'outputs_compared': sum(row['outputs'] * len(row['modes']) for row in validation['cases']),
    'all_bitwise_equal_to_native': True,
    'graph_changed_input_cases': sum(bool(r.get('graph_replay_and_changed_input_equal')) for r in validation['cases'])}
result['operator']['unprofiled_graph_benchmark'] = bench
for label in ['native', '0', '1']:
    folder, = (root / ('profile-' + label)).glob('OPPROF*')
    info, = csv_rows(folder / 'OpBasicInfo.csv')
    arithmetic = csv_rows(folder / 'ArithmeticUtilization.csv')
    pipes = csv_rows(folder / 'PipeUtilization.csv')
    backend = read(root / f'prefetch-op-{label}-source-details-baseInfo.json')
    assert backend['result'] and abs(float(backend['body']['compare']['duration']) - float(info['Task Duration(us)'])) < .001
    counts = [int(r['aic_cube_total_instr_number']) for r in arithmetic]
    assert len(counts) == 23 and set(counts) == {544}
    pipe0, = [r for r in pipes if r['block_id'] == '0']
    result['operator'][label] = {'kernel_us': float(info['Task Duration(us)']),
        'cores': 23, 'mmad_per_core': 544,
        'core0_pipe_us': {k: float(pipe0[k]) for k in ['aic_cube_time(us)', 'aic_mte1_time(us)', 'aic_mte2_time(us)', 'aic_scalar_time(us)']},
        'insight_base_info_matches_csv': True}

reference = read(root / 'model-run/results/baseline-r1-sglang-requests.json')
for label in ['baseline', 'optimized']:
    receipts = [read(root / f'model-run/results/{tag}-sglang-requests.json')
                for tag in [label + '-r1', label + '-r2', label]]
    for receipt in receipts:
        assert receipt['accuracy'] == 1 and receipt['output_tokens'] == 844
        assert [r['request'] for r in receipt['records']] == [r['request'] for r in reference['records']]
        assert [choice(r) for r in receipt['records']] == [choice(r) for r in reference['records']]
    rates = [r['throughput_tok_s'] for r in receipts]
    audit, = read(root / f'model-run/{label}-graph-audit-after.json')['results']
    assert audit['num_speculative_tokens'] == 7 and audit['draft_use_cuda_graph']
    assert audit['dtype'] == 'torch.bfloat16' and audit['tp_size'] == 1
    result['model'][label] = {'throughput_tok_s': rates,
        'median_tok_s': statistics.median(rates), 'range_tok_s': [min(rates), max(rates)],
        'output_tokens_per_pass': 844, 'questions_per_pass': 4,
        'accuracy': 1, 'all_token_ids_messages_and_finish_reasons_equal': True,
        'request_started_utc': [r['records'][0]['started_utc'] for r in receipts],
        'graph_audit': audit}
    capture = root / (label + '-profile')
    info_file, = capture.glob('profiler_info*.json')
    info = read(info_file)
    assert info['end_info']['collectionTimeEnd'] > 0
    assert info['config']['common_config']['record_shapes']
    assert info['config']['experimental_config']['_aic_metrics'] == 'ACL_AICORE_PIPE_UTILIZATION'
    exported = capture / 'ASCEND_PROFILER_OUTPUT'
    assert (exported / 'trace_view.json').exists() and list(exported.glob('*.db'))
    rows = csv_rows(exported / 'kernel_details.csv')
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['Type'], row['Name'], row['Accelerator Core'], row['Input Shapes']].append(row)
    backend = read(root / f'prefetch-{label}-backend-operators.json')
    assert backend['result'] and len(groups) == len(backend['body']['data'])
    for item in backend['body']['data']:
        item = item.get('compare', item)
        selected = groups[item['opType'], item['opName'], item['accCore'], item['inputShape']]
        assert len(selected) == int(item['count'])
        assert abs(sum(float(r['Duration(us)']) for r in selected) - float(item['totalTime'])) < .011
    events = collections.defaultdict(list)
    selected = collections.defaultdict(list)
    for index, row in enumerate(rows):
        start = int(Decimal(row['Start Time(us)'].strip()) * 1000)
        duration = int(Decimal(row['Duration(us)']) * 1000)
        group = category(row)
        selected[group].append(row)
        events[start].append((index, group, 1))
        events[start + duration].append((index, group, -1))
    active, previous = {}, None
    exclusive = collections.Counter()
    for timestamp, changes in sorted(events.items()):
        if previous is not None:
            categories = set(active.values())
            group = 'no_compute_interval' if not categories else next(iter(categories)) if len(categories) == 1 else 'other_compute'
            exclusive[group] += timestamp - previous
        for index, group, delta in changes:
            if delta > 0:
                active[index] = group
            else:
                active.pop(index)
        previous = timestamp
    span = max(events) - min(events)
    assert sum(exclusive.values()) == span
    profile = read(root / f'model-run/{label}-profile.json')
    baseline_profile = read(root / 'model-run/baseline-profile.json')
    assert profile['request'] == baseline_profile['request']
    assert choice(profile) == choice(baseline_profile)
    target = selected['target_down']
    assert len(target) == 1846
    result['profiles'][label] = {'complete': True,
        'window_ms': span / 1e6, 'scope': 'first to last compute kernel; includes prefill',
        'token_ids_equal_to_baseline': True, 'output_tokens': 128,
        'insight_groups_verified': len(groups),
        'kernel_csv_sha256': hashlib.sha256((exported / 'kernel_details.csv').read_bytes()).hexdigest(),
        'categories': {k: {'exclusive_ms': v / 1e6, 'window_percent': 100*v/span,
                          'cumulative_kernel_ms': sum(float(r['Duration(us)']) for r in selected[k])/1000}
                       for k, v in exclusive.items()},
        'target_model_call_counts': sorted(collections.Counter(r['Model ID'] for r in target).values()),
        'target_model_id_note': 'Custom task has sentinel Model ID; optimized count is aggregate, not a per-main/draft split.' if label == 'optimized' else 'Native main/draft counts are 1664 and 182.',
        'target_mean_us': {k: statistics.mean(float(r[k]) for r in target)
                           for k in ['Duration(us)', 'aic_mte2_time(us)', 'aic_mte1_time(us)', 'aic_scalar_time(us)', 'aic_mac_time(us)']}}
base, opt = result['model']['baseline'], result['model']['optimized']
result['whole_model_throughput_change_percent'] = 100 * (opt['median_tok_s']/base['median_tok_s'] - 1)
base, opt = result['profiles']['baseline'], result['profiles']['optimized']
saved = base['categories']['target_down']['exclusive_ms'] - opt['categories']['target_down']['exclusive_ms']
result['profile_explanation'] = {
    'down_saved_ms': saved,
    'down_time_reduction_percent': 100*saved/base['categories']['target_down']['exclusive_ms'],
    'down_saved_as_percent_of_baseline_window': 100*saved/base['window_ms'],
    'observed_window_saved_ms': base['window_ms'] - opt['window_ms'],
    'other_time_saved_ms': base['window_ms'] - opt['window_ms'] - saved,
    'limits': 'Separate profiled prefix and unprofiled four-question timing. Native/custom implementation differs beyond prefetch; only operator mode0/1 isolates lookahead. Pipe activity intervals overlap and are not added. Incomplete simulator capture excluded.'}
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'operator'}, ensure_ascii=False, indent=2))
