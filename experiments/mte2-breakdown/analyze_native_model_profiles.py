"""Cross-check compiled Insight groups against framework CSV and compare FFNs.

Private root contains baseline/optimized-v2/optimized-v3-profile folders and
native-m8-{baseline,v2,v3}-backend-operators.json from backend-query.cjs.
Exports counts/times only, without raw trace timestamps or tensor addresses.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root, output = map(Path, sys.argv[1:3])
result = {}
baseline_profile = json.loads((root / 'model-v2/baseline-profile.json').read_text())
baseline_ids = baseline_profile['response']['choices'][0]['token_ids']
assert len(baseline_ids) == 128
for label, folder in [('baseline', 'baseline-profile'),
                      ('v2', 'optimized-v2-profile'),
                      ('v3', 'optimized-v3-profile')]:
    path = root / folder
    info_file, = path.glob('profiler_info*.json')
    info = json.loads(info_file.read_text(encoding='utf-8'))
    assert info['end_info']['collectionTimeEnd'] > 0
    config = info['config']
    assert config['common_config']['record_shapes']
    assert config['experimental_config']['_aic_metrics'] == 'ACL_AICORE_PIPE_UTILIZATION'
    exported = path / 'ASCEND_PROFILER_OUTPUT'
    assert (exported / 'trace_view.json').exists() and list(exported.glob('*.db'))
    with (exported / 'kernel_details.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['Type'], row['Name'], row['Accelerator Core'], row['Input Shapes']].append(row)
    backend = json.loads((root / f'native-m8-{label}-backend-operators.json').read_text())
    assert backend['result']
    checked = 0
    for item in backend['body']['data']:
        item = item.get('compare', item)
        selected = groups[item['opType'], item['opName'], item['accCore'], item['inputShape']]
        assert len(selected) == int(item['count'])
        duration = sum(float(row['Duration(us)']) for row in selected)
        assert abs(duration - float(item['totalTime'])) < 0.011
        assert abs(duration / len(selected) - float(item['avgTime'])) <= 0.00501
        checked += 1
    assert checked == len(groups)
    ffn = {}
    for shape, signature in [('up', '8,5120;34816,5120'),
                              ('down', '8,17408;5120,17408')]:
        models = collections.defaultdict(list)
        for row in rows:
            if row['Type'] == 'MatMulV2' and row['Input Shapes'].strip('"') == signature:
                assert row['Input Data Types'] == 'DT_BF16;DT_BF16'
                models[row['Model ID']].append(row)
        assert len(models) == 2
        main_id = max(models, key=lambda key: len(models[key]))
        draft_id, = set(models) - {main_id}
        rounds, rem = divmod(len(models[main_id]), 64)
        assert rem == 0 and len(models[draft_id]) == rounds * 7
        ffn[shape] = {'verification_rounds': rounds}
        for role, model_id in [('main', main_id), ('draft', draft_id)]:
            selected = models[model_id]
            ffn[shape][role] = {'count': len(selected),
                               'total_kernel_us': sum(float(row['Duration(us)']) for row in selected)}
            for key in ['Duration(us)', 'aic_mte2_time(us)', 'aic_mte1_time(us)',
                        'aic_scalar_time(us)', 'aic_mac_time(us)']:
                ffn[shape][role]['mean_' + key] = statistics.mean(float(row[key]) for row in selected)
    receipt_path = {'baseline': 'model-v2/baseline-profile.json',
                    'v2': 'model-v2/optimized-profile.json',
                    'v3': 'model/optimized-profile.json'}[label]
    receipt = json.loads((root / receipt_path).read_text())
    assert receipt['request'] == baseline_profile['request']
    ids = receipt['response']['choices'][0]['token_ids']
    assert len(ids) == 128
    result[label] = {'profiler_complete': True, 'configuration': config,
                     'profile_token_ids_equal_to_baseline': ids == baseline_ids,
                     'profile_sequence_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     'insight_groups_verified': checked, 'ffn': ffn}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
