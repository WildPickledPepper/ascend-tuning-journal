"""Explain local FFN gains using existing matched Insight/PyTorch captures.

No model is launched. Input is the private native-m8-audit-20261020 directory;
output contains aggregate costs only. CPU scopes are not added to device time.
"""
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

root, destination = map(Path, sys.argv[1:3])
compute_sql = '''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
SELECT op_name, op_type, COUNT(*) AS calls, SUM(duration_ns) AS total_ns
FROM compute_view GROUP BY op_name, op_type ORDER BY total_ns DESC LIMIT 20'''


def category(row):
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


result = {'scope': 'first compute start to last compute end, same 128-token request',
          'category_policy': 'Tiny cross-category overlaps assigned to other_compute; same-category overlaps merged.',
          'gap_policy': 'No compute kernel active; may contain non-compute tasks, dispatch and synchronization.',
          'sql': compute_sql, 'runs': {}}
for label, folder in [('baseline', 'baseline-profile'), ('optimized_v3', 'optimized-v3-profile')]:
    path = root / folder
    info_file, = path.glob('profiler_info*.json')
    assert json.loads(info_file.read_text())['end_info']['collectionTimeEnd']
    out = path / 'ASCEND_PROFILER_OUTPUT'
    csv_file = out / 'kernel_details.csv'
    with csv_file.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    by_operator = collections.defaultdict(list)
    events = collections.defaultdict(list)
    cumulative_ns = collections.defaultdict(int)
    model_counts = collections.Counter()
    for index, row in enumerate(rows):
        start = int(Decimal(row['Start Time(us)'].strip()) * 1000)
        duration = int(Decimal(row['Duration(us)']) * 1000)
        group = category(row)
        events[start].append((index, group, 1))
        events[start + duration].append((index, group, -1))
        cumulative_ns[group] += duration
        by_operator[row['Name'], row['Type']].append(duration)
        if group == 'target_down':
            model_counts[row['Model ID']] += 1
    assert sorted(model_counts.values()) == [182, 1664]
    db = out / 'ascend_pytorch_profiler_0.db'
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
        sql_rows = connection.execute(compute_sql).fetchall()
    assert len(sql_rows) == 20
    for name, kind, count, total_ns in sql_rows:
        values = by_operator[name, kind]
        assert len(values) == count and sum(values) == round(total_ns)
    # Also verify all shape groups from the previously queried compiled backend.
    suffix = 'baseline' if label == 'baseline' else 'v3'
    backend = json.loads((root / f'native-m8-{suffix}-backend-operators.json').read_text())
    assert backend['result']
    shape_groups = collections.defaultdict(list)
    for row in rows:
        shape_groups[row['Type'], row['Name'], row['Accelerator Core'], row['Input Shapes']].append(row)
    for item in backend['body']['data']:
        item = item.get('compare', item)
        selected = shape_groups[item['opType'], item['opName'], item['accCore'], item['inputShape']]
        assert len(selected) == int(item['count'])
        assert abs(sum(float(r['Duration(us)']) for r in selected) - float(item['totalTime'])) < .011
    assert len(backend['body']['data']) == len(shape_groups) == 193
    exclusive_ns = collections.defaultdict(int)
    active, previous, cross_overlap_ns = {}, None, 0
    for timestamp, changes in sorted(events.items()):
        if previous is not None:
            groups = set(active.values())
            if not groups:
                group = 'no_compute_interval'
            elif len(groups) == 1:
                group = next(iter(groups))
            else:
                group = 'other_compute'
                cross_overlap_ns += timestamp - previous
            exclusive_ns[group] += timestamp - previous
        for index, group, delta in changes:
            if delta > 0:
                active[index] = group
            else:
                active.pop(index)
        previous = timestamp
    span_ns = max(events) - min(events)
    assert sum(exclusive_ns.values()) == span_ns
    assert not active and cross_overlap_ns < 20000
    result['runs'][label] = {
        'kernel_csv_sha256': hashlib.sha256(csv_file.read_bytes()).hexdigest(),
        'kernel_count': len(rows), 'insight_groups_verified': len(shape_groups),
        'sqlite_top20_verified': True, 'window_ms': span_ns / 1e6,
        'cross_category_overlap_us': cross_overlap_ns / 1000,
        'categories': {key: {'exclusive_ms': value / 1e6, 'window_percent': 100 * value / span_ns,
                             'cumulative_kernel_ms': cumulative_ns.get(key, 0) / 1e6}
                       for key, value in exclusive_ns.items()},
        'sql_top20': [{'name': name, 'type': kind, 'calls': count, 'total_ms': ns / 1e6}
                      for name, kind, count, ns in sql_rows]}
base, opt = result['runs']['baseline'], result['runs']['optimized_v3']
down_base = base['categories']['target_down']['exclusive_ms']
down_opt = opt['categories']['target_down']['exclusive_ms']
result['explanation'] = {
    'down_time_reduction_percent': 100 * (1 - down_opt / down_base),
    'down_saved_ms': down_base - down_opt,
    'observed_window_saved_ms': base['window_ms'] - opt['window_ms'],
    'predicted_window_reduction_percent_if_others_fixed': 100 * (down_base - down_opt) / base['window_ms'],
    'observed_window_reduction_percent': 100 * (1 - opt['window_ms'] / base['window_ms']),
    'remaining_time_baseline_ms': base['window_ms'] - down_base,
    'remaining_time_optimized_ms': opt['window_ms'] - down_opt,
    'limitations': 'Profile window includes prompt processing; timed four-question throughput is a separate unprofiled measurement.'}
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'categories': base['categories'], 'explanation': result['explanation']}, indent=2))
