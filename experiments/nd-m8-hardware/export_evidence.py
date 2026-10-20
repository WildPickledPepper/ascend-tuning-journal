"""Validate stopped captures and export a filtered audit without private paths."""
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:3])
load = lambda p: json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


checks = {}
for parent in sorted(root.glob('profile-*')):
    if not parent.is_dir():
        continue
    captures = sorted(parent.glob('*_ascend_pt'))
    if not captures:
        continue
    p = captures[-1]
    meta = list(p.glob('profiler_info*.json'))
    end = list(p.rglob('end_info.*'))
    d = p / 'ASCEND_PROFILER_OUTPUT'
    assert meta and end and d.exists(), parent.name
    config = load(meta[0])['config']
    assert config['common_config']['record_shapes']
    assert config['experimental_config']['_l2_cache']
    files = ['kernel_details.csv', 'trace_view.json', 'soc_pmu.csv']
    files += [x.name for x in d.glob('ascend_pytorch_profiler*.db')]
    assert len(files) == 4
    for name in files:
        assert (d / name).stat().st_size > 0
    checks[parent.name] = dict(status='valid', stop_metadata=True, end_info=True,
                              config=config,
                              kernel_csv_sha256=hashlib.sha256((d / files[0]).read_bytes()).hexdigest())
assert len(checks) == 15, len(checks)

backend_checks = []
for label in ['model-pipe', 'tlb-small', 'tlb-full', 'l2-full', 'l2-adjacent']:
    p = sorted((root / ('profile-' + label)).glob('*_ascend_pt'))[-1]
    ks = rows(p / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv')
    for b in load(root / 'backend' / (label + '-backend-matmuls.json')):
        selected = [k for k in ks if k['Input Shapes'] == b['inputShape']
                    and k['Name'] == b['opName'] and k['Type'] == b['opType']]
        assert len(selected) == int(b['count']), (label, b)
        avg = statistics.mean(float(k['Duration(us)']) for k in selected)
        assert abs(avg - float(b['avgTime'])) < .011
        backend_checks.append(dict(capture=label, shape=b['inputShape'],
                                   count=len(selected), backend_mean_us=float(b['avgTime']),
                                   csv_mean_us=avg))

model = load(root / 'model-analysis.json')
hardware = load(root / 'analysis/hardware-comparison.json')
assert not model['unmatched_matmul_keys']
assert hardware['same_addresses_verified']
model.pop('profile')  # The native directory name includes a private hostname.
request_summaries = {}
for q, record in model['requests'].items():
    groups = record['matmuls']
    up = next(k for k in groups if k['shape'].strip('"') == '8,5120;34816,5120'
              and k['calls'] % 64 == 0)
    main_id = up['model_id']
    rounds = up['calls'] // 64
    down = next(k for k in groups if k['shape'].strip('"') == '8,17408;5120,17408'
                and k['model_id'] == main_id)
    head = next(k for k in groups if k['shape'].strip('"') == '1,5120;248320,5120'
                and k['calls'] == 7 * rounds)
    assert up['smmu_identity_failures'] == down['smmu_identity_failures'] == 0
    decode = [k for k in groups if k['model_id'] in [main_id, head['model_id']]
              or (k['model_id'] == '4294967295' and k['shape'].strip('"').startswith('8,'))]
    request_summaries[q] = dict(rounds=rounds, target_up=up, target_down=down,
                               target_ffn_ms_per_round=(up['total_ms'] + down['total_ms']) / rounds,
                               draft_lm_head_ms_per_round=head['total_ms'] / rounds,
                               all_decode_matmul_ms_per_round=sum(k['total_ms'] for k in decode) / rounds)

state = load(root / 'state.json')
assert state['phase'] == 'finished' and state['baseline_api_unchanged']
assert state['restored_health'] == {'8038': 200, '8039': 200}
assert state['experiment_health'] == 0
hardware.pop('same_layer1_addresses', None)
hardware['limits'] = [
    'Controlled isolated graphs use torch.mm(out=), exported as aclnnMm; actual model uses F.linear, exported as aclnnMatmul. Their absolute time difference is not assigned to TLB or any unmeasured tiling change.',
    'No per-miss stall-cycle timing is available. Controlled cold/warm comparisons test whether increased TLB misses materially increase the observed total runtime.',
    'MemoryAccess counters describe GM/request and on-chip traffic, not a direct measurement of physical HBM transactions. Physical HBM bandwidth is taken from the separate device-6 system capture.',
    'HBM bandwidth describes isolated FFN graph phases, not complete-model bandwidth or an established hardware ceiling.',
    'No new normal model throughput result or optimization claim; model requests were profiled.',
]
result = dict(kind='Device measurements: current BF16 ND/M8 MTP7 memory diagnosis',
              device='Ascend910B2', physical_device=6, single_card=True,
              configuration=load(root / 'graph-audit.json'),
              data_checks=checks, backend_cross_checks=backend_checks,
              model_smmu_validation=load(root / 'model-smmu-validation.json'),
              model_requests=request_summaries, hardware=hardware,
              restoration=dict(restored_utc=state['restored_utc'],
                               baseline_unchanged=True, both_services_healthy=True,
                               experiment_closed=True),
              conclusions=[
                  'The old NZ/M2 translation-locality regression is not reproduced by current ND/M8. Up miss counts increase from 1.17 to 127.11 but paired FFN graph time stays 215.50 vs 215.47 us/call.',
                  'The main-model FFN kernels remain dominated by GM-to-L1/MTE2 activity. Actual target up/down times are about 297/153 us; this is distinct from a measured pure-HBM or pure-TLB duration.',
                  'Isolated FFN HBM read bandwidth is about 1.243 TB/s for both small and 64-layer working sets, with balanced units and stable 1600 MHz clocks.',
                  'Immediately repeated down projections benefit from nearly 100% L2 data hits, decreasing 154.10 to 98.01 us. This controlled cache benefit is not directly available to sequential 64-layer autoregressive execution.',
                  'The draft vocabulary projection costs about 13.34 ms/verification round, in addition to 28.82 ms of target FFN matrix multiplications. Both are major BF16 weight-reading workloads.',
              ])
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Validated', len(checks), 'captures;', len(backend_checks), 'Insight matrix groups')
