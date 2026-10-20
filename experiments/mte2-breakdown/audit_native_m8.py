"""Audit captured native M8 tiling and DBI source ranges without an NPU run.

Inputs are private captures and matching installed TBE scheduler source.
Output contains sizes, offsets within tensors, and hashes, never raw addresses.
"""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

raw_root, source_root, output = map(Path, sys.argv[1:4])
source = source_root / 'gemm_tiling_util.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
field_lists = []
for node in ast.walk(tree):
    if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == 'matmul'):
        for statement in node.body:
            if (isinstance(statement, ast.AugAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == 'tiling_format'):
                field_lists.append(ast.literal_eval(statement.value))
assert len(field_lists) == 1 and len(field_lists[0]) == 38
fields = field_lists[0]
record = struct.Struct('<6QHHBB2x')
result = {'kind': 'captured device DBI plus matching installed scheduler',
          'dtype': 'BF16', 'M': 8, 'custom_device_kernel': False,
          'sources': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in source_root.glob('*.py')}, 'operators': {}}
for shape, n, k, expected_key in [('up', 34816, 5120, 98499),
                                   ('down', 5120, 17408, 98513)]:
    spec = 'full-' + shape
    launch_file, = (raw_root / (spec + '-raw')).glob('launch*.jsonl')
    launch = json.loads(launch_file.read_text().splitlines()[0])
    assert launch['key'] == expected_key and launch['blocks'] == 23
    arguments = bytes.fromhex(launch['args_hex'])
    x, w, y = struct.unpack('<3Q', arguments[:24])
    tiling_blob = arguments[launch['tiling_data_offset']:]
    words = struct.unpack('<40I', tiling_blob)
    t = dict(zip(fields, words))
    assert (t['m_ori'], t['n_ori'], t['k_ori'], t['datatype_bf16']) == (8, n, k, 1)
    assert t['kal0_factor'] * t['k_l0'] == t['kal1_16']
    assert t['kbl0_factor'] * t['k_l0'] == t['kbl1_16']
    bm, bn, bk = t['m_l0'] * 16, t['cub_n1'] * t['n_ub_l0_time'] * 16, t['k_l0'] * 16
    ak, wk = t['kal1_16'] * 16, t['kbl1_16'] * 16
    a_bytes, b_bytes = bm * t['m_al1'] * ak * 2, bn * t['n_bl1'] * wk * 2
    raw_file, = (raw_root / (spec + '-raw')).glob('MemoryChart*.bin')
    blob = raw_file.read_bytes()
    gm = [r for r in record.iter_unpack(blob) if r[8:10] == (6, 0)]
    weights = [r for r in gm if w <= r[0] < w + n * k * 2]
    inputs = [r for r in gm if x <= r[0] < x + 8 * k * 2]
    assert len(weights) + len(inputs) == len(gm)
    # Prove all required weight bytes are read exactly once, including all cores.
    cursor = 0
    for start, size in sorted((r[0] - w, r[2]) for r in weights):
        assert start == cursor, (shape, start, cursor)
        cursor += size
    assert cursor == n * k * 2
    assert all((r[0] - w) % 512 == 0 for r in weights)
    cores = []
    for core in range(23):
        wr = [r for r in weights if r[6] == core * 3 + 2]
        ir = [r for r in inputs if r[6] == core * 3 + 2]
        assert wr and ir
        offset = wr[0][0] - w
        cores.append({'block': core, 'weight_bytes': sum(r[2] for r in wr),
                      'input_bytes': sum(r[2] for r in ir),
                      'first_record_N': offset // (k * 2),
                      'first_record_K': offset % (k * 2) // 2,
                      'weight_row_segments': len(wr),
                      'row_segment_bytes': sorted(set(r[2] for r in wr)),
                      'weight_source_row_pitch_bytes': k * 2})
    folder, = (raw_root / (spec + '-profile')).glob('OPPROF_*')
    memory = list(csv.DictReader((folder / 'Memory.csv').open()))
    assert sum(r[2] for r in gm) == sum(float(r['GM_to_L1_datas(KB)']) for r in memory) * 1024
    result['operators'][shape] = {
        'key': expected_key, 'blocks': 23, 'tiling': t,
        'tiling_first152_sha256': hashlib.sha256(tiling_blob[:152]).hexdigest(),
        'l0_allocated_shape_MNK': [bm, bn, bk],
        'l1_A_shape_MK': [bm * t['m_al1'], ak],
        'l1_B_shape_NK': [bn * t['n_bl1'], wk],
        'l1_A_buffer_bytes': a_bytes, 'l1_B_buffer_bytes': b_bytes,
        'l1_A_buffers': t['al1_db'] + 1, 'l1_B_buffers': t['bl1_db'] + 1,
        'l1_AB_reserved_bytes': a_bytes * (t['al1_db'] + 1) + b_bytes * (t['bl1_db'] + 1),
        'l0_K_chunks_per_weight_L1_copy': wk // bk,
        'weight_bytes_exactly_once': cursor,
        'input_source_bytes': sum(r[2] for r in inputs),
        'input_to_logical_weight_fraction': sum(r[2] for r in inputs) / cursor,
        'input_distinct_tensor_bytes': 8 * k * 2,
        'weight_row_segments': len(weights),
        'all_weight_segments_512B_aligned': True,
        'source_PMU_bytes_match': True,
        'DBI_sha256': hashlib.sha256(blob).hexdigest(), 'cores': cores,
    }
result['limitations'] = [
    'DBI rows describe logical contiguous source segments, not physical HBM transactions.',
    'First-record K offsets are corroborated separately using completed native-binary simulation.',
    'Allocated M=16 must not be confused with the MMAD runtime M field.',
    'Same-shape native F.linear probe is audited; the real-model profile has no 160-byte launch payload.',
    'l2_cache_flag is not treated as an effective bypass policy: the matching scheduler code is commented out.',
]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({name: {key: value[key] for key in [
    'l0_allocated_shape_MNK', 'l1_B_shape_NK', 'l1_AB_reserved_bytes',
    'l0_K_chunks_per_weight_L1_copy', 'weight_bytes_exactly_once',
    'input_source_bytes']} for name, value in result['operators'].items()}, indent=2))
