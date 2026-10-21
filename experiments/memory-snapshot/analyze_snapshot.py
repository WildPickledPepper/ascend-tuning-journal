"""Analyze our own trusted allocator dumps and compiled Insight responses.

python analyze_snapshot.py PRIVATE_ROOT FILTERED_JSON
No weights or memory contents are loaded. Input pickles are only the snapshots
collected by this experiment, never untrusted downloaded files.
"""
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import sys

root, destination = map(Path, sys.argv[1:3])
GiB = 2**30


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def clean_stack(text):
    result = []
    for line in text.splitlines():
        location, _, function = line.rpartition(' ')
        result.append(location.rsplit('/', 1)[-1]+' '+function)
    return result


phases = []
snapshots = {}
for file in sorted((root/'snapshots').glob('*.pickle')):
    snapshot = pickle.loads(file.read_bytes())
    metadata = read(file.with_suffix('.json'))
    assert metadata['msmemscope_available']
    assert len(snapshot['device_traces']) == 1
    assert len(snapshot['device_traces'][0]) < 500000, 'History capacity reached'
    allocated = sum(s['allocated_size'] for s in snapshot['segments'])
    reserved = sum(s['total_size'] for s in snapshot['segments'])
    assert allocated == metadata['allocated_bytes'] and reserved == metadata['reserved_bytes']
    phases.append({k:v for k,v in metadata.items() if k not in ['stats','msmemscope_error']})
    phases[-1].update(event_count=len(snapshot['device_traces'][0]),
                     snapshot_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    snapshots[file.stem] = snapshot
assert len(phases) == 12
final = snapshots['12-request-3']
backend_root = root/'insight-final'
complete = read(backend_root/'parse-complete.json')
assert complete['body']['snapshotParsingComplete']
assert complete['body']['snapshotSlices']['0']['eventCount'] == phases[-1]['event_count']
state = read(backend_root/'device-0-slice-0-state.json')
assert state['result']
assert sum(s['allocated'] for s in state['body']['segments']) == phases[-1]['allocated_bytes']
assert sum(s['size'] for s in state['body']['segments']) == phases[-1]['reserved_bytes']
assert len(state['body']['segments']) == len(final['segments'])
blocks = read(backend_root/'device-0-slice-0-blocks.json')
assert blocks['result'] and blocks['body']['total'] == 36076
allocations = read(backend_root/'device-0-slice-0-allocations.json')
reserved_line = read(backend_root/'device-0-slice-0-reserved.json')
assert allocations['result'] and reserved_line['result']
assert allocations['body']['allocations'][-1]['totalSize'] == phases[-1]['allocated_bytes']
assert reserved_line['body']['reservedLine'][-1]['reservedSize'] == phases[-1]['reserved_bytes']

# Validate all twelve msMemScope snapshots, independently of the PT dump parser.
msdb, = root.glob('memscope/**/memscope_dump*.db')
with sqlite3.connect(msdb.as_uri()+'?mode=ro', uri=True) as connection:
    rows = connection.execute('SELECT "Name","Attr" FROM memscope_dump WHERE "Event"=\'SNAPSHOT\'').fetchall()
assert len(rows) == 12
ms = {name: json.loads(attributes) for name, attributes in rows}
ms_free_differences = []
for phase in phases:
    for key, column in [('allocated_bytes','allocated'),('reserved_bytes','reserved'),
                        ('device_total_bytes','total')]:
        assert int(ms[phase['name']][column]) == phase[key]
    ms_free_differences.append(phase['device_free_bytes']-int(ms[phase['name']]['free']))

# Use Insight's complete event table for interval peaks; the GUI trend line is
# compressed and can omit a particular phase endpoint. These are allocator peaks.
database = root/'snapshots/12-request-3.pickle.msinsight/device_0/slice_00000.db'
endpoint_errors = []
with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
    previous = 0
    for phase in phases:
        end = phase['event_count']-1
        endpoint = connection.execute('SELECT allocated,reserved FROM trace_entry_0 WHERE id=?',(end,)).fetchone()
        peak = connection.execute('SELECT MAX(allocated),MAX(reserved) FROM trace_entry_0 WHERE id BETWEEN ? AND ?', (previous,end)).fetchone()
        error = abs(endpoint[0]-phase['allocated_bytes'])
        # Historic replay rounds sizes differently for a few small allocations;
        # bound it explicitly rather than claiming byte-exact intermediate states.
        assert error <= 8192 and endpoint[1] == phase['reserved_bytes']
        endpoint_errors.append(error)
        phase.update(interval_peak_allocated_bytes=peak[0],interval_peak_reserved_bytes=peak[1],
                     insight_endpoint_error_bytes=error)
        previous = end
    peaks = {}
    for label, start, end in [('whole_capture',0,phases[-1]['event_count']-1),
                               ('first_request',phases[8]['event_count'],phases[9]['event_count']-1)]:
        row = connection.execute('SELECT id,size,allocated,callstack FROM trace_entry_0 WHERE id BETWEEN ? AND ? ORDER BY allocated DESC LIMIT 1',(start,end)).fetchone()
        peaks[label] = {'event_id':row[0], 'trigger_allocation_bytes':row[1],
                        'allocator_allocated_bytes':row[2], 'stack_tail':clean_stack(row[3])[-12:]}

groups = collections.defaultdict(lambda: {'bytes':0,'allocations':0})
live_categories = collections.defaultdict(lambda: {'bytes':0,'requested_bytes':0,'allocations':0})
for segment in final['segments']:
    for block in segment['blocks']:
        if block['state'] != 'active_allocated':
            continue
        signature = ' > '.join(Path(frame['filename']).name+':'+frame['name'] for frame in block.get('frames',[])[:5])
        groups[signature]['bytes'] += block['size']
        groups[signature]['allocations'] += 1
        functions = {frame['name'] for frame in block.get('frames',[])}
        category = 'weight_creation' if 'create_weights' in functions else 'cache_allocation' if '_allocate_kv_cache_tensors' in functions else 'other_live'
        live_categories[category]['bytes'] += block['size']
        live_categories[category]['requested_bytes'] += block['requested_size']
        live_categories[category]['allocations'] += 1
top = sorted(({'stack':key,**value} for key,value in groups.items()), key=lambda x:x['bytes'], reverse=True)[:16]
assert sum(value['bytes'] for value in groups.values()) == phases[-1]['allocated_bytes']
fragmentation = []
for segment in final['segments']:
    inactive = [b['size'] for b in segment['blocks'] if b['state']=='inactive']
    fragmentation.append({'pool': 'graph_private' if any(segment.get('segment_pool_id',(0,0))) else 'default',
        'reserved_bytes':segment['total_size'], 'allocated_bytes':segment['allocated_size'],
        'inactive_bytes':sum(inactive), 'inactive_blocks':len(inactive),
        'largest_inactive_bytes':max(inactive,default=0),
        'awaiting_free_bytes':sum(b['size'] for b in segment['blocks'] if b['state']=='active_awaiting_free')})
last = phases[-1]
summary = {'model':'Qwen3.8-27B','hardware':'Ascend910B2','physical_device':6,
    'configuration':read(root/'graph-audit.json')['results'][0],
    'phases':phases,'insight':{'parse_complete':True,'event_count':phases[-1]['event_count'],
        'allocation_events':blocks['body']['total'],'final_state_segments':len(final['segments']),
        'final_counters_match_exactly':True,'max_intermediate_replay_error_bytes':max(endpoint_errors),
        'all_12_memscope_allocated_reserved_total_match_exactly':True,
        'msmemscope_vs_later_device_free_differences_bytes':ms_free_differences,
        'device_free_note':'msMemScope samples before pickle dump; metadata samples afterward. Device free changes outside framework pools, so these are not asserted simultaneous.'},
    'memory_breakdown_bytes':{
        'worker_initialized':phases[0]['allocated_bytes'],
        'model_loading_net_increase':phases[1]['allocated_bytes']-phases[0]['allocated_bytes'],
        'cache_initialization_net_increase':phases[5]['allocated_bytes']-phases[4]['allocated_bytes'],
        'post_cache_other_live_increase':last['allocated_bytes']-phases[5]['allocated_bytes'],
        'allocator_reserved_but_inactive':last['reserved_bytes']-last['allocated_bytes'],
        'device_used_outside_allocator_reservation':last['device_total_bytes']-last['device_free_bytes']-last['reserved_bytes'],
        'device_free':last['device_free_bytes']},
    'live_categories_by_callstack':dict(live_categories),
    'fragmentation':fragmentation,'largest_live_callstack_groups':top,'peaks':peaks,
    'request_growth':{'first_request_net_allocated_bytes':phases[9]['allocated_bytes']-phases[8]['allocated_bytes'],
        'subsequent_two_requests_net_allocated_bytes':phases[11]['allocated_bytes']-phases[9]['allocated_bytes'],
        'subsequent_two_requests_net_reserved_bytes':phases[11]['reserved_bytes']-phases[9]['reserved_bytes']},
    'cleanup':{k:v for k,v in read(root/'state.json').items() if k not in ['server_pid']},
    'limits':['Three identical 128-token requests, not a long-run leak proof.',
              'Allocator event peaks are not continuously sampled whole-process/device peaks.',
              'Outside-allocator value is device total minus free minus framework reserved; not a per-component ownership breakdown.',
              'Inactive graph-private pool reservations are not claimed to be safely releasable.',
              'Worker-local hooks and memory tracing affect startup/profiling; no throughput benchmark was run.']}
assert sum(summary['memory_breakdown_bytes'].values()) == last['device_total_bytes']
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'breakdown_GiB':{k:v/GiB for k,v in summary['memory_breakdown_bytes'].items()},
                  'peak_GiB':{k:v['allocator_allocated_bytes']/GiB for k,v in peaks.items()},
                  'request_growth':summary['request_growth'],'verification':summary['insight']},indent=2))
