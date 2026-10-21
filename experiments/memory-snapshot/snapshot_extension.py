"""One-run worker-local snapshots; no model/kernel or precision changes."""
import datetime
import functools
import json
import os
from pathlib import Path

import torch
import torch_npu
from graph_audit_base import GraphAudit
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

root = Path(os.environ['MEMORY_SNAPSHOT_RUN'])
output = root / 'snapshots'
output.mkdir(exist_ok=True)
torch_npu.npu.memory._record_memory_history(
    enabled='all', context='alloc', stacks='python', max_entries=500000)
ms_error = None
try:
    import msmemscope
    msmemscope.config(events='', level='0', data_format='db', output=str(root/'memscope'))
    msmemscope.start()
except Exception as error:
    msmemscope = None
    ms_error = type(error).__name__ + ': ' + str(error)

sequence = 0


def snapshot(label):
    global sequence
    torch_npu.npu.synchronize()
    sequence += 1
    name = f'{sequence:02d}-{label}'
    if msmemscope is not None:
        msmemscope.take_snapshot(device_mask=0, name=name)
    filename = output / (name+'.pickle')
    torch_npu.npu.memory._dump_snapshot(str(filename))
    # This torch_npu build creates dumps with mode0200; allow the owner to analyze them.
    filename.chmod(0o600)
    stats = torch_npu.npu.memory_stats()
    free, total = torch_npu.npu.mem_get_info()
    record = {'name': name, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'allocated_bytes': torch_npu.npu.memory_allocated(),
              'reserved_bytes': torch_npu.npu.memory_reserved(),
              'max_allocated_bytes': torch_npu.npu.max_memory_allocated(),
              'max_reserved_bytes': torch_npu.npu.max_memory_reserved(),
              'device_free_bytes': free, 'device_total_bytes': total,
              'stats': stats, 'msmemscope_available': msmemscope is not None,
              'msmemscope_error': ms_error}
    (output / (name+'.json')).write_text(json.dumps(record, indent=2))
    print('MEMORY_SNAPSHOT', name, record['allocated_bytes'], record['reserved_bytes'], flush=True)
    return {k:v for k,v in record.items() if k != 'stats'}


def instrument(name):
    original = getattr(NPUModelRunner, name)
    @functools.wraps(original)
    def measured(self, *args, **kwargs):
        snapshot(name+'-before')
        result = original(self, *args, **kwargs)
        snapshot(name+'-after')
        return result
    setattr(NPUModelRunner, name, measured)


for method in ['load_model', 'profile_run', 'initialize_kv_cache', 'capture_model']:
    instrument(method)


class SnapshotAudit(GraphAudit):
    def memory_snapshot(self, label='ready', finalize=False):
        record = snapshot(label)
        if finalize:
            if msmemscope is not None:
                msmemscope.stop()
            torch_npu.npu.memory._record_memory_history(enabled=None)
        return record
