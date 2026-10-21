"""One independently started whole-model A/B phase with native tiling.

Uses the unchanged public four-question GSM8K evaluator in the private run folder.
Each performance pass is unprofiled; a separate matched prefix collects Pipe data.
Start each phase with its mode set before graph capture; never hot-recapture.
"""
import json
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
from pathlib import Path
import subprocess
import sys
import time

import requests

root = Path(sys.argv[1])
url = 'http://127.0.0.1:8040'
client = root / 'client-runtime/venv/bin/python'


def benchmark(label):
    subprocess.run([str(client), '-u', str(root / 'ascend_client.py'), str(root), label],
                   check=True, timeout=900)
    for name in ['graph-audit-before.json', 'graph-audit-after.json',
                 'metrics-before.txt', 'metrics-after.txt']:
        source = root / name
        source.rename(root / (label + '-' + name))
    return json.loads((root / f'results/{label}-sglang-requests.json').read_text())


def profile(label, sample):
    audit = requests.post(url + '/collective_rpc', json={'method':'enable_profile_scopes'}, timeout=60)
    audit.raise_for_status()
    (root / 'profile-scopes.json').write_text(json.dumps(audit.json(),indent=2))
    payload = dict(sample['records'][0]['request'])
    payload.update(payload.pop('extra_body', {}))
    payload.update(max_tokens=128, return_token_ids=True)
    previous = set((root / 'profiles').glob('*_ascend_pt'))
    response = requests.post(url + '/start_profile', json={}, timeout=900)
    response.raise_for_status()
    try:
        response = requests.post(url + '/v1/chat/completions', json=payload, timeout=900)
        response.raise_for_status()
        result = response.json()
    finally:
        stopped = requests.post(url + '/stop_profile', json={}, timeout=900)
        stopped.raise_for_status()
    # Stop flushes data; allow the export process to settle before recording it.
    time.sleep(3)
    created = set((root / 'profiles').glob('*_ascend_pt')) - previous
    assert len(created) == 1
    record = {'profile': str(next(iter(created)).relative_to(root)),
              'request': payload, 'response': result}
    (root / (label + '-profile.json')).write_text(json.dumps(record, indent=2))
    print('MODEL_PROFILE_COMPLETE', label, flush=True)
    return record


def output(record):
    choice = record['response']['choices'][0]
    message = choice['message']
    return {'content': message.get('content'),
            'reasoning': message.get('reasoning', message.get('reasoning_content')),
            'token_ids': choice.get('token_ids'),
            'finish_reason': choice.get('finish_reason')}


label = sys.argv[2]
assert label == 'dflash2'
receipt = benchmark(label)
capture = profile(label, receipt)
(root / (label + '-phase-complete.json')).write_text(json.dumps({
    'label': label, 'unprofiled_tok_s': receipt['throughput_tok_s'],
    'profile': capture['profile'], 'completed': True}, indent=2))
print('MODEL_PHASE_COMPLETE', label, receipt['throughput_tok_s'], flush=True)
