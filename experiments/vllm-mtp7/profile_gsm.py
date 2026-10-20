"""After normal GSM8K timing, capture two identical 128-token prefixes."""
import datetime
import json
from pathlib import Path
import sys
import time
import requests

root = Path(sys.argv[1])
url = 'http://127.0.0.1:8040'
records = json.loads((root / 'results/mtp-sglang-requests.json').read_text())['records']
payloads = []
normal = []
for index in [0, 2]:
    payload = dict(records[index]['request'])
    payload.update(payload.pop('extra_body', {}))
    payload['max_tokens'] = 128
    payload['return_token_ids'] = True
    payloads.append((index, payload))
    start = time.perf_counter()
    response = requests.post(url + '/v1/chat/completions', json=payload, timeout=900)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    result = response.json()
    assert result['usage']['completion_tokens'] == 128
    normal.append(dict(question_index=index, request=payload, response=result, elapsed_s=elapsed))
(root / 'profile-prefix-normal.json').write_text(json.dumps(normal, ensure_ascii=False, indent=2) + '\n')
response = requests.post(url + '/collective_rpc', json={'method': 'enable_profile_scopes'}, timeout=60)
response.raise_for_status()
(root / 'profile-scope-audit.json').write_text(json.dumps(response.json(), indent=2))
rows = []
for index, payload in payloads:
    previous = set((root / 'profiles-mtp').glob('*_ascend_pt'))
    response = requests.post(url + '/start_profile', json={}, timeout=900)
    response.raise_for_status()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        response = requests.post(url + '/v1/chat/completions', json=payload, timeout=900)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        result = response.json()
    finally:
        stop = requests.post(url + '/stop_profile', json={}, timeout=900)
        stop.raise_for_status()
    assert result['usage']['completion_tokens'] == 128
    time.sleep(6)
    created = set((root / 'profiles-mtp').glob('*_ascend_pt')) - previous
    assert len(created) == 1, created
    capture = next(iter(created))
    rows.append(dict(question_index=index, request=payload, response=result, elapsed_s=elapsed,
                     started_utc=started_utc, profile_dir=str(capture.relative_to(root))))
    (root / 'profile-requests.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    print('PROFILE_COMPLETE', index, elapsed, flush=True)
