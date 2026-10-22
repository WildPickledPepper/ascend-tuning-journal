"""Read-only own-process/port receipt; never stop other device processes."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(p):
    return {'artifact':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main(root):
    service=json.loads((root/'results/service.json').read_text())
    print('service receipt keys',list(service))
    # Saved private service IDs, supplemented by captured worker folder IDs.
    own_ids={str(service[k]) for k in ('pid','service_pid') if k in service}
    for p in (root/'profiles').glob('*_ascend_pt'):
        own_ids.add(p.name.split('_')[1])
    own=[]
    for pid in sorted(own_ids):
        result=subprocess.run(['ps','-p',pid,'-o','pid='],capture_output=True,text=True)
        if result.stdout.strip():own.append(pid)
    proc=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i','7','-c','0'],text=True)
    port=subprocess.check_output(['ss','-ltn','sport = :8047'],text=True)
    now=datetime.now(timezone.utc)
    clean={'recorded_utc':now.isoformat(),'recorded_beijing':now.astimezone(timezone(timedelta(hours=8))).isoformat(),
           'own_service_stopped':json.loads((root/'results/stopped.json').read_text())['own_service_stopped'],
           'recorded_own_process_ids_absent':not own,'port8047_closed':len(port.strip().splitlines())==1,
           'card7_no_process_at_later_check':'No process in device' in proc,
           'other_device_process_present_at_later_check':'Process id:' in proc,
           'scope':'Own capture service stopped. A later unrelated device process is only observed, never inspected further or stopped.'}
    assert clean['own_service_stopped'] and clean['recorded_own_process_ids_absent'] and clean['port8047_closed']
    (root/'results/cleanup.json').write_text(json.dumps(clean,indent=2)+'\n')
    sources={'captured_sources':[digest(p) for p in sorted(root.glob('*.py'))]+[digest(root/'start.sh')],
             'private_seed':digest(root/'seed-private.json')}
    (root/'source-manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(clean,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
