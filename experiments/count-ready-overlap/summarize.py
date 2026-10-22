"""Path-free numeric comparison; raw prompts/answers/IDs stay private."""
import hashlib
import json
from pathlib import Path
import sys
import re
from latency_statistics import summarize_rows


def load(p):return json.loads(p.read_text(encoding='utf8'))
def counters(path):
    values={}
    for line in path.read_text().splitlines():
        m=re.match(r'(vllm:spec_decode_\w+_total)(\{[^}]*\})?\s+(\S+)',line)
        if m:values[m[1]+(m[2] or '')]=float(m[3])
    return values


def acceptance(root,label):
    a=counters(root/f'{label}-before-metrics.txt');b=counters(root/f'{label}-after-metrics.txt')
    d={k:b[k]-a.get(k,0) for k in b}
    def get(name):return sum(v for k,v in d.items() if k.split('{')[0]=='vllm:spec_decode_'+name+'_total')
    rounds=get('num_drafts');drafted=get('num_draft_tokens');accepted=get('num_accepted_tokens')
    assert rounds>0 and drafted==7*rounds and 0<=accepted<=drafted
    perpos=[v for k,v in d.items() if 'num_accepted_tokens_per_pos_total' in k]
    assert len(perpos)==7 and sum(perpos)==accepted
    return {'rounds':rounds,'drafted':drafted,'accepted':accepted,'accepted_per_round':accepted/rounds,
            'acceptance_fraction':accepted/drafted,'accepted_per_position':perpos}


def stats(root):
    r=root/'results';arms={};raw={}
    for label in ('baseline','ready2'):
        data=load(r/f'{label}.json');raw[label]=data
        summary=summarize_rows(data,load(r/f'{label}-window.json'))
        summary['acceptance']=acceptance(r,label)
        summary['sequence_sha256']=[hashlib.sha256(json.dumps(x['token_ids']).encode()).hexdigest() for x in data]
        summary['per_request']=[{k:x[k] for k in ('question_id','e2e_s','ttft_s','tpot_ms','output_tok_s','finish_reason')}|{'output_tokens':len(x['token_ids'])} for x in data]
        arms[label]=summary
    assert len(raw['baseline'])==len(raw['ready2'])==32
    assert all(a['token_ids']==b['token_ids'] for a,b in zip(raw['baseline'],raw['ready2']))
    a,b=[arms[k]['common'] for k in ('baseline','ready2')]
    result={'physical_card':6,'TP':1,'concurrency':1,'MTP':7,'weights':'BF16','ssm_state':'FP32',
            'same_round_dynamic_vocabulary':4096,'correctness':load(r/'correctness.json'),'arms':arms,
            'comparison':{'all32_sequences_equal':True,'faster_requests':sum(b['e2e_s']<a['e2e_s'] for a,b in zip(raw['baseline'],raw['ready2'])),
                          'throughput_change_pct':100*(b['output_throughput_by_request_e2e']/a['output_throughput_by_request_e2e']-1),
                          'e2e_change_pct':100*(b['request_e2e_sum_s']/a['request_e2e_sum_s']-1)},
            'limitations':['One sequential 32-request arm each in the same service; not repeated/interleaved trials.',
                           'Physical card 6 control is primary; historical card-7 throughput is context only.',
                           'Profiler durations are separate from unprofiled throughput.']}
    (root/'benchmark-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['comparison'],indent=2))


if __name__=='__main__':stats(Path(sys.argv[1]))
