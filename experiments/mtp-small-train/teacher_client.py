"""Generate teacher responses from train-only prompts, then capture post-norm states."""
import datetime,hashlib,json,time
from pathlib import Path
import requests
root=Path(__file__).parent;url='http://127.0.0.1:8040'
def post(path,payload,timeout=900):
 r=requests.post(url+path,json=payload,timeout=timeout);r.raise_for_status();return r.json()
def rpc(method,**kw):
 r=post('/collective_rpc',{'method':method,'kwargs':kw});result,=r['results'];return result
rows=[json.loads(x) for x in (root/'selected.jsonl').read_text().splitlines()]
assert len(rows)==72 and all(r['source_split']=='train' for r in rows)
assert json.loads((root/'data-manifest.json').read_text())['selected_test_overlap']==0
(root/'teacher-records').mkdir(exist_ok=True)
progress={'phase':'generation','total':len(rows),'generated':0,'captured':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save(): (root/'teacher-progress.json').write_text(json.dumps(progress,indent=2))
save()
for row in rows:
 i=row['sample_id'];f=root/'teacher-records'/(str(i)+'.json')
 if f.exists():continue
 messages=[{'role':'user','content':'Question: '+row['question']+'\nAnswer:'}]
 tokenized=post('/tokenize',{'model':'Qwen3.8-27B','messages':messages,'add_generation_prompt':True})
 prompt=tokenized['tokens'];assert len(prompt)<450
 payload={'model':'Qwen3.8-27B','messages':messages,'temperature':0,'top_p':1,'top_k':1,'max_tokens':min(256,500-len(prompt)),'return_token_ids':True}
 response=post('/v1/chat/completions',payload)
 ids=response['choices'][0]['token_ids'];assert ids and response['usage']['prompt_tokens']==len(prompt)
 record={'sample_id':i,'split':row['split'],'source_split':'train','question_sha256':row['question_sha256'],'prompt_ids':prompt,'generated_ids':ids,'request':payload,'response':response}
 f.write_text(json.dumps(record,ensure_ascii=False))
 progress.update(generated=i+1,output_tokens=sum(len(json.loads(x.read_text())['generated_ids']) for x in (root/'teacher-records').glob('*.json')));save()
 print('TEACHER_GENERATED',i+1,len(ids),flush=True)
progress['phase']='capture';save()
for row in rows:
 i=row['sample_id'];record=json.loads((root/'teacher-records'/(str(i)+'.json')).read_text());ids=record['prompt_ids']+record['generated_ids'];mask=[0]*len(record['prompt_ids'])+[1]*len(record['generated_ids']);assert len(ids)<=512
 rpc('prepare_capture',sample_id=i,token_ids=ids,loss_mask=mask,split=row['split'])
 result=post('/v1/completions',{'model':'Qwen3.8-27B','prompt':ids,'temperature':0,'top_p':1,'top_k':1,'max_tokens':1,'return_token_ids':True})
 captured=rpc('capture_result');next_ids=result['choices'][0].get('token_ids') or []
 if next_ids:assert next_ids[0]==captured['next_token_argmax'],(i,next_ids,captured)
 else:assert captured['next_token_argmax'] in [248044,248045]
 captured.update(greedy_next_token_verified=True)
 (root/'features'/row['split']/(str(i)+'.json')).write_text(json.dumps(captured,indent=2))
 progress.update(captured=i+1);save();print('FEATURE_CAPTURED',i+1,len(ids),flush=True)
progress.update(phase='complete',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
