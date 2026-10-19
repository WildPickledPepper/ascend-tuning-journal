"""Compare raw launch bytes without assigning undocumented tiling field names."""
import collections, hashlib, json, pathlib, struct, sys

root=pathlib.Path(sys.argv[1])
groups={}
for label,folder in [('micro','micro'),('model','model-trace')]:
    rows=[]
    for f in (root/folder).glob('launch-*.jsonl'):
        for line in f.read_text().splitlines():
            r=json.loads(line)
            b=bytes.fromhex(r['args_hex'])
            if len(b)!=200: continue
            offset=r.get('tiling_data_offset') or 40
            t=b[offset:offset+160]
            r['pointers']=[hex(p) for p in struct.unpack('<5Q',b[:40])]
            r['tiling_words']=list(struct.unpack('<40I',t))
            r['tiling_hex']=t.hex()
            r['tiling_hash']=hashlib.sha256(t).hexdigest()
            r['payload152_hash']=hashlib.sha256(t[:152]).hexdigest()
            r['M']=r['tiling_words'][1]
            r['N']=r['tiling_words'][2]
            r['K']=r['tiling_words'][0]
            rows.append(r)
    groups[label]=rows
result={'counts':{k:len(v) for k,v in groups.items()},'comparisons':[]}
for micro in groups['micro']:
    if micro['M']!=2:continue
    matches=[r for r in groups['model'] if (r['key'],r['M'],r['N'],r['K'])==(micro['key'],2,micro['N'],micro['K'])]
    details=[]
    for model in matches:
        diffs=[{'index':i,'offset':i*4,'micro':a,'model':b} for i,(a,b) in enumerate(zip(micro['tiling_words'],model['tiling_words'])) if a!=b]
        details.append({'pointers':model['pointers'],'blocks':model['blocks'],
                        'differences':diffs,'payload152_hash':model['payload152_hash']})
    result['comparisons'].append({'key':micro['key'],'M':2,'N':micro['N'],'K':micro['K'],
      'micro_pointers':micro['pointers'],'tiling_words':micro['tiling_words'],
      'matching_model_launches':len(matches),'details':details})
(root/'launch-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'counts':result['counts'],'pairs':[{k:r[k] for k in ['key','matching_model_launches']} for r in result['comparisons']]},indent=2))
