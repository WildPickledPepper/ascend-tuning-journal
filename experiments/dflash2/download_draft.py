"""Download pinned public BF16 draft weights and verify the upstream LFS hash."""
import concurrent.futures, datetime, hashlib, json, pathlib, shutil, struct, sys, time, urllib.request

root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
revision = '015e795645c74b1a0eeef3b570031fb62e769bc5'
expected_sha = '67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'
expected_size = 3848817896
modelscope_revision = '5cfe71fd56ed83d5ca27f04c1345cc7d862bbe4f'
weights_url = f'https://modelscope.cn/models/incoai/Qwen3.8-27B-DFlash2/resolve/{modelscope_revision}/model.safetensors'
base = f'https://hf-mirror.com/incoai/Qwen3.8-27B-DFlash2/resolve/{revision}/'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ['config.json', 'model.safetensors']:
    target = root / name
    if target.exists() and (name != 'model.safetensors' or target.stat().st_size == expected_size):
        continue
    partial = root / (name + '.partial')
    if name == 'model.safetensors':
        # ModelScope publishes the same SHA256. Four bounded Range streams avoid
        # the slow single-stream mirror route; final upstream hash remains mandatory.
        prefix = partial.stat().st_size if partial.exists() else 0
        parts = root / ('parts-' + str(prefix)); parts.mkdir(exist_ok=True)
        ranges = [(start, min(start+64*1024*1024, expected_size)-1)
                  for start in range(prefix, expected_size, 64*1024*1024)]
        def fetch(bounds):
            start,end = bounds; part=parts/f'{start}-{end}'
            if part.exists() and part.stat().st_size == end-start+1:return part
            for attempt in range(3):
                try:
                    req=urllib.request.Request(weights_url,headers={'User-Agent':'Mozilla/5.0','Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=120) as response:
                        assert response.status==206
                        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                        with part.open('wb') as f:shutil.copyfileobj(response,f,8*1024*1024)
                    assert part.stat().st_size==end-start+1
                    print('RANGE_COMPLETE',start,end,flush=True);return part
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(fetch,ranges))
        merged=root/(name+'.assembled')
        with merged.open('wb') as out:
            if prefix:
                with partial.open('rb') as src:shutil.copyfileobj(src,out,16*1024*1024)
            for start,end in ranges:
                with (parts/f'{start}-{end}').open('rb') as src:shutil.copyfileobj(src,out,16*1024*1024)
        assert merged.stat().st_size==expected_size
        merged.replace(target)
        if partial.exists():partial.unlink()
        for part in parts.iterdir():part.unlink()
        parts.rmdir()
        continue
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent':'Mozilla/5.0'}
        if offset: headers['Range'] = f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(base+name,headers=headers),timeout=120) as response:
                append = bool(offset and response.status == 206)
                if append: assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                done = offset if append else 0
                report = time.monotonic()
                with partial.open('ab' if append else 'wb') as out:
                    while chunk := response.read(8*1024*1024):
                        out.write(chunk); done += len(chunk)
                        if time.monotonic()-report>15:
                            print(name, done, flush=True); report=time.monotonic()
            if name=='model.safetensors': assert partial.stat().st_size==expected_size
            partial.replace(target); break
        except Exception as exc:
            print('DOWNLOAD_RETRY',name,attempt,type(exc).__name__,flush=True)
            if attempt==2: raise
            time.sleep(2)
file=root/'model.safetensors'; digest=hashlib.sha256()
with file.open('rb') as f:
    while chunk:=f.read(16*1024*1024):digest.update(chunk)
assert digest.hexdigest()==expected_sha
with file.open('rb') as f:
    size=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(size))
tensors=[v for k,v in header.items() if k!='__metadata__']
assert all(t['dtype']=='BF16' for t in tensors)
config=json.loads((root/'config.json').read_text())
assert config['architectures']==['DFlash2DraftModel']
result={'model':'incoai/Qwen3.8-27B-DFlash2','revision':revision,'weight_size_bytes':file.stat().st_size,
        'weight_download_mirror':'ModelScope','modelscope_revision':modelscope_revision,
        'weight_sha256':digest.hexdigest(),'all_weights_bf16':True,'tensor_count':len(tensors),
        'num_layers':config['num_hidden_layers'],'block_size':config['dflash_config']['block_size'],
        'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'download-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
