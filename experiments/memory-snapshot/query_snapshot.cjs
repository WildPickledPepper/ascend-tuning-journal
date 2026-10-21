// Import a private allocator snapshot and query compiled Insight without screenshots.
const fs = require('fs'), path = require('path');
const [input, output] = process.argv.slice(2);
fs.mkdirSync(output, {recursive: true});
const ws = new WebSocket('ws://127.0.0.1:9099');
let next = 83000;
const pending = new Map(), events = [];
const save = (name, value) => fs.writeFileSync(path.join(output, name+'.json'), JSON.stringify(value, null, 2));
function rpc(moduleName, command, params) {
  return new Promise((resolve, reject) => {
    const id = ++next;
    const timer = setTimeout(() => {pending.delete(id); reject(Error('RPC timeout: '+command));}, 120000);
    pending.set(id, message => {clearTimeout(timer); resolve(message);});
    ws.send(JSON.stringify({id, type:'request', moduleName, command, fileId:input, params}));
  });
}
ws.onmessage = async event => {
  const message = JSON.parse(typeof event.data==='string' ? event.data : await event.data.text());
  if (message.type === 'response' && pending.has(message.requestId)) {
    pending.get(message.requestId)(message); pending.delete(message.requestId);
  } else events.push(message);
};
ws.onopen = async () => {
  try {
    const imported = await rpc('timeline','import/action',{
      projectName:'qwen27b-memory-snapshot', path:[input],projectAction:1,isConflict:true});
    save('import', imported);
    if (!imported.result) throw Error(JSON.stringify(imported));
    const deadline = Date.now()+180000;
    let complete;
    while (!(complete = events.find(e => e.body?.module==='memsnapshot' && e.body.snapshotParsingComplete))) {
      const failure = events.find(e => e.result===false);
      if (failure) throw Error(JSON.stringify(failure));
      if (Date.now()>deadline) throw Error('Snapshot parsing timeout');
      await new Promise(resolve => setTimeout(resolve,1000));
    }
    save('parse-complete',complete);
    for (const [deviceId, metadata] of Object.entries(complete.body.snapshotSlices)) {
      for (const slice of metadata.slices) {
        if (!slice.ready) throw Error('Incomplete snapshot slice');
        const params = {deviceId, eventType:'BLOCK', sliceIndex:slice.index};
        const prefix = `device-${deviceId}-slice-${slice.index}`;
        for (const [name, command, extra] of [
          ['allocations','Memory/snapshot/allocations',{}],
          ['reserved','Memory/snapshot/allocationLines',{}],
          ['blocks','Memory/snapshot/blocks',{isTable:false,startTimestamp:slice.startEventId,endTimestamp:slice.endEventId,minSize:0,maxSize:2**50}],
          ['state','Memory/snapshot/state',{eventId:slice.endEventId}]
        ]) {
          const response = await rpc(complete.moduleName,command,{...params,...extra});
          save(prefix+'-'+name,response);
          if (!response.result) throw Error(JSON.stringify(response));
        }
      }
    }
    save('events',events);
    console.log('INSIGHT_SNAPSHOT_QUERIES_COMPLETE',Object.keys(complete.body.deviceIds));
  } catch (error) {console.error(error);save('events',events); process.exitCode=1;}
  finally {ws.close();}
};
