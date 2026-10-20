const fs=require('fs'),path=require('path');
const [input,out,label]=process.argv.slice(2);
const ws=new WebSocket('ws://127.0.0.1:9099');
let id=97000;const pending=new Map();
const timer=setTimeout(()=>{ws.close();process.exitCode=1},120000);
function rpc(moduleName,command,params){return new Promise(resolve=>{
  const n=++id;pending.set(n,resolve);
  ws.send(JSON.stringify({id:n,type:'request',moduleName,command,params}));
});}
ws.onmessage=async e=>{
  const m=JSON.parse(typeof e.data==='string'?e.data:await e.data.text());
  if(m.type==='response'&&pending.has(m.requestId)){
    pending.get(m.requestId)(m);pending.delete(m.requestId);
  }
};
ws.onopen=async()=>{try{
  const imported=await rpc('timeline','import/action',{
    projectName:'mte2-memory-'+label,path:[input],projectAction:1,isConflict:true});
  if(!imported.result)throw new Error(JSON.stringify(imported));
  const base=await rpc('source','source/details/baseInfo',{});
  if(!base.result)throw new Error(JSON.stringify(base));
  const cores=[];
  for(let block=0;block<23;block++){
    const result=await rpc('source','source/details/memoryGraph',{blockId:String(block)});
    if(!result.result||result.body.coreMemory.length!==1)throw new Error(JSON.stringify(result));
    cores.push(result.body.coreMemory[0]);
  }
  fs.writeFileSync(path.join(out,label+'-all-memory.json'),JSON.stringify({base,cores},null,2));
  console.log(label,'validated replies for',cores.length,'blocks');
}catch(e){console.error(e);process.exitCode=1}finally{clearTimeout(timer);ws.close()}};
