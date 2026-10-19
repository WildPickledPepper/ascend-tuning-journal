const fs=require('fs'),path=require('path');
const [input,out,label]=process.argv.slice(2);
const ws=new WebSocket('ws://127.0.0.1:9099');
let id=70000;const pending=new Map();
const timer=setTimeout(()=>{ws.close();process.exitCode=1},180000);
function rpc(moduleName,command,params){return new Promise(resolve=>{const n=++id;pending.set(n,resolve);ws.send(JSON.stringify({id:n,type:'request',moduleName,command,params}))})}
ws.onmessage=async e=>{const m=JSON.parse(typeof e.data==='string'?e.data:await e.data.text());if(m.type==='response'&&pending.has(m.requestId)){pending.get(m.requestId)(m);pending.delete(m.requestId)}};
ws.onopen=async()=>{try{
 const imported=await rpc('timeline','import/action',{projectName:'nz-runtime-'+label,path:[input],projectAction:1,isConflict:true});
 fs.writeFileSync(path.join(out,label+'-binary-import.json'),JSON.stringify(imported,null,2));
 if(!imported.result)throw new Error(JSON.stringify(imported));
 for(const [command,params] of [
  ['source/details/baseInfo',{}],
  ['source/details/memoryGraph',{blockId:'0'}],
  ['source/api/instructions/dynamic',{coreName:'core0.cubecore0'}]]){
   const result=await rpc('source',command,params);
   fs.writeFileSync(path.join(out,label+'-'+command.replaceAll('/','-')+'.json'),JSON.stringify(result,null,2));
   console.log(label,command,result.result,JSON.stringify(result.body).slice(0,700));
 }
 }catch(e){console.error(e);process.exitCode=1}finally{clearTimeout(timer);ws.close()}};
