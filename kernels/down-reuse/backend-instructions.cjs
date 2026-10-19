const fs=require('fs'),path=require('path');const out=process.argv[2];
function walk(p){return fs.readdirSync(p,{withFileTypes:true}).flatMap(x=>x.isDirectory()?walk(path.join(p,x.name)):[path.join(p,x.name)])}
const input=walk(path.join(out,'simulator')).find(x=>path.basename(x)==='visualize_data.bin');
const ws=new WebSocket('ws://127.0.0.1:9099');let id=2000;const p=new Map();
const timer=setTimeout(()=>{console.error('timeout');ws.close();process.exitCode=1},120000);
function rpc(moduleName,command,params){return new Promise(resolve=>{const n=++id;p.set(n,resolve);ws.send(JSON.stringify({id:n,type:'request',moduleName,command,params}))})}
ws.onmessage=async e=>{const m=JSON.parse(typeof e.data==='string'?e.data:await e.data.text());if(m.type==='response'&&p.has(m.requestId)){p.get(m.requestId)(m);p.delete(m.requestId)}};
ws.onopen=async()=>{try{
 const imported=await rpc('timeline','import/action',{projectName:'qwen38-down-pipeline-instructions',path:[input],projectAction:1,isConflict:true});
 fs.writeFileSync(path.join(out,'backend-simulator-import.json'),JSON.stringify(imported,null,2));console.log('IMPORT',imported.result,JSON.stringify(imported.body).slice(0,1000));
 await new Promise(r=>setTimeout(r,2000));
 for(const command of ['source/details/baseInfo','source/api/instructions/dynamic','source/api/instructions']){
  const data=await rpc('source',command,{coreName:'core0.cubecore0'});const file=command.replaceAll('/','-')+'.json';fs.writeFileSync(path.join(out,'backend-'+file),JSON.stringify(data,null,2));console.log(command,data.result,JSON.stringify(data.body).slice(0,1800));
 }
 }catch(e){console.error(e);process.exitCode=1}finally{clearTimeout(timer);ws.close()}};

