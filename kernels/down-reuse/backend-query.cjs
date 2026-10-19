const fs=require('fs'),path=require('path');
const root=process.argv[2];
function walk(p){return fs.readdirSync(p,{withFileTypes:true}).flatMap(x=>x.isDirectory()?walk(path.join(p,x.name)):[path.join(p,x.name)])}
const bins=walk(path.join(root,'reports')).filter(x=>path.basename(x)==='visualize_data.bin');
const ws=new WebSocket('ws://127.0.0.1:9099');let id=8000;const pending=new Map();
const timer=setTimeout(()=>{ws.close();process.exitCode=1},90000);
function rpc(moduleName,command,params){return new Promise(resolve=>{const n=++id;pending.set(n,resolve);ws.send(JSON.stringify({id:n,type:'request',moduleName,command,params}))})}
ws.onmessage=async e=>{const m=JSON.parse(typeof e.data==='string'?e.data:await e.data.text());if(m.type==='response'&&pending.has(m.requestId)){pending.get(m.requestId)(m);pending.delete(m.requestId)}};
ws.onopen=async()=>{try{for(const input of bins){
 const name=input.includes('cann-reference-profile')?'cann':input.includes('initial-profile')?'baseline':input.includes('pipeline-profile')?'pipeline':'reuse';
 const result={input};
 result.import=await rpc('timeline','import/action',{projectName:'down-'+name,path:[input],projectAction:1,isConflict:true});
 if(!result.import.result)throw new Error(JSON.stringify(result.import));
 await new Promise(r=>setTimeout(r,800));
 result.baseInfo=await rpc('source','source/details/baseInfo',{});
 result.cores=[];
 for(let block=0;block<23;block++){
  const response=await rpc('source','source/details/memoryGraph',{blockId:String(block),isCompared:false});
  if(!response.result)throw new Error(JSON.stringify(response));
  result.cores.push(response);
 }
 fs.writeFileSync(path.join(root,'backend-'+name+'.json'),JSON.stringify(result,null,2));
 console.log(name,'queried',result.cores.length,'cores',JSON.stringify(result.baseInfo.body));
}}catch(e){console.error(e);process.exitCode=1}finally{clearTimeout(timer);ws.close()}};
