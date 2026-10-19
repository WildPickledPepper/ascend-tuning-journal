const fs=require('fs'),path=require('path');
const [input,out,label]=process.argv.slice(2);fs.mkdirSync(out,{recursive:true});
const ws=new WebSocket('ws://127.0.0.1:9099');let id=12000;const pending=new Map();
const timer=setTimeout(()=>{ws.close();process.exitCode=1},90000);
function rpc(moduleName,command,params){return new Promise(resolve=>{const n=++id;pending.set(n,resolve);ws.send(JSON.stringify({id:n,type:'request',moduleName,command,params}))})}
ws.onmessage=async e=>{const m=JSON.parse(typeof e.data==='string'?e.data:await e.data.text());if(m.type==='response'&&pending.has(m.requestId)){pending.get(m.requestId)(m);pending.delete(m.requestId)}};
ws.onopen=async()=>{try{
 const imported=await rpc('timeline','import/action',{projectName:'down-model-'+label,path:[input],projectAction:1,isConflict:true});
 fs.writeFileSync(path.join(out,label+'-backend-import.json'),JSON.stringify(imported,null,2));
 if(!imported.result)throw new Error(JSON.stringify(imported));
 await new Promise(r=>setTimeout(r,1200));
 const rank=imported.body.result[0];
 const base={rankId:rank.rankId,deviceId:'7',dbPath:rank.dbPath,group:'Input Shape',topK:-1,current:1,pageSize:1000,orderBy:'totalTime',order:'descend',filters:[],isCompare:false};
 const stats=await rpc('operator','operator/statistic',base);
 if(!stats.result)throw new Error(JSON.stringify(stats));
 fs.writeFileSync(path.join(out,label+'-backend-operators.json'),JSON.stringify(stats,null,2));
 const selected=(stats.body.data||[]).map(r=>r.compare||r).filter(r=>JSON.stringify(r).includes('down_pipeline')||JSON.stringify(r).includes('DownReuse')||JSON.stringify(r).includes('98513')||(r.inputShape||'').includes('5120,17408'));
 fs.writeFileSync(path.join(out,label+'-backend-down-summary.json'),JSON.stringify(selected,null,2));
 for(const row of selected){
  if(Number(row.count)!==960)continue;
  const detail=await rpc('operator','operator/more_info',{...base,pageSize:1000,
    orderBy:'startTime',order:'ascend',opType:row.opType,opName:row.opName,
    shape:row.inputShape,accCore:row.accCore});
  fs.writeFileSync(path.join(out,label+'-backend-down-decode-details.json'),JSON.stringify(detail,null,2));
 }
 console.log(label,JSON.stringify(selected));
 }catch(e){console.error(e);process.exitCode=1}finally{clearTimeout(timer);ws.close()}};
