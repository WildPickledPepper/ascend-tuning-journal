// Read-only timeline query against an already imported compiled Insight backend.
// Usage: node backend.cjs IMPORT_RECEIPT PRIVATE_OUTPUT
const fs = require('fs'),path=require('path');
const [receipt,out]=process.argv.slice(2);
const rank=JSON.parse(fs.readFileSync(receipt)).body.result[0];
const ws=new WebSocket('ws://127.0.0.1:9099');let id=61000;
const pending=new Map();
const timer=setTimeout(()=>{ws.close();process.exitCode=1;},45000);
function rpc(command,params){return new Promise(resolve=>{const key=++id;pending.set(key,resolve);ws.send(JSON.stringify({id:key,type:'request',moduleName:'timeline',command,params}));});}
ws.onmessage=async e=>{const m=JSON.parse(typeof e.data==='string'?e.data:await e.data.text());if(m.type==='response'&&pending.has(m.requestId)){pending.get(m.requestId)(m);pending.delete(m.requestId);}};
ws.onopen=async()=>{try{
 const result=await rpc('search/all/slices',{rankId:rank.rankId,dbPath:rank.dbPath,searchContent:'aclnnMatmul_MatMulCommon_MatMulV2',isMatchCase:true,isMatchExact:true,orderBy:'startTime',order:'ascend',current:1,pageSize:20,metadataList:[]});
 fs.writeFileSync(path.join(out,'timeline-search.json'),JSON.stringify(result,null,2));
 if(!result.result)throw Error('Timeline search failed');
 const windowPath=path.join(out,'timeline-window-request.json');
 if(fs.existsSync(windowPath)){
  const slices=await rpc('unit/threadTraces',JSON.parse(fs.readFileSync(windowPath)));
  fs.writeFileSync(path.join(out,'timeline-boundary.json'),JSON.stringify(slices,null,2));
  if(!slices.result)throw Error('Timeline window failed');
  console.log(JSON.stringify(slices).slice(0,4500));
 }else console.log('Timeline search returned',result.body.searchAllSlicesDetails.length,'slices');
}catch(e){console.error(e);process.exitCode=1;}finally{clearTimeout(timer);ws.close();}};
