const fs = require('fs'), path = require('path');
const [input, out, label, device = '7'] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const ws = new WebSocket('ws://127.0.0.1:9099');
let id = 25000;
const pending = new Map();
const timer = setTimeout(() => { ws.close(); process.exitCode = 1; }, 600000);
function rpc(moduleName, command, params) {
  return new Promise(resolve => {
    const requestId = ++id;
    pending.set(requestId, resolve);
    ws.send(JSON.stringify({ id: requestId, type: 'request', moduleName, command, params }));
  });
}
ws.onmessage = async e => {
  const m = JSON.parse(typeof e.data === 'string' ? e.data : await e.data.text());
  if (m.type === 'response' && pending.has(m.requestId)) {
    pending.get(m.requestId)(m);
    pending.delete(m.requestId);
  }
};
ws.onopen = async () => {
  try {
    const imported = await rpc('timeline', 'import/action', {
      projectName: 'qwen-dynamic-decode-profile-' + label, path: [input], projectAction: 1, isConflict: true
    });
    fs.writeFileSync(path.join(out, label + '-backend-import.json'), JSON.stringify(imported, null, 2));
    if (!imported.result) throw new Error(JSON.stringify(imported));
    await new Promise(resolve => setTimeout(resolve, 1500));
    const rank = imported.body.result[0];
    const stats = await rpc('operator', 'operator/statistic', {
      rankId: rank.rankId, deviceId: device, dbPath: rank.dbPath, group: 'Input Shape',
      topK: -1, current: 1, pageSize: 1000, orderBy: 'totalTime', order: 'descend', filters: [], isCompare: false
    });
    if (!stats.result) throw new Error(JSON.stringify(stats));
    fs.writeFileSync(path.join(out, label + '-backend-operators.json'), JSON.stringify(stats, null, 2));
    const rows = (stats.body.data || []).map(r => r.compare || r);
    const matmuls = rows.filter(r => /MatMul|Gemm/i.test(r.opType || '') || /matmul|gemm/i.test(r.opName || ''));
    fs.writeFileSync(path.join(out, label + '-backend-matmuls.json'), JSON.stringify(matmuls, null, 2));
    console.log(label, 'operator groups', rows.length, 'matmul groups', matmuls.length);
    console.log(JSON.stringify(matmuls.slice(0, 12)));
  } catch (e) { console.error(e); process.exitCode = 1; }
  finally { clearTimeout(timer); ws.close(); }
};
