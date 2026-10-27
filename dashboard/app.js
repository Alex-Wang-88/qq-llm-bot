let token = '', busy = false, nextAction = null, progress = null;
let followLog = true, logRequest = 0, programmaticLogScroll = false;
const $ = id => document.getElementById(id);
function setLogFollow(value) {
  followLog=value;
  $('follow-log').setAttribute('aria-pressed',String(value));
  $('follow-log').textContent=value?'● 跟随最新':'跳至最新 ↓';
  if(value) scrollLogToBottom();
}
function scrollLogToBottom() {
  const el=$('log'); programmaticLogScroll=true;
  el.scrollTop=el.scrollHeight;
  requestAnimationFrame(()=>{programmaticLogScroll=false});
}
function drawProgress() {
  if (!progress || !busy) return;
  const elapsed=progress.elapsed+(performance.now()-progress.received)/1000;
  const remaining=Math.ceil(progress.estimate-elapsed);
  const percent=Math.min(95,Math.max(0,Math.floor(elapsed/progress.estimate*100)));
  $('operation-label').textContent=(progress.action==='stop'?'正在关闭':'正在启动')+' · 预计进度';
  $('operation-time').textContent=remaining>0?'预计还需约 '+remaining+' 秒':'比预计稍慢，请继续等待';
  $('operation-fill').style.width=percent+'%';
  $('operation-meter').setAttribute('aria-valuenow',percent);
  $('operation-meter').setAttribute('aria-valuetext',remaining>0?'预计还需约 '+remaining+' 秒':'比预计稍慢，等待实际完成');
}
async function status() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    if (!response.ok) throw Error('状态请求失败');
    const s = await response.json(); token = s.token; busy = s.operation.running;
    for (const name of ['assistant','proxy','ollama']) { const el=$(name); el.textContent=s[name]?'运行中':'未运行'; el.classList.toggle('on',s[name]); }
    nextAction = s.assistant ? 'stop' : 'start';
    $('headline').textContent=busy?'正在处理…':s.assistant?(s.proxy?'助手正在运行':'助手部分运行'):'助手已关闭';
    $('description').textContent=busy?'操作可能需要一些时间，请稍候。':s.assistant?(s.proxy?'服务端口已连接，点击电源按钮关闭。':'代理未运行，请检查服务日志。'):'需要使用时，点击电源按钮启动。';
    $('power').disabled=busy; $('power').classList.toggle('on',s.assistant); $('power').classList.toggle('busy',busy);
    $('power-label').textContent=busy?'正在处理':s.assistant?'点击关闭':'点击启动';
    $('power').setAttribute('aria-label',busy?'正在处理':s.assistant?'关闭助手':'启动助手');
    $('message').textContent=s.operation.message;
    $('operation-progress').hidden=!busy;
    progress=busy?{action:s.operation.action,estimate:s.operation.estimate||35,elapsed:s.operation.elapsed||0,received:performance.now()}:null;
    drawProgress();
    const g=s.gpu;
    if(g) { const pct=Math.min(100,Math.round(g.used/g.total*100)); $('gpu-value').textContent=(g.used/1024).toFixed(2)+' / '+(g.total/1024).toFixed(2)+' GiB'; $('gpu-name').textContent=g.name; $('gpu-percent').textContent=pct+'%'; $('gpu-fill').style.width=pct+'%'; $('gpu-meter').setAttribute('aria-valuenow',pct); }
    else { $('gpu-value').textContent='暂不可用'; $('gpu-name').textContent='无法读取显卡数据'; $('gpu-percent').textContent='—'; $('gpu-fill').style.width='0%'; $('gpu-meter').removeAttribute('aria-valuenow'); }
    const models=s.models;
    if(!Array.isArray(models)) $('model-status').textContent='模型驻留状态暂不可用';
    else if(!models.length) $('model-status').textContent='当前没有模型驻留；需要时会自动加载';
    else { const bytes=models.reduce((sum,m)=>sum+(Number(m.size_vram)||0),0); $('model-status').textContent='已加载 '+models.map(m=>m.name).join('、')+(bytes?' · 模型显存约 '+(bytes/1073741824).toFixed(2)+' GiB':''); }
  } catch (_) { nextAction=null; busy=false; progress=null; $('operation-progress').hidden=true; $('headline').textContent='无法连接控制台'; $('description').textContent='请检查本机控制台服务。'; $('power').disabled=true; $('power-label').textContent='无法连接'; }
}
async function logs() {
  const request=++logRequest, name=$('log-type').value;
  try {
    const r=await fetch('/api/log?name='+encodeURIComponent(name),{cache:'no-store'});
    if(!r.ok) throw Error();
    const data=await r.json();
    if(request!==logRequest || name!==$('log-type').value) return;
    const el=$('log');
    if(el.textContent!==data.text) { const oldTop=el.scrollTop; el.textContent=data.text; if(!followLog) el.scrollTop=oldTop; }
    if(followLog) scrollLogToBottom();
  } catch (_) { if(request===logRequest) $('log').textContent='读取日志失败。'; }
}
async function action(name) { if (!name) return; $('power').disabled=true; try { const r=await fetch('/api/'+name,{method:'POST',headers:{'X-Control-Token':token}}); if(!r.ok) throw Error('操作未被接受'); await status(); } catch (e) { await status(); $('message').textContent=e.message; } }
$('power').addEventListener('click',()=>action(nextAction)); $('refresh').addEventListener('click',()=>{status();logs()});
$('log-type').addEventListener('change',()=>{setLogFollow(true);logs()});
$('follow-log').addEventListener('click',()=>setLogFollow(!followLog));
$('log').addEventListener('scroll',()=>{const el=$('log'); if(!programmaticLogScroll&&followLog&&el.scrollHeight-el.clientHeight-el.scrollTop>30) setLogFollow(false)});
status();logs();setInterval(drawProgress,1000);setInterval(()=>{status();logs()},3000);
