const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const elements = {};
function el(id) {
  if (!elements[id]) elements[id] = {id, textContent:'', value:'assistant', style:{}, attrs:{}, scrollHeight:800, clientHeight:200, scrollTop:0,
    classList:{toggle(){}}, listeners:{}, setAttribute(k,v){this.attrs[k]=v}, removeAttribute(k){delete this.attrs[k]},
    addEventListener(k,f){this.listeners[k]=f}};
  return elements[id];
}
let content='first log';
const context = {document:{getElementById:el}, performance:{now:()=>0}, requestAnimationFrame:f=>f(), setInterval:()=>{},
  fetch:async url=>({ok:true,json:async()=>url.startsWith('/api/log')?{text:content}:{token:'test',assistant:true,proxy:true,ollama:true,gpu:{name:'fixture',used:0,total:8192},models:[],operation:{running:false,message:''}}})};
vm.createContext(context);
vm.runInContext(fs.readFileSync('dashboard/app.js','utf8'),context);
(async()=>{
  await context.logs();
  assert.equal(el('log').scrollTop,800);
  assert.match(el('model-status').textContent,/没有模型驻留/);
  el('log').scrollTop=100; el('log').listeners.scroll();
  assert.equal(el('follow-log').attrs['aria-pressed'],'false');
  content='new log'; await context.logs();
  assert.equal(el('log').scrollTop,100);
  el('follow-log').listeners.click();
  assert.equal(el('log').scrollTop,800);
  assert.equal(el('follow-log').attrs['aria-pressed'],'true');
  console.log('dashboard follow and residency: OK');
})().catch(e=>{console.error(e);process.exitCode=1});
