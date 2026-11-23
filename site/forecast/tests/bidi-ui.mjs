// Integração real com Firefox WebDriver BiDi; requer site servido por HTTP.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const ws=new WebSocket(process.env.FORECAST_BIDI??'ws://127.0.0.1:9222/session');
const url=process.env.FORECAST_URL??'http://127.0.0.1:8765/projeto4-irn/';
let sequence=0,context;const pending=new Map(),logs=[];
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
ws.onmessage=event=>{const message=JSON.parse(event.data),promise=pending.get(message.id);if(message.method==='log.entryAdded')logs.push(message.params);if(promise){pending.delete(message.id);clearTimeout(promise.timer);message.type==='error'?promise.reject(Error(JSON.stringify(message))):promise.resolve(message.result);}};
function call(method,params){return new Promise((resolve,reject)=>{const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error('Timeout: '+method));},30000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await call('script.evaluate',{expression,target:{context},awaitPromise:true});if(r.type==='exception')throw Error(JSON.stringify(r));return r.result.type==='null'?null:r.result.value;}
async function wait(expression){for(let i=0;i<250;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,120));}throw Error('Condição não satisfeita: '+expression);}
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function screenshot(name){const shot=await call('browsingContext.captureScreenshot',{context,origin:'document'});fs.writeFileSync(name,Buffer.from(shot.data,'base64'));}
async function ready(){await wait('window.__forecastDemo?.ready && !!window.__forecastDemo.state.result && !window.__forecastDemo.state.processing');}
try{
  await call('session.new',{capabilities:{}});await call('session.subscribe',{events:['log.entryAdded']});
  context=(await call('browsingContext.create',{type:'tab'})).context;
  await call('browsingContext.setViewport',{context,viewport:{width:1440,height:1100}});
  await call('browsingContext.navigate',{context,url,wait:'complete'});await ready();
  await evaluate('window.__forecastDemo.stop()');
  assert.equal(await evaluate("document.getElementById('modelStatus').textContent.includes('GRU pronta')"),true);
  assert.equal(await evaluate('window.__forecastDemo.state.result.models.gru.points.length'),15);
  assert.equal(await evaluate('Object.keys(window.__forecastDemo.state.result.models).join()'),'gru');
  assert.equal(await evaluate("document.querySelectorAll('.canvas-card').length"),2);
  assert.ok((await evaluate("document.getElementById('gruADE').textContent")).includes('px'));
  assert.ok((await evaluate("document.getElementById('gruFDE').textContent")).includes('px'));
  const reportCases=JSON.parse(await evaluate(`JSON.stringify(window.__forecastDemo.state.examples
    .map((e,index)=>({index,report_case:e.report_case,cut:e.default_cut,k:e.default_horizon,ade:e.ade_gru}))
    .filter(e=>e.report_case))`));
  assert.equal(reportCases.length,6);
  for(const example of reportCases){
    await evaluate(`document.getElementById('exampleSelect').value='${example.index}';
      document.getElementById('exampleSelect').dispatchEvent(new Event('change'))`);
    await ready();
    assert.equal(await evaluate('window.__forecastDemo.state.result.contextCount'),example.cut);
    assert.equal(await evaluate('window.__forecastDemo.state.result.horizon'),example.k);
    assert.ok(Math.abs(await evaluate('window.__forecastDemo.state.result.models.gru.metrics.ade')-example.ade)<.01);
  }
  await evaluate('window.__forecastDemo.loadExample(0,{autoplay:false})');await ready();
  await evaluate("document.getElementById('horizon').value='5';document.getElementById('horizon').dispatchEvent(new Event('change'))");await ready();
  await evaluate('window.__forecastDemo.setPrefix(16,{autoplay:false})');await ready();
  assert.equal(await evaluate('window.__forecastDemo.state.result.horizon'),5);
  assert.equal(await evaluate('window.__forecastDemo.state.prefix'),16);
  await evaluate("window.__forecastDemo.stop();window.__forecastDemo.state.reveal=0;window.__forecastDemo.render();document.getElementById('step').click()");
  assert.equal(await evaluate('window.__forecastDemo.state.reveal'),1);
  await evaluate("window.__forecastDemo.state.reveal=0;window.__forecastDemo.render();document.getElementById('speed').value='600';document.getElementById('play').click()");
  await pause(250);assert.equal(await evaluate('window.__forecastDemo.state.reveal'),0,'Reprodução lenta avançou cedo demais.');
  await wait('window.__forecastDemo.state.reveal===1');await evaluate("document.getElementById('play').click()");
  const revealed=await evaluate('window.__forecastDemo.state.reveal');await pause(700);assert.equal(await evaluate('window.__forecastDemo.state.reveal'),revealed,'Pausa não foi respeitada.');
  await evaluate("document.getElementById('all').click()");assert.equal(await evaluate('window.__forecastDemo.state.reveal'),5);
  const before=await evaluate('JSON.stringify(window.__forecastDemo.state.result.models)');
  await evaluate("document.getElementById('showTruth').click()");assert.equal(await evaluate('JSON.stringify(window.__forecastDemo.state.result.models)'),before,'Mostrar/ocultar o futuro alterou as previsões.');
  await evaluate("document.getElementById('showTruth').click();document.getElementById('horizon').value='30';document.getElementById('horizon').dispatchEvent(new Event('change'))");await ready();
  await evaluate('window.__forecastDemo.setPrefix(32,{autoplay:false})');await ready();
  assert.equal(await evaluate('window.__forecastDemo.state.result.horizon'),30);
  const original=JSON.parse(await evaluate('JSON.stringify(window.__forecastDemo.state.result.models)'));
  await evaluate("document.getElementById('translation').value='1';document.getElementById('translation').dispatchEvent(new Event('change'))");await ready();
  const shifted=JSON.parse(await evaluate('JSON.stringify(window.__forecastDemo.state.result.models)'));
  for(const model of ['gru']){
    assert.ok(Math.abs(shifted[model].metrics.ade-original[model].metrics.ade)<.02,model+' mudou demais sob translação.');
    for(let i=0;i<30;i++)assert.ok(Math.abs(shifted[model].points[i][0]-original[model].points[i][0]-120)<.03);
  }
  await evaluate("window.__forecastDemo.resetPerturbations();");await ready();
  await evaluate("document.getElementById('scale').value='1.25';document.getElementById('scale').dispatchEvent(new Event('change'))");await ready();
  assert.equal(await evaluate('window.__forecastDemo.state.result.perturb.scale'),1.25);
  await evaluate("window.__forecastDemo.resetPerturbations();");await ready();
  await evaluate("document.getElementById('shuffle').checked=true;document.getElementById('shuffle').dispatchEvent(new Event('change'))");await ready();
  assert.equal(await evaluate('window.__forecastDemo.state.result.perturb.shuffle'),true);
  await evaluate("document.getElementById('ignorePen').checked=true;document.getElementById('ignorePen').dispatchEvent(new Event('change'))");await ready();
  assert.equal(await evaluate('window.__forecastDemo.state.result.perturb.ignorePen'),true);
  await evaluate('window.__forecastDemo.resetPerturbations()');await ready();
  await evaluate("document.getElementById('category').value=String((window.__forecastDemo.state.label+1)%12);document.getElementById('category').dispatchEvent(new Event('change'))");await ready();
  assert.ok((await evaluate("document.getElementById('metricNote').textContent")).includes('Cenário alterado'));
  await evaluate("document.getElementById('all').click()");
  const pixels=await evaluate("(()=>{const d=document.getElementById('gruCanvas').getContext('2d').getImageData(0,0,600,600).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]>150&&d[i+1]<150&&d[i+2]<130&&d[i+3]>100)n++;return n})()");
  assert.ok(pixels>30,'O futuro da GRU não apareceu no canvas.');
  await screenshot('/tmp/projeto4-forecast-desktop.png');
  await evaluate("document.getElementById('drawMode').click();document.getElementById('clear').click()");
  const rect=JSON.parse(await evaluate("JSON.stringify((()=>{const r=document.getElementById('inputCanvas').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})())"));
  const actions=[];
  for(let i=0;i<=48;i++){const a=i*Math.PI*2/48;actions.push({type:'pointerMove',x:Math.round(rect.x+rect.w*.5+Math.cos(a)*rect.w*.31),y:Math.round(rect.y+rect.h*.5+Math.sin(a)*rect.h*.31),duration:20,origin:'viewport'});if(!i)actions.push({type:'pointerDown',button:0});}actions.push({type:'pointerUp',button:0});
  await call('input.performActions',{context,actions:[{type:'pointer',id:'forecast-mouse',parameters:{pointerType:'mouse'},actions}]});await ready();
  assert.equal(await evaluate('window.__forecastDemo.state.mode'),'draw');assert.equal(await evaluate('window.__forecastDemo.state.result.models.gru.metrics'),null);
  assert.ok((await evaluate("document.getElementById('metricNote').textContent")).includes('Sem futuro real'));
  await evaluate("document.getElementById('undo').click()");assert.equal(await evaluate('window.__forecastDemo.state.points.length'),0);
  await evaluate('window.__forecastDemo.loadExample(0,{autoplay:false})');await ready();
  await call('browsingContext.setViewport',{context,viewport:{width:390,height:844}});
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'Layout móvel excedeu a largura.');
  await screenshot('/tmp/projeto4-forecast-mobile.png');
  const errors=logs.filter(l=>l.level==='error'&&l.source?.context===context);assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({passed:true,report_cases_ui_verified:6,url,browser:'Firefox BiDi',checks:['inferência somente com GRU','prefixo16/32','horizonte5/30','passo exato','pausa e ritmo lento','futuro real não altera previsão','translação','escala','permutação do contexto','pen-up','categoria fornecida','desenho por mouse','sem métricas sem verdade','desfazer','390px sem overflow'],gru_pixels:pixels}));
}catch(error){try{console.log(await evaluate("JSON.stringify({status:document.getElementById('modelStatus').textContent,note:document.getElementById('predictionNote').textContent,ready:window.__forecastDemo?.ready})"));await screenshot('/tmp/projeto4-forecast-failure.png');}catch{}throw error;}
finally{try{await call('session.end',{});}catch{}ws.close();}
