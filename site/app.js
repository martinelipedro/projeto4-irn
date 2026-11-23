import {samplePoints} from './recognition.js';
const $=id=>document.getElementById(id);
const state={ready:false,mode:'example',examples:[],example:0,cut:0,points:[],raw:[],result:null,responseRevision:-1,processing:false,pending:false,revision:0,playing:false,timer:null,options:{reverse:false,hidePen:false,hideTime:false},history:[],labels:[]};
const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});let sequence=0;const pending=new Map();
worker.onmessage=event=>{const message=event.data,request=pending.get(message.id);if(!request)return;pending.delete(message.id);message.type==='error'?request.reject(Error(message.message)):request.resolve(message);};
worker.onerror=event=>{for(const request of pending.values())request.reject(Error(event.message));pending.clear();showError(event.message);};
function call(type,body={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});worker.postMessage({id,type,...body});});}
const pct=(x,digits=0)=>(100*x).toLocaleString('pt-BR',{maximumFractionDigits:digits,minimumFractionDigits:digits})+'%';
const example=()=>state.examples[state.example];
const observed=()=>state.mode==='example'?(example()?.points??[]).slice(0,state.cut):state.points;
const top=p=>p.map((value,index)=>({value,index})).sort((a,b)=>b.value-a.value).slice(0,3);
function status(text,type=''){const node=$('modelStatus');node.textContent=text;node.className='status '+type;}
function showError(message){status('Não foi possível concluir a análise.','error');$('predictionNote').textContent=message;console.error(message);}
function stop(){state.playing=false;clearTimeout(state.timer);state.timer=null;$('play').textContent='Reproduzir';$('play').setAttribute('aria-pressed','false');}
function resetPerturbations(){state.options={reverse:false,hidePen:false,hideTime:false};$('hidePen').checked=false;$('hideTime').checked=false;$('reverse').textContent='Comparar ordem dos traços';}
function changed(){state.revision++;state.pending=true;render();return pump();}
async function pump(){
 if(!state.ready||state.processing)return;
 const points=observed();
 if(points.length<2){state.pending=false;state.result=null;render();return;}
 state.processing=true;state.pending=false;const revision=state.revision;
 try{
  const result=await call('recognize',{points,options:state.options});
  if(revision===state.revision){state.result=result;state.responseRevision=revision;if(state.mode==='draw'&&!Object.values(state.options).some(Boolean)){const row={cut:points.length,gru:Math.max(...result.probabilities.gru),mlp:Math.max(...result.probabilities.mlp)};state.history=state.history.filter(r=>r.cut<row.cut);state.history.push(row);}render();}
 }catch(error){showError(error.message);}finally{state.processing=false;if(state.pending)pump();}
}
function setCut(cut){const n=example()?.points.length??0;state.cut=Math.max(2,Math.min(n,Math.round(cut)));changed();}
function loadExample(index,{prefix=.25}={}){stop();resetPerturbations();state.mode='example';state.example=index;state.result=null;state.history=[];state.cut=Math.max(2,Math.floor(example().points.length*prefix+.5));$('exampleSelect').value=String(index);changed();}
function start(){if(!state.ready||state.mode!=='example')return;if(state.playing){stop();return;}if(state.cut>=example().points.length)setCut(2);state.playing=true;$('play').textContent='Pausar';$('play').setAttribute('aria-pressed','true');schedule();}
function schedule(){state.timer=setTimeout(()=>{if(!state.playing)return;if(state.processing){schedule();return;}if(state.cut>=example().points.length){stop();return;}setCut(state.cut+1);schedule();},Number($('speed').value));}
function updateRanking(name,p){
 const rankings=$(name+'Ranking');rankings.replaceChildren();
 if(!p){$(name+'Guess').textContent='—';$(name+'Confidence').textContent='—';return;}
 const best=top(p);$(name+'Guess').textContent=state.labels[best[0].index];$(name+'Confidence').textContent=pct(best[0].value);
 for(const item of best){const row=document.createElement('div');row.className='rank';const header=document.createElement('div');header.className='rank-header';const label=document.createElement('span'),value=document.createElement('span');label.textContent=state.labels[item.index];value.textContent=pct(item.value,1);header.append(label,value);const track=document.createElement('div');track.className='rank-track';const bar=document.createElement('i');bar.style.width=pct(item.value,3).replace(',','.');track.append(bar);row.append(header,track);rankings.append(row);}
}
function drawObserved(points){
 const canvas=$('inputCanvas'),ctx=canvas.getContext('2d'),W=canvas.width;ctx.clearRect(0,0,W,W);ctx.lineCap='round';ctx.lineJoin='round';ctx.textAlign='center';
 if(!points.length){ctx.fillStyle='#849086';ctx.font='24px sans-serif';ctx.fillText(state.mode==='draw'?'Comece seu desenho aqui':'Escolha um exemplo',W/2,W/2);return;}
 let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const p of points){minX=Math.min(minX,p[0]);minY=Math.min(minY,p[1]);maxX=Math.max(maxX,p[0]);maxY=Math.max(maxY,p[1]);}
 // No modo livre o quadro mantém coordenadas fixas sob o ponteiro.
 const extent=Math.max(maxX-minX,maxY-minY,45),scale=state.mode==='draw'?W/256:W*.72/extent;
 const transform=p=>state.mode==='draw'?[p[0]*scale,p[1]*scale]:[(p[0]-(minX+maxX)/2)*scale+W/2,(p[1]-(minY+maxY)/2)*scale+W/2];
 ctx.strokeStyle='#273b46';ctx.fillStyle='#273b46';ctx.lineWidth=3.7;
 points.forEach((p,i)=>{const [x,y]=transform(p);if(i&&!points[i-1][2]){const [a,b]=transform(points[i-1]);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(x,y);ctx.stroke();}else{ctx.beginPath();ctx.arc(x,y,1.7,0,Math.PI*2);ctx.fill();}});
 const last=transform(points.at(-1));ctx.fillStyle='#cd6045';ctx.beginPath();ctx.arc(last[0],last[1],5.2,0,Math.PI*2);ctx.fill();
}
function drawRaster(image){const canvas=$('rasterCanvas'),ctx=canvas.getContext('2d');ctx.fillStyle='#f7f6ef';ctx.fillRect(0,0,128,128);if(!image)return;ctx.fillStyle='#3f4053';for(let i=0;i<1024;i++)if(image[i])ctx.fillRect((i%32)*4,Math.floor(i/32)*4,4,4);}
function drawSequence(points){const canvas=$('sequenceCanvas'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(!points.length)return;const width=canvas.width-16;ctx.strokeStyle='#d5ddcf';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(8,40);ctx.lineTo(canvas.width-8,40);ctx.stroke();for(let i=0;i<points.length;i++){const x=8+width*i/Math.max(points.length-1,1);ctx.fillStyle=points[i][2]?'#cd6045':'#b4bfb3';ctx.fillRect(x,points[i][2]?16:32,Math.max(2,width/points.length*.4),points[i][2]?48:16);}ctx.fillStyle='#cd6045';ctx.beginPath();ctx.arc(canvas.width-8,40,6,0,Math.PI*2);ctx.fill();}
function drawCurve(){
 const canvas=$('probabilityCanvas'),W=Math.max(250,canvas.getBoundingClientRect().width),H=Math.min(260,Math.max(170,W*.25)),ratio=window.devicePixelRatio||1;canvas.width=Math.round(W*ratio);canvas.height=Math.round(H*ratio);canvas.style.height=H+'px';const ctx=canvas.getContext('2d'),pad={l:42,r:17,t:20,b:33};ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,W,H);ctx.font=(W<500?'10px':'12px')+' sans-serif';ctx.textBaseline='middle';ctx.lineWidth=1;
 const xMax=state.mode==='example'?100:Math.max(20,observed().length);const xx=x=>pad.l+x/xMax*(W-pad.l-pad.r),yy=y=>H-pad.b-y*(H-pad.t-pad.b);
 ctx.fillStyle='#7b8987';
 for(const p of [0,.25,.5,.75,1]){ctx.strokeStyle='#e0e6d8';ctx.beginPath();ctx.moveTo(pad.l,yy(p));ctx.lineTo(W-pad.r,yy(p));ctx.stroke();ctx.textAlign='right';ctx.fillText(pct(p),pad.l-10,yy(p));}
 for(let i=0;i<=4;i++){const value=xMax*i/4;ctx.textAlign='center';ctx.fillText(state.mode==='example'?Math.round(value)+'%':Math.round(value),xx(value),H-pad.b+19);}
 let records=[];
 if(state.mode==='example'&&example()){
  const label=example().label;records=(example().history??[]).filter(row=>(row.cut??Math.round(row.fraction*example().points.length))<=state.cut).map(row=>({x:100*(row.cut??Math.round(row.fraction*example().points.length))/example().points.length,gru:row.gru[label],mlp:row.mlp[label]}));
  if(state.result&&state.responseRevision===state.revision){const live={x:100*state.cut/example().points.length,gru:state.result.probabilities.gru[label],mlp:state.result.probabilities.mlp[label]};records=records.filter(r=>r.x<live.x);records.push(live);}
 }else records=state.history.map(row=>({x:row.cut,gru:row.gru,mlp:row.mlp}));
 for(const [family,color] of [['gru','#cd6045'],['mlp','#7962a7']]){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2.7;ctx.beginPath();records.forEach((r,i)=>i?ctx.lineTo(xx(r.x),yy(r[family])):ctx.moveTo(xx(r.x),yy(r[family])));ctx.stroke();for(const r of records){ctx.beginPath();ctx.arc(xx(r.x),yy(r[family]),3.2,0,Math.PI*2);ctx.fill();}}
 const current=state.mode==='example'&&example()?100*state.cut/example().points.length:observed().length;ctx.strokeStyle='#7e8d7e';ctx.lineWidth=1.3;ctx.setLineDash([6,6]);ctx.beginPath();ctx.moveTo(xx(current),pad.t);ctx.lineTo(xx(current),H-pad.b);ctx.stroke();ctx.setLineDash([]);
 if(records.length<2){ctx.fillStyle='#7b8987';ctx.textAlign='center';ctx.font=(W<500?'11px':'13px')+' sans-serif';ctx.fillText('Avance os pontos para acompanhar as hipóteses.',W*.53,H*.49);}
}
function render(){
 const points=observed(),isExample=state.mode==='example',count=points.length;
 $('exampleControls').hidden=!isExample;$('prefixControls').hidden=!isExample;$('drawTools').hidden=isExample;$('inputCanvas').parentElement.classList.toggle('draw',!isExample);
 for(const [id,active] of [['exampleMode',isExample],['drawMode',!isExample]]){$(id).classList.toggle('active',active);$(id).setAttribute('aria-pressed',String(active));}
 $('eventCount').textContent=count+' eventos';$('observedTime').textContent=count?((points.at(-1)[3]-points[0][3]).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})+' s observados'):'0,0 s';
 const strokes=points.filter(p=>p[2]>.5).length+(points.length&&points.at(-1)[2]<.5?1:0);$('penInfo').textContent=strokes+' traço'+(strokes===1?'':'s');
 if(isExample&&example()){$('prefix').value=String(Math.round(100*state.cut/example().points.length));$('prefixValue').textContent=pct(state.cut/example().points.length);$('truthLabel').innerHTML='Categoria real: <strong></strong><span>Rótulo de avaliação; nunca é entrada da rede.</span>';$('truthLabel').querySelector('strong').textContent=state.labels[example().label]??example().label_name;$('exampleNote').textContent=example().selection_rule??'';$('sourceNote').textContent='Quick, Draw! · validação · ID '+example().id;}
 else{$('truthLabel').innerHTML='Desenho livre<strong>Categoria desconhecida</strong><span>As hipóteses são restritas às 12 categorias do estudo.</span>';$('exampleNote').textContent='Seus traços são processados localmente. Os classificadores não sabem reconhecer categorias fora das 12 usadas no treino.';}
 document.querySelectorAll('[data-fraction]').forEach(b=>b.classList.toggle('active',isExample&&!!example()&&Math.abs(state.cut/example().points.length-Number(b.dataset.fraction)/100)<.02));
 const result=state.result,stale=result&&state.responseRevision!==state.revision;
 drawObserved(result&&!stale&&state.options.reverse?result.points:points);drawRaster(result?.input.image);drawSequence(result?.points??points);
 for(const name of ['gru','mlp'])updateRanking(name,result?.probabilities[name]);
 $('curveNote').textContent=isExample?'Probabilidade da categoria real; curvas reveladas até o prefixo observado.':'Maior probabilidade de cada rede; a categoria principal pode mudar.';
 $('historyNote').textContent=Object.values(state.options).some(Boolean)?'Curva histórica original; último ponto reflete a perturbação.':'Cada ponto usa somente o prefixo correspondente.';
 if(!count)$('predictionNote').textContent='Comece a desenhar. As duas redes atualizarão suas hipóteses.';
 else if(count<2)$('predictionNote').textContent='Acrescente mais movimentos para reconhecer.';
 else if(stale)$('predictionNote').textContent='Analisando '+count+' eventos… As hipóteses exibidas ainda correspondem a '+result.points.length+' eventos.';
 else if(result){const a=top(result.probabilities.gru)[0],b=top(result.probabilities.mlp)[0];$('predictionNote').textContent=a.index===b.index?'As duas redes sugerem '+state.labels[a.index]+'. Probabilidade estimada não é certeza.':'A GRU sugere '+state.labels[a.index]+'; a MLP sugere '+state.labels[b.index]+'. Compare como as hipóteses evoluem.';if(state.mode==='draw'&&count>=180)$('predictionNote').textContent+=' Limite de 180 eventos: limpe o quadro para outro desenho.';}
 if(!result&&count>=2)$('predictionNote').textContent='Analisando os '+count+' eventos observados…';
 if(result?.comparison&&!stale){const c=result.comparison;$('orderResult').textContent=(c.sameImage?'Imagem binária idêntica. ':'A imagem mudou. ')+`Maior mudança nas probabilidades: GRU ${pct(c.gruChange,1)}; MLP ${pct(c.mlpChange,1)}. Referência e ordem invertida usam o mesmo relógio de 0,1 s e fecham o trecho parcial.`;}
 else $('orderResult').textContent='Inverta traços inteiros. A MLP mantém os mesmos pixels; a GRU pode mudar sua hipótese. Controle fora da distribuição de treino.';
 drawCurve();
}
$('exampleSelect').addEventListener('change',event=>loadExample(Number(event.target.value)));
$('prefix').addEventListener('input',event=>{stop();setCut(example().points.length*Number(event.target.value)/100);});
for(const button of document.querySelectorAll('[data-fraction]'))button.addEventListener('click',()=>{stop();setCut(example().points.length*Number(button.dataset.fraction)/100);});
$('play').addEventListener('click',start);$('step').addEventListener('click',()=>{stop();setCut(state.cut+1);});$('restart').addEventListener('click',()=>{stop();setCut(2);});$('speed').addEventListener('change',()=>{if(state.playing){clearTimeout(state.timer);schedule();}});
$('exampleMode').addEventListener('click',()=>loadExample(state.example));
$('drawMode').addEventListener('click',()=>{stop();resetPerturbations();state.mode='draw';state.raw=[];state.points=[];state.result=null;state.history=[];changed();});
$('reverse').addEventListener('click',()=>{stop();state.options.reverse=!state.options.reverse;$('reverse').textContent=state.options.reverse?'Ordem invertida · restaurar':'Comparar ordem dos traços';changed();});
$('resetExperiment').addEventListener('click',()=>{resetPerturbations();changed();});
$('hidePen').addEventListener('change',event=>{state.options.hidePen=event.target.checked;changed();});$('hideTime').addEventListener('change',event=>{state.options.hideTime=event.target.checked;changed();});
let pointer=null,drawingStarted=0,lastRequest=0;
const drawCanvas=$('inputCanvas');
function append(event,lift=0){const rect=drawCanvas.getBoundingClientRect();state.raw.push([(event.clientX-rect.left)/rect.width*256,(event.clientY-rect.top)/rect.height*256,lift,(performance.now()-drawingStarted)/1000]);state.points=samplePoints(state.raw);state.result=null;}
function drawRequest(){lastRequest=performance.now();changed();}
function endStroke(event){if(pointer!==event.pointerId)return;append(event,1);pointer=null;drawRequest();try{drawCanvas.releasePointerCapture(event.pointerId);}catch{}}
$('inputCanvas').addEventListener('pointerdown',event=>{if(state.mode!=='draw'||!state.ready||pointer!==null)return;event.preventDefault();pointer=event.pointerId;if(!state.raw.length)drawingStarted=performance.now();drawCanvas.setPointerCapture(pointer);append(event);drawRequest();});
$('inputCanvas').addEventListener('pointermove',event=>{if(pointer!==event.pointerId)return;event.preventDefault();append(event);drawObserved(state.points);if(performance.now()-lastRequest>130)drawRequest();});
$('inputCanvas').addEventListener('pointerup',endStroke);$('inputCanvas').addEventListener('pointercancel',endStroke);
$('clear').addEventListener('click',()=>{state.raw=[];state.points=[];state.history=[];state.result=null;resetPerturbations();changed();});
$('undo').addEventListener('click',()=>{if(!state.raw.length)return;state.raw.pop();while(state.raw.length&&!state.raw.at(-1)[2])state.raw.pop();state.points=samplePoints(state.raw);state.history=[];state.result=null;changed();});
async function getJson(name){const response=await fetch(name);if(!response.ok)throw Error('Recurso indisponível: '+name);return response.json();}
async function init(){try{
 const [definitions,examplesData,study]=await Promise.all([call('load'),getJson('assets/recognition/examples.json'),getJson('assets/recognition/study.json')]);state.labels=definitions.labels;state.examples=examplesData.examples??examplesData;state.ready=true;$('learnedCategories').textContent=state.labels.join(' · ')+'.';
 $('exampleSelect').replaceChildren();state.examples.forEach((e,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=e.title??((e.label_name??state.labels[e.label])+' · exemplo '+(i+1));$('exampleSelect').append(option);});
 for(const id of ['exampleSelect','prefix','play','step','restart','reverse','resetExperiment'])$(id).disabled=false;
 const rows=study.fractions??[];$('studyRows').replaceChildren();for(const row of rows){const tr=document.createElement('tr');for(const value of [pct(row.fraction),pct(row.gru_accuracy,1),pct(row.mlp_accuracy,1)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('studyRows').append(tr);}
 if(study.note)$('studyNote').textContent=study.note;
 status('Dois classificadores prontos · processamento local','ready');loadExample(0);
 }catch(error){showError(error.message);}}
window.__recognitionDemo={state,get ready(){return state.ready;},loadExample,setCut,stop,resetPerturbations,render,changed,observed};
window.addEventListener('resize',drawCurve);
render();init();
