import {prepareForecastPoints} from './forecast.js';

const $=id=>document.getElementById(id);
const colors={ink:'#223342',gru:'#d5684c',mlp:'#806aab',truth:'#418e83',muted:'#97a5ac'};
const state={ready:false,modelReady:false,dataReady:false,examples:[],labels:[],classes:[],mode:'example',exampleIndex:0,prefix:16,horizon:15,label:0,
  result:null,reveal:0,playing:false,processing:false,job:0,points:[],strokes:[],theme:'prefixo',startedAt:0};
let timer=null,predictTimer=null,activePointer=null;
const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
const currentExample=()=>state.examples[state.exampleIndex];
const perturbation=()=>({scale:Number($('scale').value),translation:$('translation').value==='1'?[120,-80]:[0,0],shuffle:$('shuffle').checked,ignorePen:$('ignorePen').checked});
const basePast=()=>state.mode==='example'?(currentExample()?.points.slice(0,state.prefix)??[]):state.points;
const format=(n,d=1)=>Number.isFinite(n)?n.toLocaleString('pt-BR',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
const px=n=>Number.isFinite(n)?format(n)+' px':'—';

function stopPlayback(){clearTimeout(timer);timer=null;state.playing=false;$('play').textContent='▶ Reproduzir';}
function startPlayback(){
  if(!state.result)return;
  stopPlayback();if(state.reveal>=state.horizon)state.reveal=0;
  state.playing=true;$('play').textContent='Ⅱ Pausar';
  const next=()=>{if(!state.playing)return;state.reveal=Math.min(state.horizon,state.reveal+1);render();if(state.reveal<state.horizon)timer=setTimeout(next,Number($('speed').value));else stopPlayback();};
  timer=setTimeout(next,Number($('speed').value));render();
}

function controls(){
  $('exampleControls').hidden=state.mode!=='example';$('prefixControls').hidden=state.mode!=='example';$('drawTools').hidden=state.mode!=='draw';$('truthControl').hidden=state.mode!=='example';
  for(const [id,mode] of [['exampleMode','example'],['drawMode','draw']]){$(id).classList.toggle('active',state.mode===mode);$(id).setAttribute('aria-pressed',String(state.mode===mode));}
  $('prefix').value=state.prefix;$('prefixValue').textContent=state.prefix+' pontos';$('horizon').value=state.horizon;$('category').value=state.label;
  const example=currentExample(),max=example?Math.min(64,example.points.length-state.horizon):64;
  $('prefix').max=Math.max(8,max);$('prefix').disabled=!state.ready;
  document.querySelectorAll('[data-cut]').forEach(b=>{b.disabled=!state.ready||Number(b.dataset.cut)>max;b.classList.toggle('active',Number(b.dataset.cut)===state.prefix);});
  $('predict').disabled=!state.ready||state.processing||basePast().length<8;
  for(const id of ['play','step','all','reveal'])$(id).disabled=!state.result;
  $('reveal').max=state.horizon;$('reveal').value=state.reveal;$('playValue').textContent=`t + ${state.reveal} / ${state.horizon}`;
  $('inputCanvas').style.cursor=state.mode==='draw'?'crosshair':'default';
}

function invalidate(){stopPlayback();state.result=null;state.reveal=0;render();}

function loadExample(index,{autoplay=true}={}){
  index=Math.max(0,Math.min(state.examples.length-1,Number(index)));if(!state.examples[index])return;
  state.mode='example';state.exampleIndex=index;const example=currentExample();state.label=Number(example.label);
  if(example.report_case){state.horizon=example.default_horizon;resetPerturbations(false);$('showTruth').checked=true;}
  state.prefix=Math.max(8,Math.min(example.default_cut??16,64,example.points.length-state.horizon));
  $('exampleSelect').value=index;$('sourceTag').textContent=`Quick, Draw! · ${example.source_split==='test'?'teste':'validação'} · ID ${example.id}`;
  $('exampleNote').textContent=example.explanation??example.selection_rule??'Exemplo real definido pela regra descrita no relatório.';
  invalidate();if(state.ready)requestPrediction({autoplay});
}

function setPrefix(value,{autoplay=false}={}){
  if(state.mode!=='example')return;
  const max=Math.min(64,currentExample().points.length-state.horizon);state.prefix=Math.max(8,Math.min(max,Number(value)));invalidate();controls();requestPrediction({autoplay});
}

function setMode(mode){
  if(mode==='example'){loadExample(state.exampleIndex);return;}
  state.mode='draw';state.startedAt=state.points.length?performance.now()-state.points.at(-1)[3]*1000:0;resetPerturbations(false);$('exampleNote').textContent='Seu desenho fica no navegador. A categoria deve ser escolhida por você.';invalidate();
  $('predictionNote').textContent='Desenhe alguns movimentos no primeiro quadro e peça uma continuação.';
}

function requestPrediction({autoplay=false}={}){
  clearTimeout(predictTimer);if(!state.ready)return;
  const points=basePast();if(points.length<8){$('predictionNote').textContent='Acrescente mais movimentos: o previsor precisa de pelo menos oito pontos.';return;}
  stopPlayback();state.processing=true;state.job++;state.pendingAutoplay=autoplay;state.result=null;state.reveal=0;
  $('predictionNote').textContent='As duas redes estão processando o mesmo passado…';
  worker.postMessage({type:'predict',job:state.job,points,label:state.label,horizon:state.horizon,example:state.mode==='example',
    truth:state.mode==='example'?currentExample().points.slice(state.prefix,state.prefix+state.horizon):[],perturb:perturbation()});
  render();
}

function schedulePrediction(){invalidate();clearTimeout(predictTimer);predictTimer=setTimeout(()=>requestPrediction({autoplay:false}),140);}

function selectedCondition(){
  const p=perturbation(),example=currentExample();return (state.mode==='example'&&!!example&&Number(example.label)!==state.label)||p.scale!==1||p.translation[0]!==0||p.shuffle||p.ignorePen;
}

worker.onmessage=event=>{
  const message=event.data;
  if(message.type==='ready'){
    state.modelReady=true;state.classes=message.classes;state.labels=message.labels;
    $('category').replaceChildren(...state.labels.map((label,i)=>Object.assign(document.createElement('option'),{value:i,textContent:label})));
    $('category').disabled=false;checkReady();
  }else if(message.type==='prediction'&&message.job===state.job){
    state.processing=false;state.result=message;state.reveal=0;
    const more=message.contextCount<basePast().length?` Contexto efetivo: ${message.contextCount} eventos amostrados.`:'';
    $('predictionNote').textContent=`Uma hipótese para os próximos ${state.horizon} pontos, com a categoria “${state.labels[state.label]}” fornecida.${more}`;
    $('contextNote').textContent=selectedCondition()?'Entrada alterada: este é um teste exploratório de sensibilidade, fora do protocolo principal.':'Os modelos recebem apenas os pontos até t. O futuro real serve à comparação.';
    render();if(state.pendingAutoplay)startPlayback();
  }else if(message.type==='error'){
    if(message.job!=null&&message.job!==state.job)return;
    state.processing=false;
    if(message.job==null){$('modelStatus').className='status error';$('modelStatus').textContent='Não foi possível carregar os modelos. Recarregue a página.';console.error(message.message);}
    else $('predictionNote').textContent=message.message;
    render();
  }
};
worker.onerror=event=>{state.processing=false;$('modelStatus').className='status error';$('modelStatus').textContent='Não foi possível iniciar os modelos. Recarregue a página.';console.error(event.message);controls();};

function checkReady(){
  state.ready=state.modelReady&&state.dataReady;if(!state.ready)return;
  $('modelStatus').className='status ready';$('modelStatus').innerHTML='<i></i> Dois modelos prontos · processamento local';
  $('exampleSelect').disabled=false;loadExample(0);
}

function project(points,extra=[]){
  const all=[...points,...extra].filter(p=>p?.slice(0,2).every(Number.isFinite));
  if(!all.length)return {x:0,y:0,span:256};
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const p of all){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}
  const span=Math.max(90,maxX-minX,maxY-minY)*1.25;
  return {x:(minX+maxX)/2-span/2,y:(minY+maxY)/2-span/2,span};
}

function stroke(ctx,points,projection,color,width=4.5,alpha=1){
  if(points.length<2)return;
  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.globalAlpha=alpha;
  let active=false;ctx.beginPath();
  for(let i=0;i<points.length;i++){
    const p=points[i],x=(p[0]-projection.x)/projection.span*600,y=(p[1]-projection.y)/projection.span*600;
    if(!active){ctx.moveTo(x,y);active=true;}else ctx.lineTo(x,y);
    if(p[2])active=false;
  }
  ctx.stroke();ctx.restore();
}

function drawBoards(){
  const result=state.result,past=result?.past??basePast();
  const trueFuture=result?.truth??[],gru=result?.models.gru.points??[],mlp=result?.models.mlp.points??[];
  const original=basePast();const transformed=perturbation().translation[0]!==0||perturbation().scale!==1;
  const projection=project(past,[...trueFuture,...gru,...mlp,...(transformed?original:[])]);
  const worlds={inputCanvas:state.mode==='draw'?{x:0,y:0,span:256}:projection,gruCanvas:projection,mlpCanvas:projection};
  for(const [id,world] of Object.entries(worlds)){
    const ctx=$(id).getContext('2d');ctx.clearRect(0,0,600,600);
    if(id!=='inputCanvas'&&transformed)stroke(ctx,original,world,'#C3CBC9',3,.4);
    stroke(ctx,id==='inputCanvas'&&state.mode==='draw'?state.points:past,world,colors.ink,id==='inputCanvas'?5:3.8,id==='inputCanvas'?1:.65);
    if(id==='inputCanvas'&&!past.length){ctx.fillStyle='#93A19F';ctx.font='23px sans-serif';ctx.textAlign='center';ctx.fillText(state.mode==='draw'?'Desenhe seu começo aqui':'Preparando um exemplo…',300,295);}
    if(id==='inputCanvas'||!result)continue;
    if($('showTruth').checked&&state.mode==='example')stroke(ctx,[past.at(-1),...trueFuture],world,colors.truth,3.3,.55);
    const key=id==='gruCanvas'?'gru':'mlp',model=result.models[key],visible=model.points.slice(0,state.reveal);
    stroke(ctx,[past.at(-1),...visible],world,colors[key],5.5);
    const latest=visible.at(-1);
    if(latest&&!model.ended[state.reveal-1]){ctx.beginPath();ctx.arc((latest[0]-world.x)/world.span*600,(latest[1]-world.y)/world.span*600,5,0,Math.PI*2);ctx.fillStyle=colors[key];ctx.fill();}
  }
  const count=state.mode==='example'?state.prefix:prepareForecastPoints(state.points).length;
  $('observedCount').textContent=`t = ${count}`;$('observedTime').textContent=past.length?format(past.at(-1)[3],2)+' s':'—';
  for(const model of ['gru','mlp'])$(model+'Time').textContent=result&&state.reveal?format(result.models[model].points[state.reveal-1][3],2)+' s previstos':'—';
}

function drawCurve(id,axis){
  const canvas=$(id),ctx=canvas.getContext('2d'),w=1000,h=340;ctx.clearRect(0,0,w,h);
  const result=state.result,past=result?.past??basePast(),window=past.slice(-9),start=-(window.length-1),future=result?.truth??[];
  const series=[{points:window.map((p,i)=>[start+i,p[axis]]),color:colors.ink,width:4}];
  if(result){
    for(const key of ['gru','mlp'])series.push({points:[[0,past.at(-1)[axis]],...result.models[key].points.slice(0,state.reveal).map((p,i)=>[i+1,p[axis]])],color:colors[key],width:4});
    if($('showTruth').checked&&state.mode==='example')series.push({points:[[0,past.at(-1)[axis]],...future.map((p,i)=>[i+1,p[axis]])],color:colors.truth,width:3});
  }
  const ys=[...past.slice(-9),...(result?.truth??[]),...(result?.models.gru.points??[]),...(result?.models.mlp.points??[])].map(p=>p[axis]);
  let low=ys.length?Math.min(...ys):0,high=ys.length?Math.max(...ys):256;const margin=Math.max(10,(high-low)*.13);low-=margin;high+=margin;
  const box={left:90,right:968,top:50,bottom:278},xMax=state.horizon;
  const x=v=>box.left+(v-Math.min(-8,start))/(xMax-Math.min(-8,start))*(box.right-box.left),y=v=>box.bottom-(v-low)/(high-low)*(box.bottom-box.top);
  ctx.textAlign='left';ctx.font='23px sans-serif';ctx.fillStyle=colors.ink;ctx.fillText('Coordenada '+(axis===0?'X':'Y')+' (pixels)',30,31);
  ctx.font='20px sans-serif';ctx.strokeStyle='#E2E7DF';ctx.lineWidth=1.5;
  for(let i=0;i<=3;i++){const value=low+(high-low)*i/3,yy=y(value);ctx.beginPath();ctx.moveTo(box.left,yy);ctx.lineTo(box.right,yy);ctx.stroke();ctx.fillStyle='#84938E';ctx.textAlign='right';ctx.fillText(Math.round(value),box.left-12,yy+6);}
  ctx.textAlign='center';for(const xx of [0,5,15,30].filter(v=>v<=state.horizon)){ctx.fillStyle='#84938E';ctx.fillText(xx===0?'t':'+'+xx,x(xx),310);}
  ctx.save();ctx.strokeStyle='#9DAAA2';ctx.setLineDash([6,7]);ctx.beginPath();ctx.moveTo(x(0),box.top);ctx.lineTo(x(0),box.bottom);ctx.stroke();ctx.restore();
  for(const s of series){if(s.points.length<2)continue;ctx.strokeStyle=s.color;ctx.lineWidth=s.width;ctx.beginPath();s.points.forEach((p,i)=>i?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();}
}

function render(){
  controls();drawBoards();drawCurve('curveX',0);drawCurve('curveY',1);
  for(const model of ['gru','mlp'])for(const [suffix,key] of [['ADE','ade'],['FDE','fde']])$(model+suffix).textContent=px(state.result?.models[model].metrics?.[key]);
  $('metricNote').textContent=state.mode==='draw'?'Sem futuro real conhecido: seu desenho não tem erro de previsão mensurável.':selectedCondition()?'Cenário alterado: distâncias ao futuro original transformado são exploratórias.':'Erros do horizonte completo, sobre os mesmos pontos reais. Menor é melhor.';
}

const themeCopy={
  prefixo:['O que os modelos fazem com mais contexto?','Use 8, 16, 32 e 64 para mudar o passado observado. O horizonte k conta pontos futuros. Uma continuação pode melhorar ou piorar quando novos movimentos aparecem.'],
  cruzamentos:['O caminho importa quando as linhas se cruzam?','O exemplo contém cruzamentos no passado. Compare a continuação e os estados da caneta: ler uma sequência não garante separar perfeitamente as partes do desenho.'],
  ordem:['A ordem do contexto antigo muda a hipótese?','A permutação altera a ordem de vetores já codificados, preserva o início e os últimos movimentos necessários ao referencial. É uma entrada artificial para testar sensibilidade, não outro desenho humano.'],
  translacao_escala:['Mover e ampliar são duas coisas diferentes.','A translação desaparece nas posições relativas e nos deslocamentos. Já ampliar muda Δx, Δy e as posições: usar movimento não dá invariância automática à escala.'],
  caneta:['Separar os traços também é prever.','O estado de pen-up marca uma separação entre componentes. Ocultá-lo testa a sensibilidade do encoder; o referencial geométrico original é mantido para não misturar efeitos.'],
  ambiguidade:['Um começo, mais de um futuro possível.','Compare o mesmo prefixo condicionando a previsão a Casa ou Castelo. Você fornece a categoria: isso mostra continuações condicionais, não reconhecimento automático.']
};

function selectTheme(theme){
  state.theme=theme;document.querySelectorAll('[data-theme]').forEach(b=>b.classList.toggle('active',b.dataset.theme===theme));
  $('experimentTitle').textContent=themeCopy[theme][0];$('experimentDescription').textContent=themeCopy[theme][1];
  $('alternateCategory').hidden=theme!=='ambiguidade';
  resetPerturbations(false);
  const candidates=state.examples.map((e,i)=>({e,i})).filter(({e})=>e.themes?.includes(theme));
  if(candidates.length)loadExample(candidates[0].i,{autoplay:false});
  if(theme==='ambiguidade'){
    const index=state.examples.findIndex(e=>e.category==='house'||e.label===state.classes.indexOf('house'));
    if(index>=0){loadExample(index,{autoplay:false});state.prefix=Math.min(16,currentExample().points.length-state.horizon);}
  }
  if(theme==='ordem')$('shuffle').checked=true;if(theme==='caneta')$('ignorePen').checked=false;
  if(state.ready)requestPrediction({autoplay:false});
  $('laboratorio').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});
}

function resetPerturbations(predict=true){$('scale').value='1';$('translation').value='0';$('shuffle').checked=false;$('ignorePen').checked=false;if(predict)schedulePrediction();}

$('exampleSelect').addEventListener('change',e=>loadExample(e.target.value));$('exampleMode').addEventListener('click',()=>setMode('example'));$('drawMode').addEventListener('click',()=>setMode('draw'));
$('category').addEventListener('change',e=>{state.label=Number(e.target.value);schedulePrediction();});
$('prefix').addEventListener('input',e=>{state.prefix=Number(e.target.value);schedulePrediction();});
document.querySelectorAll('[data-cut]').forEach(b=>b.addEventListener('click',()=>setPrefix(b.dataset.cut)));
$('horizon').addEventListener('change',e=>{state.horizon=Number(e.target.value);if(state.mode==='example')state.prefix=Math.min(state.prefix,currentExample().points.length-state.horizon);schedulePrediction();});
$('predict').addEventListener('click',()=>requestPrediction({autoplay:true}));
$('play').addEventListener('click',()=>state.playing?stopPlayback():startPlayback());$('step').addEventListener('click',()=>{stopPlayback();state.reveal=Math.min(state.horizon,state.reveal+1);render();});
$('all').addEventListener('click',()=>{stopPlayback();state.reveal=state.horizon;render();});$('reveal').addEventListener('input',e=>{stopPlayback();state.reveal=Number(e.target.value);render();});$('showTruth').addEventListener('change',render);
$('speed').addEventListener('change',()=>{if(state.playing){stopPlayback();startPlayback();}});
for(const id of ['scale','translation','shuffle','ignorePen'])$(id).addEventListener('change',schedulePrediction);
$('resetExperiment').addEventListener('click',()=>resetPerturbations());document.querySelectorAll('[data-theme]').forEach(b=>b.addEventListener('click',()=>selectTheme(b.dataset.theme)));
$('alternateCategory').addEventListener('click',()=>{const house=state.classes.indexOf('house'),castle=state.classes.indexOf('castle');state.label=state.label===house?castle:house;schedulePrediction();});
$('clear').addEventListener('click',()=>{state.points=[];state.strokes=[];state.startedAt=0;invalidate();$('predictionNote').textContent='Comece outro desenho no primeiro quadro.';});
$('undo').addEventListener('click',()=>{const start=state.strokes.pop();if(start!=null)state.points.length=start;invalidate();});

function pointerPoint(event){const rect=$('inputCanvas').getBoundingClientRect();return [Math.max(0,Math.min(256,(event.clientX-rect.left)/rect.width*256)),Math.max(0,Math.min(256,(event.clientY-rect.top)/rect.height*256)),0,(performance.now()-state.startedAt)/1000];}
$('inputCanvas').addEventListener('pointerdown',event=>{if(state.mode!=='draw')return;event.preventDefault();activePointer=event.pointerId;$('inputCanvas').setPointerCapture(event.pointerId);if(!state.startedAt)state.startedAt=performance.now();state.strokes.push(state.points.length);state.points.push(pointerPoint(event));invalidate();});
$('inputCanvas').addEventListener('pointermove',event=>{if(activePointer!==event.pointerId||state.mode!=='draw')return;const point=pointerPoint(event),last=state.points.at(-1);if(!last||Math.hypot(point[0]-last[0],point[1]-last[1])>=1.2){state.points.push(point);render();}});
function pointerEnd(event){if(activePointer!==event.pointerId)return;activePointer=null;if(state.points.length)state.points.at(-1)[2]=1;render();if(state.points.length>=8)requestPrediction({autoplay:false});}
$('inputCanvas').addEventListener('pointerup',pointerEnd);$('inputCanvas').addEventListener('pointercancel',pointerEnd);

async function loadAssets(){
  try{
    const read=async name=>{
      const response=await fetch(`assets/${name}.json`);
      if(!response.ok)throw Error('Exemplos indisponíveis.');
      return response.json();
    };
    const [data,curated]=await Promise.all([read('examples'),read('report_cases')]);
    const examples=Array.isArray(data)?data:data.examples;
    state.examples=[...curated,...examples];
    const gallery=document.createElement('optgroup');gallery.label='Previsões bem-sucedidas do relatório';
    const exploration=document.createElement('optgroup');exploration.label='Outros exemplos para explorar';
    state.examples.forEach((example,index)=>{
      const option=document.createElement('option');option.value=index;
      option.textContent=example.report_case?`${example.category_name} · k = ${example.default_horizon}`:
        (example.category_name??example.category??'Desenho');
      (example.report_case?gallery:exploration).append(option);
    });
    $('exampleSelect').replaceChildren(gallery,exploration);
    state.dataReady=true;checkReady();
    const studyResponse=await fetch('assets/study.json');if(studyResponse.ok)renderStudy(await studyResponse.json());
  }catch(error){$('modelStatus').className='status error';$('modelStatus').textContent='Não foi possível carregar os exemplos. Recarregue a página.';console.error(error);}
}

function renderStudy(study){
  const rows=study.horizons??[];
  if(rows.length)$('studyRows').replaceChildren(...rows.map(r=>{const tr=document.createElement('tr');for(const value of [r.k+' pontos',px(r.gru_ade),px(r.mlp_ade)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}));
  if(study.note)$('studyNote').textContent=study.note;
}

window.__forecastDemo={get ready(){return state.ready;},state,loadExample,setPrefix,predict:requestPrediction,setMode,stop:stopPlayback,render,resetPerturbations,selectTheme};
worker.postMessage({type:'load'});loadAssets();render();
