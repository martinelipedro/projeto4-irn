import {ForecastModel} from './forecast.js';

let models;
const loadJSON=async path=>{const response=await fetch(path);if(!response.ok)throw Error(`Recurso indisponível: ${path} (${response.status}).`);return response.json();};

function shuffleInterior(tokens){
  const ink=[];
  for(let i=1;i<tokens.length;i++)if(tokens[i-1][2]<.5&&tokens[i][0]**2+tokens[i][1]**2>1e-9)ink.push(i);
  const suffix=Math.max(1,(ink.slice(-3)[0]??tokens.length)-1);
  const indices=Array.from({length:Math.max(0,suffix-1)},(_,i)=>i+1);let seed=(20261118+tokens.length)>>>0;
  for(let i=indices.length-1;i>0;i--){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=seed%(i+1);[indices[i],indices[j]]=[indices[j],indices[i]];}
  return [tokens[0],...indices.map(i=>tokens[i]),...tokens.slice(suffix)];
}

function metrics(predicted,truth){
  if(!truth?.length)return null;
  const n=Math.min(predicted.length,truth.length);let ade=0,clock=0;
  for(let i=0;i<n;i++){ade+=Math.hypot(predicted[i][0]-truth[i][0],predicted[i][1]-truth[i][1]);clock+=Math.abs(predicted[i][3]-truth[i][3]);}
  return {ade:ade/n,fde:Math.hypot(predicted[n-1][0]-truth[n-1][0],predicted[n-1][1]-truth[n-1][1]),time_mae:clock/n,n};
}

function run(message){
  const {points,label,horizon,example,perturb={},truth=[]}=message;
  const scale=Number(perturb.scale??1),[dx,dy]=perturb.translation??[0,0],origin=points[0];
  const transform=p=>[origin[0]+(p[0]-origin[0])*scale+dx,origin[1]+(p[1]-origin[1])*scale+dy,p[2],p[3]];
  const transformed=points.map(transform),future=truth.map(transform).slice(0,horizon),outputs={};let contextCount=0,contextPoints=transformed;
  for(const [name,model] of Object.entries(models)){
    const encoded=model.encodePoints(transformed,{resample:!example}),reference=encoded.tokens.map(t=>t.slice());
    let tokens=reference.map(t=>t.slice());contextCount=tokens.length;contextPoints=encoded.points;
    if(perturb.shuffle)tokens=shuffleInterior(tokens);
    if(perturb.ignorePen)tokens=tokens.map(t=>t.map((v,i)=>i===2?0:v));
    if(perturb.ignoreTime)tokens=tokens.map(t=>t.map((v,i)=>i===3?0:v));
    // Nos probes de canais/ordem, só o encoder é alterado. O quadro geométrico fica fixo.
    const prediction=model.predict(tokens,label,{horizon,referenceTokens:reference});
    let time=encoded.points.at(-1)[3];
    const trajectory=prediction.tokens.map(t=>{time+=model.time_scale*Math.expm1(t[3]);return [encoded.points[0][0]+t[4]*model.position_scale,encoded.points[0][1]+t[5]*model.position_scale,t[2],time];});
    outputs[name]={points:trajectory,ended:prediction.ended,mode:prediction.mode,metrics:metrics(trajectory,future)};
  }
  return {models:outputs,past:contextPoints,truth:future,contextCount,horizon,label,perturb};
}

self.onmessage=async event=>{
  const message=event.data;
  try{
    if(message.type==='load'){
      const data=await Promise.all([loadJSON('assets/gru.json'),loadJSON('assets/mlp.json')]);
      models={gru:new ForecastModel(data[0]),mlp:new ForecastModel(data[1])};
      if(models.gru.architecture!=='trajectory'||models.mlp.architecture!=='deepsets')throw Error('Modelos incompatíveis com a comparação.');
      if(JSON.stringify(models.gru.classes)!==JSON.stringify(models.mlp.classes))throw Error('Categorias incompatíveis.');
      self.postMessage({type:'ready',classes:data[0].classes,labels:data[0].labels,models:data.map(d=>({architecture:d.architecture,hidden:d.hidden,epoch:d.epoch,seed:d.seed}))});
    }else if(message.type==='predict'){
      if(!models)throw Error('Os modelos ainda estão carregando.');
      self.postMessage({type:'prediction',job:message.job,...run(message)});
    }
  }catch(error){self.postMessage({type:'error',job:message.job,message:error.message});}
};
