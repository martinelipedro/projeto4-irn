import {RecognitionModel,represent,uniformClock,reverseStrokes} from './recognition.js';
let models;
self.onmessage=async event=>{
 const {id,type}=event.data;
 try{
  if(type==='load'){
   const definitions=await Promise.all(['gru','mlp'].map(async name=>{const r=await fetch(`assets/recognition/${name}.json`);if(!r.ok)throw Error(`Não foi possível carregar ${name}.`);return r.json();}));
   models=Object.fromEntries(definitions.map(data=>[data.family,new RecognitionModel(data)]));
   self.postMessage({id,type:'ready',labels:definitions[0].labels});return;
  }
  if(!models)throw Error('Os modelos ainda estão carregando.');
  const original=event.data.points,options=event.data.options??{};
  let points=original;
  if(options.reverse||options.uniformClock)points=uniformClock(points);
  const baseline=represent(points);
  if(options.reverse)points=reverseStrokes(points);
  const input=represent(points),tokens=input.tokens.map(t=>t.slice());
  if(options.hidePen)for(const t of tokens)t[2]=0;
  if(options.hideTime)for(const t of tokens)t[3]=0;
  const probabilities={};for(const name of ['gru','mlp'])probabilities[name]=models[name].probabilities(tokens,input.image);
  let comparison=null;
  if(options.reverse){const previous={},baselineTokens=baseline.tokens.map(t=>t.slice());for(const t of baselineTokens){if(options.hidePen)t[2]=0;if(options.hideTime)t[3]=0;}for(const name of ['gru','mlp'])previous[name]=models[name].probabilities(baselineTokens,baseline.image);comparison={baseline:previous,sameImage:baseline.image.every((v,i)=>v===input.image[i]),gruChange:Math.max(...probabilities.gru.map((v,i)=>Math.abs(v-previous.gru[i]))),mlpChange:Math.max(...probabilities.mlp.map((v,i)=>Math.abs(v-previous.mlp[i])))};}
  self.postMessage({id,type:'result',points,input:{tokens,image:input.image},probabilities,comparison});
 }catch(error){self.postMessage({id,type:'error',message:error.message});}
};
