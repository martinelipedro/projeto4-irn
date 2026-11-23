// Modelo independente de previsão. Os pesos e o banco de trajetórias são aprendidos no treino.
const f=Math.fround;
const sigmoid=x=>1/(1+Math.exp(-x));
const softplus=x=>x>20?x:Math.log1p(Math.exp(x));
function linear(x,w,b){return w.map((row,i)=>row.reduce((sum,value,j)=>sum+value*x[j],b?.[i]??0));}
function softmax(x){const peak=Math.max(...x),e=x.map(v=>Math.exp(v-peak)),sum=e.reduce((a,b)=>a+b,0);return e.map(v=>v/sum);}
function argmax(x){return x.reduce((best,v,i)=>v>x[best]?i:best,0);}
function direction(tokens){
  const ink=[];for(let i=1;i<tokens.length;i++)if(tokens[i-1][2]<.5&&tokens[i][0]**2+tokens[i][1]**2>1e-9)ink.push(tokens[i]);
  let x=0,y=0;for(const t of ink.slice(-3)){x=f(x+t[0]);y=f(y+t[1]);}const norm=f(Math.sqrt(f(f(x*x)+f(y*y))));return norm<1e-6?[1,0]:[f(x/norm),f(y/norm)];
}
export function prepareForecastPoints(points,step=14){
  if(!points.length)return [];const kept=[];
  for(let i=0;i<points.length;i++){const p=points[i],last=kept.at(-1);if(p.length<4||!p.slice(0,4).every(Number.isFinite))throw Error('Ponto inválido para a previsão.');if(i&&p[3]<points[i-1][3])throw Error('Os timestamps precisam estar em ordem.');if(!last||points[i-1]?.[2]||p[2]||Math.hypot(p[0]-last[0],p[1]-last[1])>=step-1e-5||i===points.length-1)kept.push(p);}
  return kept;
}
export class ForecastModel{
  constructor(data){Object.assign(this,data);this.w=data.weights;}
  encodePoints(points,{resample=true}={}){
    let kept=resample?prepareForecastPoints(points,this.spatial_step):points;kept=kept.slice(-this.context_events).map(p=>p.map(f));
    if(kept.length<this.min_context_events)throw Error('Acrescente mais traços: o previsor precisa de pelo menos oito pontos observados.');
    const origin=kept[0];const tokens=kept.map((p,i)=>{const previous=kept[Math.max(0,i-1)];return [f(f(p[0]-previous[0])/f(this.delta_scale)),f(f(p[1]-previous[1])/f(this.delta_scale)),p[2],f(Math.log1p(f(f(p[3]-previous[3])/f(this.time_scale)))),f(f(p[0]-origin[0])/f(this.position_scale)),f(f(p[1]-origin[1])/f(this.position_scale))];});
    return {tokens,points:kept};
  }
  step(input,state=new Float64Array(this.hidden)){
    const a=linear(input,this.w['gru.weight_ih_l0'],this.w['gru.bias_ih_l0']),b=linear(state,this.w['gru.weight_hh_l0'],this.w['gru.bias_hh_l0']),next=new Float64Array(this.hidden),H=this.hidden;
    for(let i=0;i<H;i++){const r=sigmoid(a[i]+b[i]),z=sigmoid(a[H+i]+b[H+i]),n=Math.tanh(a[2*H+i]+r*b[2*H+i]);next[i]=(1-z)*n+z*state[i];}return next;
  }
  prefix(tokens,label){
    let state=new Float64Array(this.hidden);
    const condition=this.w['embedding.weight'][label];
    if(!condition)throw Error('Categoria inválida para o previsor.');
    for(const token of tokens)state=this.step([...token,...condition],state);
    const h=Array.from(state);
    const logits=linear(h,this.w['mode.weight'],this.w['mode.bias']);
    return {h,logits};
  }

  predict(tokens,label,{horizon=30,referenceTokens=null}={}){
    if(!this.horizons.includes(horizon))throw Error('Escolha um horizonte de 5, 15 ou 30 pontos.');
    const {h,logits}=this.prefix(tokens,label);
    const mode=argmax(logits),anchor=this.w.anchors[label][mode];
    const embedding=this.w['mode_embedding.weight'][mode];
    const hidden=linear([...h,...embedding],this.w['regression_hidden.weight'],
      this.w['regression_hidden.bias']).map(Math.tanh);
    const residual=linear(hidden,this.w['regression.weight'],this.w['regression.bias']);

    const reference=referenceTokens??tokens,last=reference.at(-1);
    const [c,s]=direction(reference);
    let position=last.slice(4,6),alive=true;
    const out=[],ended=[];
    for(let k=0;k<horizon;k++){
      const r=residual.slice(k*6,k*6+6),a=anchor[k];
      const x=a[0]+.35*Math.tanh(r[0]),y=a[1]+.35*Math.tanh(r[1]);
      const pen=a.slice(2,5).map((p,i)=>Math.log(Math.max(.02,p))+r[i+2]);
      const end=argmax(pen)===2;
      const lift=softmax(pen.slice(0,2))[1]>(this.pen_threshold??.5);
      alive=alive&&!end;
      const next=alive?[last[4]+x*c-y*s,last[5]+x*s+y*c]:position;
      const clock=alive?softplus(Math.log(Math.expm1(Math.max(.02,a[5])))+r[5]):0;
      const [dx,dy]=next.map((v,i)=>(v-position[i])*this.position_scale/this.delta_scale);
      out.push([dx,dy,alive?+lift:1,clock,...next]);
      ended.push(!alive);
      position=next;
    }
    return {tokens:out,ended,mode,mode_probabilities:softmax(logits)};
  }
}
