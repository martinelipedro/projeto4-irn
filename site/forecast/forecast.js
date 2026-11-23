// Modelo independente de previsão. Os pesos e o banco de trajetórias são aprendidos no treino.
const f=Math.fround;
const sigmoid=x=>1/(1+Math.exp(-x));
const softplus=x=>x>20?x:Math.log1p(Math.exp(x));
function linear(x,w,b){return w.map((row,i)=>row.reduce((sum,value,j)=>sum+value*x[j],b?.[i]??0));}
function softmax(x,temperature=1){const peak=Math.max(...x),e=x.map(v=>Math.exp((v-peak)/temperature)),sum=e.reduce((a,b)=>a+b,0);return e.map(v=>v/sum);}
function argmax(x){return x.reduce((best,v,i)=>v>x[best]?i:best,0);}
function randomSource(seed){let value=seed>>>0,spare=null;const uniform=()=>{value+=0x6D2B79F5;let t=value;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};return {uniform,normal(){if(spare!==null){const n=spare;spare=null;return n;}const r=Math.sqrt(-2*Math.log(Math.max(1e-12,uniform()))),angle=2*Math.PI*uniform();spare=r*Math.sin(angle);return r*Math.cos(angle);},choose(p){let r=uniform();for(let i=0;i<p.length;i++){r-=p[i];if(r<=0)return i;}return p.length-1;}};}
function direction(tokens){
  const ink=[];for(let i=1;i<tokens.length;i++)if(tokens[i-1][2]<.5&&tokens[i][0]**2+tokens[i][1]**2>1e-9)ink.push(tokens[i]);
  let x=0,y=0;for(const t of ink.slice(-3)){x=f(x+t[0]);y=f(y+t[1]);}const norm=f(Math.sqrt(f(f(x*x)+f(y*y))));return norm<1e-6?[1,0]:[f(x/norm),f(y/norm)];
}
const roundEven=x=>{const floor=Math.floor(x),part=x-floor;return part===.5?(floor%2?floor+1:floor):Math.round(x);};
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
  rasterImage(tokens){
    const [c,s]=direction(tokens),last=tokens.at(-1),local=tokens.map(t=>{const x=f(t[4]-last[4]),y=f(t[5]-last[5]);return [f(f(x*c)+f(y*s)),f(f(-x*s)+f(y*c))];});
    const extent=f(Math.max(.5,...local.flat().map(Math.abs))*f(1.1)),image=new Float64Array(2*32*32);
    for(let i=1;i<local.length;i++){
      if(tokens[i-1][2]>=.5)continue;const a=local[i-1],b=local[i],dx=f(b[0]-a[0]),dy=f(b[1]-a[1]);
      const recency=f(.1+(i-1)*(.9/(local.length-2||1)));
      for(const v of this.raster_fraction){const x=f(f(a[0]+f(dx*v))/extent),y=f(f(a[1]+f(dy*v))/extent);const px=Math.max(0,Math.min(31,roundEven(f(f(f(f(x+1)*29)/2)+1)))),py=Math.max(0,Math.min(31,roundEven(f(f(f(f(y+1)*29)/2)+1))));const index=py*32+px;image[index]=1;image[1024+index]=Math.max(image[1024+index],recency);}
    }
    return {image,extent};
  }
  convolution(input,height,width,key){
    const weights=this.w[key+'.weight'],bias=this.w[key+'.bias'],outH=Math.floor((height+1)/2),outW=Math.floor((width+1)/2),output=new Float64Array(weights.length*outH*outW);
    for(let oc=0;oc<weights.length;oc++)for(let y=0;y<outH;y++)for(let x=0;x<outW;x++){let sum=bias[oc];for(let ic=0;ic<weights[oc].length;ic++)for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++){const iy=y*2+ky-1,ix=x*2+kx-1;if(iy>=0&&iy<height&&ix>=0&&ix<width)sum+=weights[oc][ic][ky][kx]*input[ic*height*width+iy*width+ix];}output[oc*outH*outW+y*outW+x]=Math.max(0,sum);}
    return {values:output,height:outH,width:outW};
  }
  prefix(tokens,label){
    let state=new Float64Array(this.hidden);const condition=this.w['embedding.weight'][label];if(!condition)throw Error('Categoria inválida para o previsor.');
    let h;
    if(this.architecture==='deepsets'){
      const pooled=Array(this.hidden).fill(0);
      for(const token of tokens){const first=linear([...token,...condition],this.w['set_phi.0.weight'],this.w['set_phi.0.bias']).map(Math.tanh),encoded=linear(first,this.w['set_phi.2.weight'],this.w['set_phi.2.bias']).map(Math.tanh);for(let j=0;j<pooled.length;j++)pooled[j]+=encoded[j]/tokens.length;}
      h=linear([...pooled,tokens.length/this.context_events],this.w['set_rho.weight'],this.w['set_rho.bias']).map(Math.tanh);
    }else{
      for(const token of tokens)state=this.step([...token,...condition,...(this.architecture==='planned'?Array(this.latent).fill(0):[])],state);
      h=Array.from(state);
    }
    if(this.raster){const {image,extent}=this.rasterImage(tokens);let layer={values:image,height:32,width:32};for(const key of ['visual.0','visual.2','visual.4'])layer=this.convolution(layer.values,layer.height,layer.width,key);const visual=linear(layer.values,this.w['visual.7.weight'],this.w['visual.7.bias']).map(Math.tanh);h=linear([...h,...visual,extent],this.w['fusion.weight'],this.w['fusion.bias']).map(Math.tanh);}
    let logits;if(this.architecture==='trajectory'||this.architecture==='deepsets'){logits=linear(h,this.w['mode.weight'],this.w['mode.bias']);if(this.raster)logits=logits.slice(label*this.n_modes,(label+1)*this.n_modes);}else logits=linear(h,this.w[this.architecture==='planned'?'prior.weight':'head.weight'],this.w[this.architecture==='planned'?'prior.bias':'head.bias']);
    return {h,logits};
  }
  predict(tokens,label,{horizon=30,seed=0,temperature=.4,referenceTokens=null}={}){
    if(!this.horizons.includes(horizon))throw Error('Escolha um horizonte de 5, 15 ou 30 pontos.');
    const {h,logits}=this.prefix(tokens,label),random=randomSource(seed),reference=referenceTokens??tokens,last=reference.at(-1),out=[],ended=[];let position=last.slice(4,6),alive=true;
    if(this.architecture==='trajectory'||this.architecture==='deepsets'){
      const mode=seed?random.choose(softmax(logits,temperature)):argmax(logits),anchor=this.w.anchors[label][mode],embedding=this.w['mode_embedding.weight'][mode+(this.raster?label*this.n_modes:0)];
      const hidden=linear([...h,...embedding],this.w['regression_hidden.weight'],this.w['regression_hidden.bias']).map(Math.tanh),residual=linear(hidden,this.w['regression.weight'],this.w['regression.bias']),[c,s]=direction(reference);
      for(let k=0;k<horizon;k++){
        const r=residual.slice(k*6,k*6+6),a=anchor[k],x=a[0]+.35*Math.tanh(r[0]),y=a[1]+.35*Math.tanh(r[1]);const pen=a.slice(2,5).map((p,i)=>Math.log(Math.max(.02,p))+r[i+2]),end=argmax(pen)===2,status=end?2:+(softmax(pen.slice(0,2))[1]>(this.pen_threshold??.5));alive=alive&&status!==2;
        const next=alive?[last[4]+x*c-y*s,last[5]+x*s+y*c]:position;const clock=alive?softplus(Math.log(Math.expm1(Math.max(.02,a[5])))+r[5]):0;
        out.push([(next[0]-position[0])*this.position_scale/this.delta_scale,(next[1]-position[1])*this.position_scale/this.delta_scale,alive?+(status===1):1,clock,...next]);ended.push(!alive);position=next;
      }
      return {tokens:out,ended,mode,mode_probabilities:softmax(logits)};
    }
    let z=[],state=h;
    if(this.architecture==='planned'){z=logits.slice(0,this.latent).map((v,i)=>v+(seed?random.normal()*Math.exp(.5*Math.max(-5,Math.min(3,logits[this.latent+i])))*Math.sqrt(temperature):0));const plan=linear(z,this.w['plan.weight'],this.w['plan.bias']).map(Math.tanh);state=h.map((v,i)=>v+plan[i]);}
    let params=linear(state,this.w['head.weight'],this.w['head.bias']);
    for(let k=0;k<horizon;k++){
      const M=this.mixtures,weights=Array.from({length:M},(_,i)=>params[i*6]),j=seed?random.choose(softmax(weights,temperature)):argmax(weights),o=j*6;let dx=params[o+1],dy=params[o+2];
      if(seed){const zx=random.normal(),zy=random.normal(),rho=.95*Math.tanh(params[o+5]);dx+=Math.exp(Math.max(-4,Math.min(2,params[o+3])))*Math.sqrt(temperature)*zx;dy+=Math.exp(Math.max(-4,Math.min(2,params[o+4])))*Math.sqrt(temperature)*(rho*zx+Math.sqrt(1-rho*rho)*zy);}
      const pen=params.slice(6*M,6*M+3),status=seed?random.choose(softmax(pen,temperature)):argmax(pen)===2?2:+(softmax(pen.slice(0,2))[1]>(this.pen_threshold??.5));alive=alive&&status!==2;if(!alive){dx=0;dy=0;}const clock=alive?softplus(params.at(-1)):0;position=[position[0]+dx*this.delta_scale/this.position_scale,position[1]+dy*this.delta_scale/this.position_scale];const token=[dx,dy,alive?+(status===1):1,clock,...position];out.push(token);ended.push(!alive);state=this.step([...token,...this.w['embedding.weight'][label],...z],state);params=linear(state,this.w['head.weight'],this.w['head.bias']);
    }
    return {tokens:out,ended};
  }
  forecast(points,label,options={}){
    const encoded=this.encodePoints(points),prediction=this.predict(encoded.tokens,label,options),last=encoded.points.at(-1),origin=encoded.points[0],first=points[0],visible=[],trajectory=[];let previous=last.slice(0,2),time=last[3];
    prediction.tokens.forEach((token,i)=>{const x=origin[0]+token[4]*this.position_scale,y=origin[1]+token[5]*this.position_scale,dt=this.time_scale*Math.expm1(token[3]);time+=dt;trajectory.push([x,y,token[2],time]);if(!prediction.ended[i])visible.push([(x-previous[0])/128,(y-previous[1])/128,token[2],(x-first[0])/128,(y-first[1])/128,Math.log1p(dt)]);previous=[x,y];});
    return {tokens:visible,forecast_trajectory:trajectory,forecast_ended:prediction.ended,forecast_horizon:options.horizon??30,forecast_context_events:encoded.tokens.length};
  }
}
