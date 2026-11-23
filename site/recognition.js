// Inferência dos classificadores treinados. Não existe entrada de categoria.
const f=Math.fround;
const sigmoid=x=>1/(1+Math.exp(-x));
export function softmax(values){const peak=Math.max(...values),out=Array.from(values,x=>Math.exp(x-peak)),sum=out.reduce((a,b)=>a+b,0);return out.map(x=>x/sum);}
function linear(x,weights,bias){const out=new Float64Array(weights.length);for(let i=0;i<weights.length;i++){const row=weights[i];let sum=bias?.[i]??0;for(let j=0;j<row.length;j++)sum+=row[j]*x[j];out[i]=sum;}return out;}

export function represent(points){
  if(!points.length||points.length>180)throw Error('Use entre 1 e 180 eventos observados.');
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(const p of points){if(p.length<4||!p.slice(0,4).every(Number.isFinite))throw Error('Ponto inválido.');minX=Math.min(minX,p[0]);minY=Math.min(minY,p[1]);maxX=Math.max(maxX,p[0]);maxY=Math.max(maxY,p[1]);}
  const extent=Math.max(maxX-minX,maxY-minY,1),midX=(minX+maxX)/2,midY=(minY+maxY)/2;
  const xy=points.map(p=>[(p[0]-midX)/extent,(p[1]-midY)/extent]);
  const tokens=points.map((p,i)=>{const a=xy[Math.max(i-1,0)],previous=points[Math.max(i-1,0)];return [f(xy[i][0]-a[0]),f(xy[i][1]-a[1]),f(p[2]),f(Math.log1p(Math.max(0,p[3]-previous[3])/.1)),f(xy[i][0]),f(xy[i][1])];});
  const pixels=xy.map(p=>[p[0]*27+15.5,p[1]*27+15.5]),image=new Uint8Array(1024);
  function stamp(x,y){const xx=Math.floor(x+.5),yy=Math.floor(y+.5);for(const [dx,dy] of [[0,0],[-1,0],[1,0],[0,-1],[0,1]]){const a=xx+dx,b=yy+dy;if(a>=0&&a<32&&b>=0&&b<32)image[b*32+a]=1;}}
  for(const [x,y] of pixels)stamp(x,y);
  for(let i=1;i<pixels.length;i++){if(points[i-1][2]>=.5)continue;const a=pixels[i-1],b=pixels[i],dx=b[0]-a[0],dy=b[1]-a[1],n=Math.max(1,Math.ceil(Math.max(Math.abs(dx),Math.abs(dy))*2));for(let j=0;j<=n;j++)stamp(a[0]+dx*(j/n),a[1]+dy*(j/n));}
  return {tokens,image:Array.from(image),extent,center:[midX,midY]};
}

// Controle OOD: mesmo relógio e mesmo fechamento do trecho parcial nas duas ordens.
export function uniformClock(points){return points.map((p,i)=>[f(p[0]),f(p[1]),i===points.length-1?1:f(p[2]),f(i*.1)]);}
export function reverseStrokes(points){
  const strokes=[];let start=0;
  for(let i=0;i<points.length;i++)if(points[i][2]>.5||i===points.length-1){const stroke=points.slice(start,i+1).map(p=>p.map(f));if(stroke.length){stroke.at(-1)[2]=1;strokes.push(stroke);}start=i+1;}
  if(strokes.length<2)return points.map(p=>p.slice());
  return uniformClock(strokes.reverse().flat());
}

export function samplePoints(points,step=14){
  const kept=[];
  for(let i=0;i<points.length;i++){const p=points[i],last=kept.at(-1);if(!last||points[i-1]?.[2]||p[2]||Math.hypot(p[0]-last[0],p[1]-last[1])>=step||i===points.length-1)kept.push(p.slice());}
  return kept.slice(0,180);
}

export class RecognitionModel{
  constructor(data){Object.assign(this,data);this.w=data.weights;if(!['gru','mlp'].includes(this.family))throw Error('Família de classificador desconhecida.');}
  probabilities(tokens,image){
    if(this.family==='mlp'){
      const h=linear(image,this.w['layers.0.weight'],this.w['layers.0.bias']);for(let i=0;i<h.length;i++)h[i]=Math.max(0,h[i]);
      const z=linear(h,this.w['layers.3.weight'],this.w['layers.3.bias']);for(let i=0;i<z.length;i++)z[i]=Math.max(0,z[i]);
      return softmax(linear(z,this.w['layers.6.weight'],this.w['layers.6.bias']));
    }
    const H=this.hidden;let state=new Float64Array(H);
    for(const token of tokens){const a=linear(token,this.w['gru.weight_ih_l0'],this.w['gru.bias_ih_l0']),b=linear(state,this.w['gru.weight_hh_l0'],this.w['gru.bias_hh_l0']),next=new Float64Array(H);for(let i=0;i<H;i++){const r=sigmoid(a[i]+b[i]),z=sigmoid(a[H+i]+b[H+i]),n=Math.tanh(a[2*H+i]+r*b[2*H+i]);next[i]=(1-z)*n+z*state[i];}state=next;}
    return softmax(linear(state,this.w['head.1.weight'],this.w['head.1.bias']));
  }
  recognize(points){const input=represent(points);return {probabilities:this.probabilities(input.tokens,input.image),...input};}
}
