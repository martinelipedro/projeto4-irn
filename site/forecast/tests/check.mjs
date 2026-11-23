import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {ForecastModel} from '../forecast.js';

const directory=fileURLToPath(new URL('../',import.meta.url));
const read=name=>JSON.parse(fs.readFileSync(path.join(directory,name),'utf8'));
for(const name of ['index.html','style.css','app.js','worker.js','forecast.js','icon.svg','relatorio.pdf','assets/gru.json','assets/mlp.json','assets/gru.reference.json','assets/mlp.reference.json','assets/examples.json','assets/study.json'])assert.ok(fs.existsSync(path.join(directory,name)),`Recurso obrigatório ausente: ${name}`);
for(const name of ['index.html','style.css','app.js','worker.js','forecast.js']){
  const content=fs.readFileSync(path.join(directory,name),'utf8');
  const refs=[...content.matchAll(/(?:src|href)=["']([^"']+)["']/g),...content.matchAll(/\bfrom\s*["']([^"']+)["']/g)].map(match=>match[1]);
  for(const ref of refs){if(/^(https?:|data:|#)/.test(ref))continue;assert.ok(!ref.startsWith('/'),`Caminho absoluto incompatível com Pages: ${ref}`);assert.ok(fs.existsSync(path.resolve(directory,ref)),`Referência ausente: ${ref}`);}
}
assert.equal(fs.readFileSync(path.join(directory,'relatorio.pdf')).subarray(0,5).toString(),'%PDF-');
const definitions={gru:read('assets/gru.json'),mlp:read('assets/mlp.json')};
assert.equal(definitions.gru.architecture,'trajectory');assert.equal(definitions.mlp.architecture,'deepsets');
assert.equal(definitions.gru.raster,false);assert.equal(definitions.mlp.raster,false);
assert.deepEqual(definitions.gru.classes,definitions.mlp.classes);
assert.equal(definitions.gru.seed,42);assert.equal(definitions.mlp.seed,42);
let maxError=0,fixtures=0;
const compare=(a,b,label,tolerance=.001)=>{assert.equal(a.length,b.length,label);for(let i=0;i<a.length;i++){const delta=Math.abs(a[i]-b[i]);maxError=Math.max(maxError,delta);assert.ok(delta<tolerance,`${label}[${i}] erro ${delta}`);}};
for(const name of ['gru','mlp']){
  const model=new ForecastModel(definitions[name]);
  const refs=read(`assets/${name}.reference.json`);
  for(const ref of refs){
    const encoded=model.encodePoints(ref.points,{resample:false});encoded.tokens.forEach((token,i)=>compare(token,ref.input[i],name+' entrada',1e-5));
    const state=model.prefix(ref.input,ref.label);compare(state.h,ref.h,name+' estado');compare(state.logits,ref.logits,name+' logits');
    const output=model.predict(ref.input,ref.label,{horizon:30});output.tokens.forEach((token,i)=>compare(token,ref.tokens[i],name+' previsão'));assert.deepEqual(output.ended,ref.ended);
    const five=model.predict(ref.input,ref.label,{horizon:5});assert.deepEqual(five.tokens,output.tokens.slice(0,5));
    const translated=ref.points.map(p=>[p[0]+120,p[1]-80,p[2],p[3]]);const shifted=model.encodePoints(translated,{resample:false});
    shifted.tokens.forEach((token,i)=>compare(token,encoded.tokens[i],name+' translação',2e-5));
    if(name==='mlp'){
      const tokens=ref.input.map(t=>t.slice()),permuted=[tokens[0],...tokens.slice(1,-4).reverse(),...tokens.slice(-4)];
      compare(model.prefix(tokens,ref.label).h,model.prefix(permuted,ref.label).h,'pooling invariável à ordem',1e-7);
    }
    fixtures++;
  }
}
const examplesData=read('assets/examples.json'),examples=examplesData.examples??examplesData;
assert.ok(examples.length>=12,'Esperados exemplos reais de todas as categorias.');
const categories=new Set();
for(const e of examples){
  assert.ok(['validation','test'].includes(e.source_split));assert.ok(e.id&&e.selection_rule,'Exemplo precisa de origem e regra de seleção.');
  assert.ok(Number.isInteger(e.label)&&e.label>=0&&e.label<definitions.gru.classes.length);categories.add(e.label);
  assert.ok(e.points.length>=38);assert.ok(e.points.every(p=>p.length>=4&&p.slice(0,4).every(Number.isFinite)));
  for(let i=1;i<e.points.length;i++)assert.ok(e.points[i][3]>=e.points[i-1][3],'Timestamp fora de ordem.');
}
assert.equal(categories.size,definitions.gru.classes.length);
const study=read('assets/study.json');assert.equal(study.horizons.length,3);assert.ok(study.horizons.every(r=>[5,15,30].includes(r.k)&&Number.isFinite(r.gru_ade)&&Number.isFinite(r.mlp_ade)));
console.log(JSON.stringify({passed:true,fixtures,maximum_absolute_error:maxError,categories:categories.size,examples:examples.length,relative_assets:true,model_parity:true,backend_required:false}));
