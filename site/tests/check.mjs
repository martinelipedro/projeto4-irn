import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {RecognitionModel,represent,reverseStrokes,uniformClock} from '../recognition.js';
const directory=fileURLToPath(new URL('../',import.meta.url));
const read=name=>JSON.parse(fs.readFileSync(path.join(directory,name),'utf8'));
for(const name of ['index.html','style.css','app.js','worker.js','recognition.js','icon.svg','relatorio.pdf','assets/recognition/gru.json','assets/recognition/mlp.json','assets/recognition/reference.json','assets/recognition/examples.json','assets/recognition/study.json'])assert.ok(fs.existsSync(path.join(directory,name)),`Recurso ausente: ${name}`);
assert.equal(fs.readFileSync(path.join(directory,'relatorio.pdf')).subarray(0,5).toString(),'%PDF-');
for(const name of ['index.html','style.css','app.js','worker.js','recognition.js']){
 const content=fs.readFileSync(path.join(directory,name),'utf8');
 for(const match of content.matchAll(/(?:src|href)=["']([^"']+)["']|\bfrom\s*["']([^"']+)["']/g)){
  const ref=match[1]??match[2];if(/^(https?:|data:|#)/.test(ref))continue;assert.ok(!ref.startsWith('/'),'Caminho absoluto: '+ref);assert.ok(fs.existsSync(path.resolve(directory,ref)),'Referência ausente: '+ref);
 }
}
const definitions=Object.fromEntries(['gru','mlp'].map(name=>[name,read(`assets/recognition/${name}.json`)]));
assert.equal(definitions.gru.family,'gru');assert.equal(definitions.mlp.family,'mlp');assert.equal(definitions.gru.seed,42);assert.equal(definitions.mlp.seed,42);assert.equal(definitions.gru.classes.length,12);assert.deepEqual(definitions.gru.classes,definitions.mlp.classes);
assert.ok('gru.weight_ih_l0' in definitions.gru.weights);assert.ok('layers.0.weight' in definitions.mlp.weights);assert.equal(definitions.mlp.weights['layers.0.weight'][0].length,1024);
assert.ok(!Object.keys(definitions.mlp.weights).some(key=>/gru|set_phi|set_rho|embedding|conv/i.test(key)),'MLP não pode conter recorrência, Deep Sets, embedding ou convolução.');
const models=Object.fromEntries(['gru','mlp'].map(name=>[name,new RecognitionModel(definitions[name])]));
let maxError=0,fixtures=0;
function close(actual,expected,tolerance,label){assert.equal(actual.length,expected.length,label);for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);maxError=Math.max(maxError,error);assert.ok(error<tolerance,`${label}[${i}]: ${error}`);}}
for(const ref of read('assets/recognition/reference.json')){
 const input=represent(ref.points);assert.deepEqual(input.image,ref.image.flat());
 const tokens=ref.tokens.slice(0,ref.points.length);input.tokens.forEach((row,i)=>close(row,tokens[i],2e-6,'token'));
 for(const name of ['gru','mlp'])close(models[name].probabilities(input.tokens,input.image),ref[name],.0002,name+' probabilidade');
 const translated=represent(ref.points.map(p=>[p[0]+120,p[1]-80,p[2],p[3]]));assert.deepEqual(translated.image,input.image);translated.tokens.forEach((row,i)=>close(row,input.tokens[i],2e-6,'translação'));
 const baseline=represent(uniformClock(ref.points)),reordered=represent(reverseStrokes(uniformClock(ref.points)));
 assert.deepEqual(baseline.image,reordered.image,'Permutar traços inteiros deve preservar o raster.');
 assert.deepEqual(models.mlp.probabilities(baseline.tokens,baseline.image),models.mlp.probabilities(reordered.tokens,reordered.image));fixtures++;
}
const examples=read('assets/recognition/examples.json').examples;
assert.ok(examples.length>=12);assert.equal(new Set(examples.map(e=>e.label)).size,12);
for(const row of examples){assert.equal(row.source_split,'validation');assert.ok(row.selection_rule);assert.ok(row.points.length>=12&&row.points.length<=180);assert.ok(row.history.length>=4);for(const item of row.history){assert.ok(item.cut<=row.points.length);assert.equal(item.gru.length,12);assert.equal(item.mlp.length,12);}}
const study=read('assets/recognition/study.json');assert.equal(study.fractions.length,4);assert.ok(study.fractions.every(row=>row.gru_accuracy>=0&&row.gru_accuracy<=1&&row.mlp_accuracy>=0&&row.mlp_accuracy<=1));
console.log(JSON.stringify({passed:true,fixtures,maximum_absolute_error:maxError,categories:12,examples:examples.length,conventional_mlp:true,category_not_input:true,same_image_order_control:true,relative_assets:true,model_parity:true,backend_required:false}));
