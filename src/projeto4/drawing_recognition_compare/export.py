"""Pesos reais para JavaScript, exemplos de validação e controles do reconhecimento."""
from pathlib import Path

import numpy as np
import torch

from ..io import ROOT, read_json, write_json
from ..drawing_forecast_compare.data import load
from .data import represent, permute_strokes
from .model import make_model
from .study import metrics, setup


def load_models(output):
    models={}
    for family in ['gru','mlp']:
        spec=read_json(output/family/'frozen_selection.json')['best']['spec']
        checkpoint=torch.load(output/family/f'h{spec["hidden"]}_lr{spec["lr"]}_seed42'/'best.pt',
                              map_location='cpu',weights_only=True)
        model=make_model(family,spec['hidden']).eval();model.load_state_dict(checkpoint['state_dict'])
        models[family]=model
    return models


@torch.inference_mode()
def predict(models, points=None, tokens=None, images=None, lengths=None):
    if points is not None:
        represented=[represent(p) for p in points]
        tokens=np.stack([v[0] for v in represented]);images=np.stack([v[1] for v in represented])
        lengths=np.array([len(p) for p in points])
    result={}
    for family,model in models.items():
        x=torch.from_numpy(np.asarray(tokens if family=='gru' else images,dtype=np.float32))
        ln=torch.tensor(lengths,dtype=torch.long)
        parts=[]
        for start in range(0,len(x),256):
            xb=x[start:start+256];lens=ln[start:start+256]
            if family=='gru':xb=xb[:,:int(lens.max())]
            parts.append(model(xb,lens).softmax(-1).numpy())
        result[family]=np.concatenate(parts)
    return result


def crossings(points):
    if len(points)<4:return 0
    a,b=points[:-1,:2],points[1:,:2]
    valid=points[:-1,2]<.5
    def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    a,b=a[:,None,:],b[:,None,:];c,d=a.transpose(1,0,2),b.transpose(1,0,2)
    proper=(cross(b-a,c-a)*cross(b-a,d-a)<-1e-7)&(cross(d-c,a-c)*cross(d-c,b-c)<-1e-7)
    proper&=valid[:,None]&valid[None,:]
    return int(np.triu(proper,k=2).sum())


def build(config=None):
    config=config or read_json(ROOT/'configs/recognition.json')
    output=ROOT/'results'/config['name'];source=ROOT/'results'/config['source_experiment']
    setup(42,config['threads'],device='cpu')
    models=load_models(output)
    meta=read_json(output/'data_manifest.json')
    destination=ROOT/'site/assets/recognition';destination.mkdir(parents=True,exist_ok=True)
    for family,model in models.items():
        spec=read_json(output/family/'frozen_selection.json')['best']['spec']
        write_json(destination/(family+'.json'),{'family':family,'hidden':spec['hidden'],
                  'classes':meta['classes'],'labels':meta['labels'],'seed':42,
                  'weights':{key:value.detach().numpy().tolist() for key,value in model.state_dict().items()}})
    rows=load(source);validation=[r for r in rows if r['split']=='validation'];test_rows=[r for r in rows if r['split']=='test']
    y=np.array([r['label'] for r in validation]);prefixes=[r['points'][:max(2,int(np.floor(len(r['points'])*.5+.5)))] for r in validation]
    validation_p=predict(models,points=prefixes)
    full_p=predict(models,points=[r['points'] for r in validation])
    indices=[];rules=[]
    for label in range(12):
        eligible=np.flatnonzero((y==label)&(full_p['gru'].argmax(-1)==y)&(full_p['mlp'].argmax(-1)==y))
        if not len(eligible):eligible=np.flatnonzero(y==label)
        confidence=(validation_p['gru'][eligible,label]+validation_p['mlp'][eligible,label])/2
        chosen=eligible[np.argsort(confidence,kind='stable')[len(eligible)//2]]
        indices.append(int(chosen));rules.append('Um por categoria: mediana da confiança média no prefixo de 50%, entre acertos completos de ambas quando disponíveis; somente validação.')
    gap=validation_p['gru'][np.arange(len(y)),y]-validation_p['mlp'][np.arange(len(y)),y]
    for family,ordered in [('GRU',np.argsort(-gap)),('MLP',np.argsort(gap))]:
        chosen=0
        own='gru' if family=='GRU' else 'mlp';other='mlp' if own=='gru' else 'gru'
        exclusive=(validation_p[own].argmax(-1)==y)&(validation_p[other].argmax(-1)!=y)
        if exclusive.sum()>=2:
            ordered=[i for i in ordered if exclusive[i]]
        for index in ordered:
            if int(index) in indices:continue
            indices.append(int(index));rules.append('Caso extremo ilustrativo de validação: acerto exclusivo quando disponível e maior diferença de probabilidade da classe verdadeira em 50% favorece '+family+'. Não representa a média do teste.')
            chosen+=1
            if chosen==2:break
    examples=[]
    for i,rule in zip(indices,rules):
        row=validation[i];fractions=np.arange(.1,1.01,.1);cuts=[max(2,int(np.floor(len(row['points'])*f+.5))) for f in fractions]
        probs=predict(models,points=[row['points'][:cut] for cut in cuts])
        suffix='favorece GRU em 50%' if 'favorece GRU' in rule else 'favorece MLP em 50%' if 'favorece MLP' in rule else 'mediana de validação'
        examples.append({'id':row['id'],'label':row['label'],'label_name':meta['labels'][row['label']],
                         'title':meta['labels'][row['label']]+' · '+suffix,
                         'source_split':'validation','points':row['points'].tolist(),'selection_rule':rule,
                         'history':[{'fraction':float(f),'cut':cut,'gru':probs['gru'][j].tolist(),
                                     'mlp':probs['mlp'][j].tolist()} for j,(f,cut) in enumerate(zip(fractions,cuts))]})
    write_json(destination/'examples.json',{'examples':examples})
    references=[]
    for i in indices[:12]:
        row=validation[i]
        for f in [.25,.75]:
            points=row['points'][:max(2,int(np.floor(len(row['points'])*f+.5)))]
            tokens,image=represent(points);p=predict(models,points=[points])
            references.append({'points':points.tolist(),'tokens':tokens[:len(points)].tolist(),
                               'image':image.tolist(),'gru':p['gru'][0].tolist(),'mlp':p['mlp'][0].tolist()})
    write_json(destination/'reference.json',references)
    print('PESOS E EXEMPLOS EXPORTADOS',len(examples),flush=True)

    selected=np.concatenate([np.flatnonzero(y==label)[:20] for label in range(12)])
    points=[validation[i]['points'][:max(2,int(np.floor(len(validation[i]['points'])*.75+.5)))] for i in selected]
    represented=[represent(p) for p in points]
    tokens=np.stack([r[0] for r in represented]);images=np.stack([r[1] for r in represented]);lengths=np.array([len(p) for p in points])
    scenarios={'original':predict(models,tokens=tokens,images=images,lengths=lengths)}
    for scenario,column in [('hidden_penup',2),('zero_clock',3)]:
        x=tokens.copy();x[:,:,column]=0
        scenarios[scenario]=predict(models,tokens=x,images=images,lengths=lengths)
    uniform=[];reordered=[];bitmap_equal=True;multistroke=0
    max_geometry_error=0.
    for p in points:
        before=p.copy();before[:,3]=np.arange(len(p),dtype=np.float32)*.1
        # O fragmento final pode estar aberto no corte; fechar nas DUAS ordens.
        before[-1,2]=1
        after=permute_strokes(before)
        np.testing.assert_array_equal(np.sort(before[:,2]),np.sort(after[:,2]))
        if np.count_nonzero(p[:-1,2])>0:multistroke+=1
        bitmap_equal &= np.array_equal(represent(before)[1],represent(after)[1])
        uniform.append(before);reordered.append(after)
        changed=p.copy();changed[:,:2]=p[:,:2]*2+[128,-64]
        a,b=represent(p),represent(changed)
        max_geometry_error=max(max_geometry_error,float(np.max(np.abs(a[0]-b[0]))))
        if not np.array_equal(a[1],b[1]):raise AssertionError('Bitmap mudou em normalização geométrica.')
    if not bitmap_equal:raise AssertionError('Ordem de traços alterou a imagem de controle.')
    scenarios['uniform_clock']=predict(models,points=uniform)
    scenarios['reordered_strokes_uniform_clock']=predict(models,points=reordered)
    probes=[]
    for scenario,probabilities in scenarios.items():
        baseline=scenarios['uniform_clock' if scenario=='reordered_strokes_uniform_clock' else 'original']
        for family,p in probabilities.items():
            m=metrics(y[selected],p)
            probes.append({'family':family,'scenario':scenario,'n':len(selected),
                           'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                           'mean_abs_probability_change':float(np.abs(p-baseline[family]).mean()),
                           'prediction_change_fraction':float(np.mean(p.argmax(-1)!=baseline[family].argmax(-1)))})
    paired=[];stable=[];crossing_groups=[];rng=np.random.default_rng(20261123)
    probabilities={}
    for family in models:
        with np.load(output/family/'test_seed42.npz') as array:
            probabilities[family]=array['probabilities'];truth=array['truth']
        if not np.array_equal(truth,[r['label'] for r in test_rows]):raise AssertionError('Alinhamento de IDs de teste.')
    for j,f in enumerate(config['fractions']):
        correct={family:p[:,j].argmax(-1)==truth for family,p in probabilities.items()}
        delta=correct['gru'].astype(float)-correct['mlp'].astype(float)
        samples=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(1)
        paired.append({'fraction':f,'delta_accuracy':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
                       'n_drawings':len(truth),'seed':42})
    counts=np.array([crossings(r['points'][:max(2,int(np.floor(len(r['points'])*.75+.5)))]) for r in test_rows])
    for family,p in probabilities.items():
        correct=p.argmax(-1)==truth[:,None];stable_correct=np.logical_and.accumulate(correct[:,::-1],axis=1)[:,::-1]
        stable.append({'family':family,'n':len(truth),'stable_by_fraction':{str(f):float(stable_correct[:,j].mean()) for j,f in enumerate(config['fractions'])},
                       'never_stably_correct':float((~stable_correct.any(1)).mean())})
        for group,mask in [('sem_cruzamentos',counts==0),('com_cruzamentos',counts>0)]:
            m=metrics(truth[mask],p[mask,2])
            crossing_groups.append({'family':family,'group':group,'n':int(mask.sum()),'accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
    diagnostics={'paired_bootstrap':paired,'probes':probes,'crossing_groups':crossing_groups,'early_stable':stable,
                 'controls':{'same_bitmap':bool(bitmap_equal),'n_probe':len(selected),'n_multistroke':multistroke,
                             'max_geometry_token_difference':max_geometry_error,'translation':[128,-64],'scale':2,
                             'order_baseline':'uniform_clock','order_probe':'reordered_strokes_uniform_clock',
                             'order_baseline_final_penup_forced':True,
                             'probe_split':'validation','probe_sampling':'Primeiros 20 IDs de cada categoria no manifesto, prefixo de 75%, sem selecionar por erro.',
                             'test_ids':[r['id'] for r in test_rows]},
                 'bootstrap_note':'2000 reamostragens de desenhos pareados, seed42 fixa; unidades0..1; não inclui incerteza entre treinamentos.',
                 'test_previously_observed':True}
    write_json(output/'diagnostics.json',diagnostics)
    test=read_json(output/'test_metrics.json')
    write_json(destination/'study.json',{'fractions':[{'fraction':f,'gru_accuracy':test['gru']['mean'][str(f)]['accuracy'],
                  'mlp_accuracy':test['mlp']['mean'][str(f)]['accuracy'],'gru_f1':test['gru']['mean'][str(f)]['macro_f1'],
                  'mlp_f1':test['mlp']['mean'][str(f)]['macro_f1']} for f in config['fractions']],
                  'note':'Médias das três inicializações 42/43/44 no teste previamente observado. Demonstração com seed 42; exemplos exclusivamente de validação.',
                  'probes':probes,'controls':{k:v for k,v in diagnostics['controls'].items() if k!='test_ids'}})
    write_json(output/'export_audit.json',{'passed':True,'fixtures':len(references),'examples':len(examples),
               'all_examples_validation':True,'classes_provided_as_input':False,'image_has_time_channel':False,
               'same_bitmap_in_order_probe':bool(bitmap_equal),'normalization_control':max_geometry_error})
    print('DIAGNÓSTICOS E REFERÊNCIAS PRONTOS',flush=True)


if __name__=='__main__':build()
