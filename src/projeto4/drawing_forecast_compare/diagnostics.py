"""Diagnósticos de previsão GRU × DeepSets, posteriores à seleção por validação.

As sensibilidades não retreinam pesos e não são intervenções causais sobre o
comportamento de pessoas. Nenhuma tarefa de reconhecimento é avaliada aqui.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..io import ROOT,read_json,write_json,write_csv
from .data import load,tokens_from_points
from .trajectory import frame,to_global
from .model import new_model

MODELS=('gru','mlp')
LABELS={'gru':'GRU recorrente','mlp':'MLP DeepSets'}
COLORS={'gru':'#007F89','mlp':'#7959A0'}
INK='#203448';REAL='#D75B42';BOOTSTRAP=500


def _save(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def prefix_descriptors(points):
    """Descritores apenas do passado; cruzamentos próprios, excluindo extremos."""
    points=np.asarray(points);xy=points[:,:2];delta=np.diff(xy,axis=0)
    valid=(points[:-1,2]<.5)&(np.linalg.norm(delta,axis=1)>1e-8)
    ink=np.flatnonzero(valid);crossings=0
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for at,i in enumerate(ink):
        later=ink[at+1:];later=later[later>i+1]
        if not len(later):continue
        a=xy[i];r=delta[i];b=xy[later];s=delta[later]
        denominator=cross(r,s);nonparallel=np.abs(denominator)>1e-8
        t=np.divide(cross(b-a,s),denominator,out=np.zeros(len(later)),where=nonparallel)
        u=np.divide(cross(b-a,r),denominator,out=np.zeros(len(later)),where=nonparallel)
        crossings+=int(np.sum(nonparallel&(t>1e-6)&(t<1-1e-6)&(u>1e-6)&(u<1-1e-6)))
    return {'crossings':crossings,'pen_up_count':int(np.sum(points[:,2]>.5)),
            'length':len(points),'ink_length':float(np.linalg.norm(delta[valid],axis=1).sum()),
            'past_extent':float(np.ptp(xy,axis=0).max())}


def _window_metrics(tokens,ended,truth,metadata):
    error=(tokens[:,4:6]-truth[:,4:6])*metadata['position_scale']
    distance=np.linalg.norm(error,axis=1);y=truth[:,2]>.5;p=tokens[:,2]>.5
    dt=metadata['time_scale']*np.expm1(tokens[:,3]);real_dt=metadata['time_scale']*np.expm1(truth[:,3])
    return {'ade':float(distance.mean()),'fde':float(distance[-1]),
        'clock_mae':float(np.abs(dt-real_dt).mean()),'elapsed_error':float(abs(dt.sum()-real_dt.sum())),
        'pen_tp':int(np.sum(y&p)),'pen_fp':int(np.sum(~y&p)),'pen_fn':int(np.sum(y&~p)),
        'premature_end':float(np.any(ended))}


def _aggregate(rows,repetitions=BOOTSTRAP):
    if not rows:raise ValueError('Grupo sem observações.')
    ids,inverse=np.unique([r['id'] for r in rows],return_inverse=True)
    n=np.bincount(inverse);rng=np.random.default_rng(20261118)
    draws=rng.integers(0,len(ids),size=(repetitions,len(ids)))
    ade=np.array([r['ade'] for r in rows]);sums=np.bincount(inverse,weights=ade)
    sample=sums[draws].sum(1)/n[draws].sum(1)
    tp=sum(r['pen_tp'] for r in rows);fp=sum(r['pen_fp'] for r in rows);fn=sum(r['pen_fn'] for r in rows)
    return {'n_prefixes':len(rows),'n_drawings':len(ids),'ade':float(ade.mean()),
        'fde':float(np.mean([r['fde'] for r in rows])),'median_ade':float(np.median(ade)),
        'p90_ade':float(np.quantile(ade,.9)),'clock_mae':float(np.mean([r['clock_mae'] for r in rows])),
        'elapsed_error':float(np.mean([r['elapsed_error'] for r in rows])),
        'pen_precision':tp/max(1,tp+fp),'pen_recall':tp/max(1,tp+fn),'pen_f1':2*tp/max(1,2*tp+fp+fn),
        'premature_end':float(np.mean([r['premature_end'] for r in rows])),
        'ade_ci95':[float(v) for v in np.quantile(sample,[.025,.975])]}


def _paired(gru,mlp,repetitions=BOOTSTRAP):
    key=lambda r:(r['id'],r['cut'],r['k'])
    a={key(r):r for r in gru};b={key(r):r for r in mlp}
    if a.keys()!=b.keys():raise ValueError('As redes não têm exatamente os mesmos IDs/cortes/horizontes.')
    keys=sorted(a);ids,inverse=np.unique([k[0] for k in keys],return_inverse=True)
    delta=np.array([a[k]['ade']-b[k]['ade'] for k in keys]);counts=np.bincount(inverse)
    sums=np.bincount(inverse,weights=delta);rng=np.random.default_rng(20261118)
    draws=rng.integers(0,len(ids),size=(repetitions,len(ids)));sample=sums[draws].sum(1)/counts[draws].sum(1)
    return {'n_prefixes':len(keys),'n_drawings':len(ids),'delta_ade':float(delta.mean()),
            'delta_ci95':[float(v) for v in np.quantile(sample,[.025,.975])],
            'gru_win_fraction':float(np.mean(delta<0))}


def artifact_records(directory,rows,metadata,seed=42):
    records=[];predictions={}
    for family in MODELS:
        values=read_json(directory/'models'/family/f'test_predictions_seed{seed}.json')
        predictions[family]={(p['id'],p['cut']):p for p in values}
        for pred in values:
            row=rows[pred['id']];cut=pred['cut'];tokens=np.asarray(pred['tokens']);end=np.asarray(pred['ended'])
            descriptors=prefix_descriptors(row['points'][:cut])
            for k in [5,15,30]:
                if len(row['x'])<cut+k:continue
                records.append({'model':family,'seed':seed,'id':pred['id'],'label':row['label'],'cut':cut,'k':k,
                    **descriptors,**_window_metrics(tokens[:k],end[:k],row['x'][cut:cut+k],metadata)})
    return records,predictions


def load_models(config,directory,seed=42,*,families=MODELS):
    models={}
    for family in families:
        folder=directory/'models'/family;selection=read_json(folder/'selection.json')
        if selection.get('test_used_for_selection',False):raise ValueError('Seleção declarou uso de teste.')
        chosen=selection['best'];spec=chosen['spec'];checkpoint=torch.load(folder/'runs'/f"{spec['id']}_seed{seed}"/'best.pt',map_location='cpu',weights_only=True)
        model=new_model(config,spec);model.load_state_dict(checkpoint['state_dict']);model.eval()
        model.pen_threshold=chosen.get('pen_threshold',.2);models[family]=model
    return models


@torch.inference_mode()
def fixed_reference_forecast(model,prefix,reference,labels,metadata,steps):
    """Mesmo decoder; somente o encoder vê o prefixo perturbado. Referência congelada."""
    model.eval();h,logits=model(prefix,labels);mode=logits.argmax(-1)
    xy,pen,clock,_=model.decode(h,labels,mode)
    last=reference[:,-1,4:6];position=last.clone()
    coordinates=to_global(xy,last,frame(reference))
    end=pen.argmax(-1)==2;lift=pen[:,:,:2].softmax(-1)[:,:,1]>model.pen_threshold
    status=torch.where(end,torch.full_like(end,2,dtype=torch.long),lift.long());alive=(status!=2).long().cumprod(1).bool()
    output=[]
    for j in range(steps):
        new=torch.where(alive[:,j,None],coordinates[:,j],position)
        delta=(new-position)*metadata['position_scale']/metadata['delta_scale']
        up=torch.where(alive[:,j],(status[:,j]==1).float(),torch.ones(len(prefix),device=prefix.device))
        output.append(torch.cat([delta,up[:,None],(clock[:,j]*alive[:,j])[:,None],new],-1));position=new
    return torch.stack(output,1),~alive[:,:steps]


def _contexts(rows,cut,metadata,window=None,scale=1.,translation=(0.,0.)):
    xs=[];targets=[]
    for row in rows:
        start=0 if window is None else cut-window
        points=row['points'][start:cut].copy();origin=points[0,:2].copy()
        points[:,:2]=origin+scale*(points[:,:2]-origin)+np.array(translation)
        x=tokens_from_points(points,metadata)
        target=row['x'][cut:cut+15].copy();target[:,4:6]=(row['points'][cut:cut+15,:2]-origin)*scale/metadata['position_scale']
        xs.append(x);targets.append(target)
    return np.stack(xs),np.stack(targets)


def shuffle_fixed_suffix(prefix,seed=20261118):
    """Permuta vetores prontos; mantém primeiro e últimos três segmentos de tinta."""
    result=prefix.clone();changed=[];rng=np.random.default_rng(seed)
    for b in range(len(prefix)):
        valid=(prefix[b,:-1,2]<.5)&(prefix[b,1:,:2].square().sum(-1)>1e-9)
        endpoints=torch.where(valid)[0]+1
        start=max(1,int(endpoints[-3])-1) if len(endpoints)>=3 else 1
        if start>2:
            indexes=np.arange(1,start);permutation=rng.permutation(indexes)
            result[b,torch.from_numpy(indexes)]=prefix[b,torch.from_numpy(permutation)]
            changed.append(int(np.any(indexes!=permutation)))
        else:changed.append(0)
    torch.testing.assert_close(frame(prefix),frame(result),rtol=0,atol=1e-6)
    torch.testing.assert_close(prefix[:,-1,4:6],result[:,-1,4:6],rtol=0,atol=0)
    return result,np.array(changed)


@torch.inference_mode()
def sensitivities(config,models,all_rows,metadata,per_class=20):
    """Subamostra fixa balanceada; nenhuma escolha por acerto/erro dos modelos."""
    sample=[]
    for label in range(len(config['classes'])):
        eligible=sorted([r for r in all_rows if r['split']=='test' and r['label']==label and len(r['x'])>=47],key=lambda r:r['id'])
        sample+=eligible[:per_class]
    cut=32;k=15;x,y=_contexts(sample,cut,metadata);prefix=torch.from_numpy(x)
    labels=torch.tensor([r['label'] for r in sample]);lengths=torch.full((len(sample),),cut)
    translated,_=_contexts(sample,cut,metadata,translation=(120.,-80.))
    scaled075,y075=_contexts(sample,cut,metadata,scale=.75);scaled125,y125=_contexts(sample,cut,metadata,scale=1.25)
    shuffled,changed=shuffle_fixed_suffix(prefix)
    masked_pen=prefix.clone();masked_pen[:,:,2]=0
    masked_time=prefix.clone();masked_time[:,:,3]=0
    conditions=[('baseline',prefix,y,1.,False),('translate_120_-80',torch.from_numpy(translated),y,1.,False),
        ('scale_075',torch.from_numpy(scaled075),y075,.75,False),('scale_125',torch.from_numpy(scaled125),y125,1.25,False),
        ('shuffle_interior_fixed_suffix',shuffled,y,1.,True),('mask_pen_fixed_frame',masked_pen,y,1.,True),
        ('mask_time',masked_time,y,1.,False)]
    results=[];controls={'translation':{'max_feature_delta':float(np.max(np.abs(translated-x))),
                    'expected':'Controle da codificação relativa; não é uma invariância aprendida pela rede.'},
        'permutation':{'n_changed':int(changed.sum()),'n_total':len(sample),'max_frame_delta':float((frame(prefix)-frame(shuffled)).abs().max()),
                    'operation':'Vetores já codificados; primeiro e sufixo geométrico preservados. Sequência pode não corresponder a movimentos físicos.'},
        'decoder_parity':{}}
    for family,model in models.items():
        base,end=model.forecast(prefix,lengths,labels,metadata,k)
        fixed,fixed_end=fixed_reference_forecast(model,prefix,prefix,labels,metadata,k)
        torch.testing.assert_close(base,fixed,rtol=1e-6,atol=1e-6);assert torch.equal(end,fixed_end)
        controls['decoder_parity'][family]=float((base-fixed).abs().max())
        base_values=[_window_metrics(base[i].numpy(),end[i].numpy(),y[i],metadata) for i in range(len(sample))]
        base_ade=float(np.mean([r['ade'] for r in base_values]));_,base_logits=model(prefix,labels)
        for name,xx,target,scale,frozen in conditions:
            if frozen:pred,ended=fixed_reference_forecast(model,xx,prefix,labels,metadata,k)
            else:pred,ended=model.forecast(xx,lengths,labels,metadata,k)
            rows=[{'id':r['id'],'cut':cut,'k':k,**_window_metrics(pred[i].numpy(),ended[i].numpy(),target[i],metadata)} for i,r in enumerate(sample)]
            aggregate=_aggregate(rows)
            rng=np.random.default_rng(20261118);draws=rng.integers(0,len(rows),size=(BOOTSTRAP,len(rows)))
            a=np.array([r['ade']/scale for r in rows]);b=np.array([r['ade'] for r in base_values])
            ratios=a[draws].mean(1)/b[draws].mean(1)
            max_delta=float(((pred[:,:,4:6]/scale-base[:,:,4:6])*metadata['position_scale']).abs().max())
            results.append({'model':family,'scenario':name,'cut':cut,'k':k,**aggregate,
                'ade_rescaled':aggregate['ade']/scale,'ratio_vs_baseline':aggregate['ade']/scale/base_ade,
                'ratio_ci95':[float(v) for v in np.quantile(ratios,[.025,.975])],
                'max_feature_delta':float((xx-prefix).abs().max()),'max_prediction_delta':max_delta,
                'frame_frozen':frozen,'n_order_actually_changed':int(changed.sum()) if name.startswith('shuffle') else None})
            if name=='translate_120_-80':controls['translation'][family+'_max_prediction_delta_pixels']=max_delta
            if name.startswith('shuffle'):
                _,logits=model(xx,labels)
                controls['permutation'][family+'_max_logit_delta']=float((logits-base_logits).abs().max())
                controls['permutation'][family+'_mode_changed_fraction']=float((logits.argmax(-1)!=base_logits.argmax(-1)).float().mean())
                controls['permutation'][family+'_max_prediction_delta_pixels']=max_delta
    return results,controls,sample


@torch.inference_mode()
def fixed_origin_windows(config,models,all_rows,metadata,per_class=20):
    sample=[]
    for label in range(len(config['classes'])):
        eligible=sorted([r for r in all_rows if r['split']=='test' and r['label']==label and len(r['x'])>=94],key=lambda r:r['id'])
        sample+=eligible[:per_class]
    results=[]
    for window in [8,16,32,64]:
        x,y=_contexts(sample,64,metadata,window=window);prefix=torch.from_numpy(x)
        labels=torch.tensor([r['label'] for r in sample]);lengths=torch.full((len(sample),),window)
        for family,model in models.items():
            pred,end=model.forecast(prefix,lengths,labels,metadata,15)
            values=[{'id':r['id'],'cut':64,'k':15,**_window_metrics(pred[i].numpy(),end[i].numpy(),y[i],metadata)} for i,r in enumerate(sample)]
            results.append({'model':family,'origin':64,'window':window,'k':15,**_aggregate(values)})
    return results


@torch.inference_mode()
def site_manifest(config,models,all_rows,metadata,output):
    """Exemplos de validação. Regras simples predefinidas; vitórias têm rótulo explícito."""
    validation=sorted([r for r in all_rows if r['split']=='validation' and len(r['x'])>=62],key=lambda r:r['id'])
    pool=[]
    for label in range(len(config['classes'])):pool.extend([r for r in validation if r['label']==label][:40])
    x,y=_contexts(pool,32,metadata);prefix=torch.from_numpy(x);labels=torch.tensor([r['label'] for r in pool]);lengths=torch.full((len(pool),),32)
    # A comparação para as demonstrações usa k=30, como o site; os probes usam15.
    y30=np.stack([r['x'][32:62] for r in pool]);metrics={};predictions={}
    for family,model in models.items():
        pred,end=model.forecast(prefix,lengths,labels,metadata,30);predictions[family]=pred.numpy()
        metrics[family]={r['id']:_window_metrics(pred[i].numpy(),end[i].numpy(),y30[i],metadata) for i,r in enumerate(pool)}
    descriptors={r['id']:prefix_descriptors(r['points'][:32]) for r in pool};selected={}
    def add(row,theme,rule,explanation):
        key=row['id']
        if key not in selected:
            selected[key]={'id':key,'label':row['label'],'category':config['classes'][row['label']],
                'category_name':config['labels'][row['label']],'source_split':'validation','points':row['points'].tolist(),
                'default_cut':32,'default_horizon':30,'themes':[],'selection_rule':[],'explanation':[],
                'prefix_descriptors':descriptors[key],'metrics':{family:{**metrics[family][key],
                    'ade_pixels':metrics[family][key]['ade'],'fde_pixels':metrics[family][key]['fde']} for family in MODELS}}
        selected[key]['themes'].append(theme);selected[key]['selection_rule'].append(rule);selected[key]['explanation'].append(explanation)
    for label in range(len(config['classes'])):
        row=next(r for r in pool if r['label']==label)
        add(row,'prefixo','Primeiro ID elegível da categoria na validação, pelo menos62eventos.','Compare a mesma realização humana com os dois futuros previstos. Prefixos maiores mudam também a origem da previsão.')
    rules=[('cruzamentos',lambda r:descriptors[r['id']]['crossings']>=2,'Primeiro ID de validação com pelo menos dois cruzamentos próprios no prefixo32.','Cruzamentos descrevem a geometria do passado; não demonstram imunidade a sobreposição visual.'),
        ('caneta',lambda r:descriptors[r['id']]['pen_up_count']>=3,'Primeiro ID de validação com pelo menos três eventos pen-up no prefixo32.','As separações entre traços e os saltos também fazem parte da sequência.'),
        ('ordem',lambda r:descriptors[r['id']]['pen_up_count']<=1,'Primeiro ID de validação com no máximo um pen-up no prefixo32.','O teste de ordem permuta vetores já codificados e preserva o quadro do decoder; pode produzir sequências não físicas.'),
        ('translacao_escala',lambda r:len(r['x'])>=94,'Primeiro ID de validação com pelo menos94eventos.','A translação cancela nas features relativas de ambas as redes. Escala altera os deslocamentos e não é invariância garantida.'),
        ('mlp_vence',lambda r:metrics['mlp'][r['id']]['ade']<metrics['gru'][r['id']]['ade'],'Primeiro ID de validação em que a MLP tem menor ADE30, não o melhor caso.','Exemplo escolhido na validação para mostrar uma falha relativa da GRU; não é uma estimativa de desempenho em teste.'),
        ('gru_vence',lambda r:metrics['gru'][r['id']]['ade']<metrics['mlp'][r['id']]['ade'],'Primeiro ID de validação em que a GRU tem menor ADE30, não o melhor caso.','Exemplo escolhido na validação para mostrar uma vantagem relativa da GRU, sem representar todos os desenhos.'),
        ('continuidade',lambda r:metrics['gru'][r['id']]['ade']<40 and metrics['mlp'][r['id']]['ade']<40,'Primeiro ID de validação com ADE30 abaixo de40pixels para ambas as redes.','Caso curado na validação para ilustrar previsões próximas; não foi escolhido no teste e não resume a distribuição de erros.')]
    missing=[]
    for theme,condition,rule,explanation in rules:
        candidates=sorted([r for r in pool if condition(r)],key=lambda r:r['id'])
        if candidates:add(candidates[0],theme,rule,explanation)
        else:missing.append(theme)
    for entry in selected.values():
        entry['selection_rules']=entry['selection_rule'];entry['selection_rule']=' '.join(entry['selection_rule'])
        entry['explanations']=entry['explanation'];entry['explanation']=' '.join(entry['explanation'])
    manifest={'version':1,'classes':config['classes'],'labels':config['labels'],
        'selection_note':'Somente validação; primeiros IDs elegíveis, descritores do passado e vitórias explicitamente rotuladas. Nenhum caso foi escolhido por ser o melhor acerto de teste.',
        'examples':list(selected.values()),'unavailable_themes':missing}
    write_json(output/'site_examples.json',manifest)
    return manifest


def _tables(records,config):
    overall=[];paired_overall=[];prefixes=[];groups=[];paired_groups=[];categories=[]
    for k in config['horizons']:
        candidates=[r for r in records if r['k']==k and r['cut'] in [16,32]]
        for family in MODELS:overall.append({'model':family,'seed':42,'k':k,**_aggregate([r for r in candidates if r['model']==family])})
        paired_overall.append({'k':k,**_paired(*[[r for r in candidates if r['model']==family] for family in MODELS])})
        for cut in [8,16,32,64]:
            for family in MODELS:
                chosen=[r for r in records if r['model']==family and r['cut']==cut and r['k']==k]
                if chosen:prefixes.append({'model':family,'seed':42,'cut':cut,'k':k,**_aggregate(chosen)})
    base=[r for r in records if r['cut']==32 and r['k']==15]
    descriptors=[('crossings','0',lambda r:r['crossings']==0),('crossings','1',lambda r:r['crossings']==1),
        ('crossings','≥2',lambda r:r['crossings']>=2),('pen_up','0',lambda r:r['pen_up_count']==0),
        ('pen_up','1–2',lambda r:1<=r['pen_up_count']<=2),('pen_up','≥3',lambda r:r['pen_up_count']>=3)]
    for descriptor,group,condition in descriptors:
        chosen=[r for r in base if condition(r)]
        if not chosen:continue
        for family in MODELS:
            subset=[r for r in chosen if r['model']==family]
            groups.append({'model':family,'seed':42,'descriptor':descriptor,'group':group,'cut':32,'k':15,
                'per_class_n':{config['classes'][label]:sum(r['label']==label for r in subset) for label in range(len(config['classes']))},**_aggregate(subset)})
        paired_groups.append({'descriptor':descriptor,'group':group,'cut':32,'k':15,
            **_paired(*[[r for r in chosen if r['model']==family] for family in MODELS])})
    for label in range(len(config['classes'])):
        for family in MODELS:
            chosen=[r for r in base if r['model']==family and r['label']==label]
            if chosen:categories.append({'model':family,'label':label,'category':config['classes'][label],'k':15,'cut':32,**_aggregate(chosen)})
    return overall,paired_overall,prefixes,groups,paired_groups,categories


def figures(findings,records,manifest,models,metadata,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','text.color':INK,'axes.labelcolor':INK,'legend.frameon':False,'grid.alpha':.17})
    overall=findings['overall'];ks=[5,15,30]
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for family in MODELS:
        rows=[next(r for r in overall if r['model']==family and r['k']==k) for k in ks]
        for ax,field,title in zip(axes,['ade','fde'],['Erro da trajetória inteira','Erro no último evento previsto']):
            ax.plot(ks,[r[field] for r in rows],'o-',color=COLORS[family],label=LABELS[family],lw=2)
            if field=='ade':ax.fill_between(ks,[r['ade_ci95'][0] for r in rows],[r['ade_ci95'][1] for r in rows],color=COLORS[family],alpha=.15)
            ax.set(title=title,xlabel='Eventos futuros k',ylabel=field.upper()+' (pixels)',xticks=ks);ax.grid()
    axes[0].legend();_save(fig,output,'01_horizons_comparison')

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for family in MODELS:
        rows=[r for r in findings['prefixes'] if r['model']==family and r['k']==15];rows.sort(key=lambda r:r['cut'])
        axes[0].plot([r['cut'] for r in rows],[r['ade'] for r in rows],'o-',color=COLORS[family],label=LABELS[family],lw=2)
        if family=='gru':
            for r in rows:axes[0].annotate('n='+str(r['n_prefixes']),(r['cut'],r['ade']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
        fixed=[r for r in findings['context_windows'] if r['model']==family]
        axes[1].plot([r['window'] for r in fixed],[r['ade'] for r in fixed],'o-',color=COLORS[family],label=LABELS[family],lw=2)
    axes[0].set(title='Prefixo desde o início · coortes e alvos mudam',xlabel='Último evento observado t',ylabel='ADE15 (pixels)',xticks=[8,16,32,64]);axes[0].legend(fontsize=8)
    axes[1].set(title='Origem fixa t=64 · mesmo futuro',xlabel='Eventos mantidos no contexto',ylabel='ADE15 (pixels)',xticks=[8,16,32,64])
    for ax in axes:ax.grid()
    _save(fig,output,'02_prefix_cohorts')

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,descriptor,title in zip(axes,['crossings','pen_up'],['Cruzamentos próprios do prefixo','Eventos de caneta levantada no prefixo']):
        groups=list(dict.fromkeys(r['group'] for r in findings['groups'] if r['descriptor']==descriptor));x=np.arange(len(groups))
        for family,offset in [('gru',-.18),('mlp',.18)]:
            rows=[next(r for r in findings['groups'] if r['descriptor']==descriptor and r['group']==group and r['model']==family) for group in groups]
            ax.bar(x+offset,[r['ade'] for r in rows],width=.34,color=COLORS[family],label=LABELS[family])
        for j,group in enumerate(groups):
            r=next(r for r in findings['groups'] if r['descriptor']==descriptor and r['group']==group)
            ax.text(j,ax.get_ylim()[1]*.96,'n='+str(r['n_prefixes']),ha='center',va='top',fontsize=8)
        ax.set(title=title,xticks=x,xticklabels=groups,ylabel='ADE15 (pixels)');ax.grid(axis='y')
    axes[0].legend(fontsize=8);_save(fig,output,'03_geometry_groups')

    scenarios=['scale_075','scale_125','shuffle_interior_fixed_suffix','mask_pen_fixed_frame','mask_time']
    names=['Escala\n0,75','Escala\n1,25','Ordem dos\nvetores','Caneta\nzerada','Tempo\nzerado']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');x=np.arange(len(scenarios))
    for family,offset in [('gru',-.17),('mlp',.17)]:
        rows=[next(r for r in findings['sensitivities'] if r['model']==family and r['scenario']==s) for s in scenarios]
        values=np.array([r['ratio_vs_baseline']-1 for r in rows])*100
        low=np.array([r['ratio_ci95'][0]-1 for r in rows])*100;high=np.array([r['ratio_ci95'][1]-1 for r in rows])*100
        axes[0].errorbar(x+offset,values,yerr=np.maximum(0,np.vstack([values-low,high-values])),fmt='o',capsize=4,color=COLORS[family],label=LABELS[family])
        base=next(r for r in findings['sensitivities'] if r['model']==family and r['scenario']=='baseline')
        axes[1].bar(x+offset,[r['clock_mae']-base['clock_mae'] for r in rows],width=.32,color=COLORS[family],label=LABELS[family])
    axes[0].set(title='Variação espacial · corrigida pela escala',ylabel='Variação de (ADE / fator) (%)');axes[0].legend(fontsize=8)
    axes[1].set(title='Variação no erro do relógio',ylabel='MAE de intervalo − original (s)')
    for ax in axes:ax.set_xticks(x,names);ax.axhline(0,color='#9EACB6',ls='--',lw=1);ax.grid(axis='y')
    _save(fig,output,'04_sensitivities')

    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained');base=[r for r in records if r['cut'] in [16,32] and r['k']==15]
    for family in MODELS:
        values=np.sort([r['ade'] for r in base if r['model']==family]);axes[0].plot(values,100*np.arange(1,len(values)+1)/len(values),color=COLORS[family],label=LABELS[family],lw=2)
    a={(r['id'],r['cut']):r['ade'] for r in base if r['model']=='gru'};b={(r['id'],r['cut']):r['ade'] for r in base if r['model']=='mlp'}
    delta=np.array([a[key]-b[key] for key in sorted(a)])
    axes[1].hist(delta,bins=70,color='#A3B7C3',edgecolor='white',linewidth=.3);axes[1].axvline(0,color=REAL,ls='--')
    axes[0].set(title='Distribuição do erro entre prefixos',xlabel='ADE15 do prefixo (pixels)',ylabel='Prefixos acumulados (%)');axes[0].legend(fontsize=8)
    axes[1].set(title='Comparação pareada dos mesmos prefixos',xlabel='ADE GRU − ADE MLP (pixels)',ylabel='Prefixos')
    for ax in axes:ax.grid(axis='y')
    _save(fig,output,'05_paired_errors')

    _example_figure(manifest,models,metadata,output)
    example=manifest['examples'][0];points=np.asarray(example['points'])[:32]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    axes[0].plot(points[:,0],points[:,1],color=INK,label='Prefixo original')
    axes[0].plot(points[:,0]+120,points[:,1]-80,color=REAL,label='Translação (+120, −80)')
    axes[0].invert_yaxis();axes[0].set_aspect('equal');axes[0].set(title='Coordenadas absolutas mudam',xlabel='X (pixels)',ylabel='Y (pixels)');axes[0].legend(fontsize=8)
    axes[1].axis('off');translation=findings['controls']['translation']
    lines=['As features relativas cancelam a translação.','Controle numérico em todos os casos do probe:','',f"Máxima mudança nas features: {translation['max_feature_delta']:.3g}"]
    lines +=[LABELS[f]+': '+f"{translation[f+'_max_prediction_delta_pixels']:.3g} px de mudança" for f in MODELS]
    lines+=['','Este controle vale para ambas as redes.','Não é uma invariância aprendida no treinamento.']
    axes[1].text(.02,.95,'\n'.join(lines),va='top',fontsize=11,linespacing=1.65)
    _save(fig,output,'07_translation_control')

    fig,ax=plt.subplots(figsize=(12,4.6),layout='constrained');categories=config_labels=manifest['labels'];x=np.arange(len(categories))
    for family,offset in [('gru',-.18),('mlp',.18)]:
        rows=[next(r for r in findings['categories'] if r['model']==family and r['label']==label) for label in range(len(categories))]
        ax.bar(x+offset,[r['ade'] for r in rows],width=.34,color=COLORS[family],label=LABELS[family])
    ax.set(xticks=x,xticklabels=categories,ylabel='ADE15 (pixels)',title='Erro por categoria fornecida · t=32');ax.tick_params(axis='x',rotation=35,labelsize=9);ax.legend();ax.grid(axis='y')
    _save(fig,output,'08_category_comparison')

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for family in MODELS:
        rows=[next(r for r in overall if r['model']==family and r['k']==k) for k in ks]
        for ax,key,title in zip(axes,['clock_mae','pen_f1','premature_end'],['MAE de intervalo (s)','F1 de caneta levantada','Encerramento prematuro (%)']):
            multiplier=100 if key=='premature_end' else 1
            ax.plot(ks,[r[key]*multiplier for r in rows],'o-',color=COLORS[family],label=LABELS[family],lw=2)
            ax.set(title=title,xlabel='Eventos futuros k',xticks=ks);ax.grid()
    axes[0].legend(fontsize=8);_save(fig,output,'09_time_pen_quality')


@torch.inference_mode()
def _example_figure(manifest,models,metadata,output):
    from .drawing import draw
    chosen=[]
    for theme in ['continuidade','cruzamentos','mlp_vence','gru_vence']:
        options=[e for e in manifest['examples'] if theme in e['themes'] and e['id'] not in [x['id'] for x in chosen]]
        if options:chosen.append(options[0])
    for entry in manifest['examples']:
        if len(chosen)>=4:break
        if entry['id'] not in [e['id'] for e in chosen]:chosen.append(entry)
    fig,axes=plt.subplots(len(chosen),3,figsize=(12,3*len(chosen)),layout='constrained',squeeze=False,gridspec_kw={'width_ratios':[.85,1,1]})
    for i,entry in enumerate(chosen):
        points=np.array(entry['points']);x=tokens_from_points(points[:32],metadata);truth=tokens_from_points(points,metadata)[32:62]
        prediction={}
        for family,model in models.items():prediction[family]=model.forecast(torch.from_numpy(x[None]),torch.tensor([32]),torch.tensor([entry['label']]),metadata,30)[0][0].numpy()
        scale=metadata['position_scale'];ax=axes[i,0]
        draw(ax,x[:,4:6]*scale,x[:,2],'#9AAAB5',1.7)
        draw(ax,np.vstack([x[-1,4:6],truth[:,4:6]])*scale,np.r_[x[-1,2],truth[:,2]],INK,1.6)
        for family in MODELS:
            pred=prediction[family];draw(ax,np.vstack([x[-1,4:6],pred[:,4:6]])*scale,np.r_[x[-1,2],pred[:,2]],COLORS[family],1.8)
        ax.invert_yaxis();ax.set_aspect('equal');ax.axis('off');ax.set_title(entry['category_name']+' · validação',fontsize=11)
        for j,coordinate in enumerate([0,1],1):
            ax=axes[i,j];ax.plot(range(-8,1),x[-9:,4+coordinate]*scale,color='#9AAAB5',label='Passado')
            ax.plot(range(31),np.r_[x[-1,4+coordinate],truth[:,4+coordinate]]*scale,color=INK,label='Futuro real')
            for family in MODELS:ax.plot(range(31),np.r_[x[-1,4+coordinate],prediction[family][:,4+coordinate]]*scale,color=COLORS[family],label=LABELS[family])
            ax.axvline(0,ls='--',color='#A7B4BD',lw=1);ax.set(title='X' if coordinate==0 else 'Y',xlabel='Eventos após t',ylabel='Posição (pixels)');ax.grid()
    axes[0,1].legend(fontsize=8);_save(fig,output,'06_examples_2d_xy')


def analyze(directory,output=None,per_class=20):
    directory=Path(directory);output=Path(output) if output else directory/'diagnostics'
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    for family in MODELS:
        folder=directory/'models'/family
        if not (folder/'frozen_selection.json').exists():raise RuntimeError('Congele primeiro a seleção de ambas as famílias: '+str(folder))
        if not (folder/'test_metrics.json').exists():raise RuntimeError('Aguarde a avaliação final de ambas as famílias: '+str(folder))
    config=read_json(directory/'config.json');metadata=read_json(directory/'data/metadata.json')
    all_rows=load(directory);test={r['id']:r for r in all_rows if r['split']=='test'}
    records,predictions=artifact_records(directory,test,metadata)
    if predictions['gru'].keys()!=predictions['mlp'].keys():raise ValueError('Os manifestos de previsão não têm os mesmos prefixos.')
    cuts={p[1] for p in predictions['gru']}
    if cuts!={8,16,32,64}:raise ValueError('Esperadas predições extras nos cortes8/16/32/64; recebido '+str(cuts))
    print('Registros pareados de teste:',len(records),flush=True)
    overall,paired,prefixes,groups,paired_groups,categories=_tables(records,config)
    models=load_models(config,directory)
    probes,controls,sample=sensitivities(config,models,all_rows,metadata,per_class)
    print('Sensibilidades CPU:',len(sample),'desenhos; nenhum peso ajustado.',flush=True)
    windows=fixed_origin_windows(config,models,all_rows,metadata,per_class)
    manifest=site_manifest(config,models,all_rows,metadata,output)
    captions={
        '01_horizons_comparison':'Mesmos desenhos, cortes16/32 e horizontes em ambas as redes, execução42. ADE mede a trajetória inteira; FDE mede o último ponto. Faixas de95% do ADE reamostram IDs de desenhos, mantendo seus prefixos juntos.',
        '02_prefix_cohorts':'À esquerda, prefixos8/16/32/64 desde o início: coortes e alvos mudam, portanto não é uma ablação causal de contexto. À direita, origem fixa64 e alvos comuns, variando apenas a janela observada. Essas janelas internas não eram origens de treino; é análise de sensibilidade com coorte fixa de desenhos longos.',
        '03_geometry_groups':'Erro de previsão condicionado a descritores do passado, t=32 e k=15. Cruzamentos são interseções próprias entre segmentos de tinta não adjacentes; extremos compartilhados e sobreposições colineares não são contados. Classes e estilos podem ter distribuições diferentes entre grupos. Não é demonstração de imunidade a borrões ou sobreposição visual.',
        '04_sensitivities':'Intervenções sintéticas na entrada, sem retreino, t=32 e k=15. O erro espacial é dividido pelo fator de escala antes de comparar, evitando confundir aumento geométrico com perda de consistência. A ordem permuta vetores prontos e pode produzir sequências não físicas. Caneta zerada usa quadro geométrico original congelado; tempo zerado significa intervalos nulos, não informação ausente.',
        '05_paired_errors':'Distribuição dos erros e diferenças GRU−MLP sobre os mesmos IDs/cortes16/32 em k=15. Valores negativos favorecem a GRU; valores positivos favorecem a MLP. Nenhum prefixo foi escolhido por seu acerto.',
        '06_examples_2d_xy':'Casos de validação: primeiras ocorrências elegíveis por regras declaradas, incluindo previsões próximas e vitórias de cada rede quando disponíveis. Não são os melhores acertos do teste. A representação2D usa o prefixo32 completo; X/Y mostram os últimos9 observados e os30 futuros. Categoria verdadeira fornecida.',
        '07_translation_control':'Controle de pré-processamento. Transladar todas as coordenadas por um deslocamento comum cancela nas features relativas; isso beneficia ambas as arquiteturas. Os resíduos numéricos são reportados sem atribuir a invariância ao treinamento.',
        '08_category_comparison':'Erro de previsão em15 eventos, origem32, separado pela categoria fornecida ao modelo. Não é acurácia de reconhecimento; as duas redes recebem a mesma condição de categoria e os mesmos desenhos.',
        '09_time_pen_quality':'Previsão temporal e de estado, execução42 nos cortes16/32. F1 é da classe pen-up; encerramento prematuro ocorre quando a rede encerra antes dos eventos futuros disponíveis. Acurácia total da caneta não é usada para ocultar desequilíbrio.'}
    hypotheses=[
        {'id':'H1','hypothesis':'Poucos traços observados podem admitir continuações geométricas distintas.','operation':'Erros nos cortes8/16/32/64 e exemplos2D/X/Y de validação.','evidence':'prefixes e06_examples_2d_xy','limitations':'Não há rótulo de ambiguidade nem várias realizações futuras do mesmo prefixo. A categoria é fornecida. Não avaliamos classificação.'},
        {'id':'H2','hypothesis':'Cruzamentos e múltiplos traços podem estar associados a maior dificuldade de previsão.','operation':'Grupos de cruzamentos próprios e pen-up no passado, t=32/k15, mesmos pares GRU/MLP.','evidence':'groups e paired_groups','limitations':'Associação descritiva; categoria, escala e estilo podem confundir grupos. A MLP é pontual, não uma CNN de imagem.'},
        {'id':'H3','hypothesis':'Mais contexto pode melhorar a previsão, sem garantia de queda monotônica do erro.','operation':'Prefixos crescentes com n por coorte e janela8/16/32/64 em origem fixa64.','evidence':'prefixes e context_windows','limitations':'Prefixos crescentes mudam o futuro-alvo. Janelas internas são uma mudança de distribuição diante do treino com prefixos de início.'},
        {'id':'H4','hypothesis':'Translação deve cancelar na codificação relativa; escala não é invariância garantida.','operation':'Deslocamento(+120,−80), fatores0.75/1.25 e ADE dividido pelo fator.','evidence':'controls.translation e sensitivities','limitations':'Controle sintético de pré-processamento. Fatores ficam fora da augmentação0.8–1.2; não demonstram generalização arbitrária de escala.'},
        {'id':'H5','hypothesis':'Ordem dos eventos, caneta e tempo podem influenciar a previsão.','operation':'Permutação de vetores com primeiro/sufixo preservados; canais caneta/tempo zerados; quadro original congelado quando necessário.','evidence':'sensitivities e controls.permutation','limitations':'Sem retreino e com entradas potencialmente não físicas. Δx/Δy, posições, tempo e o decoder também carregam informação local na MLP; não é ausência total de informação temporal.'}]
    findings={'version':1,'source':{'directory':str(directory),'primary_seed':42},
        'protocol':{'task':'Somente previsão de eventos futuros, com categoria fornecida.',
            'models':'GRU pura e MLP DeepSets com decoder comum; sem CNN neste benchmark.',
            'test_previously_observed':True,'bootstrap_repetitions':BOOTSTRAP,'bootstrap_seed':20261118,
            'bootstrap_unit':'ID do desenho, mantendo seus cortes juntos; intervalos condicionais aos pesos da execução42.',
            'sensitivity_sample':'Primeiros IDs elegíveis por categoria, até'+str(per_class)+'cada; nenhum caso selecionado pelo erro.',
            'sensitivity_sample_ids':[r['id'] for r in sample],
            'case_selection':'Somente validação; regras e casos rotulados no manifest. Não altera métricas de teste.',
            'scales_tested':[.75,1.25],'training_scale_augmentation':[.8,1.2]},
        'overall':overall,'paired_overall':paired,'prefixes':prefixes,'groups':groups,'paired_groups':paired_groups,
        'categories':categories,'sensitivities':probes,'context_windows':windows,'controls':controls,
        'hypotheses':hypotheses,'captions':captions,'examples_manifest':str(output/'site_examples.json'),
        'n_site_examples':len(manifest['examples'])}
    write_csv(output/'per_prefix_metrics.csv',records);write_csv(output/'overall.csv',overall)
    write_csv(output/'prefixes.csv',prefixes);write_csv(output/'sensitivities.csv',probes)
    figures(findings,records,manifest,models,metadata,output)
    write_json(output/'findings.json',findings)
    write_json(output/'diagnostics_audit.json',{'passed':True,'primary_seed':42,'matched_prediction_keys':len(predictions['gru']),
        'common_cut_values':sorted(cuts),'no_weights_updated':True,'decoder_parity':controls['decoder_parity'],
        'frame_preserved_for_permutation':controls['permutation']['max_frame_delta'],'figures':len(list(output.glob('*.pdf')))})
    print('Diagnósticos comparativos de previsão:',output,flush=True)
    return findings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/drawing_forecast_compare_v1')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--per-class',type=int,default=20)
    args=parser.parse_args();analyze(args.directory,args.output,args.per_class)


if __name__=='__main__':main()
