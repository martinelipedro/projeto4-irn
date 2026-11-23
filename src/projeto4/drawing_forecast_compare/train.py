"""Busca por hiperparâmetros e avaliação da previsão com GRU."""
import numpy as np
import torch

from ..io import read_json,write_json
from .anchors import learn_anchors
from .metrics import evaluate_rollouts
from .metrics import ending_metrics
from .training_core import configure,CPUInterface,train_one,trials
from .data import prepare,load
from .model import new_model


def search(config):
    configure(config);output=prepare(config);rows=load(output)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];metadata=read_json(output/'data/metadata.json')
    anchors=learn_anchors(train,config,metadata,output)
    for family,definition in config['families'].items():
        folder=output/'models'/family;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'frozen_selection.json').exists():continue
        runs=[];all_trials=[]
        for spec in definition['candidates']:
            runs.append(train_one(config,folder,spec,42,train,val,metadata,anchors))
            all_trials+=trials(config,folder,spec,42,val,metadata)
        screening=[min([r for r in all_trials if r['spec']==spec],key=lambda r:r['score']) for spec in definition['candidates']]
        finalists=[r['spec'] for r in sorted(screening,key=lambda r:r['score'])[:2]]
        write_json(folder/'screening.json',{'seed':42,'best_trials':screening,'runs':runs,'finalists':finalists})
        for spec in finalists:
            runs.append(train_one(config,folder,spec,43,train,val,metadata,anchors))
            all_trials+=trials(config,folder,spec,43,val,metadata)
        final_scores=[]
        for spec in finalists:
            for threshold in [.2,.35,.5]:
                repeated=[r for r in all_trials if r['spec']==spec and r['pen_threshold']==threshold]
                first=next(r for r in repeated if r['seed']==42)
                final_scores.append({**first,'score':float(np.mean([r['score'] for r in repeated])),
                    'seed_scores':[r['score'] for r in repeated],'selection_seeds':[42,43]})
        best=min(final_scores,key=lambda r:r['score'])
        runs.append(train_one(config,folder,best['spec'],44,train,val,metadata,anchors))
        selection={'family':family,'label':definition['label'],'best':best,'screening':screening,'trials':all_trials,
            'runs':runs,'finalists':finalists,'finalist_scores':final_scores,'selection_seeds':[42,43],'replication_seed':44,
            'all_models_from_scratch':True,'no_transfer_learning':True,'same_decoder_and_anchors':True,
            'test_used_for_selection':False,'test_previously_observed':True,
            'criterion':'Triagem de4candidatos seed42; top2 treinadosnovamentedozerocom43; melhor média42/43 com limiarcomum. Réplica44nãoescolhemodelo.',
            'checkpoint_criterion':'Menor média ADE/256 +0,1*fimprecoce emk5/15/30, primeiros100desenhosval/classe; limiar externo na validação completa acrescenta0,02*(1-F1caneta).'}
        write_json(folder/'selection.json',selection);write_json(folder/'frozen_selection.json',selection)
        print('FAMÍLIA CONGELADA',family,best['spec']['id'],best['score'],flush=True)
    write_json(output/'frozen_comparison.json',{'families':{name:read_json(output/'models'/name/'selection.json')['best'] for name in config['families']},
        'same_data_and_decoder':True,'test_previously_observed':True,'decision':'As duas escolhas de família foram congeladas antes das avaliações desta rodada.'})
    return output


def evaluate(config):
    configure(config);output=prepare(config)
    if not (output/'frozen_comparison.json').exists():raise RuntimeError('Congele a seleção antes do teste.')
    test=[r for r in load(output) if r['split']=='test'];metadata=read_json(output/'data/metadata.json');comparison={}
    for family,definition in config['families'].items():
        folder=output/'models'/family
        if (folder/'test_metrics.json').exists():comparison[family]=read_json(folder/'test_metrics.json');continue
        selected=read_json(folder/'selection.json')['best'];spec=selected['spec'];runs=[]
        for seed in config['seeds']:
            saved=torch.load(folder/'runs'/f"{spec['id']}_seed{seed}"/'best.pt',map_location='cpu',weights_only=True)
            model=new_model(config,spec).cuda();model.load_state_dict(saved['state_dict']);model.pen_threshold=selected['pen_threshold'];model.eval();adapter=CPUInterface(model)
            metrics=evaluate_rollouts(adapter,test,config,metadata,detailed=True)
            metrics.pop('details')
            runs.append({'seed':seed,'checkpoint_epoch':saved['epoch'],**metrics,'endings':ending_metrics(adapter,test,config,metadata)})
            print('TESTE COMPARAÇÃO',family,seed,{k:round(v['ade_pixels'],2) for k,v in metrics['horizons'].items()},flush=True)
        horizons={}
        for k in map(str,config['horizons']):
            horizons[k]={key:float(np.mean([r['horizons'][k][key] for r in runs])) for key in runs[0]['horizons'][k]}
            horizons[k]['n_prefixes']=runs[0]['horizons'][k]['n_prefixes'];horizons[k]['ade_std_between_seeds']=float(np.std([r['horizons'][k]['ade_pixels'] for r in runs],ddof=1))
        summary={'family':family,'label':definition['label'],'selected':spec,'pen_threshold':selected['pen_threshold'],
            'n_test':len(test),'n_classes':len(config['classes']),'known_category':True,'horizons':horizons,'seeds':runs,
            'main_cuts':config['validation_cuts'],'all_models_from_scratch':True,
            'test_previously_observed':True,'notes':['Categoria fornecida à GRU. Métricas nos prefixos de 16 e 32 eventos.']}
        write_json(folder/'test_metrics.json',summary);comparison[family]=summary
    write_json(output/'comparison.json',{'families':comparison,'main_cuts':config['validation_cuts'],'horizons':config['horizons'],
        'selected_before_test':True,'test_previously_observed':True,'no_transfer_learning':True,'reference':'Previsão condicionada da GRU.'})
    return comparison
