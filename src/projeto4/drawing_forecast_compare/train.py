"""Busca CUDA por família, mantendo dados e decoder iguais entre GRU e DeepSets."""
import json
import shutil

import numpy as np
import torch

from ..io import ROOT,read_json,write_json
from .anchors import learn_anchors
from .metrics import evaluate_rollouts
from .metrics import ending_metrics
from .training_core import configure,CPUInterface,train_one,trials
from .data import prepare,load,sha256
from .model import new_model


def reuse_screening(config,output):
    if (output/'reuse_manifest.json').exists():return
    source_name=config.get('reuse_gru_screening_from')
    if not source_name:return
    source=ROOT/'results'/source_name
    if not (source/'config.json').exists():
        print('Fonte de triagem não incluída: todas as GRUs serão treinadas do zero.',flush=True)
        return
    old=read_json(source/'config.json')
    keys=['classes','embedding','context_events','horizons','validation_cuts','batch_size','weight_decay','spatial_step','position_scale','time_scale']
    for key in keys:
        if old[key]!=config[key]:raise ValueError('Reutilização incompatível em '+key)
    destination=output/'models/gru';manifest=[]
    for spec in config['families']['gru']['candidates']:
        if spec not in old['candidates']:raise ValueError('ConfiguraçãoGRU sem execução idêntica de origem.')
        src=source/'runs'/f"{spec['id']}_seed42";dst=destination/'runs'/src.name
        summary=read_json(src/'summary.json')
        if summary.get('warm_started') or summary['epochs_run']!=spec['epochs']:raise ValueError('ExecuçãoGRU não atende cold start/orçamento.')
        if not dst.exists():shutil.copytree(src,dst)
        trial=source/'validation_trials'/f"{spec['id']}_seed42.json";new=destination/'validation_trials'/trial.name;new.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(trial,new)
        manifest.append({'spec':spec,'seed':42,'source_directory':str(src.relative_to(ROOT)),
            'checkpoint_sha256':sha256(src/'best.pt'),'copied_checkpoint_sha256':sha256(dst/'best.pt'),
            'epochs_run':summary['epochs_run'],'warm_started':False,
            'meaning':'Reutilização de execução completa já treinada do zero emCUDA com protocolo exatamente igual; não inicializa outra rede com esses pesos.'})
    write_json(output/'reuse_manifest.json',{'data_npz_sha256':sha256(output/'data/sequences.npz'),
        'source_data_npz_sha256':sha256(source/'data/sequences.npz'),'anchors_sha256':sha256(output/'anchors_64.npz'),
        'runs':manifest,'test_note':'Partição de teste já observada em estudos prévios; nova escolha por família usa somente validação.'})


def search(config):
    configure(config);output=prepare(config);rows=load(output)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];metadata=read_json(output/'data/metadata.json')
    anchors=learn_anchors(train,config,metadata,output);reuse_screening(config,output)
    for family,definition in config['families'].items():
        folder=output/'models'/family;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'frozen_selection.json').exists():continue
        runs=[];all_trials=[]
        for spec in definition['candidates']:
            runs.append(train_one(config,folder,spec,42,train,val,metadata,anchors,model_factory=new_model))
            all_trials+=trials(config,folder,spec,42,val,metadata,model_factory=new_model)
        screening=[min([r for r in all_trials if r['spec']==spec],key=lambda r:r['score']) for spec in definition['candidates']]
        finalists=[r['spec'] for r in sorted(screening,key=lambda r:r['score'])[:2]]
        write_json(folder/'screening.json',{'seed':42,'best_trials':screening,'runs':runs,'finalists':finalists})
        for spec in finalists:
            runs.append(train_one(config,folder,spec,43,train,val,metadata,anchors,model_factory=new_model))
            all_trials+=trials(config,folder,spec,43,val,metadata,model_factory=new_model)
        final_scores=[]
        for spec in finalists:
            for threshold in [.2,.35,.5]:
                repeated=[r for r in all_trials if r['spec']==spec and r['pen_threshold']==threshold]
                first=next(r for r in repeated if r['seed']==42)
                final_scores.append({**first,'score':float(np.mean([r['score'] for r in repeated])),
                    'seed_scores':[r['score'] for r in repeated],'selection_seeds':[42,43]})
        best=min(final_scores,key=lambda r:r['score'])
        runs.append(train_one(config,folder,best['spec'],44,train,val,metadata,anchors,model_factory=new_model))
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
    if not (output/'frozen_comparison.json').exists():raise RuntimeError('Congele ambas as famílias antes do teste.')
    test=[r for r in load(output) if r['split']=='test'];metadata=read_json(output/'data/metadata.json');comparison={}
    for family,definition in config['families'].items():
        folder=output/'models'/family
        if (folder/'test_metrics.json').exists():comparison[family]=read_json(folder/'test_metrics.json');continue
        selected=read_json(folder/'selection.json')['best'];spec=selected['spec'];runs=[]
        for seed in config['seeds']:
            saved=torch.load(folder/'runs'/f"{spec['id']}_seed{seed}"/'best.pt',map_location='cpu',weights_only=True)
            model=new_model(config,spec).cuda();model.load_state_dict(saved['state_dict']);model.pen_threshold=selected['pen_threshold'];model.eval();adapter=CPUInterface(model)
            metrics=evaluate_rollouts(adapter,test,config,metadata,detailed=True);main_predictions=metrics.pop('details')
            predictions=main_predictions
            extra={**config,'validation_cuts':[c for c in config['diagnostic_cuts'] if c not in config['validation_cuts']]}
            if extra['validation_cuts']:
                diagnostic=evaluate_rollouts(adapter,test,extra,metadata,detailed=True);predictions+=diagnostic['details']
            (folder/f'test_predictions_seed{seed}.json').write_text(json.dumps(predictions,separators=(',',':')))
            runs.append({'seed':seed,'checkpoint_epoch':saved['epoch'],**metrics,'endings':ending_metrics(adapter,test,config,metadata)})
            print('TESTE COMPARAÇÃO',family,seed,{k:round(v['ade_pixels'],2) for k,v in metrics['horizons'].items()},flush=True)
        horizons={}
        for k in map(str,config['horizons']):
            horizons[k]={key:float(np.mean([r['horizons'][k][key] for r in runs])) for key in runs[0]['horizons'][k]}
            horizons[k]['n_prefixes']=runs[0]['horizons'][k]['n_prefixes'];horizons[k]['ade_std_between_seeds']=float(np.std([r['horizons'][k]['ade_pixels'] for r in runs],ddof=1))
        summary={'family':family,'label':definition['label'],'selected':spec,'pen_threshold':selected['pen_threshold'],
            'n_test':len(test),'n_classes':len(config['classes']),'known_category':True,'horizons':horizons,'seeds':runs,
            'main_cuts':config['validation_cuts'],'exported_diagnostic_cuts':config['diagnostic_cuts'],'all_models_from_scratch':True,
            'test_previously_observed':True,'notes':['Métricas principais emt16/32; export contém também8/64para diagnósticos.',
                'Categoria verdadeira fornecida; nenhumclassificador participa.','Mesmosdados,âncoras,decoder,perda,augmentaçãoe20épocas.',
                'MLPencodersemrecorrência, masfeaturesΔ/tempo/posição eframecausal retêminformação local deordem.']}
        write_json(folder/'test_metrics.json',summary);comparison[family]=summary
    write_json(output/'comparison.json',{'families':comparison,'main_cuts':config['validation_cuts'],'horizons':config['horizons'],
        'selected_before_test':True,'test_previously_observed':True,'no_transfer_learning':True,'reference':'GRU pura e DeepSets compartilham somente o decoder e dados, não pesos de encoder.'})
    return comparison
