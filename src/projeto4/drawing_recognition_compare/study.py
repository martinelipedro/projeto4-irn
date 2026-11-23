"""Treino, busca e avaliação reproduzíveis; sem Deep Sets no reconhecimento."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss

from ..io import ROOT, read_json, write_json, write_csv
from ..drawing_forecast_compare.data import load
from .data import represent, permute_strokes
from .model import make_model


def setup(seed, threads=4, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível. Use ambiente com acesso à GPU.')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists() and read_json(output/'config.json') != config:
        raise ValueError('Configuração congelada; crie outro nome.')
    write_json(output/'config.json', config)
    if all((output/(split+'.npz')).exists() for split in ['train','validation','test']):
        return
    source = ROOT/'results'/config['source_experiment']
    rows = load(source)
    meta = read_json(source/'config.json')
    manifest = {'classes':meta['classes'], 'labels':meta['labels'], 'counts':{},
                'source_experiment':config['source_experiment'], 'source_manifest_sha256':
                hashlib.sha256((source/'data/records.json').read_bytes()).hexdigest(),
                'normalization':'Somente bbox do prefixo observado, centrada e dividida pela maior extensão; compartilhada pelas duas apresentações.',
                'image':'32x32 binária, DDA com passos de meio pixel e pincel cruz de raio1; união de pixels sem tempo ou ordem.',
                'test_previously_observed':True}
    for split in ['train','validation','test']:
        subset = [r for r in rows if r['split'] == split]
        n, v = len(subset), len(config['fractions'])
        tokens = np.zeros((n,v,180,6), np.float32)
        images = np.zeros((n,v,32,32), np.uint8)
        lengths = np.zeros((n,v), np.int64)
        for i,row in enumerate(subset):
            for j,fraction in enumerate(config['fractions']):
                cut=max(2,int(np.floor(len(row['points'])*fraction+.5)))
                tokens[i,j],images[i,j] = represent(row['points'][:cut])
                lengths[i,j] = cut
            if i%6000==0:
                print('PREPARAR',split,i,'/',n,flush=True)
        np.savez_compressed(output/(split+'.npz'), tokens=tokens, images=images,
                            lengths=lengths, labels=np.array([r['label'] for r in subset]),
                            ids=np.array([r['id'] for r in subset]))
        manifest['counts'][split]=n
    write_json(output/'data_manifest.json',manifest)


def dataset(path, family, device):
    with np.load(path) as arrays:
        x = arrays['tokens' if family=='gru' else 'images'].astype(np.float32)
        return (torch.from_numpy(x).to(device), torch.from_numpy(arrays['lengths']).to(device),
                torch.from_numpy(arrays['labels']).long().to(device))


@torch.inference_mode()
def evaluate(model, data, batch_size=256):
    model.eval()
    x,lengths,labels=data
    probabilities=[]
    # Desenho majoritário; nenhuma informação do desenho seguinte é compartilhada.
    x=x.flatten(0,1);lengths=lengths.flatten()
    for start in range(0,len(x),batch_size):
        xb=x[start:start+batch_size];ln=lengths[start:start+batch_size]
        if xb.ndim==3 and xb.shape[-1]==6:xb=xb[:,:int(ln.max())]
        probabilities.append(model(xb,ln).softmax(-1).cpu().numpy())
    p=np.concatenate(probabilities).reshape(len(labels),4,12)
    y=labels.cpu().numpy()
    scores=[log_loss(y,p[:,i],labels=np.arange(12)) for i in range(4)]
    return float(np.mean(scores)),p


def train_one(config, output, family, hidden, lr, seed, train, val):
    spec={'family':family,'hidden':hidden,'lr':lr}
    directory=output/family/f'h{hidden}_lr{lr}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():return read_json(directory/'summary.json')
    setup(seed,config['threads'])
    model=make_model(family,hidden).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=lr*.1)
    rng=torch.Generator().manual_seed(seed)
    x,lengths,labels=train;n=len(labels);history=[];best=float('inf');best_epoch=0
    started=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train();order=torch.randperm(n,generator=rng);views=torch.randint(4,(n,),generator=rng)
        total=0.;correct=0
        for idx in order.split(config['batch_size']):
            take=idx.cuda();view=views[idx].cuda();xb=x[take,view];ln=lengths[take,view]
            if family=='gru':xb=xb[:,:int(ln.max())]
            logits=model(xb,ln);loss=nn.functional.cross_entropy(logits,labels[take])
            if not torch.isfinite(loss):raise RuntimeError('Perda não finita.')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(idx);correct+=int((logits.argmax(-1)==labels[take]).sum())
        score,p=evaluate(model,val,config['batch_size'])
        row={'epoch':epoch,'lr':optimizer.param_groups[0]['lr'],'train_ce':total/n,
             'train_accuracy':correct/n,'val_ce':score,
             'val_full_accuracy':accuracy_score(val[2].cpu(),p[:,-1].argmax(-1))}
        history.append(row)
        if score<best:
            best=score;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'spec':spec,'seed':seed,'epoch':epoch,
                        'val_ce':score,'class_label_in_input':False},directory/'best.pt')
        scheduler.step();write_csv(directory/'history.csv',history)
        print(f'{family} H{hidden} lr{lr} seed{seed} {epoch}/{config["epochs"]} '
              f'val CE={score:.4f} full acc={row["val_full_accuracy"]:.3f} '
              f'{time.perf_counter()-started:.1f}s',flush=True)
    result={'spec':spec,'seed':seed,'score':best,'best_epoch':best_epoch,
            'parameters':sum(p.numel() for p in model.parameters()),
            'seconds':time.perf_counter()-started,'directory':str(directory.relative_to(output)),
            'epochs':config['epochs'],'gpu':torch.cuda.get_device_name(),'from_scratch':True}
    write_json(directory/'summary.json',result)
    return result


def search(config,output):
    for family in config['families']:
        if (output/family/'frozen_selection.json').exists():continue
        train=dataset(output/'train.npz',family,'cuda')
        val=dataset(output/'validation.npz',family,'cuda')
        screening=[train_one(config,output,family,h,lr,42,train,val)
                   for h in config['families'][family] for lr in config['learning_rates']]
        finalists=sorted(screening,key=lambda r:r['score'])[:2]
        repeats=[train_one(config,output,family,r['spec']['hidden'],r['spec']['lr'],43,train,val) for r in finalists]
        comparisons=[{'spec':a['spec'],'score':(a['score']+b['score'])/2,
                      'seed_scores':[a['score'],b['score']]} for a,b in zip(finalists,repeats)]
        winner=min(comparisons,key=lambda r:r['score'])
        s=winner['spec']
        replica=train_one(config,output,family,s['hidden'],s['lr'],44,train,val)
        selection={'best':winner,'screening':screening,'finalists':comparisons,
                   'runs':screening+repeats+[replica], 'criterion':config['selection'],
                   'test_used':False,'test_previously_observed':True}
        write_json(output/family/'frozen_selection.json',selection)
        del train,val;torch.cuda.empty_cache()


def metrics(y,p):
    predicted=p.argmax(-1)
    confidence=p.max(-1);correct=predicted==y
    ece=0.
    bins=np.minimum((confidence*10).astype(np.int64),9)
    for index in range(10):
        selected=bins==index
        if selected.any():ece+=selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {'accuracy':float(correct.mean()),'macro_f1':float(f1_score(y,predicted,average='macro',zero_division=0)),
            'top3':float(np.any(np.argsort(p,axis=1)[:,-3:]==y[:,None],axis=1).mean()),
            'ce':float(log_loss(y,p,labels=np.arange(12))), 'ece':float(ece),
            'brier':float(((p-np.eye(12)[y])**2).sum(-1).mean()),
            'confusion':confusion_matrix(y,predicted,labels=np.arange(12)).tolist()}


def test(config,output):
    if not all((output/f/'frozen_selection.json').exists() for f in config['families']):
        raise RuntimeError('Congele as duas seleções antes de consultar teste.')
    results={}
    for family in config['families']:
        spec=read_json(output/family/'frozen_selection.json')['best']['spec']
        data=dataset(output/'test.npz',family,'cuda');y=data[2].cpu().numpy();rows=[]
        for seed in config['seeds']:
            checkpoint=torch.load(output/family/f'h{spec["hidden"]}_lr{spec["lr"]}_seed{seed}'/'best.pt',map_location='cpu',weights_only=True)
            model=make_model(family,spec['hidden']).cuda();model.load_state_dict(checkpoint['state_dict'])
            _,p=evaluate(model,data,config['batch_size'])
            np.savez_compressed(output/family/f'test_seed{seed}.npz',truth=y,probabilities=p)
            rows.append({'seed':seed,'fractions':{str(f):metrics(y,p[:,i]) for i,f in enumerate(config['fractions'])}})
        aggregate={str(f):{key:float(np.mean([r['fractions'][str(f)][key] for r in rows]))
                          for key in ['accuracy','macro_f1','top3','ce','ece','brier']} for f in config['fractions']}
        results[family]={'spec':spec,'rows':rows,'mean':aggregate}
        del data;torch.cuda.empty_cache()
    write_json(output/'test_metrics.json',results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','search','test','all'])
    parser.add_argument('--config',default='configs/recognition_reproduce.json');args=parser.parse_args()
    config=read_json(ROOT/args.config);output=ROOT/'results'/config['name']
    setup(42,config['threads'],device='cpu' if args.action=='prepare' else 'cuda')
    if args.action in ['prepare','all']:prepare(config,output)
    if args.action in ['search','all']:search(config,output)
    if args.action in ['test','all']:test(config,output)


if __name__=='__main__':main()
