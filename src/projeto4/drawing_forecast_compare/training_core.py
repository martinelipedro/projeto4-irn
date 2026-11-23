"""Núcleo CUDA de previsão; extração fiel do backend já validado, sem tarefas de classificação."""
import os
import time
import numpy as np
import torch
from ..io import read_json,write_json,write_csv
from .trajectory import TrajectoryGRU,targets
from .metrics import evaluate_rollouts
from .model import new_model

def configure(config):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Defina CUBLAS_WORKSPACE_CONFIG=:4096:8 antes de iniciar o processo.')
    if not torch.cuda.is_available():raise RuntimeError('Esta rodada exige CUDA; não permite fallback silencioso para CPU.')
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

class CPUInterface:
    """A avaliação antiga recebe tensoresCPU; somente inferência é enviada aCUDA."""
    def __init__(self,model):self.model=model

    @torch.inference_mode()
    def forecast(self,prefix,lengths,labels,metadata,steps,decoder='greedy',seed=2026):
        if decoder!='greedy':raise ValueError('Busca determinística usa somente o modo mais provável.')
        out,ended=self.model.forecast(prefix.cuda(),lengths.cuda(),labels.cuda(),metadata,steps,decoder,seed)
        return out.cpu(),ended.cpu()

def train_one(config,output,spec,seed,train_rows,val_rows,metadata,anchors,model_factory=None):
    if spec.get('warm_start') or config.get('warm_start'):raise AssertionError('Pesos pré-treinados não são permitidos.')
    folder=output/'runs'/f"{spec['id']}_seed{seed}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return read_json(folder/'summary.json')
    configure(config);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=(model_factory(config,spec,anchors) if model_factory else TrajectoryGRU(len(config['classes']),anchors,spec['hidden'],config['embedding'],raster=spec.get('raster',False))).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=config['weight_decay'])
    epochs=spec['epochs'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=spec['lr']*.1)
    generator=torch.Generator().manual_seed(seed);lengths=torch.tensor([len(r['x']) for r in train_rows])
    history=[];best=float('inf');best_epoch=0;elapsed=0.;start=time.perf_counter()
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['generator']);torch.random.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];elapsed=saved['seconds']
    for epoch in range(len(history)+1,epochs+1):
        model.train();sums={};total=0;order=torch.randperm(len(train_rows),generator=generator);groups=[]
        for bucket in order.split(config['batch_size']*8):groups.extend(bucket[torch.argsort(lengths[bucket])].split(config['batch_size']))
        groups=[groups[i] for i in torch.randperm(len(groups),generator=generator)]
        for indexes in groups:
            upper=max(9,min(65,int(lengths[indexes].min())-1));cut=int(torch.randint(8,upper,(1,),generator=generator))
            scale=.8+.4*torch.rand(len(indexes),1,1,generator=generator)
            cpu_batch=targets(train_rows,indexes,cut,max(config['horizons']),metadata,scale)
            past,xy,pen,clock,valid,labels=[v.cuda() for v in cpu_batch]
            objective,parts=model(past,labels,xy,pen,clock,valid)
            if not torch.isfinite(objective):raise RuntimeError('Perda não finita.')
            opt.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            n=len(indexes);total+=n
            for key,value in {'objective':objective,**parts}.items():sums[key]=sums.get(key,0.)+float(value.detach())*n
        free=evaluate_rollouts(CPUInterface(model),val_rows,config,metadata,limit_per_class=100)
        row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],**{'train_'+key:value/total for key,value in sums.items()},
             'val_rollout_objective':free['objective'],**{'val_ade_'+key:value['ade_pixels'] for key,value in free['horizons'].items()}}
        history.append(row)
        state={key:value.detach().cpu() for key,value in model.state_dict().items()}
        if free['objective']<best:
            best=free['objective'];best_epoch=epoch
            torch.save({'state_dict':state,'spec':spec,'seed':seed,'epoch':epoch,'validation':free},folder/'best.pt')
        scheduler.step();seconds=elapsed+time.perf_counter()-start
        torch.save({'model':state,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'generator':generator.get_state(),
            'cpu_rng':torch.random.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'history':history,'best':best,
            'best_epoch':best_epoch,'seconds':seconds},folder/'last.pt');write_csv(folder/'history.csv',history)
        print(f"GPU previsão {spec['id']} seed{seed} época{epoch}: Jval={free['objective']:.4f}, ADE30={free['horizons']['30']['ade_pixels']:.1f}; {seconds:.0f}s",flush=True)
    summary={'spec':spec,'seed':seed,'best_epoch':best_epoch,'epochs_run':len(history),'validation_objective':best,
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':elapsed+time.perf_counter()-start,
        'directory':str(folder.relative_to(output)),'device':torch.cuda.get_device_name(),'warm_started':False,'all_models_from_scratch':True,
        'checkpoint_criterion':'ADE/256 +0,1*encerramento prematuro, média k5/15/30,100desenhos de validação por classe.'}
    write_json(folder/'summary.json',summary);return summary

def trials(config,output,spec,seed,validation,metadata,model_factory=None):
    path=output/'validation_trials'/f"{spec['id']}_seed{seed}.json"
    if path.exists():return read_json(path)
    folder=output/'runs'/f"{spec['id']}_seed{seed}";saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model=(model_factory(config,spec,None) if model_factory else new_model(config,spec)).cuda();model.load_state_dict(saved['state_dict']);model.eval();result=[]
    for threshold in [.2,.35,.5]:
        model.pen_threshold=threshold;metrics=evaluate_rollouts(CPUInterface(model),validation,config,metadata)
        score=metrics['objective']+.02*np.mean([1-m['pen_f1'] for m in metrics['horizons'].values()])
        result.append({'spec':spec,'seed':seed,'directory':str(folder.relative_to(output)),'checkpoint_epoch':saved['epoch'],
            'pen_threshold':threshold,'score':float(score),'validation':metrics,
            'parameters':sum(p.numel() for p in model.parameters())})
    write_json(path,result);return result
