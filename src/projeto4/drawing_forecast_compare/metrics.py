"""Métricas de previsão pura, extraídas sem alterar fórmulas nem elegibilidade."""
import numpy as np
import torch
from .inference import rollout

def ink_distance(a,b,pen_a,pen_b):
    def ink(p,pen):
        cloud=[]
        for i in range(1,len(p)):
            if pen[i-1]:continue
            n=max(1,int(np.ceil(np.linalg.norm(p[i]-p[i-1])/8)))
            cloud.extend(p[i-1]+np.arange(1,n+1)[:,None]/n*(p[i]-p[i-1]))
        return np.array(cloud) if cloud else p
    a=ink(a,pen_a);b=ink(b,pen_b)
    d=np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(-1))
    return float((d.min(0).mean()+d.min(1).mean())/2)

@torch.inference_mode()
def evaluate_rollouts(model,rows,config,metadata,decoder='greedy',detailed=False,limit_per_class=None):
    selected=rows
    if limit_per_class is not None:selected=sum([[r for r in rows if r['label']==label][:limit_per_class] for label in range(len(config['classes']))],[])
    sums={str(k):[] for k in config['horizons']};details=[]
    for cut in config['validation_cuts']:
        eligible=[r for r in selected if len(r['x'])>cut+min(config['horizons'])]
        for start in range(0,len(eligible),config['batch_size']):
            batch=eligible[start:start+config['batch_size']]
            x=torch.from_numpy(np.stack([r['x'][:cut] for r in batch]));labels=torch.tensor([r['label'] for r in batch]);lengths=torch.full((len(batch),),cut,dtype=torch.long)
            pred,ended=rollout(model,x,lengths,labels,metadata,max(config['horizons']),decoder,seed=2026+start)
            p=pred.numpy();ended=ended.numpy()
            for i,row in enumerate(batch):
                available=min(len(row['x'])-cut,max(config['horizons']));truth=row['x'][cut:cut+available]
                pos=p[i,:available,4:6]*metadata['position_scale'];actual=truth[:,4:6]*metadata['position_scale']
                last=row['x'][cut-1,4:6]*metadata['position_scale'];velocity=row['x'][cut-1,:2]*metadata['delta_scale']
                for k in config['horizons']:
                    if len(truth)<k:continue
                    distances=np.linalg.norm(pos[:k]-actual[:k],axis=1)
                    hold=np.linalg.norm(last-actual[:k],axis=1);linear=np.linalg.norm(last+np.arange(1,k+1)[:,None]*velocity-actual[:k],axis=1)
                    dt=metadata['time_scale']*np.expm1(p[i,:k,3]);true_dt=metadata['time_scale']*np.expm1(truth[:k,3]);past_dt=metadata['time_scale']*np.expm1(row['x'][:cut,3]);positive=past_dt[past_dt>0];median_dt=float(np.median(positive)) if len(positive) else .04
                    guessed=p[i,:k,2]>.5;actual_pen=truth[:k,2]>.5
                    result={'ade_pixels':float(distances.mean()),'fde_pixels':float(distances[-1]),'rmse_pixels':float(np.sqrt(((pos[:k]-actual[:k])**2).mean())),
                        'hold_ade_pixels':float(hold.mean()),'velocity_ade_pixels':float(linear.mean()),'clock_mae_seconds':float(np.abs(dt-true_dt).mean()),
                        'clock_median_mae_seconds':float(np.abs(median_dt-true_dt).mean()),'elapsed_error_seconds':float(abs(dt.sum()-true_dt.sum())),'pen_accuracy':float((p[i,:k,2]==truth[:k,2]).mean()),'premature_end':bool(ended[i,:k].any()),'pen_tp':int((guessed&actual_pen).sum()),'pen_fp':int((guessed&~actual_pen).sum()),'pen_fn':int((~guessed&actual_pen).sum()),'pen_tn':int((~guessed&~actual_pen).sum())}
                    if detailed:result['ink_chamfer_pixels']=ink_distance(np.vstack([last,pos[:k]]),np.vstack([last,actual[:k]]),np.r_[row['x'][cut-1,2],p[i,:k,2]],np.r_[row['x'][cut-1,2],truth[:k,2]])
                    sums[str(k)].append(result)
                if detailed:details.append({'id':row['id'],'label':row['label'],'cut':cut,'tokens':p[i].tolist(),'ended':ended[i].tolist()})
    result={k:{'n_prefixes':len(values),**{metric:float(np.mean([v[metric] for v in values])) for metric in values[0]}} for k,values in sums.items() if values}
    for k,values in sums.items():
        if not values:continue
        tp=sum(v['pen_tp'] for v in values);fp=sum(v['pen_fp'] for v in values);fn=sum(v['pen_fn'] for v in values);tn=sum(v['pen_tn'] for v in values)
        for field in ['pen_tp','pen_fp','pen_fn','pen_tn']:result[k][field]=sum(v[field] for v in values)
        result[k]['pen_f1']=2*tp/max(1,2*tp+fp+fn);result[k]['pen_balanced_accuracy']=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp))
    # Seleção dá peso igual a cada horizonte. O futuro real só calcula a métrica.
    objective=float(np.mean([v['ade_pixels']/metadata['position_scale']+.1*v['premature_end'] for v in result.values()]))
    return {'objective':objective,'horizons':result,'details':details if detailed else []}

@torch.inference_mode()
def ending_metrics(model,rows,config,metadata):
    truth=[];prediction=[];errors=[]
    for cut in config['validation_cuts']:
        eligible=[r for r in rows if len(r['x'])>cut]
        for start in range(0,len(eligible),config['batch_size']):
            batch=eligible[start:start+config['batch_size']];x=torch.from_numpy(np.stack([r['x'][:cut] for r in batch]));labels=torch.tensor([r['label'] for r in batch])
            _,ended=rollout(model,x,torch.full((len(batch),),cut),labels,metadata,30)
            for i,row in enumerate(batch):
                n=len(row['x'])-cut;actual=n<30;positions=torch.where(ended[i])[0];guess=int(positions[0])+1 if len(positions) else 31
                truth.append(actual);prediction.append(guess<=30)
                if actual:errors.append(abs(guess-(n+1)))
    t=np.array(truth);p=np.array(prediction);tp=int((t&p).sum());fp=int((~t&p).sum());fn=int((t&~p).sum())
    return {'n_prefixes':len(t),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'ending_event_mae':float(np.mean(errors)) if errors else None}
