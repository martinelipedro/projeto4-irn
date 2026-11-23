"""Banco comum de âncoras: exclusivamente dados de treino."""
import numpy as np
import torch
from ..io import write_json
from .trajectory import targets

def learn_anchors(rows,config,metadata,output,n_modes=64):
    path=output/f'anchors_{n_modes}.npz'
    if path.exists():return np.load(path)['anchors']
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    horizon=max(config['horizons']);rng=np.random.default_rng(20261117);all_anchors=[];ids=[]
    with threadpool_limits(limits=2):
        for label in range(len(config['classes'])):
            indexes=[i for i,r in enumerate(rows) if r['label']==label];features=[]
            for index in indexes:
                row=rows[index];upper=min(len(row['x'])-1,64)
                for _ in range(3):
                    cut=int(rng.integers(8,upper+1));past,xy,pen,clock,valid,labels=targets(rows,[index],cut,horizon,metadata)
                    onehot=torch.nn.functional.one_hot(pen,3).float()
                    features.append(np.r_[xy.numpy().ravel(),.15*onehot.numpy().ravel(),.05*clock.numpy().ravel()])
                ids.append(row['id'])
            data=np.array(features,np.float32)
            cluster=MiniBatchKMeans(n_clusters=n_modes,batch_size=1024,n_init=3,max_iter=150,random_state=42).fit(data)
            centers=cluster.cluster_centers_;xy=centers[:,:2*horizon].reshape(n_modes,horizon,2)
            pen=np.clip(centers[:,2*horizon:5*horizon].reshape(n_modes,horizon,3)/.15,0,1);pen/=pen.sum(-1,keepdims=True).clip(1e-8)
            clock=np.maximum(0,centers[:,5*horizon:].reshape(n_modes,horizon,1)/.05)
            all_anchors.append(np.concatenate([xy,pen,clock],-1));print('âncoras',config['classes'][label],len(features),flush=True)
    anchors=np.array(all_anchors,np.float32);np.savez_compressed(path,anchors=anchors)
    write_json(output/f'anchors_{n_modes}_provenance.json',{'fit_split':'train','drawings':len(ids),'training_ids':ids,'n_modes_per_class':n_modes,'examples_per_drawing':3,'seed':20261117,'method':'MiniBatchKMeans sobre a trajetória futura no referencial causal do último segmento observado. Nenhum desenho de validação ou teste participa.'})
    return anchors
