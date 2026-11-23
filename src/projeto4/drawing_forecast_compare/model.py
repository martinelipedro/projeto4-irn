import numpy as np
import torch
from torch import nn

from .trajectory import TrajectoryGRU


class DeepSetsForecast(TrajectoryGRU):
    """Mesmo decoder do TrajectoryGRU; troca somente o encoder por pooling de eventos."""
    def __init__(self,classes,anchors,hidden=128,embedding=16,context_events=64):
        super().__init__(classes,anchors,hidden,embedding,raster=False)
        # Os heads comuns têm exatamente a inicialização do construtor da GRU.
        # A GRU descartada não fica no state_dict nem recebe otimização.
        del self.gru
        self.context_events=context_events
        self.set_phi=nn.Sequential(nn.Linear(6+embedding,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.set_rho=nn.Linear(hidden+1,hidden)

    def encode_masked(self,prefix,labels,mask=None):
        condition=self.embedding(labels)[:,None,:].expand(-1,prefix.shape[1],-1)
        representation=self.set_phi(torch.cat([prefix,condition],dim=-1))
        if mask is None:mask=torch.ones(prefix.shape[:2],device=prefix.device,dtype=prefix.dtype)
        mask=mask.to(prefix.dtype);count=mask.sum(1,keepdim=True).clamp_min(1)
        pooled=(representation*mask[:,:,None]).sum(1)/count
        return torch.tanh(self.set_rho(torch.cat([pooled,count/self.context_events],dim=-1)))

    def encode(self,prefix,labels):return self.encode_masked(prefix,labels)


def new_model(config,spec,anchors=None):
    if anchors is None:anchors=np.zeros((len(config['classes']),spec.get('n_modes',64),max(config['horizons']),6),np.float32)
    if spec['architecture']=='deepsets':
        return DeepSetsForecast(len(config['classes']),anchors,spec['hidden'],config['embedding'],config['context_events'])
    if spec['architecture']=='trajectory' and not spec.get('raster',False):
        return TrajectoryGRU(len(config['classes']),anchors,spec['hidden'],config['embedding'],raster=False)
    raise ValueError('Benchmark principal admite apenas GRU pura e DeepSets, sem CNN.')
