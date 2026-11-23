"""Previsão direta multimodal t+1..t+30; âncoras aprendidas somente no treino."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def frame(prefix):
    # A direção usa somente os últimos segmentos de tinta já observados.
    delta=prefix[:,1:,:2];valid=(prefix[:,:-1,2]<.5)&(delta.square().sum(-1)>1e-9)
    ranks=valid.long().cumsum(1);mask=valid&(ranks>ranks[:,-1:]-3)
    velocity=(delta*mask[:,:,None]).sum(1);norm=velocity.norm(dim=-1,keepdim=True)
    direction=velocity/norm.clamp_min(1e-6)
    return torch.where((norm<1e-6).expand(-1,2),torch.tensor([1.,0.],device=prefix.device),direction)


def to_local(xy,last,direction):
    delta=xy-last[:,None,:];c=direction[:,0,None];s=direction[:,1,None]
    return torch.stack([delta[:,:,0]*c+delta[:,:,1]*s,-delta[:,:,0]*s+delta[:,:,1]*c],-1)


def to_global(local,last,direction):
    c=direction[:,0,None];s=direction[:,1,None]
    return last[:,None,:]+torch.stack([local[:,:,0]*c-local[:,:,1]*s,local[:,:,0]*s+local[:,:,1]*c],-1)


class TrajectoryGRU(nn.Module):
    def __init__(self,classes,anchors,hidden=192,embedding=16,mode_embedding=24):
        super().__init__();self.hidden=hidden;self.horizon=anchors.shape[2];self.n_modes=anchors.shape[1]
        self.register_buffer('anchors',torch.as_tensor(anchors,dtype=torch.float32))
        self.embedding=nn.Embedding(classes,embedding);self.gru=nn.GRU(6+embedding,hidden,batch_first=True)
        self.mode=nn.Linear(hidden,self.n_modes);self.mode_embedding=nn.Embedding(self.n_modes,mode_embedding)
        self.regression_hidden=nn.Linear(hidden+mode_embedding,128);self.regression=nn.Linear(128,self.horizon*6)
        nn.init.zeros_(self.regression.weight);nn.init.zeros_(self.regression.bias)

    def encode(self,prefix,labels):
        condition=self.embedding(labels)[:,None,:].expand(-1,prefix.shape[1],-1)
        _,state=self.gru(torch.cat([prefix,condition],-1))
        return state[0]

    def decode(self,h,labels,mode):
        anchor=self.anchors[labels,mode];joined=torch.cat([h,self.mode_embedding(mode)],-1)
        residual=self.regression(torch.tanh(self.regression_hidden(joined))).reshape(-1,self.horizon,6)
        offset=.35*torch.tanh(residual[:,:,:2]);xy=anchor[:,:,:2]+offset
        pen=(anchor[:,:,2:5].clamp_min(.02)).log()+residual[:,:,2:5]
        base=anchor[:,:,5].clamp_min(.02);clock=F.softplus(torch.log(torch.expm1(base))+residual[:,:,5])
        return xy,pen,clock,offset

    def forward(self,prefix,labels,local_targets=None,pen_targets=None,clock_targets=None,valid=None):
        h=self.encode(prefix,labels);logits=self.mode(h)
        if local_targets is None:return h,logits
        anchors=self.anchors[labels]
        # Um modo para a trajetória inteira, sem escolher um futuro diferente em cada ponto.
        xy_error=(anchors[:,:,:,:2]-local_targets[:,None]).square().mean((-1,-2))
        end_error=(anchors[:,:,:,4]-(pen_targets==2)[:,None].float()).square().mean(-1)
        mode=(xy_error+.04*end_error).argmin(-1)
        xy,pen,clock,offset=self.decode(h,labels,mode)
        coordinate=(F.smooth_l1_loss(xy,local_targets,reduction='none').mean(-1)*valid).sum(1)/valid.sum(1).clamp_min(1)
        velocity=F.smooth_l1_loss(torch.diff(xy,dim=1),torch.diff(local_targets,dim=1),reduction='none').mean(-1)
        velocity=(velocity*valid[:,1:]).sum(1)/valid[:,1:].sum(1).clamp_min(1)
        pen_loss=F.cross_entropy(pen.transpose(1,2),pen_targets,reduction='none').mean(1)
        clock_loss=(F.smooth_l1_loss(clock,clock_targets,reduction='none')*valid).sum(1)/valid.sum(1).clamp_min(1)
        ce=F.cross_entropy(logits,mode)
        smooth=torch.diff(offset,n=2,dim=1).square().mean()
        objective=ce+8*coordinate.mean()+.4*velocity.mean()+pen_loss.mean()+.15*clock_loss.mean()+.2*smooth
        return objective,{'mode_ce':ce,'coordinate_huber':coordinate.mean(),'velocity_huber':velocity.mean(),'pen_ce':pen_loss.mean(),'clock_huber':clock_loss.mean()}

    @torch.inference_mode()
    def forecast(self,prefix,lengths,labels,metadata,steps,decoder='greedy',seed=2026):
        if steps>self.horizon:raise ValueError('Horizonte maior que o treinado.')
        self.eval()
        if torch.all(lengths==prefix.shape[1]):h,logits=self(prefix,labels)
        else:
            parts=[self(prefix[i:i+1,:n],labels[i:i+1]) for i,n in enumerate(lengths.tolist())];h=torch.cat([a for a,b in parts]);logits=torch.cat([b for a,b in parts])
        if decoder=='greedy':mode=logits.argmax(-1)
        else:
            temperature={'sample020':.2,'sample040':.4,'sample070':.7}[decoder];generator=torch.Generator().manual_seed(seed);mode=torch.multinomial((logits/temperature).softmax(-1),1,generator=generator).squeeze(-1)
        xy,pen,clock,_=self.decode(h,labels,mode);direction=frame(prefix) if torch.all(lengths==prefix.shape[1]) else torch.cat([frame(prefix[i:i+1,:n]) for i,n in enumerate(lengths.tolist())]);last=prefix[torch.arange(len(prefix)),lengths-1,4:6]
        global_xy=to_global(xy,last,direction);end=pen.argmax(-1)==2;lift=pen[:,:,:2].softmax(-1)[:,:,1]>getattr(self,'pen_threshold',.5);status=torch.where(end,torch.full_like(end,2,dtype=torch.long),lift.long());alive=(status!=2).long().cumprod(1).bool()
        position=last;out=[]
        for k in range(steps):
            next_position=torch.where(alive[:,k,None],global_xy[:,k],position)
            delta=(next_position-position)*metadata['position_scale']/metadata['delta_scale'];lift=torch.where(alive[:,k],(status[:,k]==1).float(),torch.ones(len(prefix),device=prefix.device))
            out.append(torch.cat([delta,lift[:,None],(clock[:,k]*alive[:,k])[:,None],next_position],-1));position=next_position
        return torch.stack(out,1),~alive[:,:steps]


def targets(rows,indexes,cut,horizon,metadata,scale=None,offsets=None):
    past=[];future=[];pens=[];valid=[];labels=[]
    for j,index in enumerate(indexes):
        row=rows[int(index)];x=row['x'];start=int(offsets[j]) if offsets is not None else 0;stop=start+cut;p=x[start:stop].copy();n=min(horizon,len(x)-stop);f=np.zeros((horizon,6),np.float32);f[:n]=x[stop:stop+n];f[n:,4:6]=x[-1,4:6];anchor=x[start,4:6].copy();p[:,4:6]-=anchor;f[:,4:6]-=anchor;p[0,:2]=0;p[0,3]=0;pp=np.full(horizon,2,np.int64);pp[:n]=f[:n,2].astype(np.int64)
        past.append(p);future.append(f);pens.append(pp);valid.append(np.arange(horizon)<n);labels.append(row['label'])
    past=torch.from_numpy(np.stack(past));future=torch.from_numpy(np.stack(future));pen=torch.from_numpy(np.stack(pens));valid=torch.from_numpy(np.stack(valid)).float();labels=torch.tensor(labels)
    if scale is not None:
        past[:,:,[0,1,4,5]]*=scale;future[:,:,[0,1,4,5]]*=scale
    local=to_local(future[:,:,4:6],past[:,-1,4:6],frame(past))
    return past,local,pen,future[:,:,3],valid,labels
