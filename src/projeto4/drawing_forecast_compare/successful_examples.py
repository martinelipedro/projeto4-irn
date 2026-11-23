"""Seleção explícita de bons casos de validação para ilustrar previsões t+k."""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..io import ROOT, read_json, write_json
from .data import load
from .diagnostics import load_models


HORIZONS = [5, 15, 30]
LIMITS = {5: (8., 15.), 15: (15., 25.), 30: (25., 40.)}
INK = '#203448'; REAL = '#D96646'; PREDICTED = '#087F8C'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def search(directory=None):
    directory = Path(directory or ROOT/'results/drawing_forecast_compare_v1')
    config = read_json(directory/'config.json'); metadata = read_json(directory/'data/metadata.json')
    output = directory/'successful_examples'; output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    models = load_models(config, directory, seed=42)
    validation = [r for r in load(directory) if r['split']=='validation']
    candidates = []; counts = {str(k): 0 for k in HORIZONS}; accepted = dict(counts)
    for cut in [16, 32, 64]:
        rows = [r for r in validation if len(r['x'])>=cut+5 and r['points'][cut-1,2]<.5]
        for start in range(0,len(rows),128):
            batch = rows[start:start+128]
            x = torch.from_numpy(np.stack([r['x'][:cut] for r in batch]))
            labels = torch.tensor([r['label'] for r in batch]); lengths = torch.full((len(batch),),cut)
            predictions = {}
            for family, model in models.items():
                p, ended = model.forecast(x,lengths,labels,metadata,30)
                predictions[family] = (p.numpy(),ended.numpy())
            for i,row in enumerate(batch):
                extent = max(float(np.ptp(row['points'][:cut,:2],axis=0).max()),14.)
                origin = row['points'][0,:2]
                for k in HORIZONS:
                    if len(row['points'])<cut+k:continue
                    counts[str(k)] += 1
                    truth = row['points'][cut:cut+k]
                    p,end = predictions['gru']; guessed = p[i,:k]
                    xy = guessed[:,4:6]*metadata['position_scale'] + origin
                    distances = np.linalg.norm(xy-truth[:,:2],axis=1)
                    actual_path = np.vstack([row['points'][cut-1,:2],truth[:,:2]])
                    guessed_path = np.vstack([actual_path[0],xy])
                    real_length = float(np.linalg.norm(np.diff(actual_path,axis=0),axis=1).sum())
                    pred_length = float(np.linalg.norm(np.diff(guessed_path,axis=0),axis=1).sum())
                    ratio = pred_length/max(real_length,1e-8)
                    ade,fde = float(distances.mean()),float(distances[-1])
                    limit_ade,limit_fde = LIMITS[k]
                    good = (not end[i,:k].any() and ade<=limit_ade and fde<=limit_fde
                            and ade/extent<=.075 and distances.max()<=3*limit_ade
                            and real_length>=8*k and .6<=ratio<=1.4
                            and not truth[:-1,2].any()
                            and np.array_equal(guessed[:,2]>.5,truth[:,2]>.5))
                    if not good:continue
                    accepted[str(k)] += 1
                    metrics={}; future={}
                    for family,(all_p,all_end) in predictions.items():
                        tokens = all_p[i,:k]; coords = tokens[:,4:6]*metadata['position_scale']+origin
                        error = np.linalg.norm(coords-truth[:,:2],axis=1)
                        dt = metadata['time_scale']*np.expm1(tokens[:,3])
                        times = row['points'][cut-1,3]+np.cumsum(dt)
                        future[family] = np.column_stack([coords,tokens[:,2],times]).tolist()
                        metrics[family] = {'ade':float(error.mean()),'fde':float(error[-1]),
                                           'max_distance':float(error.max()),'premature_end':bool(all_end[i,:k].any()),
                                           'pen_exact':bool(np.array_equal(tokens[:,2]>.5,truth[:,2]>.5))}
                    candidates.append({'id':row['id'],'label':row['label'],'category':config['labels'][row['label']],
                                       'source_split':'validation','cut':cut,'k':k,'points':row['points'].tolist(),
                                       'forecast':future,'metrics':metrics,'real_ink_length':real_length,
                                       'predicted_length_ratio':ratio,'observed_extent':extent,
                                       'score':ade+fde*.25})
        print('Validação examinada: corte',cut,'bons casos por horizonte',accepted,flush=True)
    # Limita somente o material de revisão; as contagens acima usam toda a validação elegível.
    review=[]
    for k in HORIZONS:
        for label in range(12):
            chosen=sorted([r for r in candidates if r['k']==k and r['label']==label],
                          key=lambda r:(r['score'],r['id'],r['cut']))[:3]
            review.extend(chosen)
    checkpoints={}
    for family in models:
        spec=read_json(directory/'models'/family/'selection.json')['best']['spec']
        path=directory/'models'/family/'runs'/f'{spec["id"]}_seed42'/'best.pt'
        checkpoints[family]=sha(path)
    record={'split':'validation','seed':42,'decoder':'greedy','cuts':[16,32,64],
            'thresholds':{str(k):{'ade_max':LIMITS[k][0],'fde_max':LIMITS[k][1]} for k in HORIZONS},
            'additional_filters':'ADE/extensão observada<=0.075; sem fim precoce; pen-up correto; trecho contínuo; comprimento real>=8k pixels; razão de comprimentos0.6..1.4; erro máximo<=3limiteADE.',
            'candidate_counts':counts,'qualified_counts':accepted,'checkpoint_sha256':checkpoints,
            'selection_is_qualitative_not_test_estimate':True,'candidates':review}
    write_json(output/'candidate_review.json',record)
    contact_sheets(output,review)
    return record


def plot_ink(ax,points,color,lw=1.8,**kwargs):
    p=np.asarray(points)
    for i in range(1,len(p)):
        if p[i-1,2]<.5:ax.plot(p[i-1:i+1,0],p[i-1:i+1,1],color=color,lw=lw,**kwargs)


def contact_sheets(output,candidates):
    for k in HORIZONS:
        cases=sorted([r for r in candidates if r['k']==k],key=lambda r:r['score'])[:18]
        if not cases:continue
        fig,axes=plt.subplots(int(np.ceil(len(cases)/3)),3,figsize=(12,3*int(np.ceil(len(cases)/3))),squeeze=False)
        for ax,case in zip(axes.flat,cases):
            p=np.asarray(case['points']);cut=case['cut'];future=np.asarray(case['forecast']['gru'])
            plot_ink(ax,p[:cut],INK)
            plot_ink(ax,p[cut-1:cut+k],REAL,lw=2.7,alpha=.6)
            plot_ink(ax,np.vstack([p[cut-1],future]),PREDICTED,lw=1.8,linestyle='--')
            ax.set_title(f"{case['category']} · {case['id']}\nt={cut}, k={k}, ADE={case['metrics']['gru']['ade']:.1f}, FDE={case['metrics']['gru']['fde']:.1f}",fontsize=9)
            ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.axis('off')
        for ax in list(axes.flat)[len(cases):]:ax.axis('off')
        fig.tight_layout();fig.savefig(output/f'review_k{k}.png',dpi=140);plt.close(fig)


def make_pages(source,report_output):
    folder=source/'successful_examples'
    selected=read_json(folder/'selected_cases.json')
    review=read_json(folder/'candidate_review.json')
    assert selected['review_checkpoint_sha256']==review['checkpoint_sha256']
    for family,digest in selected['review_checkpoint_sha256'].items():
        spec=read_json(source/'models'/family/'selection.json')['best']['spec']
        assert sha(source/'models'/family/'runs'/f'{spec["id"]}_seed42'/'best.pt')==digest
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    pages=[]
    for index,case in enumerate(selected['cases'],1):
        assert case['source_split']=='validation' and not case['metrics']['gru']['premature_end']
        p=np.asarray(case['points']);cut=case['cut'];k=case['k']
        truth=p[cut:cut+k];prediction=np.asarray(case['forecast']['gru']);last=p[cut-1]
        extent=max(float(np.ptp(p[:cut+k,:2],axis=0).max()),1.)
        center=(p[:cut+k,:2].min(0)+p[:cut+k,:2].max(0))/2
        fig=plt.figure(figsize=(7.1,7.4),facecolor='white')
        grid=fig.add_gridspec(3,6,left=.10,right=.98,bottom=.08,top=.95,
                             height_ratios=[1.45,1.1,1.0],hspace=.52,wspace=.75)
        for column,title in enumerate(['Passado observado até t','Passado + futuro real','Passado + futuro da GRU']):
            ax=fig.add_subplot(grid[0,column*2:column*2+2])
            plot_ink(ax,p[:cut],INK,lw=1.7)
            if column==1:plot_ink(ax,np.vstack([last,truth]),REAL,lw=2.)
            if column==2:plot_ink(ax,np.vstack([last,prediction]),PREDICTED,lw=2.)
            ax.scatter(last[0],last[1],s=14,color=INK,zorder=3)
            ax.set(xlim=(center[0]-.6*extent,center[0]+.6*extent),
                   ylim=(center[1]+.6*extent,center[1]-.6*extent),aspect='equal')
            ax.axis('off');ax.set_title(title,fontsize=8.4,pad=9)
        ax=fig.add_subplot(grid[1,:])
        plot_ink(ax,np.vstack([last,truth]),REAL,lw=2.5,alpha=.6)
        plot_ink(ax,np.vstack([last,prediction]),PREDICTED,lw=1.8,linestyle='--')
        ax.scatter(truth[:,0],truth[:,1],s=7,color=REAL,zorder=3)
        ax.scatter(prediction[:,0],prediction[:,1],s=7,color=PREDICTED,zorder=3)
        ax.set_title('Zoom no trecho t+1 até t+k: real e previsto sobrepostos',fontsize=9)
        ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.14)
        ax.tick_params(labelsize=7);ax.set_xlabel('X (pixels)',fontsize=8);ax.set_ylabel('Y (pixels)',fontsize=8)
        tail=min(8,cut-1);past=p[cut-1-tail:cut]
        for column,coordinate in enumerate([0,1]):
            ax=fig.add_subplot(grid[2,column*3:column*3+3])
            ax.plot(np.arange(-tail,1),past[:,coordinate],color=INK,label='Passado',lw=1.5)
            ax.plot(np.arange(0,k+1),np.r_[last[coordinate],truth[:,coordinate]],color=REAL,label='Real',lw=1.8)
            ax.plot(np.arange(0,k+1),np.r_[last[coordinate],prediction[:,coordinate]],color=PREDICTED,label='GRU',lw=1.6,ls='--')
            ax.axvline(0,color='#84929A',ls=':',lw=1)
            ax.set(title=('X' if coordinate==0 else 'Y')+' ao longo dos eventos',
                   xlabel='Evento relativo a t',ylabel='Pixels')
            ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.title.set_size(9)
            ax.grid(alpha=.17)
            if column==0:ax.legend(fontsize=7,ncol=3,loc='best',columnspacing=.7,handlelength=1.3)
        stem=f'forecast_success_{index:02d}_{case["id"]}_k{k}'
        path=report_output/'figures'/f'{stem}.pdf'
        for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=200)
        plt.close(fig)
        m=case['metrics']['gru']
        number=lambda v:f'{v:.2f}'.replace('.',',')
        pages.append({'title':f'Previsão {index} — {case["category"]}, horizonte k={k}',
            'text':f'Categoria fornecida: {case["category"]}. A GRU observa {cut} eventos e prevê os próximos {k}. '
                   f'ADE de {number(m["ade"])} px e FDE de {number(m["fde"])} px. Validação, ID {case["id"]}, seed 42.',
            'figure':path,
            'caption':'Caso positivo selecionado na validação e revisado visualmente. O horizonte contém somente os próximos k eventos, não o desenho inteiro. '
                      'A curva esperada e a curva prevista usam os mesmos índices de eventos; o estado da caneta também coincide neste trecho. '
                      'O zoom revela diferenças residuais que a visão geral pode esconder. O caso não representa o erro médio no conjunto de teste.'})
    write_json(folder/'gallery_manifest.json',{'cases':len(pages),'horizons':HORIZONS,'split':'validation',
              'seed':42,'decoder':'greedy','success_gallery_only':True,'aggregate_test_unchanged':True})
    return pages


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path);args=parser.parse_args()
    search(args.directory)
