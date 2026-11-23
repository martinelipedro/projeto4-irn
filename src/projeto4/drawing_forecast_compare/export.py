"""Pesos e fixtures CPU para verificar a inferência JavaScript sem servidor."""
import json

import numpy as np
import torch

from ..io import read_json,write_json
from .data import tokens_from_points
from .data import prepare,load
from .model import new_model


def export(config,families=None):
    output=prepare(config)
    families=list(config['families']) if families is None else families
    if not all((output/'models'/name/'frozen_selection.json').exists() for name in families):raise RuntimeError('Congele a família antes de exportar pesos.')
    torch.set_num_threads(config['threads']);metadata=read_json(output/'data/metadata.json')
    rows=load(output);validation=[r for r in rows if r['split']=='validation']
    examples=[];chosen=[]
    for label in range(len(config['classes'])):
        eligible=[r for r in validation if r['label']==label and len(r['x'])>=62]
        chosen.append(eligible[0])
        for row in eligible[:2]:
            examples.append({'id':row['id'],'label':label,'points':np.round(row['points'].astype(np.float64),3).tolist(),
                             'source_split':'validation','selection_rule':'First two validation drawings per class with at least62events; no selection by forecast error.'})
    write_json(output/'examples.json',examples)
    for family in families:
        folder=output/'models'/family;selected=read_json(folder/'selection.json')['best'];spec=selected['spec']
        saved=torch.load(folder/'runs'/f"{spec['id']}_seed42"/'best.pt',map_location='cpu',weights_only=True)
        model=new_model(config,spec);model.load_state_dict(saved['state_dict']);model.pen_threshold=selected['pen_threshold'];model.eval()
        exported={'format':'drawing-forecast-compare-v1','family':family,'architecture':spec['architecture'],'raster':False,
            'hidden':spec['hidden'],'embedding':config['embedding'],'n_modes':spec['n_modes'],'horizon':max(config['horizons']),
            'horizons':config['horizons'],'context_events':config['context_events'],'min_context_events':8,
            'delta_scale':metadata['delta_scale'],'position_scale':metadata['position_scale'],'time_scale':metadata['time_scale'],
            'spatial_step':config['spatial_step'],'classes':config['classes'],'labels':config['labels'],'seed':42,'epoch':saved['epoch'],
            'pen_threshold':selected['pen_threshold'],'known_category':True,
            'features':['dx/delta_scale','dy/delta_scale','pen_up','log1p(dt/time_scale)','x_relative/256','y_relative/256'],
            'weights':{key:np.round(value.detach().numpy().astype(np.float64),7).tolist() for key,value in model.state_dict().items()}}
        (folder/'model.json').write_text(json.dumps(exported,ensure_ascii=False,separators=(',',':')))
        references=[]
        for row in chosen:
            points=np.round(row['points'][:16].astype(np.float64),3);x=tokens_from_points(points,metadata)
            prefix=torch.from_numpy(x[None]);labels=torch.tensor([row['label']]);lengths=torch.tensor([len(x)])
            with torch.inference_mode():
                h,logits=model(prefix,labels);tokens,ended=model.forecast(prefix,lengths,labels,metadata,30)
            references.append({'id':row['id'],'label':row['label'],'points':points.tolist(),'input':x.tolist(),
                'h':h[0].tolist(),'logits':logits[0].tolist(),'tokens':tokens[0].tolist(),'ended':ended[0].tolist()})
        write_json(folder/'javascript_reference.json',references)
        print('EXPORT',family,'seed42',folder/'model.json',flush=True)
    write_json(output/'example_manifest.json',{'source_split':'validation','rule':'First two drawings per class with >=62events; IDs chosen without errors.',
        'ids':[r['id'] for r in examples],'reference_fixture_ids':[r['id'] for r in chosen],'fixed_prefixes':[16,32],'horizons':config['horizons']})
