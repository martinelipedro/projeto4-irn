"""Verifica decisões de validação, arquiteturas e métricas publicadas do reconhecimento."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from projeto4.drawing_recognition_compare.study import metrics
from projeto4.drawing_recognition_compare.model import make_model


def read(path):return json.loads(path.read_text())


def audit():
    config=read(ROOT/'configs/recognition.json');output=ROOT/'results'/config['name']
    manifest=read(output/'data_manifest.json')
    source=ROOT/'results'/config['source_experiment']
    assert hashlib.sha256((source/'data/records.json').read_bytes()).hexdigest()==manifest['source_manifest_sha256']
    records=read(source/'data/records.json');test=[r for r in records if r['split']=='test']
    truth=np.array([r['label'] for r in test]);assert len(truth)==3000
    result=read(output/'test_metrics.json')
    for family in ['gru','mlp']:
        selection=read(output/family/'frozen_selection.json')
        assert selection['test_used'] is False
        assert len(selection['screening'])==4
        assert len(selection['finalists'])==2
        assert selection['best']==min(selection['finalists'],key=lambda row:row['score'])
        for candidate in selection['finalists']:
            spec=candidate['spec']
            runs=[r for r in selection['runs'] if r['spec']==spec and r['seed'] in [42,43]]
            assert sorted(r['seed'] for r in runs)==[42,43]
            np.testing.assert_allclose(candidate['score'],np.mean([r['score'] for r in runs]),rtol=1e-12)
        for run in selection['runs']:
            assert run['from_scratch'] and run['epochs']==config['epochs']
            with (output/run['directory']/'history.csv').open() as stream:
                history=list(csv.DictReader(stream))
            assert len(history)==config['epochs']
            winner=min(history,key=lambda r:float(r['val_ce']))
            assert int(winner['epoch'])==run['best_epoch']
            np.testing.assert_allclose(float(winner['val_ce']),run['score'])
            checkpoint=torch.load(output/run['directory']/'best.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==run['best_epoch'] and checkpoint['class_label_in_input'] is False
        spec=selection['best']['spec'];model=make_model(family,spec['hidden'])
        weights=read(ROOT/f'site/assets/recognition/{family}.json')['weights']
        checkpoint=torch.load(output/family/f'h{spec["hidden"]}_lr{spec["lr"]}_seed42'/'best.pt',map_location='cpu',weights_only=True)
        for key,array in checkpoint['state_dict'].items():np.testing.assert_array_equal(array.numpy(),np.asarray(weights[key],np.float32))
        if family=='mlp':
            assert not any(isinstance(m,(torch.nn.GRU,torch.nn.LSTM,torch.nn.Conv2d)) for m in model.modules())
            assert model.layers[0].in_features==1024
        else:assert model.gru.input_size==6
        for seed,row in zip(config['seeds'],result[family]['rows']):
            assert seed==row['seed']
            with np.load(output/family/f'test_seed{seed}.npz') as arrays:
                np.testing.assert_array_equal(truth,arrays['truth']);p=arrays['probabilities']
            assert p.shape==(3000,4,12)
            for j,f in enumerate(config['fractions']):
                actual=metrics(truth,p[:,j]);saved=row['fractions'][str(f)]
                for key in ['accuracy','macro_f1','top3','ce','ece','brier']:np.testing.assert_allclose(actual[key],saved[key])
                assert actual['confusion']==saved['confusion']
    diagnostic=read(output/'diagnostics.json');assert diagnostic['controls']['same_bitmap'] is True
    assert diagnostic['controls']['test_ids']==[r['id'] for r in test]
    examples=read(ROOT/'site/assets/recognition/examples.json')['examples']
    valid={r['id'] for r in records if r['split']=='validation'}
    assert all(e['id'] in valid and e['source_split']=='validation' for e in examples)
    assert len({e['label'] for e in examples})==12
    return {'passed':True,'classifiers_from_scratch':True,'ordinary_mlp_pixels':True,
            'category_not_in_input':True,'selection_validation_only':True,'epochs_per_run':config['epochs'],
            'test_drawings':3000,'seeds':config['seeds'],'export_matches_checkpoints':True,
            'metrics_recomputed':True,'all_demo_examples_validation':True,'same_bitmap_order_probe':True}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
