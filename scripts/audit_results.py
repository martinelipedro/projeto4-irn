"""Auditoria CPU dos dados e artefatos congelados, sem retreinar ou usar a GPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from projeto4.drawing_forecast_compare.data import sampled_points


def read(path):return json.loads(path.read_text())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def audit(directory,raw_directory=None):
    config=read(directory/'config.json');records=read(directory/'data/records.json')
    raw_manifest=read(directory/'data_sources/raw_manifest.json')
    assert sha(directory/'data/sequences.npz')==raw_manifest['prepared_npz_sha256']
    ids={split:{r['id'] for r in records if r['split']==split} for split in ['train','validation','test']}
    geometry={split:{r['geometry_sha256'] for r in records if r['split']==split} for split in ids}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not ids[a]&ids[b],f'IDs cruzando {a}/{b}'
        assert not geometry[a]&geometry[b],f'Geometrias cruzando {a}/{b}'
    assert [len(ids[k]) for k in ids]==[36000,3000,3000]
    anchor_provenance=read(directory/'anchors_64_provenance.json')
    assert anchor_provenance['fit_split']=='train'
    assert set(anchor_provenance['training_ids'])<=ids['train']
    reuse=read(directory/'reuse_manifest.json')
    assert sha(directory/'anchors_64.npz')==reuse['anchors_sha256']
    for row in reuse['runs']:
        checkpoint=directory/'models/gru/runs'/f"{row['spec']['id']}_seed42"/'best.pt'
        assert sha(checkpoint)==row['checkpoint_sha256']==row['copied_checkpoint_sha256']
        assert not row['warm_started']
    definitions={};counts={}
    for family in ['gru','mlp']:
        folder=directory/'models'/family;selection=read(folder/'selection.json');metrics=read(folder/'test_metrics.json')
        assert selection['test_used_for_selection'] is False
        assert selection['all_models_from_scratch'] is True
        assert selection['best']==min(selection['finalist_scores'],key=lambda x:x['score'])
        assert [r['seed'] for r in metrics['seeds']]==[42,43,44]
        assert metrics['known_category'] is True
        for run in selection['runs']:
            assert not run.get('warm_started',False)
            history_path=folder/run['directory']/'history.csv'
            with history_path.open() as f:history=list(csv.DictReader(f))
            winner=min(history,key=lambda r:float(r['val_rollout_objective']))
            assert int(winner['epoch'])==run['best_epoch']
            assert len(history)==run['spec']['epochs']
        for k in config['horizons']:
            expected=sum(r['length']>cut+min(config['horizons']) and r['length']>=cut+k
                         for cut in config['validation_cuts'] for r in records if r['split']=='test')
            h=metrics['horizons'][str(k)];assert h['n_prefixes']==expected
            np.testing.assert_allclose(h['ade_pixels'],np.mean([r['horizons'][str(k)]['ade_pixels'] for r in metrics['seeds']]))
            assert np.isfinite([h['ade_pixels'],h['fde_pixels'],h['rmse_pixels']]).all()
            counts[f'{family}_k{k}']=expected
        definitions[family]=read(folder/'model.json')
    a,b=definitions['gru'],definitions['mlp']
    for key in ['classes','features','embedding','n_modes','horizon','context_events','delta_scale','position_scale','time_scale']:
        assert a[key]==b[key],f'Representação/decoder divergente: {key}'
    np.testing.assert_array_equal(a['weights']['anchors'],b['weights']['anchors'])
    assert not any(key.startswith('gru.') for key in b['weights'])
    assert not a['raster'] and not b['raster']
    raw_checked=0
    if raw_directory is not None:
        with np.load(directory/'data/sequences.npz') as arrays:
            for label,category in enumerate(config['classes']):
                wanted=dict([(r['id'],i) for i,r in enumerate(records) if r['label']==label][:3])
                with (raw_directory/(category+'.ndjson')).open() as f:
                    for line in f:
                        try:row=json.loads(line)
                        except json.JSONDecodeError:continue
                        if row['key_id'] not in wanted:continue
                        index=wanted.pop(row['key_id'])
                        np.testing.assert_array_equal(sampled_points(row['drawing'],config['spatial_step']),arrays[f'points_{index}'])
                        raw_checked+=1
                        if not wanted:break
                assert not wanted
    return {'passed':True,'dataset_sha256':raw_manifest['prepared_npz_sha256'],
            'split_sizes':{k:len(v) for k,v in ids.items()},'ids_and_geometries_disjoint':True,
            'anchors_fit_only_train':True,'shared_representation_and_anchors':True,
            'checkpoint_selection_by_validation':True,'same_main_target_counts':counts,
            'raw_reconstruction_examples_checked':raw_checked}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/drawing_forecast_compare_v1')
    parser.add_argument('--raw-directory',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.results,args.raw_directory),ensure_ascii=False,indent=2))
