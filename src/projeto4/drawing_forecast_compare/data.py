"""Manifesto por ID e reconstrução de dados sem depender de partições de outros estudos."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from ..io import ROOT,read_json,write_json



def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def prepare(config,download=False):
    output=ROOT/'results'/config['name'];output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() and read_json(output/'config.json')!=config:raise ValueError('Configuração congelada: use outro nome.')
    write_json(output/'config.json',config)
    folder=output/'data';sources=output/'data_sources';folder.mkdir(exist_ok=True);sources.mkdir(exist_ok=True)
    if (folder/'sequences.npz').exists():return output
    source=ROOT/'results'/config['source_experiment']
    if (source/'data/sequences.npz').exists():
        for name in ['sequences.npz','records.json','metadata.json']:shutil.copy2(source/'data'/name,folder/name)
        for name in ['anchors_64.npz','anchors_64_provenance.json']:shutil.copy2(source/name,output/name)
        source_config=read_json(source/'config.json');metadata=read_json(folder/'metadata.json')
        if (source/'data_sources/raw_manifest.json').exists():
            manifest=read_json(source/'data_sources/raw_manifest.json')
        else:
            raw=ROOT/source_config['raw_directory']
            manifest={'dataset':'Google Quick, Draw!, raw moderated NDJSON','license':'CC BY 4.0','files':{}}
            for category in config['classes']:
                manifest['files'][category]={'url':f'https://storage.googleapis.com/quickdraw_dataset/full/raw/{category}.ndjson',
                    'bytes':(raw/(category+'.ndjson')).stat().st_size,'sha256':metadata['source_files'][category]}
        manifest.update({
                  'records':'../data/records.json','prepared_npz_bytes':(folder/'sequences.npz').stat().st_size,
                  'prepared_npz_sha256':sha256(folder/'sequences.npz'),'partition_note':'Same fixed IDs/splits as the source study; tests previously observed. No repartitioning.',
                  'sampling':{'spatial_step':config['spatial_step'],'position_scale':config['position_scale'],'time_scale':config['time_scale']}})
        write_json(sources/'raw_manifest.json',manifest)
        metadata['partition_manifest']='records.json';metadata['test_previously_observed']=True
        metadata['prepared_data_origin']=config['source_experiment'];write_json(folder/'metadata.json',metadata)
        return output
    if not (sources/'raw_manifest.json').exists():raise FileNotFoundError('Inclua data_sources/raw_manifest.json ou o dataset NPZ preparado.')
    reconstruct(config,output,download)
    return output


def sampled_points(drawing,step=14.):
    points=[];previous_time=-np.inf
    for stroke in drawing:
        kept=[]
        for x,y,stamp in zip(*stroke):
            stamp=float(stamp)/1000
            if not np.isfinite([x,y,stamp]).all() or stamp<previous_time:raise ValueError('Traço inválido.')
            previous_time=stamp
            if not kept or np.hypot(x-kept[-1][0],y-kept[-1][1])>=step:kept.append([float(x),float(y),stamp])
        if not kept:continue
        end=[float(stroke[0][-1]),float(stroke[1][-1]),float(stroke[2][-1])/1000]
        if end!=kept[-1]:kept.append(end)
        points.extend([[x,y,int(i==len(kept)-1),stamp] for i,(x,y,stamp) in enumerate(kept)])
    points=np.asarray(points,np.float32);points[:,3]-=points[0,3]
    return points


def reconstruct(config,output,download=False):
    """Reconstrói apenas IDs do manifesto; não consulta regras previous_experiments."""
    manifest=read_json(output/'data_sources/raw_manifest.json');records=read_json(output/'data/records.json')
    raw=output/'data_sources/raw';raw.mkdir(exist_ok=True);arrays={}
    by_id={r['id']:i for i,r in enumerate(records)}
    for label,category in enumerate(config['classes']):
        source=manifest['files'][category];path=raw/(category+'.ndjson')
        if not path.exists():
            if not download:raise FileNotFoundError(f'{path}: use prepare --download-raw para baixar os recortes do manifesto.')
            temporary=path.with_suffix('.part')
            request=urllib.request.Request(source['url'],headers={'Range':f"bytes=0-{source['bytes']-1}"})
            with urllib.request.urlopen(request) as response,temporary.open('wb') as stream:
                remaining=source['bytes']
                while remaining:
                    chunk=response.read(min(1024*1024,remaining))
                    if not chunk:raise RuntimeError('Download incompleto.')
                    stream.write(chunk);remaining-=len(chunk)
            temporary.replace(path)
        if path.stat().st_size!=source['bytes'] or sha256(path)!=source['sha256']:raise ValueError('Recorte bruto difere do manifesto: '+category)
        needed={r['id'] for r in records if r['label']==label}
        with path.open() as stream:
            for line in stream:
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if row['key_id'] not in needed:continue
                index=by_id[row['key_id']];points=sampled_points(row['drawing'],config['spatial_step'])
                if len(points)!=records[index]['length']:raise ValueError('Comprimento divergente: '+row['key_id'])
                arrays[f'points_{index}']=points;needed.remove(row['key_id'])
                if not needed:break
        if needed:raise ValueError('IDs não encontrados: '+category)
    if len(arrays)!=len(records):raise AssertionError('Dataset reconstruído incompleto.')
    temporary=output/'data/sequences.partial.npz';np.savez_compressed(temporary,**arrays);temporary.replace(output/'data/sequences.npz')
    write_json(output/'data_sources/reconstruction.json',{'drawings':len(arrays),'partition_ids_preserved':True,'sampling':'Causal, mesmas regras do manifesto; sem consulta a experiências antigas.'})


def tokens_from_points(points,metadata):
    p=np.array(points,dtype=np.float32)
    delta=np.diff(p[:,:2],axis=0,prepend=p[:1,:2])/metadata['delta_scale']
    relative=(p[:,:2]-p[:1,:2])/metadata['position_scale']
    dt=np.diff(p[:,3],prepend=p[0,3])
    return np.column_stack([delta,p[:,2],np.log1p(dt/metadata['time_scale']),relative]).astype(np.float32)


def load(output):
    metadata=read_json(output/'data/metadata.json');records=read_json(output/'data/records.json')
    with np.load(output/'data/sequences.npz') as arrays:
        return [{**r,'points':arrays[f'points_{i}'],'x':tokens_from_points(arrays[f'points_{i}'],metadata)} for i,r in enumerate(records)]
