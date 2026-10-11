"""Hash-bound physical context over unchanged candidate arrays and labels."""
import json
import os
from pathlib import Path
import numpy as np
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece as load_base

STORAGE_FORMAT='axial_context_overlay_v1'
FEATURE_VERSION='typed_contact_axial_context_v1'
HALO=256


def write_piece(folder,base_folder,sequences,provenance):
    folder,base_folder=Path(folder).resolve(),Path(base_folder).resolve()
    base=load_base(base_folder)
    if len(base['sequences'])!=len(sequences):raise ValueError('Sequence count changed')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['provenance']!=provenance or manifest['base_manifest_sha256']!=sha256(base_folder/'manifest.json'):
            raise ValueError('Existing context build differs')
        load_piece(folder);return manifest
    values=[];indices=[]
    for i,(old,new) in enumerate(zip(base['sequences'],sequences)):
        if old.keys()!=new.keys()-{'left_context','right_context'}:raise ValueError('Core metadata keys changed')
        for key in old:
            if not np.array_equal(old[key],new[key]):raise ValueError('Core candidate changed: '+key)
        if old['type']=='vertical':
            item=np.stack([new['left_context'],new['right_context']])
            if item.shape!=(2,HALO,84) or item.dtype!=np.float32 or not np.isfinite(item).all():
                raise ValueError('Invalid axial context')
            if not (item[:,:,66]>.5).all():raise ValueError('Context type changed')
            values.append(item);indices.append(i)
        elif 'left_context' in new or 'right_context' in new:raise ValueError('Horizontal context not allowed')
    array=np.stack(values) if values else np.empty((0,2,HALO,84),np.float32)
    np.save(folder/'axial_context.npy',array,allow_pickle=False)
    manifest={'storage_format':STORAGE_FORMAT,'feature_version':FEATURE_VERSION,'halo':HALO,
        'base_folder':os.path.relpath(base_folder,folder),'base_manifest_sha256':sha256(base_folder/'manifest.json'),
        'vertical_indices':indices,'context_shape':list(array.shape),'context_sha256':sha256(folder/'axial_context.npy'),
        'provenance':provenance,'label_policy':'Original arrays, unknown masks and output extent unchanged.'}
    pending=folder/'manifest.pending.json';pending.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    pending.replace(folder/'manifest.json');load_piece(folder);return manifest


def load_piece(folder,verify=True):
    folder=Path(folder).resolve();m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if m['storage_format']!=STORAGE_FORMAT or m['feature_version']!=FEATURE_VERSION or m['halo']!=HALO:
        raise ValueError('Unsupported axial cache')
    parent=(folder/m['base_folder']).resolve()
    if not parent.is_relative_to(Path.cwd().resolve()):raise ValueError('Base outside project')
    if sha256(parent/'manifest.json')!=m['base_manifest_sha256']:raise ValueError('Base cache changed')
    base=load_base(parent,verify=verify);path=folder/'axial_context.npy'
    if verify and sha256(path)!=m['context_sha256']:raise ValueError('Context changed')
    values=np.load(path,mmap_mode='r' if m['context_shape'][0] else None,allow_pickle=False)
    wanted=[i for i,s in enumerate(base['sequences']) if s['type']=='vertical']
    if wanted!=m['vertical_indices'] or values.shape!=(len(wanted),2,HALO,84) or list(values.shape)!=m['context_shape'] or values.dtype!=np.float32:
        raise ValueError('Context shape or coverage changed')
    if not np.isfinite(values).all() or (len(values) and not (values[:,:,:,66]>.5).all()):raise ValueError('Invalid context values')
    sequences=[dict(s) for s in base['sequences']]
    for index,value in zip(wanted,values):sequences[index].update(left_context=value[0],right_context=value[1])
    return {**base,'sequences':sequences,'record':{**base['record'],'feature_version':FEATURE_VERSION},
            'cache_manifest_sha256':sha256(folder/'manifest.json')}
