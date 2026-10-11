"""Read-only, hash-verified sequence arrays that do not preload whole datasets."""
import gc
import json
from pathlib import Path
import numpy as np
from training.labels import sha256

STORAGE_FORMAT = 'sequence_memmap_v1'


def compact_truth(truth):
    output=[]
    for weld in truth:
        segments=[]
        for segment in weld['segments']:
            points=segment['points']
            if segment['kind']=='line' and len(points)>2:
                p=np.asarray(points,float);direction=p[-1]-p[0]
                length=float(np.linalg.norm(direction))
                if length>0 and np.linalg.norm(np.cross(p-p[0],direction/length),axis=1).max()<1e-5:
                    points=[points[0],points[-1]]
            segments.append({**segment,'points':points})
        output.append({**weld,'segments':segments})
    return output


def write_piece(folder,piece,provenance):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['provenance']!=provenance:raise ValueError('Existing cache provenance differs: '+str(folder))
        load_piece(folder,verify=True)
        return existing
    sequences=piece['sequences'];targets=piece['targets']
    if not sequences or len(sequences)!=len(targets):raise ValueError('Empty or inconsistent sequence set')
    size=sum(len(s['xyz']) for s in sequences);channels=sequences[0]['features'].shape[1]
    shape={'features':(size,channels),'xyz':(size,3),'targets':(size,7)}
    arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=v) for k,v in shape.items()}
    entries=[];start=0
    for seq,target in zip(sequences,targets):
        stop=start+len(seq['xyz'])
        if len(target)!=stop-start:raise ValueError('Target length differs')
        source={'features':seq['features'],'xyz':seq['xyz'],'targets':target}
        for name,value in source.items():
            if value.dtype!=np.float32:raise ValueError('Cache conversion must preserve float32 values exactly')
            arrays[name][start:stop]=value
            if not np.array_equal(arrays[name][start:stop],value):raise ValueError('Array conversion differs: '+name)
        entries.append({'start':start,'stop':stop,'type':seq['type'],
                        'direction':np.asarray(seq['direction']).tolist(),'sample_step_mm':seq['sample_step_mm']})
        start=stop
    for array in arrays.values():array.flush()
    del array;arrays.clear();gc.collect()
    result={'storage_format':STORAGE_FORMAT,'provenance':provenance,
            'files':{k+'.npy':{'sha256':sha256(folder/(k+'.npy')),'shape':list(v),'dtype':'float32'} for k,v in shape.items()},
            'sequences':entries,'truth':compact_truth(piece['truth']),'record':piece['record'],
            'array_content_check':'Every feature, coordinate and target compared exactly before manifest commit'}
    temporary=folder/'manifest.pending.json'
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(manifest_path)
    return result


def load_piece(folder,verify=True):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['storage_format']!=STORAGE_FORMAT:raise ValueError('Unknown mapped cache version')
    arrays={}
    for name,record in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Cache paths must be local filenames')
        path=folder/name
        if verify and sha256(path)!=record['sha256']:raise ValueError('Cache hash differs: '+str(path))
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(array.shape)!=record['shape'] or str(array.dtype)!=record['dtype']:raise ValueError('Cache shape/dtype differs')
        arrays[Path(name).stem]=array
    sequences=[];targets=[];previous=0
    for entry in manifest['sequences']:
        start,stop=entry['start'],entry['stop']
        if start!=previous or stop<=start:raise ValueError('Noncontiguous sequence bounds')
        sequences.append({'xyz':arrays['xyz'][start:stop],'features':arrays['features'][start:stop],
                          'direction':np.array(entry['direction'],float),'type':entry['type'],
                          'sample_step_mm':entry['sample_step_mm']})
        targets.append(arrays['targets'][start:stop]);previous=stop
    if previous!=len(arrays['features']):raise ValueError('Sequence bounds omit data')
    return {'sequences':sequences,'targets':targets,'truth':manifest['truth'],'record':manifest['record'],
            'cache_manifest_sha256':sha256(folder/'manifest.json')}
