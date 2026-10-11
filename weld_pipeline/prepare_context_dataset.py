"""Add reference-free point geometry context to the frozen development split."""
import argparse,gc,json,shutil,time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.mapped_cache import load_piece,write_piece
from weld_pipeline.frame_context_features import append_frame_context,FEATURE_VERSION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',default='weld_pipeline/data/expanded_surface_v3')
    ap.add_argument('--outdir',default='weld_pipeline/data/context_surface_v1');args=ap.parse_args()
    parent,out=Path(args.parent),Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    if json.loads((parent/'preprocessing_complete.json').read_text())['status']!='READY':raise ValueError('Parent not ready')
    original=json.loads((parent/'split.json').read_text(encoding='utf-8'))
    if original['feature_version']!='typed_contact_surface_v3':raise ValueError('Unexpected parent features')
    sources=[Path(__file__),*[Path('weld_pipeline')/n for n in ['frame_context_features.py','indexed_surface_features.py','local_height_features.py','mapped_cache.py']]]
    hashes={str(p):sha256(p) for p in sources};snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in sources:
        target=snapshot/p.name
        if target.exists() and sha256(target)!=sha256(p):raise ValueError('Use a new dataset version')
        shutil.copyfile(p,target)
    split={**original,'feature_version':FEATURE_VERSION,'feature_channels':102,
        'parent_split_sha256':sha256(parent/'split.json'),'context_sources':hashes}
    if (out/'split.json').exists() and json.loads((out/'split.json').read_text(encoding='utf-8'))!=split:raise ValueError('Frozen split differs')
    (out/'split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=[]
    for name in split['train']+split['validation']:
        dest=out/name;source=parent/original.get('piece_roots',{}).get(name,name)
        if (dest/'manifest.json').exists():
            saved=load_piece(dest);record=saved['record'];del saved
        else:
            p=load_piece(source);pcd=Path(p['record']['input'])
            if sha256(pcd)!=p['record']['pcd_sha256']:raise ValueError('Point source changed')
            start=time.perf_counter();xyz=read_xyz_pcd(pcd)*1000.
            seq,stats=append_frame_context(xyz,p['sequences'])
            for old,new in zip(p['sequences'],seq):
                if not np.array_equal(old['features'],new['features'][:,:84]):raise ValueError('Original features changed')
                if not np.array_equal(old['xyz'],new['xyz']):raise ValueError('Original coordinates changed')
                if not np.isfinite(new['features']).all():raise ValueError('Nonfinite context')
            record={**p['record'],'feature_version':FEATURE_VERSION,'feature_channels':102,
                    'context_sources':hashes,'context_statistics':stats,'context_seconds':time.perf_counter()-start,
                    'context_input_policy':'Only raw XYZ and original point-derived sequences passed to feature function; labels retained from verified parent.',
                    'base_features_and_coordinates_exact':True,'parent_manifest_sha256':sha256(source/'manifest.json')}
            write_piece(dest,{'sequences':seq,'targets':p['targets'],'truth':p['truth'],'record':record},
                {'parent_manifest_sha256':sha256(source/'manifest.json'),'original_pcd_sha256':record['pcd_sha256'],'context_sources':hashes})
            del p,seq,xyz;gc.collect()
        audit.append({'name':name,'context_seconds':record['context_seconds'],'base_exact':record['base_features_and_coordinates_exact'],
                       'truth':record['mapping']['total_reference_welds']})
        print('CONTEXT_PREPARED',json.dumps(audit[-1],ensure_ascii=False),flush=True)
        (out/'context_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'preprocessing_complete.json').write_text(json.dumps({'status':'READY','training_groups':len(split['train']),
        'validation_groups':len(split['validation']),'base_geometry_and_targets_unchanged':True,
        'scope':'Feature expansion only, not model accuracy; original representation oracle inherited unchanged.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
