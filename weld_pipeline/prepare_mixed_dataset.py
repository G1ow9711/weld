"""Extend the frozen 20/4 development split with a complete arc training piece."""
import argparse,gc,json,shutil,time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.mapped_cache import load_piece as load_parent
from weld_pipeline.curved_cache import write_piece,load_piece,STORAGE_FORMAT
from weld_pipeline.mixed_sequences import append_curved_contacts,make_mixed_sequences,mixed_training_targets,FEATURE_VERSION
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.truth import load_truth
from weld_pipeline.evaluation import score_welds

SOURCES=['prepare_mixed_dataset.py','mixed_sequences.py','curved_sequences.py','arc_tangent_sequences.py',
         'arc_surface_candidates.py','local_tangent_circles.py','tangent_circle_candidates.py','circle_candidates.py',
         'point_frame.py','curved_cache.py','mapped_cache.py','bounded_sequences.py','robust_sequences.py',
         'robust_surface_planes.py','surface_features.py','typed_sequences.py','surface_planes_local.py',
         'surface_planes.py','geometry_candidates.py','typed_contact_model.py','truth.py','evaluation.py',
         'customer_label_corrections.py','process_label_semantics.py']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',default='weld_pipeline/data/expanded_surface_v3')
    ap.add_argument('--outdir',default='weld_pipeline/data/mixed_surface_v1');args=ap.parse_args()
    parent,out=Path(args.parent),Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    if json.loads((parent/'preprocessing_complete.json').read_text())['status']!='READY':raise ValueError('Parent not ready')
    original=json.loads((parent/'split.json').read_text(encoding='utf-8'))
    split={**original,'train':original['train']+['TD2-LB8A'],'feature_version':FEATURE_VERSION,
           'storage_format':STORAGE_FORMAT,'parent_split_sha256':sha256(parent/'split.json'),
           'scope':'21 real training pieces, unchanged 4 development validation pieces and reserved IDs. Not blind acceptance.',
           'type_coverage':'One supplemental arc training piece; independent arc test data still absent.'}
    split_path=out/'split.json'
    if split_path.exists() and json.loads(split_path.read_text(encoding='utf-8'))!=split:raise ValueError('Frozen split changed')
    split_path.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[Path('weld_pipeline')/name for name in SOURCES];hashes={str(p):sha256(p) for p in sources}
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for source in sources:
        dest=snap/source.name
        if dest.exists() and sha256(dest)!=sha256(source):raise ValueError('Preparation source changed; use a new data version')
        shutil.copy2(source,dest)
    audit=[]
    # Check the newly admitted piece first, before writing the other 24 caches.
    order=['TD2-LB8A']+original['train']+original['validation']
    for name in order:
        dest=out/name
        if (dest/'manifest.json').exists():
            piece=load_piece(dest);record=piece['record'];del piece
        else:
            print('PREPARING_MIXED',name,flush=True);start=time.perf_counter();parent_hash=None
            if name=='TD2-LB8A':
                folder=Path('data/supplemental/TD2-LB8A');pcd=next(folder.glob('*AllFaces.pcd'))
                xyz=read_xyz_pcd(pcd)*1000;sequences,stats=make_mixed_sequences(xyz)
                truth,missing=load_truth(folder)
                if missing:raise ValueError('Missing reference curves')
                reference_hashes={str(p):sha256(p) for p in [*folder.glob('*.xlsx'),*folder.glob('*welds.json')]}
                base_exact=True
            else:
                source=parent/original.get('piece_roots',{}).get(name,name);base=load_parent(source)
                pcd=Path(base['record']['input'])
                if sha256(pcd)!=base['record']['pcd_sha256']:raise ValueError('Raw PCD changed')
                xyz=read_xyz_pcd(pcd)*1000;sequences,stats=append_curved_contacts(xyz,base['sequences'])
                truth=base['truth'];reference_hashes=base['record']['reference_hashes'];parent_hash=sha256(source/'manifest.json')
                base_exact=all(np.array_equal(a['features'],b['features']) and np.array_equal(a['xyz'],b['xyz'])
                               for a,b in zip(base['sequences'],sequences))
                if not base_exact:raise ValueError('Parent geometry changed')
                for path,h in reference_hashes.items():
                    if sha256(path)!=h:raise ValueError('Reference changed')
            if any(not np.isfinite(s['features']).all() for s in sequences):raise ValueError('Nonfinite features')
            targets,mapping=mixed_training_targets(sequences,truth)
            oracle=score_welds(decode_curved(sequences,targets),truth)
            record={'name':name,'input':str(pcd),'pcd_sha256':sha256(pcd),'reference_hashes':reference_hashes,
                    'feature_version':FEATURE_VERSION,'feature_channels':84,'mapping':mapping,'representation_oracle':oracle,
                    'source_hashes':hashes,'geometry':stats,'parent_geometry_exact':base_exact,
                    'feature_seconds':time.perf_counter()-start}
            write_piece(dest,{'sequences':sequences,'targets':targets,'truth':truth,'record':record},
                        {'parent_manifest_sha256':parent_hash,'source_hashes':hashes,'pcd_sha256':record['pcd_sha256']})
            del xyz,sequences,targets,truth
            if name!='TD2-LB8A':del base
            gc.collect()
        row={'name':name,'reference':record['mapping']['total_reference_welds'],
             'mapped':record['mapping']['mapped_to_sequences'],'oracle_valid':record['representation_oracle']['valid_welds'],
             'oracle_false':record['representation_oracle']['false_or_duplicate_predictions'],'seconds':record['feature_seconds']}
        audit.append(row);print('MIXED_PREPARED',json.dumps(row,ensure_ascii=False),flush=True)
        (out/'representation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    ready=all(r['mapped']==r['reference']==r['oracle_valid'] and r['oracle_false']==0 for r in audit)
    (out/'preprocessing_complete.json').write_text(json.dumps({'status':'READY' if ready else 'REPRESENTATION_GAPS',
        'training_groups':len(split['train']),'validation_groups':len(split['validation']),
        'scope':'Representation check only, not model performance.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
