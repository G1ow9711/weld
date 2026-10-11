"""Add audited real pieces without changing the four development validation pieces."""
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.typed_sequences import make_sequences,training_targets
from weld_pipeline.surface_features import append_surface_features,FEATURE_VERSION
from weld_pipeline.typed_contact_model import decode_typed
from weld_pipeline.truth import load_truth
from weld_pipeline.evaluation import score_welds
from weld_pipeline.mapped_cache import STORAGE_FORMAT,write_piece,load_piece

SOURCE_NAMES=['typed_sequences.py','surface_features.py','surface_planes_local.py','surface_planes.py',
              'geometry_candidates.py','typed_contact_model.py','truth.py','evaluation.py','mapped_cache.py','prepare_expanded_pilot.py']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent',default='weld_pipeline/data/surface_mapped_v1')
    ap.add_argument('--candidates',default='weld_pipeline/reports/expanded_training_candidates.json')
    ap.add_argument('--outdir',default='weld_pipeline/data/expanded_surface_v1')
    ap.add_argument('--groups',nargs='+')
    args=ap.parse_args();parent,out=Path(args.parent),Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    candidates=json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    original=json.loads((parent/'split.json').read_text(encoding='utf-8'))
    names=[r['name'] for r in candidates['additional_train_candidates']]
    if set(names)&set(original['train']+original['validation']+original['reserved']):raise ValueError('Split overlap')
    if original['train']!=candidates['current_train'] or original['validation']!=candidates['validation_unchanged']:raise ValueError('Parent split changed')
    split={**original,'status':'FROZEN_EXPANDED_DEVELOPMENT_SPLIT','storage_format':STORAGE_FORMAT,
           'feature_version':FEATURE_VERSION,'train':original['train']+names,
           'parent_split_sha256':sha256(parent/'split.json'),'candidate_audit_sha256':sha256(args.candidates),
           'piece_roots':{n:os.path.relpath(parent/n,out) for n in original['train']+original['validation']},
           'scope':'21 real training pieces and unchanged 4 development validation pieces. Two remain reserved for this model; not a new blind set.',
           'admission':'source-label completeness and point-cloud coverage; no selection by model score'}
    split_path=out/'split.json'
    if split_path.exists() and json.loads(split_path.read_text(encoding='utf-8'))!=split:raise ValueError('Frozen expanded split differs')
    split_path.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[Path('weld_pipeline')/n for n in SOURCE_NAMES];hashes={str(p):sha256(p) for p in sources}
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in sources:
        target=snapshot/p.name
        if target.exists() and sha256(target)!=sha256(p):raise ValueError('Preprocessing source changed; use a new version')
        if not target.exists():shutil.copy2(p,target)
    audit=[]
    for entry in candidates['additional_train_candidates']:
        name=entry['name']
        if args.groups and name not in args.groups:continue
        folder=out/name
        if not folder.resolve().is_relative_to(out.resolve()):raise ValueError('Invalid group path')
        if (folder/'manifest.json').exists():
            piece=load_piece(folder,verify=True);record=piece['record'];del piece
            print('EXISTING',name,flush=True)
        else:
            pcd=Path(entry['input']);start=time.perf_counter()
            print('PREPARING',name,'raw_points',entry['point_count'],flush=True)
            if sha256(pcd)!=entry['source_hashes'][str(pcd)]:raise ValueError('Audited PCD changed')
            xyz=read_xyz_pcd(pcd)*1000.
            sequences,geometry=make_sequences(xyz)
            sequences,surface=append_surface_features(xyz,sequences)
            feature_seconds=time.perf_counter()-start
            # Only original point-derived values participate before this line.
            for path,wanted in entry['source_hashes'].items():
                if sha256(path)!=wanted:raise ValueError('Audited reference changed: '+path)
            truth,missing=load_truth(entry['folder'])
            if missing or len(truth)!=entry['declared_welds']:raise ValueError('Audited truth completeness changed')
            targets,mapping=training_targets(sequences,truth)
            oracle=score_welds(decode_typed(sequences,targets),truth)
            record={'name':name,'input':str(pcd),'pcd_sha256':sha256(pcd),'source_hashes':hashes,
                    'feature_version':FEATURE_VERSION,'feature_channels':84,'feature_seconds':feature_seconds,
                    'features_before_truth_read':True,'geometry':geometry,'surface_features':surface,
                    'mapping':mapping,'missing_source_curves':missing,'representation_oracle':oracle,
                    'reference_hashes':{p:s for p,s in entry['source_hashes'].items() if p!=str(pcd)}}
            (out/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            if sequences:
                write_piece(folder,{'sequences':sequences,'targets':targets,'truth':truth,'record':record},
                            {'original_pcd_sha256':record['pcd_sha256'],'source_hashes':hashes})
            else:print('NO_SEQUENCES',name,flush=True)
            del xyz,sequences,targets,truth;gc.collect()
        summary={'name':name,'truth':record['mapping']['total_reference_welds'],
                 'mapped':record['mapping']['mapped_to_sequences'],'oracle_valid':record['representation_oracle']['valid_welds'],
                 'oracle_false':record['representation_oracle']['false_or_duplicate_predictions'],
                 'feature_seconds':record['feature_seconds']}
        audit.append(summary)
        print('PREPARED',json.dumps(summary,ensure_ascii=False),flush=True)
        (out/'additional_representation_audit.partial.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if all((out/n/'manifest.json').exists() for n in names):
        records=[json.loads((out/n/'manifest.json').read_text(encoding='utf-8'))['record'] for n in names]
        complete=all(r['representation_oracle']['valid_welds']==r['mapping']['total_reference_welds'] and r['representation_oracle']['false_or_duplicate_predictions']==0 for r in records)
        (out/'preprocessing_complete.json').write_text(json.dumps({'additional_groups':len(names),'training_groups':len(split['train']),
            'all_additional_reference_curves_represented':complete,'status':'READY' if complete else 'REPRESENTATION_GAPS_REQUIRE_REVIEW',
            'reference_assisted_representation_check_not_model_accuracy':True},indent=2),encoding='utf-8')


if __name__=='__main__':main()
