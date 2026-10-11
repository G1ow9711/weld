"""Rebuild one point-only feature version for all admitted train/validation pieces."""
import argparse,gc,json,shutil,time
from pathlib import Path
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.robust_sequences import training_targets
from weld_pipeline.bounded_sequences import make_sequences
from weld_pipeline.surface_features import append_surface_features
from weld_pipeline.typed_contact_model import decode_typed
from weld_pipeline.truth import load_truth
from weld_pipeline.evaluation import score_welds
from weld_pipeline.mapped_cache import STORAGE_FORMAT,write_piece,load_piece

FEATURE_VERSION='typed_contact_surface_v3'
SOURCE_NAMES=['bounded_sequences.py','typed_sequences.py','surface_planes_local.py','surface_planes.py','point_frame.py','robust_sequences.py','surface_features.py','robust_surface_planes.py',
              'geometry_candidates.py','typed_contact_model.py','truth.py','evaluation.py','mapped_cache.py',
              'prepare_robust_dataset.py']
HOLD='N8C-FR174Q-0917-李欣'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent',default='weld_pipeline/data/surface_mapped_v1')
    ap.add_argument('--candidates',default='weld_pipeline/reports/expanded_training_candidates.json')
    ap.add_argument('--outdir',default='weld_pipeline/data/expanded_surface_v3')
    args=ap.parse_args();parent,out=Path(args.parent),Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    original=json.loads((parent/'split.json').read_text(encoding='utf-8'))
    candidates=json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    entries={}
    for name in original['train']+original['validation']:
        record=json.loads((parent/name/'manifest.json').read_text(encoding='utf-8'))['record']
        entries[name]={'name':name,'input':record['input'],'folder':str(Path(record['input']).parent),
                       'source_hashes':{record['input']:record['pcd_sha256'],**record['reference_hashes']}}
    for e in candidates['additional_train_candidates']:
        if e['name'] in entries:raise ValueError('Duplicate assembly')
        entries[e['name']]=e
    additional=[e['name'] for e in candidates['additional_train_candidates'] if e['name']!=HOLD]
    split={**original,'status':'FROZEN_EXPANDED_DEVELOPMENT_SPLIT','storage_format':STORAGE_FORMAT,
           'feature_version':FEATURE_VERSION,'train':original['train']+additional,
           'parent_split_sha256':sha256(parent/'split.json'),'candidate_audit_sha256':sha256(args.candidates),
           'label_alignment_review':[{ 'name':HOLD,'reference_welds':2,
               'reason':'Reference parallel weld pair is offset from the sampled upright plate faces. Dense baseplate proximity alone did not establish matching geometry. Held before training; no model scores used.',
               'evidence':'weld_pipeline/reports/expansion_gap_diagnostics.json'}],
           'scope':'20 real training pieces and unchanged 4 development validation pieces. Two previously reserved pieces remain untouched for this model; not blind acceptance.',
           'admission':'Source-label completeness, point-derived representation, and geometric alignment review before training.'}
    dest=out/'split.json'
    if dest.exists() and json.loads(dest.read_text(encoding='utf-8'))!=split:raise ValueError('Frozen split differs')
    dest.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[Path('weld_pipeline')/n for n in SOURCE_NAMES];hashes={str(p):sha256(p) for p in sources}
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in sources:
        target=snap/p.name
        if target.exists() and sha256(target)!=sha256(p):raise ValueError('Preprocessing changed; use a new output version')
        if not target.exists():shutil.copy2(p,target)
    audit=[]
    for name in split['train']+split['validation']:
        entry=entries[name];folder=out/name
        if (folder/'manifest.json').exists():
            piece=load_piece(folder);record=piece['record'];del piece
        else:
            pcd=Path(entry['input']);start=time.perf_counter()
            if sha256(pcd)!=entry['source_hashes'][str(pcd)]:raise ValueError('PCD changed')
            print('PREPARING',name,flush=True)
            xyz=read_xyz_pcd(pcd)*1000.;seq,geometry=make_sequences(xyz)
            seq,surface=append_surface_features(xyz,seq);seconds=time.perf_counter()-start
            for path,wanted in entry['source_hashes'].items():
                if sha256(path)!=wanted:raise ValueError('Reference changed: '+path)
            truth,missing=load_truth(entry['folder'])
            if missing:raise ValueError('Incomplete source curves')
            target,mapping=training_targets(seq,truth);oracle=score_welds(decode_typed(seq,target),truth)
            record={'name':name,'input':str(pcd),'pcd_sha256':sha256(pcd),'source_hashes':hashes,
                    'feature_version':FEATURE_VERSION,'feature_channels':84,'feature_seconds':seconds,
                    'features_before_truth_read':True,'geometry':geometry,'surface_features':surface,
                    'mapping':mapping,'missing_source_curves':missing,'representation_oracle':oracle,
                    'reference_hashes':{p:s for p,s in entry['source_hashes'].items() if p!=str(pcd)}}
            write_piece(folder,{'sequences':seq,'targets':target,'truth':truth,'record':record},
                        {'original_pcd_sha256':sha256(pcd),'source_hashes':hashes})
            del xyz,seq,target,truth;gc.collect()
        row={'name':name,'truth':record['mapping']['total_reference_welds'],
             'mapped':record['mapping']['mapped_to_sequences'],'oracle_valid':record['representation_oracle']['valid_welds'],
             'oracle_false':record['representation_oracle']['false_or_duplicate_predictions'],
             'feature_seconds':record['feature_seconds']}
        audit.append(row);print('PREPARED',json.dumps(row,ensure_ascii=False),flush=True)
        (out/'representation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    ready=all(r['truth']==r['mapped']==r['oracle_valid'] and r['oracle_false']==0 for r in audit)
    (out/'preprocessing_complete.json').write_text(json.dumps({'training_groups':len(split['train']),
        'validation_groups':len(split['validation']),'source_alignment_hold':[HOLD],
        'all_admitted_reference_curves_represented':ready,'status':'READY' if ready else 'REPRESENTATION_GAPS_REQUIRE_REVIEW',
        'reference_assisted_representation_check_not_model_accuracy':True},indent=2),encoding='utf-8')


if __name__=='__main__':main()
