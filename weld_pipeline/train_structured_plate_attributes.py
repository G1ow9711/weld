"""Fixed TRAIN-driven correction of endpoint shortcuts in thickness prediction."""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.structured_plate_attributes import VERSION,thickness_features,predict_samples
from weld_pipeline.train_expanded_plate_attributes import load_records
from weld_pipeline.train_derived_plate_attributes import options


def evaluate(bundle,records,role):
    counts={k:0 for k in ['thickness_1','thickness_2','normal_1','normal_2']};results=[]
    for r in records:
        ref=r['reference'];points=np.array([s['xyz_mm'] for s in ref['samples']]);u=points[-1]-points[0];u/=np.linalg.norm(u)
        got=predict_samples(bundle,r['sequence'],points,u);errors=[]
        for observed,wanted in zip(got,ref['samples']):
            pair=wanted['candidate_pairs'][0];e={}
            if observed['state']=='estimated_candidate':
                for i in [0,1]:
                    e['thickness_'+str(i+1)]=abs(observed['thickness_mm'][i]-pair[i]['thickness_candidate_mm'])
                    e['normal_'+str(i+1)]=float(np.degrees(np.arccos(np.clip(np.array(observed['material_outward_normals'][i])@np.array(pair[i]['outward_normal_world']),-1.,1.))))
            errors.append(e)
        flags={k:all(k in e and e[k]<=(1. if k.startswith('thickness') else 3.) for e in errors) for k in counts}
        for k in counts:
            counts[k]+=flags[k]
        results.append({'piece':r['piece'],'parent':ref['weld_id'],'child':ref['child_index'],'flags':flags,'errors':errors})
    return {'role':role,'children':len(records),'counts':counts,'results':results,'full_contract_pass':False,'reviewed_customer_truth':False}


def attach_structure(records,split):
    cache={};indices={}
    def key(seq):
        axes=seq['features'][len(seq['features'])//2,60:66]
        return (seq['type'],tuple(seq['xyz'][[0,-1]].ravel()),tuple(axes))
    for r in records:
        name=r['piece']
        if name not in cache:
            path=Path(split['structure_roots'][name])/(name+'.npz')
            with np.load(path) as values:
                cache[name]=values['structural'].copy()
            from weld_pipeline.curved_cache import load_piece
            p=load_piece(split['piece_roots'][name],verify=True)
            mapping={}
            for i,s in enumerate(p['sequences']):
                signature=key(s)
                if signature in mapping and not np.array_equal(cache[name][i],cache[name][mapping[signature]]):
                    raise ValueError('Duplicate physical frame has different plate context')
                mapping[signature]=i
            indices[name]=mapping
        seq=r['sequence'];index=indices[name][key(seq)]
        r['sequence']={**seq,'plate_context':cache[name][index]}


def main():
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    root=Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')
    out=Path('weld_pipeline/runs/structured_plate_attributes_seed134');out.mkdir(parents=True,exist_ok=True)
    if (out/'heads.joblib').exists():
        raise ValueError('Frozen structure attribute run')
    records,excluded,hashes=load_records(root,split,[n for n in split['train'] if '__' not in n])
    attach_structure(records,split)
    x,y,groups=[],[],[]
    for r in records:
        for row,sample in zip(r['rows'],r['reference']['samples']):
            pair=sample['candidate_pairs'][0]
            for transformed in options(row.astype(np.float32),r['reference']['type']):
                normals=transformed[60:66].reshape(2,3).astype(float);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
                targets=[]
                for n in normals:
                    matches=[p for p in pair if abs(n@np.array(p['outward_normal_world']))>.999]
                    if len(matches)!=1:
                        raise ValueError('Reference board order mismatch')
                    targets.append(matches[0]['thickness_candidate_mm'])
                x.append(thickness_features(transformed,r['sequence']['plate_context']));y.append(targets);groups.append(r['piece'])
    n={g:groups.count(g) for g in set(groups)};weights=np.array([1./n[g] for g in groups])
    parent=Path('weld_pipeline/runs/expanded_plate_attributes_seed132/heads.joblib')
    old=joblib.load(parent)
    sources={p:sha256(p) for p in [__file__,'weld_pipeline/structured_plate_attributes.py',
        'weld_pipeline/point_plate_context.py','weld_pipeline/train_expanded_plate_attributes.py',
        'weld_pipeline/train_derived_plate_attributes.py','weld_pipeline/learned_plate_attributes.py']}
    config={'seed':134,'version':VERSION,'trees':256,'sources':sources,'reference_hashes':hashes,
        'children':len(records),'augmented_rows':len(x),'material_direction_head_frozen_sha256':sha256(parent),
        'regression_features':'63 local point-distance channels, 50 canonical plate support/parallel-face features, 3 gravity features; no endpoint distances or world position.',
        'train_pieces':sorted(n),'confidence_threshold':.8,'full_contract_pass':False,'reviewed_customer_truth':False,
        'unseen_training_or_selection_used':False}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for p in sources:
        shutil.copy2(p,archive/Path(p).name)
    start=time.perf_counter();reg=ExtraTreesRegressor(n_estimators=256,max_features=1.,n_jobs=4,random_state=134)
    reg.fit(np.stack(x),np.array(y),sample_weight=weights)
    bundle={'thickness':reg,'material_sign':old['material_sign'],'config':config}
    joblib.dump(bundle,out/'heads.joblib');print('STRUCTURED_THICKNESS_TRAINED',len(records),len(x),flush=True)
    fit=evaluate(bundle,records,'FIT_ON_UNREVIEWED_CAD_CANDIDATES')
    (out/'fit_reference_report.json').write_text(json.dumps(fit,ensure_ascii=False,indent=2),encoding='utf8')
    unseen,missing,check_hashes=load_records(root,split,split['validation']);attach_structure(unseen,split)
    check=evaluate(bundle,unseen,'UNTRAINED_REFERENCE_DIAGNOSTIC_NOT_E2')
    (out/'unseen_reference_report.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf8')
    report={'fit_candidates':fit['children'],'fit_candidate_counts':fit['counts'],'unseen_candidates':check['children'],
        'unseen_candidate_counts':check['counts'],'elapsed_seconds':time.perf_counter()-start,'head_sha256':sha256(out/'heads.joblib'),
        'full_contract_pass':False,'reviewed_customer_truth':False}
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('STRUCTURED_ATTRIBUTE_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
