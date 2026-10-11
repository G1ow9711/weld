"""Multiple FIT STEP-derived references, sealed unseen-reference diagnostics."""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.learned_plate_attributes import VERSION, features, at_positions, predict_samples
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_derived_plate_attributes import options


def load_records(root,split,names):
    rows=[];excluded=[];hashes={}
    for name in names:
        path=root/(name+'_attribute_candidates.json')
        if not path.exists():
            excluded.append({'piece':name,'reason':'NO_SUPPORTED_CAD_REFERENCE_POSE'});continue
        ref=json.loads(path.read_text('utf8'));hashes[str(path)]=sha256(path)
        piece=load_piece(split['piece_roots'][name],verify=True)
        mapping={w['id']:i for i,ws in mapped_references(piece).items() for w in ws}
        for r in ref['children']:
            if r['state']!='geometric_candidate_all_samples':
                excluded.append({'piece':name,'parent':r['weld_id'],'child':r['child_index'],'reason':r['state']});continue
            if r['weld_id'] not in mapping:
                excluded.append({'piece':name,'parent':r['weld_id'],'child':r['child_index'],'reason':'NO_POINT_CANDIDATE_MAPPING'});continue
            seq=piece['sequences'][mapping[r['weld_id']]]
            points=[s['xyz_mm'] for s in r['samples']]
            sample_rows=at_positions(seq,points)
            supported=True
            for row,sample in zip(sample_rows,r['samples']):
                normals=row[60:66].reshape(2,3).astype(float);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
                pair=sample['candidate_pairs'][0]
                if any(len([p for p in pair if abs(n@np.array(p['outward_normal_world']))>.999])!=1 for n in normals):
                    supported=False;break
            if not supported:
                excluded.append({'piece':name,'parent':r['weld_id'],'child':r['child_index'],'reason':'POINT_CONTACT_AXES_DISAGREE_WITH_CAD_REFERENCE'});continue
            rows.append({'piece':name,'reference':r,'sequence':seq,'rows':sample_rows})
    return rows,excluded,hashes


def evaluate(bundle,records,role):
    evaluated=[]
    for record in records:
        ref,seq=record['reference'],record['sequence']
        points=np.array([s['xyz_mm'] for s in ref['samples']]);tangent=points[-1]-points[0]
        tangent/=np.linalg.norm(tangent);pred=predict_samples(bundle,seq,points,tangent)
        errors=[]
        for got,wanted in zip(pred,ref['samples']):
            e=[]
            if got['state']=='estimated_candidate':
                for i,p in enumerate(wanted['candidate_pairs'][0]):
                    dt=abs(got['thickness_mm'][i]-p['thickness_candidate_mm'])
                    angle=float(np.degrees(np.arccos(np.clip(np.array(got['material_outward_normals'][i])@np.array(p['outward_normal_world']),-1.,1.))))
                    e.append({'thickness_error_mm':dt,'normal_angle_deg':angle,'thickness_pass':dt<=1.,'normal_pass':angle<=3.})
            errors.append(e)
        flags={field+'_'+str(i+1):all(len(e)==2 and e[i][field+'_pass'] for e in errors)
               for field in ['thickness','normal'] for i in [0,1]}
        evaluated.append({'piece':record['piece'],'parent':ref['weld_id'],'child':ref['child_index'],
                          'candidate_reference_flags':flags,'predictions':pred,'errors':errors})
    keys=['thickness_1','thickness_2','normal_1','normal_2']
    counts={k:sum(r['candidate_reference_flags'][k] for r in evaluated) for k in keys}
    return {'role':role,'comparable_candidate_children':len(records),'candidate_reference_counts':counts,
        'results':evaluated,'reviewed_customer_truth':False,'geometry_misses_included':False,'full_contract_pass':False}


def main():
    root=Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')
    split_path=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(split_path.read_text('utf8'))
    if not (root/'complete.json').exists():
        raise ValueError('Reference audit unfinished')
    out=Path('weld_pipeline/runs/expanded_plate_attributes_seed132');out.mkdir(parents=True,exist_ok=True)
    if (out/'heads.joblib').exists():
        raise ValueError('Existing expanded attribute run')
    train_names=[n for n in split['train'] if '__' not in n]
    records,excluded,reference_hashes=load_records(root,split,train_names)
    x,thickness,sign,groups=[],[],[],[]
    for r in records:
        for row,sample in zip(r['rows'],r['reference']['samples']):
            pair=sample['candidate_pairs'][0]
            for transformed in options(row.astype(np.float32),r['reference']['type']):
                normals=transformed[60:66].reshape(2,3).astype(float)
                normals/=np.linalg.norm(normals,axis=1,keepdims=True)
                labels_t,labels_s=[],[]
                for n in normals:
                    match=[p for p in pair if abs(n@np.array(p['outward_normal_world']))>.999]
                    if len(match)!=1:
                        raise ValueError('Augmented normal reference identity mismatch')
                    target=match[0];labels_t.append(target['thickness_candidate_mm'])
                    labels_s.append(int(n@np.array(target['outward_normal_world'])>0))
                x.append(features(transformed));thickness.append(labels_t);sign.append(labels_s);groups.append(r['piece'])
    counts={n:groups.count(n) for n in set(groups)}
    weights=np.array([1./counts[n] for n in groups])
    sources={p:sha256(p) for p in [__file__,'weld_pipeline/learned_plate_attributes.py',
        'weld_pipeline/train_derived_plate_attributes.py','weld_pipeline/sequence_augmentation.py',
        'weld_pipeline/candidate_frame_tree_model.py','weld_pipeline/vertical_frame_augmentation.py']}
    config={'version':VERSION,'seed':132,'trees':256,'sources':sources,'reference_hashes':reference_hashes,
        'split_sha256':sha256(split_path),'children':len(records),'augmented_rows':len(x),'train_pieces':sorted(counts),
        'unseen_check_pieces':split['validation'],'unseen_selection_or_training_used':False,'confidence_threshold':.8,
        'role':'FIT_ON_UNREVIEWED_CAD_REFERENCE_CANDIDATES','full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'reference_exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for p in sources:
        shutil.copy2(p,archive/Path(p).name)
    start=time.perf_counter();regressor=ExtraTreesRegressor(n_estimators=256,max_features=1.,n_jobs=4,random_state=132)
    classifier=ExtraTreesClassifier(n_estimators=256,max_features=1.,n_jobs=4,random_state=132)
    regressor.fit(np.stack(x),np.array(thickness),sample_weight=weights)
    classifier.fit(np.stack(x),np.array(sign),sample_weight=weights)
    bundle={'thickness':regressor,'material_sign':classifier,'config':config};joblib.dump(bundle,out/'heads.joblib')
    print('EXPANDED_ATTRIBUTE_WEIGHTS',len(records),len(x),len(counts),flush=True)
    fit=evaluate(bundle,records,'FIT_ON_DERIVED_REFERENCES')
    (out/'fit_reference_report.json').write_text(json.dumps(fit,ensure_ascii=False,indent=2),encoding='utf8')
    # Unseen references are first read only after fitting and all settings are fixed.
    unseen_records,unseen_excluded,unseen_hashes=load_records(root,split,split['validation'])
    unseen=evaluate(bundle,unseen_records,'UNTRAINED_REFERENCE_DIAGNOSTIC_NOT_E2')
    unseen.update(reference_hashes=unseen_hashes,excluded=unseen_excluded)
    (out/'unseen_reference_report.json').write_text(json.dumps(unseen,ensure_ascii=False,indent=2),encoding='utf8')
    report={'head_sha256':sha256(out/'heads.joblib'),'elapsed_seconds':time.perf_counter()-start,
        'fit_candidates':fit['comparable_candidate_children'],'fit_candidate_counts':fit['candidate_reference_counts'],
        'unseen_candidates':unseen['comparable_candidate_children'],'unseen_candidate_counts':unseen['candidate_reference_counts'],
        'reviewed_customer_truth':False,'full_contract_pass':False}
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('EXPANDED_ATTRIBUTE_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
