"""Train on traceable CAD candidates; never relabel them as reviewed truth."""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.candidate_frame_tree_model import frames
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.learned_plate_attributes import VERSION,features,at_positions,predict_samples
from weld_pipeline.sequence_augmentation import change_frame
from weld_pipeline.train_candidate_tree import mapped_references


def options(row,typ):
    seq={'features':row[None,:]}
    if typ=='vertical':
        return [s['features'][0] for s,reverse in frames(seq)]
    output=[]
    for reverse in [False,True]:
        for flip in [False,True]:
            x=row[:,None].copy();target=np.zeros((7,1),np.float32);valid=np.ones((1,1),np.float32)
            change_frame(x,target,valid,reverse=reverse,flip_normal=flip,flip_vertical_cross=False)
            output.append(x[:,0])
    return output


def main():
    root=Path('weld_pipeline/reports/step_alignment_20260930')
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    out=Path('weld_pipeline/runs/derived_plate_attributes_seed130')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'heads.joblib').exists():
        raise ValueError('Existing attribute training')
    x,thickness,sign,groups,records=[],[],[],[],[]
    reference_hashes={}
    for name in ['224-GR24A1','527-FR146A']:
        if name not in split['train']:
            raise ValueError('Attribute source is not FIT TRAIN')
        file=root/(name+'_attribute_candidates.json')
        ref=json.loads(file.read_text('utf8'));reference_hashes[str(file)]=sha256(file)
        piece=load_piece(split['piece_roots'][name],verify=True)
        mapping={w['id']:i for i,ws in mapped_references(piece).items() for w in ws}
        for r in ref['children']:
            if r['state']!='geometric_candidate_all_samples':
                continue
            seq=piece['sequences'][mapping[r['weld_id']]]
            points=[s['xyz_mm'] for s in r['samples']]
            sample_rows=at_positions(seq,points)
            record={'piece':name,'reference':r,'sequence':seq,'rows':sample_rows}
            records.append(record)
            for row,sample in zip(sample_rows,r['samples']):
                pair=sample['candidate_pairs'][0]
                for transformed in options(row.astype(np.float32),r['type']):
                    normals=transformed[60:66].reshape(2,3).astype(float)
                    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
                    labels_t,labels_s=[],[]
                    for n in normals:
                        match=[p for p in pair if abs(n@np.array(p['outward_normal_world']))>.999]
                        if len(match)!=1:
                            raise ValueError('CAD contact does not match point-derived contact frame')
                        target=match[0]
                        labels_t.append(target['thickness_candidate_mm'])
                        labels_s.append(int(n@np.array(target['outward_normal_world'])>0))
                    x.append(features(transformed));thickness.append(labels_t);sign.append(labels_s);groups.append(name)
    unique={n:groups.count(n) for n in set(groups)}
    weights=np.array([1./unique[n] for n in groups])
    paths=[__file__,'weld_pipeline/learned_plate_attributes.py','weld_pipeline/sequence_augmentation.py',
           'weld_pipeline/candidate_frame_tree_model.py','weld_pipeline/vertical_frame_augmentation.py']
    config={'version':VERSION,'seed':130,'trees':256,'sources':{p:sha256(p) for p in paths},
            'reference_hashes':reference_hashes,'children':len(records),'augmented_rows':len(x),
            'role':'FIT_ON_CAD_DERIVED_CANDIDATES_NOT_CUSTOMER_REVIEWED_TRUTH','confidence_threshold':.8,
            'features':'63 point-distance channels, two point-derived candidate endpoint distances, tangent/normal gravity components. No XYZ/names/STEP/runtime reference inputs.',
            'train_pieces':sorted(unique),'independent_attribute_validation':False,'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for p in paths:
        shutil.copy2(p,archive/Path(p).name)
    start=time.perf_counter()
    regressor=ExtraTreesRegressor(n_estimators=256,max_features=1.,n_jobs=4,random_state=130)
    classifier=ExtraTreesClassifier(n_estimators=256,max_features=1.,n_jobs=4,random_state=130)
    regressor.fit(np.stack(x),np.array(thickness),sample_weight=weights)
    classifier.fit(np.stack(x),np.array(sign),sample_weight=weights)
    bundle={'thickness':regressor,'material_sign':classifier,'config':config}
    joblib.dump(bundle,out/'heads.joblib')
    evaluated=[]
    for row in records:
        ref,seq=row['reference'],row['sequence']
        points=np.array([s['xyz_mm'] for s in ref['samples']])
        tangent=points[-1]-points[0];tangent/=np.linalg.norm(tangent)
        prediction=predict_samples(bundle,seq,points,tangent)
        errors=[]
        for got,wanted in zip(prediction,ref['samples']):
            e=[]
            if got['state']=='estimated_candidate':
                for i,p in enumerate(wanted['candidate_pairs'][0]):
                    dt=abs(got['thickness_mm'][i]-p['thickness_candidate_mm'])
                    angle=float(np.degrees(np.arccos(np.clip(np.array(got['material_outward_normals'][i])@np.array(p['outward_normal_world']),-1.,1.))))
                    e.append({'thickness_error_mm':dt,'normal_angle_deg':angle,'thickness_pass':dt<=1.,'normal_pass':angle<=3.})
            errors.append(e)
        flags={field+'_'+str(i+1):all(len(e)==2 and e[i][field+'_pass'] for e in errors) for field in ['thickness','normal'] for i in [0,1]}
        evaluated.append({'piece':row['piece'],'parent':ref['weld_id'],'child':ref['child_index'],
                          'candidate_fit_flags':flags,'predictions':prediction,'errors':errors})
    report={'role':config['role'],'comparable_candidate_children':len(records),
            'candidate_fit_counts':{k:sum(r['candidate_fit_flags'][k] for r in evaluated) for k in flags},
            'unknown_or_unreviewed_remaining_in_contract_denominator':True,
            'full_contract_pass':False,'elapsed_seconds':time.perf_counter()-start,'results':evaluated}
    (out/'candidate_fit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'complete.json').write_text(json.dumps({'elapsed_seconds':report['elapsed_seconds'],'candidate_fit_counts':report['candidate_fit_counts'],
        'head_sha256':sha256(out/'heads.joblib'),'full_contract_pass':False},indent=2),encoding='utf8')
    print('ATTRIBUTE_HEAD_TRAINED',len(records),len(x),report['candidate_fit_counts'],round(report['elapsed_seconds'],2),flush=True)


if __name__=='__main__':
    main()
