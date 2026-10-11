"""Train only actual known endpoint labels; missing cuts and Any remain missing."""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import child_units, score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.endpoint_process_attributes import VERSION, FIELD_FAMILY, child_profiles, predict_field
from weld_pipeline.full_fit_hybrid import infer_from_sequences
from weld_pipeline.prepare_contract_child import child_attributes, FIELDS
from weld_pipeline.process_label_semantics import distance_label
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_horizontal_patch_tree import context


def known_label(field, attr):
    if not attr['truth_available']:
        return None
    if FIELD_FAMILY[field] in ['eave','side_split']:
        return {'state':'true' if bool(attr['value']) else 'false','value':bool(attr['value'])}
    parsed = distance_label(FIELDS[field], attr['value'])
    if parsed['state']=='exact':
        return {'state':'exact','value':float(parsed['exact_mm'])}
    if parsed['state'] in ['no_hole','over_50']:
        return {'state':parsed['state'],'value':0. if parsed['state']=='no_hole' else 51.}
    return None


def records(split, names):
    result=[]; coverage={}; manifest_hashes={}
    for name in names:
        piece=load_piece(split['piece_roots'][name],verify=True)
        structural,mean,covariance=context(split,name)
        mapping={w['id']:i for i,refs in mapped_references(piece).items() for w in refs}
        count=Counter()
        for child in child_units(piece['truth']):
            if child['kind']!='line' or child['parent_id'] not in mapping:
                count['unsupported_curve_or_contact']+=1;continue
            attrs=child_attributes(piece['truth'][child['parent_index']],child['segment_index'])
            labels={field:known_label(field,attrs[field]) for field in FIELD_FAMILY}
            si=mapping[child['parent_id']]
            profiles=child_profiles(piece['sequences'][si],child['points'],structural[si],mean,covariance)
            result.append({'piece':name,'child_id':child['id'],'labels':labels,'profiles':profiles})
            for field,label in labels.items():
                count[field+('_known' if label is not None else '_unknown')]+=1
        coverage[name]=dict(count)
        manifest_hashes[name]=sha256(Path(split['piece_roots'][name])/'manifest.json')
    return result,coverage,manifest_hashes


def accepted(got, wanted):
    return (wanted is not None and got['state']==wanted['state'] and got.get('value') is not None
        and (abs(float(got['value'])-float(wanted['value']))<=3. if wanted['state']=='exact'
             else got['value']==wanted['value']))


def conditional(bundle, rows, role):
    counts={f:{'known':0,'valid':0,'unknown':0} for f in FIELD_FAMILY}
    results=[]
    for row in rows:
        flags={}
        for field in FIELD_FAMILY:
            wanted=row['labels'][field]
            if wanted is None:
                counts[field]['unknown']+=1;continue
            counts[field]['known']+=1
            got=predict_field(bundle,field,row['profiles'][field])
            ok=accepted(got,wanted);counts[field]['valid']+=ok
            flags[field]={'pass':ok,'prediction':got,'reference':wanted}
        results.append({'piece':row['piece'],'child_id':row['child_id'],'fields':flags})
    return {'role':role,'counts':counts,'results':results,'geometry_misses_included':False,
        'complete_along_line_truth':False,'full_contract_pass':False}


def joint(bundle,geometry_state,model,split,names,role):
    results=[]; totals={f:{'truth':0,'known_truth':0,'valid':0} for f in FIELD_FAMILY}
    for name in names:
        piece=load_piece(split['piece_roots'][name],verify=True)
        structural,mean,covariance=context(split,name)
        with np.load(Path(split['structure_roots'][name])/(name+'.npz')) as value:
            process=value['process'].copy()
        pred,_=infer_from_sequences(geometry_state,model,piece['sequences'],structural,process)
        score=score_children(pred,piece['truth'])
        pc,tc=child_units(pred),child_units(piece['truth'])
        matches={m['truth_index']:m for m in score['matches']}
        mapping={w['id']:i for i,refs in mapped_references(piece).items() for w in refs}
        for child_index,truth in enumerate(tc):
            attrs=child_attributes(piece['truth'][truth['parent_index']],truth['segment_index'])
            flags={}; profiles=None
            if child_index in matches and truth['kind']=='line' and truth['parent_id'] in mapping:
                match=matches[child_index];points=np.array(pc[match['prediction_index']]['points'])
                if match['reversed']:
                    points=points[::-1]
                si=mapping[truth['parent_id']]
                try:
                    profiles=child_profiles(piece['sequences'][si],points,structural[si],mean,covariance)
                except ValueError:
                    profiles=None
            for field in FIELD_FAMILY:
                label=known_label(field,attrs[field]); totals[field]['truth']+=1
                totals[field]['known_truth']+=label is not None
                got=(predict_field(bundle,field,profiles[field]) if profiles is not None else
                    {'state':'unknown','value':None,'reason':'NO_VALID_COMPARABLE_PREDICTED_GEOMETRY'})
                ok=accepted(got,label);totals[field]['valid']+=ok
                flags[field]={'pass':ok,'prediction':got,'reference':label}
            results.append({'piece':name,'child_id':truth['id'],'fields':flags})
    return {'role':role,'counts':totals,'results':results,'geometry_misses_included':True,
        'feature_contact_for_comparison':'Reference-matched point candidate; diagnostic only, standalone contact selection not yet validated.',
        'complete_along_line_truth':False,'side_schema_frozen':False,'full_contract_pass':False}


def main():
    torch.set_num_threads(3)
    split_path=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(split_path.read_text('utf8'));names=[n for n in split['train'] if '__' not in n]
    out=Path('weld_pipeline/runs/endpoint_process_attributes_seed137');out.mkdir(parents=True,exist_ok=True)
    if (out/'heads.joblib').exists():
        raise ValueError('Existing fixed endpoint attribute run')
    sources={p:sha256(p) for p in [__file__,'weld_pipeline/endpoint_process_attributes.py',
        'weld_pipeline/horizontal_patch_tree.py','weld_pipeline/prepare_contract_child.py',
        'weld_pipeline/process_label_semantics.py','weld_pipeline/customer_label_corrections.py',
        'weld_pipeline/sequence_augmentation.py']}
    config={'seed':137,'version':VERSION,'trees':128,'max_features':.75,'min_samples_leaf':1,
        'state_confidence_threshold':.8,'train':names,'untrained_check':split['validation'],
        'split_sha256':sha256(split_path),'sources':sources,'input_features':521,
        'endpoint_scope_only':True,'front_back_endpoint_semantics_frozen':False,
        'unknown_policy':'No parent Any broadcasting, no internal cut interpolation, exclude ambiguous wall -1/negative/height0.',
        'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for p in sources:
        shutil.copy2(p,archive/Path(p).name)
    started=time.perf_counter();rows,coverage,hashes=records(split,names)
    (out/'training_coverage.json').write_text(json.dumps({'coverage':coverage,'data_hashes':hashes},indent=2),encoding='utf8')
    heads={}; family_counts={}
    for fi,family in enumerate(sorted(set(FIELD_FAMILY.values()))):
        rr=[(r['piece'],r['profiles'][field],r['labels'][field]) for r in rows for field in FIELD_FAMILY
            if FIELD_FAMILY[field]==family and r['labels'][field] is not None]
        family_counts[family]=len(rr)
        if not rr:
            continue
        x=np.stack([r[1] for r in rr]); states=np.array([r[2]['state'] for r in rr])
        counts=Counter(r[0] for r in rr);weights=np.array([1./counts[r[0]] for r in rr])
        classifier=ExtraTreesClassifier(n_estimators=128,max_features=.75,n_jobs=3,random_state=137+fi)
        classifier.fit(x,states,sample_weight=weights)
        exact=states=='exact';regressor=None
        if exact.any():
            regressor=ExtraTreesRegressor(n_estimators=128,max_features=.75,n_jobs=3,random_state=137+fi)
            regressor.fit(x[exact],np.array([r[2]['value'] for r in rr])[exact],sample_weight=weights[exact])
        heads[family]={'state':classifier,'value':regressor}
        print('PROCESS_FAMILY_TRAINED',family,len(rr),int(exact.sum()),flush=True)
    bundle={'heads':heads,'config':config};joblib.dump(bundle,out/'heads.joblib')
    fit=conditional(bundle,rows,'FIT_REFERENCE_ENDPOINTS_NOT_E3')
    (out/'fit_conditional.json').write_text(json.dumps(fit,indent=2),encoding='utf8')
    del rows
    # All weights and choices fixed before any check reference is read.
    unseen,unseen_coverage,unseen_hashes=records(split,split['validation'])
    check=conditional(bundle,unseen,'UNTRAINED_REFERENCE_ENDPOINTS_NOT_E3')
    (out/'check_conditional.json').write_text(json.dumps(check,indent=2),encoding='utf8')
    state=torch.load('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt',map_location='cpu')
    model=from_checkpoint(state['horizontal']).cuda().eval()
    fit_joint=joint(bundle,state,model,split,names,'FIT_JOINT_ENDPOINT_DIAGNOSTIC')
    (out/'fit_joint.json').write_text(json.dumps(fit_joint,indent=2),encoding='utf8')
    check_joint=joint(bundle,state,model,split,split['validation'],'UNTRAINED_JOINT_ENDPOINT_DIAGNOSTIC')
    (out/'check_joint.json').write_text(json.dumps(check_joint,indent=2),encoding='utf8')
    report={'elapsed_seconds':time.perf_counter()-started,'head_sha256':sha256(out/'heads.joblib'),
        'families':family_counts,'fit_conditional':fit['counts'],'untrained_conditional':check['counts'],
        'fit_joint':fit_joint['counts'],'untrained_joint':check_joint['counts'],
        'untrained_data_hashes':unseen_hashes,'full_contract_pass':False,
        'not_integrated_into_v4':True,'standalone_contact_selection_verified':False,'complete_along_line_truth':False}
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PROCESS_HEADS_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
