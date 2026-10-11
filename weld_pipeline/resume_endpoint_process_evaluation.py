"""Resume scoring saved137 heads in batches without changing weights or gates."""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import child_units, score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.endpoint_process_attributes import FIELD_FAMILY, child_profiles, predict_field
from weld_pipeline.full_fit_hybrid import infer_from_sequences
from weld_pipeline.prepare_contract_child import child_attributes
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_endpoint_process_attributes import records, known_label, accepted
from weld_pipeline.train_horizontal_patch_tree import context


def predict_batches(bundle, rows):
    output=[{f:{'state':'unknown','value':None,'reason':'NO_VALID_COMPARABLE_PREDICTED_GEOMETRY'}
        for f in FIELD_FAMILY} for row in rows]
    for family,head in bundle['heads'].items():
        selected=[(i,field,row['profiles'][field]) for i,row in enumerate(rows) if row['profiles'] is not None
            for field in FIELD_FAMILY if FIELD_FAMILY[field]==family]
        if not selected:
            continue
        x=np.stack([r[2] for r in selected]);probability=head['state'].predict_proba(x)
        confidence=probability.max(axis=1);states=head['state'].classes_[probability.argmax(axis=1)]
        exact=(states=='exact')&(confidence>=.8);values=np.zeros(len(selected))
        if exact.any():
            values[exact]=head['value'].predict(x[exact])
        for j,(i,field,xrow) in enumerate(selected):
            state=str(states[j]);c=float(confidence[j])
            if c<.8:
                got={'state':'unknown','value':None,'reason':'LOW_PROCESS_STATE_CONFIDENCE','confidence':c}
            else:
                value=(float(values[j]) if state=='exact' else 51. if state=='over_50' else
                    0. if state=='no_hole' else state=='true' if state in ['false','true'] else None)
                if value is None:
                    raise ValueError('Unknown process state')
                got={'state':state,'value':value,'confidence':c,
                    'source':'PCD_profiles_supervised_by_explicit_FIT_Excel_endpoints',
                    'scope':'child_endpoint_only_not_along_line_acceptance','side_schema_frozen':False}
            output[i][field]=got
    return output


def score_rows(bundle,rows,role,include_geometry):
    predictions=predict_batches(bundle,rows);results=[]
    totals={f:({'truth':len(rows),'known_truth':0,'valid':0} if include_geometry
        else {'known':0,'valid':0,'unknown':0}) for f in FIELD_FAMILY}
    for row,got in zip(rows,predictions):
        fields={}
        for field in FIELD_FAMILY:
            label=row['labels'][field]
            if include_geometry:
                totals[field]['known_truth']+=label is not None
            elif label is not None:
                totals[field]['known']+=1
            else:
                totals[field]['unknown']+=1
            ok=bool(accepted(got[field],label));totals[field]['valid']+=ok
            fields[field]={'pass':ok,'prediction':got[field],'reference':label}
        results.append({'piece':row['piece'],'child_id':row['child_id'],'fields':fields})
    return {'role':role,'counts':totals,'results':results,'geometry_misses_included':include_geometry,
        'feature_contact_for_comparison':'Reference-matched PCD candidate; diagnostic only, standalone selection not validated.',
        'complete_along_line_truth':False,'side_schema_frozen':False,'full_contract_pass':False}


def joint_rows(state,model,split,names):
    results=[]
    for name in names:
        piece=load_piece(split['piece_roots'][name],verify=True)
        structural,mean,covariance=context(split,name)
        with np.load(Path(split['structure_roots'][name])/(name+'.npz')) as value:
            process=value['process'].copy()
        pred,_=infer_from_sequences(state,model,piece['sequences'],structural,process)
        score=score_children(pred,piece['truth']);pc,tc=child_units(pred),child_units(piece['truth'])
        matches={m['truth_index']:m for m in score['matches']}
        mapping={w['id']:i for i,refs in mapped_references(piece).items() for w in refs}
        for index,truth in enumerate(tc):
            attrs=child_attributes(piece['truth'][truth['parent_index']],truth['segment_index'])
            labels={field:known_label(field,attrs[field]) for field in FIELD_FAMILY}
            profiles=None
            if index in matches and truth['kind']=='line' and truth['parent_id'] in mapping:
                match=matches[index];points=np.array(pc[match['prediction_index']]['points'])
                if match['reversed']:
                    points=points[::-1]
                si=mapping[truth['parent_id']]
                try:
                    profiles=child_profiles(piece['sequences'][si],points,structural[si],mean,covariance)
                except ValueError:
                    profiles=None
            results.append({'piece':name,'child_id':truth['id'],'labels':labels,'profiles':profiles})
        print('JOINT_CONTACTS',name,score['valid_children'],'/',score['truth_count'],flush=True)
    return results


def main():
    torch.set_num_threads(3)
    out=Path('weld_pipeline/runs/endpoint_process_attributes_seed137')
    if (out/'complete.json').exists():
        raise ValueError('Do not overwrite completed evaluation')
    started=time.perf_counter();head_path=out/'heads.joblib';head_sha=sha256(head_path)
    config=json.loads((out/'config.json').read_text('utf8'))
    for source,wanted in config['sources'].items():
        assert sha256(source)==wanted,'Frozen137 source changed'
    bundle=joblib.load(head_path)
    assert bundle['config']==config
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    rows,coverage,hashes=records(split,config['train'])
    for name,wanted in json.loads((out/'training_coverage.json').read_text('utf8'))['data_hashes'].items():
        assert hashes[name]==wanted
    probes=rows[:16];batch=predict_batches(bundle,probes);comparisons=0
    for row,got in zip(probes,batch):
        for field in FIELD_FAMILY:
            expected=predict_field(bundle,field,row['profiles'][field]);actual=got[field]
            assert expected['state']==actual['state']
            if expected.get('value') is not None:
                assert abs(float(expected['value'])-float(actual['value']))<=1e-9
            assert abs(expected.get('confidence',1.)-actual.get('confidence',1.))<=1e-12
            comparisons+=1
    print('PASS_BATCH_SCALAR_EQUIVALENCE',comparisons,flush=True)
    fit=score_rows(bundle,rows,'FIT_REFERENCE_ENDPOINTS_NOT_E3',False)
    (out/'fit_conditional.json').write_text(json.dumps(fit,indent=2),encoding='utf8')
    del rows
    unseen,check_coverage,check_hashes=records(split,config['untrained_check'])
    check=score_rows(bundle,unseen,'UNTRAINED_REFERENCE_ENDPOINTS_NOT_E3',False)
    (out/'check_conditional.json').write_text(json.dumps(check,indent=2),encoding='utf8')
    del unseen
    state=torch.load('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt',map_location='cpu')
    model=from_checkpoint(state['horizontal']).cuda().eval()
    fit_joint=score_rows(bundle,joint_rows(state,model,split,config['train']),'FIT_JOINT_ENDPOINT_DIAGNOSTIC',True)
    (out/'fit_joint.json').write_text(json.dumps(fit_joint,indent=2),encoding='utf8')
    check_joint=score_rows(bundle,joint_rows(state,model,split,config['untrained_check']),'UNTRAINED_JOINT_ENDPOINT_DIAGNOSTIC',True)
    (out/'check_joint.json').write_text(json.dumps(check_joint,indent=2),encoding='utf8')
    assert sha256(head_path)==head_sha
    family_counts={f:sum(v['known'] for k,v in fit['counts'].items() if FIELD_FAMILY[k]==f) for f in bundle['heads']}
    report={'head_sha256':head_sha,'families':family_counts,'fit_conditional':fit['counts'],
        'untrained_conditional':check['counts'],'fit_joint':fit_joint['counts'],'untrained_joint':check_joint['counts'],
        'untrained_data_hashes':check_hashes,'full_contract_pass':False,'not_integrated_into_v4':True,
        'standalone_contact_selection_verified':False,'complete_along_line_truth':False,
        'evaluation_recovered_from_saved_heads':True,'evaluator_source':__file__,'evaluator_sha256':sha256(__file__),
        'batch_scalar_equivalent_comparisons':comparisons,'batch_evaluation_seconds':time.perf_counter()-started,
        'scalar_evaluator_stop':'Controlled interrupt after all six trained heads saved; excessive single-row thread dispatch overhead.'}
    shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PROCESS_EVALUATION_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
