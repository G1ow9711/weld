"""FIT-only feature replay of the frozen 146 head; counterfactuals are audits."""
import json
from pathlib import Path

import joblib
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.contract_child_evaluation import child_units
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.continuous_contact_sampling import ContinuousJointBases
from weld_pipeline.full_fit_hybrid import point_structure
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline import continuous_contextual_height_model as height_model


def main():
    name='207-FR79M';out=Path('weld_pipeline/reports/height_runtime_features_146_20261001')
    if out.exists():
        raise ValueError('Frozen FIT replay audit exists')
    out.mkdir(parents=True)
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    piece=load_piece(split['piece_roots'][name],verify=True)
    raw=Path('weld_pipeline/reports/continuous_sampling_20261001')/('raw_'+name)
    predictions=json.loads((raw/'predictions.json').read_text('utf8'))
    old_heights=json.loads((raw/'height_candidates.json').read_text('utf8'))
    score=json.loads((raw/'joint_height_diagnostic.json').read_text('utf8'))
    root=Path('weld_pipeline/runs/continuous_local_height_seed146')
    fixed=json.loads((root/('FIT_'+name+'_reference_support.json')).read_text('utf8'))['rows']
    originals={(r['parent'],r['child'],r['endpoint']):r for r in fixed if r['jitter_mm']==0. and r['supported']}
    with np.load(root/'FIT_training.npz',allow_pickle=False) as saved:
        reference_x=saved['features'].copy()
    bundle=joblib.load(root/'heads.joblib');head_digest=sha256(root/'heads.joblib')
    xyz=read_xyz_pcd(piece['record']['input'])*1000.
    sequences,_=make_mixed_sequences(xyz);structural,process=point_structure(xyz,sequences)
    captured=[];original_combine=height_model.combine_features
    def capture(*args):
        feature=original_combine(*args);captured.append(feature.copy());return feature
    height_model.combine_features=capture
    try:
        fresh=height_model.estimate_children(bundle,FiniteContactPlates(xyz),ContinuousJointBases(xyz),sequences,predictions,structural,process)
    finally:
        height_model.combine_features=original_combine
    indexed={};cursor=0
    for row in fresh:
        for field in ('start_height','end_height'):
            record=row['fields'][field]
            if 'physical_span_baseline_mm' in record:
                indexed[(row['parent_id'],row['child_index'],field)]=captured[cursor];cursor+=1
    if cursor!=len(captured):
        raise ValueError('Captured feature-to-field mapping incomplete')
    pc,tc=child_units(predictions),child_units(piece['truth'])
    raw_x,ref_x,targets,rows=[],[],[],[]
    for match in score['geometry']['matches']:
        predicted,truth=pc[match['prediction_index']],tc[match['truth_index']]
        for endpoint in ('start','end'):
            source=originals.get((truth['parent_id'],truth['segment_index'],endpoint))
            field=('end' if endpoint=='start' else 'start')+'_height' if match['reversed'] else endpoint+'_height'
            feature=indexed.get((predicted['parent_id'],predicted['segment_index'],field))
            if source is None or feature is None:
                continue
            raw_x.append(feature);ref_x.append(reference_x[source['sample_index']]);targets.append(source['reference_height_mm'])
            rows.append(dict(parent=truth['parent_id'],child=truth['segment_index'],endpoint=endpoint))
    raw_x,ref_x,target=np.array(raw_x),np.array(ref_x),np.array(targets)
    original_prediction,_=height_model.predict(bundle,raw_x)
    trained_prediction,_=height_model.predict(bundle,ref_x)
    groups=dict(profile39=(0,39),structure50=(39,89),physical_sizes6=(89,95),normal3=(95,98),inward3=(98,101))
    trials={}
    for group,(a,b) in groups.items():
        diagnostic_x=raw_x.copy();diagnostic_x[:,a:b]=ref_x[:,a:b]
        diagnostic_prediction,_=height_model.predict(bundle,diagnostic_x)
        trials[group]=dict(max_abs_input_difference=float(np.max(np.abs(raw_x[:,a:b]-ref_x[:,a:b]))),
            rows_with_any_difference=int((np.abs(raw_x[:,a:b]-ref_x[:,a:b])>1e-7).any(1).sum()),
            within_3_under_training_reference_feature_replacement=int((np.abs(diagnostic_prediction-target)<=3.).sum()),
            production_input=False)
    result=dict(piece=name,role='FIT_RUNTIME_FEATURE_DIAGNOSTIC',paired_endpoints=len(rows),
        raw_features_within_3=int((np.abs(original_prediction-target)<=3.).sum()),
        trained_original_features_within_3=int((np.abs(trained_prediction-target)<=3.).sum()),
        groups=trials,rows=rows,counterfactuals_include_training_reference_features=True,
        counterfactual_results_are_not_runtime_acceptance=True,
        raw_value_replay_equal=(json.dumps(fresh,sort_keys=True)==json.dumps(old_heights,sort_keys=True)),
        heads_unchanged=sha256(root/'heads.joblib')==head_digest,
        pcd_unchanged=sha256(piece['record']['input'])==piece['record']['pcd_sha256'],
        source_sha256=sha256(__file__),observed_check_read=False,full_contract_pass=False)
    np.savez(out/'paired_features.npz',raw_features=raw_x,trained_reference_features=ref_x,
             targets=target,raw_predictions=original_prediction,trained_predictions=trained_prediction)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('HEIGHT_FEATURE_REPLAY',json.dumps({key:value for key,value in result.items() if key!='rows'}),flush=True)


if __name__=='__main__':
    main()
