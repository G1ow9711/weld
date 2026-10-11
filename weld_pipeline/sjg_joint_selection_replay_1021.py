"""Freeze joint-graph selection before scoring a single completed TRAIN case."""
import argparse
import gc
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc
from weld_pipeline.sjg_joint_interval_selection_1020 import select, controls
from weld_pipeline.sjg_local_event_score_932 import geometric_match


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    assert read(OUT/'run197/local_round_review.json')['status']=='SINGLE_TRAIN_FEASIBILITY_FAILED'
    prediction=OUT/'run198/source_21';proposals=OUT/'run185/source_21'
    original=read(prediction/'input_output_freeze.json');binding=read(proposals/'input_freeze.json')
    assert digest(prediction/'predictions.json')==original['files']['predictions.json']
    assert digest(proposals/'profiles.json')==binding['files']['profiles.json']
    protocol=dict(stage='SINGLE_TRAIN_INPUT_ONLY_JOINT_SELECTION_REPLAY',original_eight_G=496,supplemental_TRAIN_G=4,
        source_key='21',prediction_run='run198',checkpoint_sha256=digest(OUT/'run197/final.pt'),
        prediction_sha256=original['files']['predictions.json'],source_sha256=digest(__file__),
        optimizer_updates=0,independent_test=False,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);write(dest/'controls.json',controls())
    kept,audit=select(read(prediction/'predictions.json'),read(proposals/'profiles.json'))
    write(dest/'predictions.json',kept);write(dest/'joint_selection.json',audit)
    write(dest/'output_freeze_before_target_read.json',dict(prediction_sha256=digest(dest/'predictions.json'),target_reads=0))
    cohort={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']}
    case=cohort['21'];assert digest(OUT/case['target_file'])==case['target_sha256']
    score=geometric_match(kept,read(OUT/case['target_file']));write(dest/'quick_geometry_score.json',score)
    # These inputs are deliberately reference-derived software controls. They
    # test preservation of true adjacent/gapped children, never model accuracy.
    controls_rows=[]
    for row in read(OUT/'run186/training_manifest.json')['cases']:
        key=row['source_key'];assert key not in ('64','24')
        sup=read(OUT/'run186'/('source_'+key+'_supervision.json'))
        target_case=cohort[key];assert digest(OUT/target_case['target_file'])==target_case['target_sha256']
        truth=read(OUT/target_case['target_file'])
        profile_path=OUT/'run185'/('source_'+key)/'profiles.json'
        assert digest(profile_path)==read(profile_path.parent/'input_freeze.json')['files']['profiles.json']
        profiles=read(profile_path);truth_controls=[]
        for child in sup['children']:
            representation=truth['parents'][child['parent']]['children'][child['child']]['representation']
            truth_controls.append(dict(representation=representation,profile=child['profiles'][0],confidence=.75))
        surviving,_=select(truth_controls,profiles)
        assert len(surviving)==len(truth_controls)==row['original_G']
        controls_rows.append(dict(source_key=key,reference_derived_controls=len(truth_controls),retained=len(surviving)))
        del profiles,truth,truth_controls,surviving;gc.collect()
    write(dest/'TRAIN_reference_structure_controls.json',dict(rows=controls_rows,reference_derived=True,
        not_model_accuracy=True,original_G=363,all_original_children_retained=True))
    write(dest/'receipt.json',dict(status='JOINT_SELECTION_REPLAY_COMPLETE',score=score,
        TRAIN_reference_controls_not_accuracy=True,full_contract_pass=False,optimizer_updates=0,end_utc=utc()))
    write(dest/'local_round_review.json',dict(score=score,full_contract_pass=False,
        next_action='Verify identical selection in fresh pure-XYZ single-case inference; then continue original small-pilot gates.',
        external_consultation_sent=False))
    print('JOINT_SELECTION_REPLAY','TP',score['TP'],'/',score['G'],'FP',score['FP'],
          'TRAIN_STRUCTURE_CONTROLS',sum(r['retained'] for r in controls_rows),'NOT_FULL_CONTRACT',flush=True)


if __name__=='__main__':main()
