"""Fresh single-case XYZ inference, then input/hash and geometric validation."""
import argparse
import sys
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_local_event_infer_931 import main as infer
from weld_pipeline.sjg_local_event_score_932 import geometric_match


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    assert read(OUT/'run199/receipt.json')['status']=='JOINT_SELECTION_REPLAY_COMPLETE'
    sys.argv.extend(['--sources','21','--joint-selection'])
    infer(default_data_run='run186',default_training_run='run197',default_proposal_run='run185',default_feature_run='run166')
    folder=OUT/args.run/'source_21';binding=read(folder/'input_output_freeze.json')
    previous=read(OUT/'run198/source_21/input_output_freeze.json')
    same={name:digest(folder/name)==previous['files'][name] for name in
          ('probability.npy','offset.npy','q_mm.npy','profile_latent.npy','profile_offsets.npy')}
    assert all(same.values())
    assert digest(folder/'predictions_before_joint_selection.json')==previous['files']['predictions.json']
    assert digest(folder/'predictions.json')==binding['files']['predictions.json']
    write(OUT/args.run/'fresh_inference_consistency.json',dict(numeric_outputs_identical_to_run198=same,
        old_decoded_predictions_identical=True,selection_changes_only_hypothesis_set=True,target_reads=0))
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
    assert digest(OUT/case['target_file'])==case['target_sha256']
    score=geometric_match(read(folder/'predictions.json'),read(OUT/case['target_file']))
    write(OUT/args.run/'quick_geometry_score.json',score)
    passed=score['TP_over_G']>.95 and score['FP_over_G']<=.05
    write(OUT/args.run/'local_round_review.json',dict(status='SINGLE_TRAIN_GEOMETRY_PASS' if passed else 'SINGLE_TRAIN_GEOMETRY_FAIL',
        score=score,fresh_feature_to_output_seconds=binding['elapsed_seconds'],backbone_feature_generation_not_timed=True,
        optimizer_updates=0,full_contract_pass=False,independent_test=False,external_consultation_sent=False,
        remaining=['multi-case TRAIN and held-out small-pilot validation','weld type and parent structure',
                   'complete E16 attributes','uncached end-to-end time','formal independent splits'],end_utc=utc()))
    print('FRESH_SINGLE_CASE_RESULT','TP',score['TP'],'/',score['G'],'FP',score['FP'],
          'SINGLE_TRAIN_GEOMETRY_PASS',passed,'FULL_CONTRACT_PENDING',flush=True)


if __name__=='__main__':main()
