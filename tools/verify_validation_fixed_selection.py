"""Exercise held-role separation, complete denominators and family selection."""
import argparse
import copy
import math
from pathlib import Path

from tools.continue_validation_fixed_test import validation_merit, choose, RULE
from tools.evaluate_full_weld_predictions import verify_final_selection
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/full_dataset_v1'


def main(args):
    if Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('An independent local control name is required')
    out=BASE/args.experiment_name;out.mkdir(exist_ok=False)
    expected={key:dict(role='MODEL_SELECTION_ONLY',family_id=family,children=10,
        family_equal_version_evaluation_weight=weight)
        for key,family,weight in [('a','same_family',.25),('b','same_family',.25),('c','other_family',.5)]}
    def versions(counts):
        return [dict(version_key=key,family_id=expected[key]['family_id'],role='MODEL_SELECTION_ONLY',
            overall=dict(G=10,TP=tp,FP=fp,predicted=tp+fp)) for key,(tp,fp) in zip(expected,counts)]
    rows=versions([(10,0),(0,0),(6,2)])
    merit=validation_merit(rows,expected)
    assert math.isclose(merit['family_equal_reference_utility'],.45,abs_tol=1.e-12)
    assert math.isclose(merit['family_equal_recall'],.55,abs_tol=1.e-12)
    assert math.isclose(merit['family_equal_false_positive'],.1,abs_tol=1.e-12)
    failures=[]
    for role in ['TRAIN','PROTECTED_DIAGNOSTIC_ONLY','FINAL_TEST_ONLY']:
        bad=copy.deepcopy(rows);bad[0]['role']=role
        try:validation_merit(bad,expected)
        except ValueError:failures.append(role)
        else:raise AssertionError('A non-selection role entered selection')
    wrong=[rows[:-1],rows+[rows[0]]]
    for kind in ['G','FP','TP','predicted']:
        bad=copy.deepcopy(rows)
        bad[0]['overall'][kind]=11 if kind in ['G','TP'] else -1
        wrong.append(bad)
    for bad in wrong:
        try:validation_merit(bad,expected)
        except ValueError:pass
        else:raise AssertionError('A missing/duplicate/malformed denominator was accepted')
    one=dict(continuation_name='one',query_checkpoint_sha256='a'*64,selector_checkpoint_sha256='b'*64,merit=merit)
    two=dict(continuation_name='two',query_checkpoint_sha256='c'*64,selector_checkpoint_sha256='d'*64,
             merit=validation_merit(versions([(4,0),(4,0),(4,0)]),expected))
    assert choose([two,one])['continuation_name']=='one'
    assert choose([one,two])['continuation_name']=='one'
    tie=dict(one,continuation_name='tie',query_checkpoint_sha256='0'*64)
    assert choose([one,tie])['continuation_name']=='tie'
    assert choose([tie,one])['continuation_name']=='tie'
    assessment=(BASE/args.train_assessment_run).resolve();assessment.relative_to(ROOT)
    summary=read(assessment/'summary.json')
    if summary['role']!='TRAIN':
        raise ValueError('The negative control must use an actual saved TRAIN assessment')
    fixture=out/'rejected_train_selection.json'
    write(fixture,dict(status='FINAL_MODEL_SELECTED_VALIDATION_ONLY_BEFORE_TEST',
        diagnostic_used_for_selection=False,test_evaluations_for_selection=0,
        query_checkpoint_sha256='a'*64,selector_checkpoint_sha256='b'*64,
        validation_summary_path=str((assessment/'summary.json').relative_to(ROOT)),
        validation_summary_sha256=digest(assessment/'summary.json'),synthetic_negative_fixture=True))
    try:verify_final_selection(fixture,'a'*64,'b'*64,digest(BASE/'evaluation_registry.json'))
    except ValueError:pass
    else:raise AssertionError('Actual TRAIN output was accepted as final validation')
    write(out/'receipt.json',dict(status='PASS_VALIDATION_ONLY_SELECTION_RULE_AND_REAL_TRAIN_REJECTION',
        selection_rule=RULE,driver_sha256=digest(ROOT/'tools/continue_validation_fixed_test.py'),
        source_sha256=digest(__file__),protected_and_test_roles_rejected=True,
        complete_views_required=True,all_G_TP_FP_prediction_identities_checked=True,
        family_weights_not_version_count_bias=True,selection_order_independent=True,
        real_train_summary_rejected_as_validation=True,
        real_train_summary_sha256=digest(assessment/'summary.json'),
        fit_calls=0,TEST_label_geometry_reads=0,final_model_selected=False))
    print('VALIDATION_FIXED_RULE_AND_REAL_TRAIN_REJECTION_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--train-assessment-run',required=True)
    main(parser.parse_args())
