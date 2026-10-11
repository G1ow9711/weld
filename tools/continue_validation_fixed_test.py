"""Freeze a validation-selected complete model, then assess every TEST version.

The selection criterion is declared before testing. Only complete validation
geometry participates; protected diagnostics and TEST scores never select a
model. Fields are authenticated and reported separately without inventing a
business-frame or millimetre acceptance threshold. This is model assessment,
not automatic promotion of a new best model.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.evaluate_full_weld_predictions import verify_final_selection, SOURCE_NAMES
from tools.assess_full_weld_output_fields import select_records, SOURCES as FIELD_SOURCES
from tools.record_training_experiment import git_head
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/full_dataset_v1'
RULE = 'FAMILY_EQUAL_COMPLETE_D3_UTILITY_THEN_RECALL_THEN_LOWER_FP_THEN_MODEL_HASH'
SOURCES = list(dict.fromkeys(['tools/continue_validation_fixed_test.py',
                              'tools/verify_validation_fixed_selection.py', *SOURCE_NAMES,
                              *FIELD_SOURCES, 'tools/continue_full_recall_training.py']))


def local_name(value):
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('Independent local experiment names are required')
    return value


def atomic(path, value):
    temporary=path.with_name(path.name+'.incomplete')
    write(temporary,value)
    os.replace(temporary,path)


def validation_merit(versions, expected):
    """One complete reference utility per view, using frozen family weights."""
    if sorted(v['version_key'] for v in versions) != sorted(expected):
        raise ValueError('Validation views are omitted, duplicated or from another role')
    utility=recall=false_positive=mass=0.
    for view in versions:
        specification=expected[view['version_key']]
        if specification['role'] != 'MODEL_SELECTION_ONLY' or view['role'] != 'MODEL_SELECTION_ONLY':
            raise ValueError('Only validation may participate in model selection')
        if view['family_id'] != specification['family_id']:
            raise ValueError('Validation family ownership changed')
        values=view['overall']
        if any(type(values[k]) is not int or values[k] < 0 for k in ('G','TP','FP','predicted')):
            raise ValueError('Complete integer scoring denominators are required')
        if (values['G'] != specification['children'] or values['TP'] > values['G']
                or values['TP'] > values['predicted'] or values['FP'] != values['predicted']-values['TP']):
            raise ValueError('Validation score/denominator identity is malformed')
        weight=specification['family_equal_version_evaluation_weight']
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('A positive frozen family weight is required')
        denominator=max(1,values['G'])
        utility+=weight*(2*values['TP']-values['predicted'])/denominator
        recall+=weight*values['TP']/denominator
        false_positive+=weight*values['FP']/denominator
        mass+=weight
    if not expected or not math.isclose(mass,1.,rel_tol=0.,abs_tol=1.e-12):
        raise ValueError('The complete validation role must have unit family weight')
    return dict(family_equal_reference_utility=utility,
                family_equal_recall=recall,family_equal_false_positive=false_positive)


def candidate(name, dataset):
    """Authenticate a complete model and all validation artifacts, never TEST."""
    continuation=BASE/local_name(name)
    completion=read(continuation/'receipt.json')
    pair_file=continuation/'model_pair_binding.json';pair=read(pair_file)
    if (completion['status'] != 'FULL_QUERY_PARENT_AND_SELECTION_DIAGNOSTICS_COMPLETE_NOT_PROMOTED'
            or completion['model_pair_sha256'] != digest(pair_file)
            or not completion['final_test_not_executed']):
        raise ValueError('An authentic complete query/parent/held continuation is required')
    run=BASE/(name+'_selection')
    paths,summary_file=select_records(run,dataset,'MODEL_SELECTION_ONLY',None)
    summary=read(summary_file);protocol=read(run/'protocol.json')
    if (summary['protocol_sha256'] != digest(run/'protocol.json')
            or protocol['role'] != 'MODEL_SELECTION_ONLY'
            or protocol['partial_control_not_final_evaluation']
            or protocol['full_train_fit'] is None
            or protocol['sources'] != {n:digest(ROOT/n) for n in protocol['sources']}
            or summary['query_checkpoint_sha256'] != pair['query_checkpoint_sha256']
            or summary['selector_checkpoint_sha256'] != pair['selector_checkpoint_sha256']):
        raise ValueError('Complete validation model/source binding differs')
    versions=[]
    for path in paths:
        receipt=read(path)
        for name2,sha in receipt['artifacts'].items():
            if Path(name2).name != name2 or digest(path.parent/name2) != sha:
                raise ValueError('Frozen validation artifacts changed')
        for view in receipt['versions']:
            versions.append(dict(version_key=view['version_key'],family_id=view['family_id'],
                role=receipt['role'],overall=view['geometry_score']['categories']['overall']))
    expected={key:value for key,(_,value) in dataset.versions.items()
              if value['role']=='MODEL_SELECTION_ONLY'}
    merit=validation_merit(versions,expected)
    field_dir=BASE/(name+'_selection_fields')
    fields,field_protocol=read(field_dir/'summary.json'),read(field_dir/'protocol.json')
    if (fields['status'] != 'ALL_FROZEN_RECORDS_FIELDS_ASSESSED_NOT_PROMOTED'
            or fields['role'] != 'MODEL_SELECTION_ONLY' or fields['fit_calls'] != 0
            or fields['source_protocol_sha256'] != digest(field_dir/'protocol.json')
            or field_protocol['prior_complete_role_summary_sha256'] != digest(summary_file)
            or field_protocol['prediction_run'] != str(run.relative_to(ROOT))
            or field_protocol['prediction_protocol_sha256'] != digest(run/'protocol.json')
            or field_protocol['sources'] != {n:digest(ROOT/n) for n in field_protocol['sources']}):
        raise ValueError('Complete validation field evidence differs')
    actual=[]
    for item in fields['records']:
        path=(ROOT/item['path']).resolve();path.relative_to(field_dir.resolve())
        if digest(path) != item['sha256'] or item['checkpoint_sha256'] != pair['selector_checkpoint_sha256']:
            raise ValueError('Validation field record/model changed')
        actual.extend(item['versions'])
    if sorted(actual) != sorted(expected):
        raise ValueError('Validation field assessment omitted or duplicated views')
    return dict(continuation_name=name,**pair,merit=merit,
        validation_summary_path=str(summary_file.relative_to(ROOT)),
        validation_summary_sha256=digest(summary_file),
        validation_field_summary_path=str((field_dir/'summary.json').relative_to(ROOT)),
        validation_field_summary_sha256=digest(field_dir/'summary.json'))


def choose(candidates):
    if not candidates:
        raise ValueError('At least one complete validation candidate is required')
    return min(candidates,key=lambda value:(-value['merit']['family_equal_reference_utility'],
        -value['merit']['family_equal_recall'],value['merit']['family_equal_false_positive'],
        value['query_checkpoint_sha256'],value['selector_checkpoint_sha256'],value['continuation_name']))


def verify_bindings(protocol,control):
    if protocol['sources'] != {n:digest(ROOT/n) for n in SOURCES}:
        raise ValueError('Bound final assessment source changed')
    if (protocol['registry_sha256'] != digest(BASE/'evaluation_registry.json')
            or protocol['admission_control_sha256'] != digest(control)):
        raise ValueError('Frozen role registry or selection admission control changed')


def launch(out,protocol,control,module,arguments,stage):
    verify_bindings(protocol,control)
    with (out/(stage+'.log')).open('a') as stream:
        child=subprocess.Popen([sys.executable,'-u','-m',module,*arguments],
                               cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        atomic(out/'progress.json',dict(status='RUNNING_VALIDATION_FIXED_FINAL_ASSESSMENT',
            controller_pid=os.getpid(),child_pid=child.pid,stage=stage,child_module=module))
        if child.wait():
            raise RuntimeError('Final assessment child failed: '+stage)


def main(args):
    if args.poll_seconds < 5:
        raise ValueError('An explicit conservative polling interval is required')
    out=BASE/local_name(args.experiment_name);out.mkdir(exist_ok=True)
    names=sorted(set(local_name(n) for n in args.candidate_continuation))
    if len(names) != len(args.candidate_continuation):
        raise ValueError('Duplicate continuation candidates are not allowed')
    control=Path(args.admission_control).resolve();control.relative_to(ROOT)
    proof=read(control)
    if (proof['status'] != 'PASS_VALIDATION_ONLY_SELECTION_RULE_AND_REAL_TRAIN_REJECTION'
            or proof['selection_rule'] != RULE
            or proof['driver_sha256'] != digest(Path(__file__))
            or proof['source_sha256'] != digest(ROOT/'tools/verify_validation_fixed_selection.py')
            or not proof['protected_and_test_roles_rejected'] or not proof['complete_views_required']
            or not proof['real_train_summary_rejected_as_validation']
            or proof['fit_calls'] != 0 or proof['TEST_label_geometry_reads'] != 0):
        raise ValueError('Current validation selection needs its numerical and role controls')
    protocol=dict(status='PREDECLARED_VALIDATION_ONLY_SELECTION_THEN_COMPLETE_FINAL_TEST',
        source_commit_at_initialization=(read(out/'protocol.json')['source_commit_at_initialization']
            if (out/'protocol.json').exists() else git_head(ROOT)),
        candidate_continuations=names,selection_rule=RULE,
        registry_sha256=digest(BASE/'evaluation_registry.json'),
        sources={n:digest(ROOT/n) for n in SOURCES},admission_control_sha256=digest(control),
        admission_control_path=str(control.relative_to(ROOT)),
        test_used_for_selection=False,
        diagnostic_used_for_selection=False,all_final_test_versions_required=True,
        automatic_best_model_promotion=False)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'protocol.json').exists():
            if read(out/'protocol.json') != protocol:
                raise ValueError('Final test protocol changed; use a new explicit experiment')
        else:
            atomic(out/'protocol.json',dict(protocol))
        while not all((BASE/n/'receipt.json').exists() for n in names):
            verify_bindings(protocol,control)
            for name in names:
                if (BASE/name/'failure.json').exists():
                    raise RuntimeError('Candidate continuation actually failed: '+name)
            atomic(out/'progress.json',dict(status='WAITING_COMPLETE_TRAIN_AND_VALIDATION_CANDIDATES',
                controller_pid=os.getpid(),candidate_continuations=names,
                test_reference_reads=0,optimizer_updates=0,unix_time=time.time()))
            time.sleep(args.poll_seconds)
        verify_bindings(protocol,control)
        dataset=EvaluationShardDataset(ROOT)
        values=[candidate(n,dataset) for n in names]
        selected=choose(values);selection=out/'final_selection.json'
        if selection.exists():
            fixed=read(selection)
            if (fixed['selection_rule'] != RULE or fixed['validation_candidates'] != values
                    or fixed['selected_continuation'] != selected['continuation_name']):
                raise ValueError('Immutable validation selection changed after testing began')
        else:
            fixed=dict(status='FINAL_MODEL_SELECTED_VALIDATION_ONLY_BEFORE_TEST',
                recorded_utc=datetime.now(timezone.utc).isoformat(),selection_rule=RULE,
                selected_continuation=selected['continuation_name'],validation_candidates=values,
                query_checkpoint_sha256=selected['query_checkpoint_sha256'],
                selector_checkpoint_sha256=selected['selector_checkpoint_sha256'],
                validation_summary_path=selected['validation_summary_path'],
                validation_summary_sha256=selected['validation_summary_sha256'],
                validation_field_summary_path=selected['validation_field_summary_path'],
                validation_field_summary_sha256=selected['validation_field_summary_sha256'],
                diagnostic_used_for_selection=False,test_evaluations_for_selection=0,
                fields_not_used_to_invent_unfrozen_acceptance_thresholds=True,
                model_fixed_for_assessment_not_promoted=True,full_contract_pass=False)
            stage=out/'final_selection.pending.json';write(stage,fixed)
            verify_final_selection(stage,selected['query_checkpoint_sha256'],
                                   selected['selector_checkpoint_sha256'],dataset.registry_sha256)
            os.replace(stage,selection)
        selection_sha=verify_final_selection(selection,selected['query_checkpoint_sha256'],
                                             selected['selector_checkpoint_sha256'],dataset.registry_sha256)
        continuation=BASE/selected['continuation_name'];cp=read(continuation/'protocol.json')
        parent=BASE/local_name(cp['parent_training_name']);query=BASE/local_name(cp['query_training_name'])
        geometry_name=out.name+'_geometry';geometry=BASE/geometry_name
        if not (geometry/'summary.json').exists():
            launch(out,protocol,control,'tools.evaluate_full_weld_predictions',[
                '--experiment-name',geometry_name,'--query-checkpoint',str(continuation/'query_checkpoint.pt'),
                '--query-sha256',selected['query_checkpoint_sha256'],'--selector-checkpoint',str(ROOT/selected['selector_checkpoint']),
                '--selector-sha256',selected['selector_checkpoint_sha256'],'--selector-protocol',str(parent/'protocol.json'),
                '--query-training-receipt',str(query/'receipt.json'),'--selector-training-receipt',str(parent/'receipt.json'),
                '--role','FINAL_TEST_ONLY','--final-selection',str(selection),'--wait-ready'],'final_test_geometry')
        fields_name=out.name+'_fields';fields=BASE/fields_name
        if not (fields/'summary.json').exists():
            launch(out,protocol,control,'tools.assess_full_weld_output_fields',[
                '--prediction-run',str(geometry),'--experiment-name',fields_name,'--role','FINAL_TEST_ONLY',
                '--final-selection',str(selection)],'final_test_fields')
        verify_bindings(protocol,control)
        select_records(geometry,dataset,'FINAL_TEST_ONLY',selection)
        summary=read(fields/'summary.json');field_protocol=read(fields/'protocol.json')
        required=sorted(k for k,(_,v) in dataset.versions.items() if v['role']=='FINAL_TEST_ONLY')
        scored=[]
        if (summary['status'] != 'ALL_FROZEN_RECORDS_FIELDS_ASSESSED_NOT_PROMOTED'
                or summary['role'] != 'FINAL_TEST_ONLY' or summary['fit_calls'] != 0
                or summary['source_protocol_sha256'] != digest(fields/'protocol.json')
                or field_protocol['prior_complete_role_summary_sha256'] != digest(geometry/'summary.json')
                or field_protocol['prediction_run'] != str(geometry.relative_to(ROOT))
                or field_protocol['prediction_protocol_sha256'] != digest(geometry/'protocol.json')
                or field_protocol['registry_sha256'] != dataset.registry_sha256
                or field_protocol['sources'] != {n:digest(ROOT/n) for n in field_protocol['sources']}):
            raise ValueError('Complete TEST field assessment binding differs')
        for item in summary['records']:
            path=(ROOT/item['path']).resolve();path.relative_to(fields.resolve())
            if digest(path) != item['sha256'] or item['checkpoint_sha256'] != selected['selector_checkpoint_sha256']:
                raise ValueError('Final TEST field record or model changed')
            scored.extend(item['versions'])
        if sorted(scored) != required or digest(selection) != selection_sha:
            raise ValueError('Final TEST omitted a version or changed the fixed selection')
        atomic(out/'receipt.json',dict(status='ALL_REGISTERED_FINAL_TEST_GEOMETRY_AND_FIELDS_EVALUATED_NOT_PROMOTED',
            protocol_sha256=digest(out/'protocol.json'),final_selection_sha256=selection_sha,
            geometry_summary_path=str((geometry/'summary.json').relative_to(ROOT)),
            geometry_summary_sha256=digest(geometry/'summary.json'),
            field_summary_path=str((fields/'summary.json').relative_to(ROOT)),
            field_summary_sha256=digest(fields/'summary.json'),scored_test_versions=required,
            test_evaluations_used_for_selection=0,fit_calls=0,independent_generalization_certified=False,
            original_five_regression_business_frames_csv_and_raw_timing_remaining=True,full_contract_pass=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--candidate-continuation',required=True,action='append')
    parser.add_argument('--admission-control',required=True)
    parser.add_argument('--poll-seconds',type=float,default=10.)
    args=parser.parse_args()
    try:
        main(args)
    except BaseException:
        folder=BASE/local_name(args.experiment_name);folder.mkdir(exist_ok=True)
        atomic(folder/'failure.json',dict(status='INCOMPLETE_VALIDATION_FIXED_FINAL_TEST_NOT_PROMOTED',
            traceback=traceback.format_exc(),unix_time=time.time()))
        raise
