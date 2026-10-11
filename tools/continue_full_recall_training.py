"""Continue full-query fitting into matching parent fitting and held roles.

Waits for authentic all-TRAIN query evidence, then starts the complete parent
controller using one immutable query snapshot. Selection and diagnostics are
evaluated separately. Final TEST remains gated by an explicit validation-fixed
model selection, so this controller cannot choose a model using TEST results.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_query_checkpoint_model import load_query_model, verify_complete_query_fit
from weld_pipeline.server_reference_recall_weld_model import ARCHITECTURE
from weld_pipeline.server_full_evaluation_dataset import ROLES
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCES = ['tools/continue_full_recall_training.py', 'tools/train_full_recall_parent_dataset.py',
    'tools/prepare_full_recall_parent_pool.py', 'tools/evaluate_full_weld_predictions.py',
    'weld_pipeline/server_query_checkpoint_model.py', 'weld_pipeline/server_reference_recall_weld_model.py',
    'tools/train_full_weld_reference_recall.py', 'tools/assess_full_weld_output_fields.py',
    'weld_pipeline/server_complete_output_assessment.py',
    'weld_pipeline/server_full_evaluation_dataset.py']


def held_stages():
    stages = [('MODEL_SELECTION_ONLY', 'selection'),
              ('PROTECTED_DIAGNOSTIC_ONLY', 'protected_diagnostic')]
    if any(role not in ROLES for role, _ in stages):
        raise ValueError('Continuation roles differ from the actual evaluation interface')
    return stages


def atomic_json(path, value):
    stage = path.with_name(path.name + '.incomplete')
    write(stage, value)
    os.replace(stage, path)


def checked_name(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Controller names must be local independent experiment names')
    return name


def run_child(out, protocol, module, arguments, stage):
    if protocol['sources'] != {name:digest(ROOT/name) for name in SOURCES}:
        raise ValueError('Bound continuation source changed; preserve and declare a new version')
    with (out / (stage + '.log')).open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', module, *arguments],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(out / 'progress.json', dict(status='RUNNING_BOUND_CHILD', stage=stage,
            controller_pid=os.getpid(), child_pid=child.pid, child_module=module, unix_time=time.time()))
        print('FULL_RECALL_CONTINUATION_CHILD_STARTED', stage, child.pid, flush=True)
        code = child.wait()
    if code:
        raise RuntimeError('Bound continuation child failed: '+stage+' exit '+str(code))
    print('FULL_RECALL_CONTINUATION_CHILD_COMPLETE', stage, flush=True)


def main(args):
    out = BASE / checked_name(args.experiment_name)
    query_dir = BASE / checked_name(args.query_training_name)
    parent_name = checked_name(args.parent_training_name)
    out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name:digest(ROOT/name) for name in SOURCES}
        selector = Path(args.selector_checkpoint).resolve() if args.selector_checkpoint else None
        if selector:
            selector.relative_to(ROOT)
            if not args.selector_sha256 or digest(selector) != args.selector_sha256:
                raise ValueError('Frozen selector initialization fingerprint differs')
        elif args.selector_sha256:
            raise ValueError('Selector initialization fingerprint lacks a checkpoint')
        protocol = dict(status='FULL_RECALL_QUERY_TO_PARENT_AND_HELD_CONTINUATION_PROTOCOL',
            query_training_name=args.query_training_name, parent_training_name=parent_name,
            passes=args.passes, steps_per_job=args.steps_per_job,
            selector_initialization=str(selector.relative_to(ROOT)) if selector else None,
            selector_initialization_sha256=args.selector_sha256 if selector else None,
            selector_initializer_not_accepted_best_model=True, sources=sources,
            full_query_fit_required=True, final_test_auto_selection_allowed=False,
            evaluation_roles=[role for role, _ in held_stages()],
            complete_role_fields_assessed_after_geometry=True,
            held_or_diagnostic_fit_calls=0)
        if (out / 'protocol.json').exists():
            if read(out / 'protocol.json') != protocol:
                raise ValueError('Continuation protocol changed; use a new experiment')
        else:
            atomic_json(out / 'protocol.json', protocol)
        receipt_file = query_dir / 'receipt.json'
        while not receipt_file.exists():
            if sources != {name:digest(ROOT/name) for name in SOURCES}:
                raise ValueError('Bound continuation source changed while waiting')
            atomic_json(out / 'progress.json', dict(status='WAITING_AUTHENTIC_ALL_TRAIN_QUERY_FIT',
                controller_pid=os.getpid(), query_training_name=args.query_training_name,
                actual_parent_fit_started=False, test_reference_reads=0, unix_time=time.time()))
            time.sleep(10)
        qr = read(receipt_file)
        checkpoint = out / 'query_checkpoint.pt'
        if not checkpoint.exists():
            stage = checkpoint.with_suffix('.incomplete')
            shutil.copyfile(query_dir / 'last.pt', stage)
            if digest(stage) != qr['checkpoint_sha256']:
                raise ValueError('Final query fitting receipt and snapshot differ')
            os.replace(stage, checkpoint)
        if digest(checkpoint) != qr['checkpoint_sha256']:
            raise ValueError('Final query snapshot changed')
        query = torch.load(checkpoint, map_location='cpu', weights_only=False)
        _, architecture = load_query_model(query)
        if architecture != ARCHITECTURE:
            raise ValueError('Continuation requires the explicit recall query architecture')
        query_fit = verify_complete_query_fit(ROOT, query, qr['checkpoint_sha256'], receipt_file)
        atomic_json(out / 'query_binding.json', dict(query_checkpoint_sha256=qr['checkpoint_sha256'],
            query_checkpoint=str(checkpoint.relative_to(ROOT)), query_full_fit=query_fit))
        parent_dir = BASE / parent_name
        if not (parent_dir / 'receipt.json').exists():
            arguments = ['--experiment-name',parent_name,'--passes',str(args.passes),
                '--steps-per-job',str(args.steps_per_job),'--query-checkpoint',str(checkpoint),
                '--query-sha256',qr['checkpoint_sha256'],'--query-training-receipt',str(receipt_file)]
            if selector:
                arguments += ['--selector-checkpoint',str(selector),'--selector-sha256',args.selector_sha256]
            run_child(out, protocol, 'tools.train_full_recall_parent_dataset', arguments, 'parent_fit')
        parent_receipt = read(parent_dir / 'receipt.json')
        parent_protocol = read(parent_dir / 'protocol.json')
        if (parent_receipt['status'] != 'ALL_REGISTERED_TRAIN_PARENT_FIT_COMPLETE_NOT_PROMOTED'
                or parent_receipt['protocol_sha256'] != digest(parent_dir/'protocol.json')
                or parent_protocol['query_checkpoint_sha256'] != qr['checkpoint_sha256']
                or parent_protocol['sources'] != {name:digest(ROOT/name) for name in parent_protocol['sources']}):
            raise ValueError('Complete matching parent training evidence differs')
        resume = read(parent_dir / 'resume.json')
        final_selector = (parent_dir / resume['checkpoint']).resolve()
        final_selector.relative_to(parent_dir.resolve())
        if digest(final_selector) != resume['sha256']:
            raise ValueError('Final parent model differs from durable resume binding')
        atomic_json(out / 'model_pair_binding.json', dict(query_checkpoint_sha256=qr['checkpoint_sha256'],
            selector_checkpoint_sha256=resume['sha256'], selector_checkpoint=str(final_selector.relative_to(ROOT)),
            selector_training_receipt_sha256=digest(parent_dir/'receipt.json'),
            selector_training_protocol_sha256=digest(parent_dir/'protocol.json'), model_not_promoted=True))
        field_bindings = []
        for role, suffix in held_stages():
            evaluation_name = args.experiment_name + '_' + suffix
            arguments = ['--experiment-name',evaluation_name,'--query-checkpoint',str(checkpoint),
                '--query-sha256',qr['checkpoint_sha256'],'--selector-checkpoint',str(final_selector),
                '--selector-sha256',resume['sha256'],'--selector-protocol',str(parent_dir/'protocol.json'),
                '--query-training-receipt',str(receipt_file),'--selector-training-receipt',str(parent_dir/'receipt.json'),
                '--role',role,'--wait-ready']
            run_child(out, protocol, 'tools.evaluate_full_weld_predictions', arguments, suffix)
            field_name = evaluation_name + '_fields'
            field_dir = BASE / field_name
            if not (field_dir / 'summary.json').exists():
                run_child(out, protocol, 'tools.assess_full_weld_output_fields',
                    ['--prediction-run', str(BASE / evaluation_name), '--experiment-name', field_name,
                     '--role', role], suffix + '_fields')
            fields, field_protocol = read(field_dir / 'summary.json'), read(field_dir / 'protocol.json')
            geometry_dir = BASE / evaluation_name
            required = sorted(v['key'] for job in read(BASE / 'evaluation_registry.json')['cases']
                              for v in job['versions'] if v['role'] == role)
            actual = []
            if (fields['status'] != 'ALL_FROZEN_RECORDS_FIELDS_ASSESSED_NOT_PROMOTED'
                    or fields['role'] != role or fields['fit_calls'] != 0
                    or fields['source_protocol_sha256'] != digest(field_dir / 'protocol.json')
                    or field_protocol['prediction_run'] != str(geometry_dir.relative_to(ROOT))
                    or field_protocol['prediction_protocol_sha256'] != digest(geometry_dir / 'protocol.json')
                    or field_protocol['prior_complete_role_summary_sha256'] != digest(geometry_dir / 'summary.json')):
                raise ValueError('Complete-role process assessment differs from the frozen geometry run')
            for record in fields['records']:
                path = (ROOT / record['path']).resolve()
                path.relative_to(field_dir.resolve())
                if digest(path) != record['sha256'] or record['checkpoint_sha256'] != resume['sha256']:
                    raise ValueError('Saved process-field result differs from this model pair')
                actual.extend(record['versions'])
            if sorted(actual) != required:
                raise ValueError('Process assessment omitted or duplicated a registered held version')
            field_bindings.append(dict(role=role, summary_path=str((field_dir / 'summary.json').relative_to(ROOT)),
                                       summary_sha256=digest(field_dir / 'summary.json')))
        atomic_json(out / 'receipt.json', dict(status='FULL_QUERY_PARENT_AND_SELECTION_DIAGNOSTICS_COMPLETE_NOT_PROMOTED',
            query_full_fit=query_fit, model_pair_sha256=digest(out/'model_pair_binding.json'),
            process_field_assessments=field_bindings, final_test_not_executed=True,
            baseline_regression_field_tolerance_business_frames_csv_and_acceptance_remaining=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--query-training-name', default='full_weld_reference_recall_v3')
    parser.add_argument('--parent-training-name', required=True)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--steps-per-job', type=int, default=8)
    parser.add_argument('--selector-checkpoint')
    parser.add_argument('--selector-sha256')
    args = parser.parse_args()
    try:
        if min(args.passes,args.steps_per_job) < 1:
            raise ValueError('Complete positive parent visit budgets are required')
        main(args)
    except BaseException:
        out = BASE / checked_name(args.experiment_name)
        out.mkdir(exist_ok=True)
        atomic_json(out / 'failure.json', dict(status='INCOMPLETE_CONTINUATION_NOT_PROMOTED',
            traceback=traceback.format_exc(), unix_time=time.time()))
        raise
