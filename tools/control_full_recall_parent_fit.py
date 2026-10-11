"""Consume one entire actual recall pool and exercise complete parent fitting.

This is a TRAIN control, not full-dataset training or model selection. It waits
for the already running producer's authenticated pool, retains every candidate
and every registered label view of the input, and uses the same complete-set
loss as the all-TRAIN controller. It never treats an observation timeout as a
failed producer or starts another producer.
"""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback

import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_query_checkpoint_model import load_query_model
from weld_pipeline.server_reference_recall_weld_model import ARCHITECTURE
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.record_training_experiment import git_head
from tools.train_full_predicted_parent_selector import load_job
from tools.train_full_recall_parent_dataset import (
    SOURCE_NAMES, atomic_json, fit_job, post_job_result, save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def local_name(value):
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('An independent local experiment name is required')
    return value


def process_binding(pid):
    """Distinguish this actual producer from a later reuse of its PID."""
    folder = Path('/proc') / str(pid)
    stat = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    command = (folder / 'cmdline').read_bytes().split(b'\0')
    return dict(pid=pid, start_ticks=stat[19], state=stat[0],
                argv=[item.decode() for item in command if item])


def main(args):
    if args.steps < 1 or args.producer_pid < 1:
        raise ValueError('Positive actual producer PID and fit budget required')
    pools = BASE / local_name(args.pool_name)
    out = BASE / local_name(args.experiment_name)
    if pools == out:
        raise ValueError('A control cannot modify its producer directory')
    out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in dict.fromkeys(
        [str(Path(__file__).relative_to(ROOT)), *SOURCE_NAMES])}
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    dataset = FullShardDataset(ROOT)
    if args.job not in dataset.jobs:
        raise ValueError('Control requires one registered TRAIN input')
    checkpoint = pools / 'query_checkpoint.pt'
    if digest(checkpoint) != args.query_sha256:
        raise ValueError('Control query differs from the declared immutable snapshot')
    query = torch.load(checkpoint, map_location='cpu', weights_only=False)
    _, architecture = load_query_model(query)
    if architecture != ARCHITECTURE:
        raise ValueError('Actual recall query architecture required')
    initial = Path(args.selector_checkpoint).resolve()
    initial.relative_to(ROOT)
    if digest(initial) != args.selector_sha256:
        raise ValueError('Selector initialization fingerprint differs')
    receipt_path = pools / 'pools' / args.job / 'receipt.json'
    binding = process_binding(args.producer_pid) if not receipt_path.exists() else None
    if binding:
        argv = binding['argv']
        required = ['tools.prepare_full_recall_parent_pool', args.pool_name, args.job,
                    args.query_sha256]
        if not all(value in argv for value in required):
            raise ValueError('The specified live producer belongs to another experiment')
    protocol = dict(status='COMPLETE_ACTUAL_TRAIN_RECALL_POOL_FIT_CONTROL',
        source_commit_at_initialization=git_head(ROOT),
        job_key=args.job, registered_label_versions=[v['key'] for v in dataset.jobs[args.job]['versions']],
        registry_sha256=dataset.registry_sha256, query_checkpoint_sha256=args.query_sha256,
        query_architecture=architecture, query_actual_new_optimizer_steps=query['optimizer_steps'],
        selector_initialization_sha256=args.selector_sha256, selector_initialization_not_bound_final_model=True,
        producer_binding=binding, steps=args.steps, sources=sources,
        complete_candidate_pool_required=True, all_same_input_label_views_required=True,
        no_new_producer_started=True, reference_read_only_after_complete_pool_freeze=True,
        partial_query_control_not_full_registered_training=True,
        held_or_diagnostic_fit_calls=0, independent_accuracy_or_best_model_not_claimed=True)
    write(out / 'protocol.json', protocol)
    try:
        while not receipt_path.exists():
            try:
                current = process_binding(args.producer_pid)
            except FileNotFoundError:
                if receipt_path.exists():
                    break
                raise RuntimeError('Authenticated producer exited without a frozen complete pool')
            if (current['start_ticks'] != binding['start_ticks']
                    or current['argv'] != binding['argv'] or current['state'] == 'Z'):
                raise RuntimeError('Authenticated producer is terminal without a frozen complete pool')
            if sources != {name: digest(ROOT / name) for name in sources}:
                raise ValueError('Control source changed while waiting; declare a new control')
            atomic_json(out / 'progress.json', dict(status='WAITING_VERIFIED_RUNNING_POOL_PRODUCER',
                controller_pid=os.getpid(), producer_pid=args.producer_pid,
                producer_start_ticks=current['start_ticks'], reference_geometry_reads=0,
                optimizer_updates=0, unix_time=time.time()))
            time.sleep(10)
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise ValueError('Bound control source changed')
        receipt = read(receipt_path)
        if receipt['query_checkpoint_sha256'] != args.query_sha256:
            raise ValueError('Frozen pool has a different actual query checkpoint')
        atomic_json(out / 'progress.json', dict(status='COMPILING_ALL_ACTUAL_PARENT_TOKENS_AND_REFERENCE_COSTS',
            controller_pid=os.getpid(), job_key=args.job, children=receipt['children'],
            parents=receipt['parents'], pool_receipt_sha256=digest(receipt_path), unix_time=time.time()))
        priors = FrozenRun446Priors(ROOT)
        cached = CachedRun446RelationPrior(priors)
        with threadpool_limits(limits=2):
            row = load_job(dataset, args.job, pools, cached, reference_chunk_size=1024)
            if len(row['pool']['parents']) != receipt['parents'] or len(row['counts']) != receipt['parents']:
                raise ValueError('Actual complete parent pool was truncated')
            expected = {v['key'] for v in dataset.jobs[args.job]['versions']}
            if {v['key'] for v in row['views']} != expected:
                raise ValueError('A same-input reference view was omitted')
            write(out / 'capacity.json', dict(job_key=args.job, complete_pool_receipt_sha256=digest(receipt_path),
                children=receipt['children'], parents=receipt['parents'],
                views=[dict(version_key=v['key'], G=v['G'],
                            all_pool_geometry_reference_capacity=v['pool_capacity_reference_evaluation'])
                       for v in row['views']],
                all_pool_FP_counts_are_unselected_hypotheses_not_final_predictions=True,
                global_original_D3_assignment=True, independent_extraction_accuracy=False))
            state = dict(optimizer_steps=0, updated_versions=[], processed_versions=sorted(expected),
                job_visits=[], completed=[], passes=1, passes_completed=0, query_full_fit=None)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = OrderedParentSetModel().to(device)
            model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=False)['state_dict'], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001, eps=1e-6)
            history = fit_job(model, optimizer, row, args.steps, state, out, dataset, 0)
            state['job_visits'] = [args.job]
            state['completed'] = [dict(pass_index=0, job_key=args.job)]
            result_checkpoint = out / 'model.pt'
            save_checkpoint(result_checkpoint, model, optimizer, dict(state,
                protocol_sha256=digest(out / 'protocol.json'), query_checkpoint_sha256=args.query_sha256))
            evaluations = post_job_result(model, row, result_checkpoint, out)
            write(out / 'history.json', history)
            write(out / 'receipt.json', dict(status='COMPLETE_ACTUAL_POOL_TRAIN_CONTROL_FINISHED_NOT_FULL_DATASET',
                job_key=args.job, actual_optimizer_updates=state['optimizer_steps'],
                scheduled_local_steps=args.steps, complete_input_pool_children=receipt['children'],
                complete_input_pool_parents=receipt['parents'], reference_versions=sorted(expected),
                query_checkpoint_sha256=args.query_sha256, selector_checkpoint_sha256=digest(result_checkpoint),
                protocol_sha256=digest(out / 'protocol.json'), capacity_sha256=digest(out / 'capacity.json'),
                prediction_freeze_sha256=digest(out / 'prediction_freeze.json'),
                training_output_evaluations= evaluations,
                held_or_diagnostic_fit_calls=0, full_registered_fit_or_independent_accuracy=False,
                best_model_promoted=False, full_contract_pass=False))
        atomic_json(out / 'progress.json', dict(status='COMPLETE_ACTUAL_POOL_TRAIN_CONTROL_FINISHED',
            controller_pid=os.getpid(), actual_optimizer_updates=state['optimizer_steps'], unix_time=time.time()))
        del row, model
        gc.collect()
    except BaseException as exc:
        atomic_json(out / 'failure.json', dict(status='CONTROL_FAILED_WITH_ORIGINAL_EVIDENCE_PRESERVED',
            error=str(exc), traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--pool-name', required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--query-sha256', required=True)
    parser.add_argument('--selector-checkpoint', required=True)
    parser.add_argument('--selector-sha256', required=True)
    parser.add_argument('--steps', type=int, default=8)
    main(parser.parse_args())
