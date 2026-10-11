"""Stream complete ordered-parent fitting over every registered TRAIN cloud.

The query checkpoint is immutable. Each complete input pool is frozen before
opening labels, fitted with every label version, recorded and then evicted from
this controller's own rebuildable cache. Immutable per-job checkpoints permit
recovery without keeping every large candidate pool on disk. Training case
results belong to their per-job checkpoints, not the final model.
"""
import argparse
import fcntl
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors, decoded_subset
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set
from weld_pipeline.server_offline_parent_set_comparator import reference_comparator
from weld_pipeline.server_offline_parent_set_reference import evaluate_reference_subset
from weld_pipeline.server_streaming_parent_set_fit import backward_complete_hinge
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_predicted_parent_pool import initialize, prepare, save_gzip
from tools.train_full_predicted_parent_selector import load_job, forward_scores, packed_batch, expected_utility
from tools.record_training_experiment import record as record_experiment, git_head

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCE_NAMES = ['tools/train_full_parent_dataset.py', 'tools/prepare_full_predicted_parent_pool.py',
    'tools/train_full_predicted_parent_selector.py', 'weld_pipeline/server_cached_parent_relation_prior.py',
    'weld_pipeline/server_full_shard_dataset.py', 'weld_pipeline/server_full_prediction_runtime.py',
    'weld_pipeline/server_predicted_parent_pool.py', 'weld_pipeline/server_ordered_parent_set_model.py',
    'weld_pipeline/server_lazy_parent_competition.py', 'weld_pipeline/server_streaming_parent_set_fit.py',
    'weld_pipeline/server_offline_parent_set_reference.py', 'weld_pipeline/server_offline_parent_set_comparator.py',
    'weld_pipeline/server_reference_parent_selection.py', 'weld_pipeline/server_legacy_anchored_full_weld_model.py',
    'weld_pipeline/sjg_parent_set_features_1160.py', 'weld_pipeline/sjg_additive_parent_precision_1254.py',
    'tools/record_training_experiment.py', 'weld_pipeline/contract_curve_geometry.py',
    'weld_pipeline/world_set_runtime_285.py', 'weld_pipeline/world_set_evaluation_285.py',
    'weld_pipeline/sjg_native_d3_adapter_1056.py', 'weld_pipeline/contract_geometry_d3_409.py',
    'weld_pipeline/server_runtime_aligned_training_samples.py', 'weld_pipeline/server_full_weld_model.py',
    'weld_pipeline/server_joint_reference_query_model.py', 'weld_pipeline/server_full_training_samples.py']


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, state):
    if path.exists():
        raise ValueError('Immutable job checkpoint already exists')
    temporary = path.with_name(path.name + '.incomplete')
    torch.save(dict(state, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state()), temporary)
    check = torch.load(temporary, map_location='cpu', weights_only=False)
    assert check['optimizer_steps'] == state['optimizer_steps'] and check['completed'] == state['completed']
    os.replace(temporary, path)


def semantic_fingerprints(folder):
    # gzip container timestamps can change when rebuilding a pool; fingerprint
    # the complete decompressed input prediction contents as well as file bytes.
    result = {}
    for name in ('child_proposals.json.gz', 'parent_pool.json.gz'):
        h = hashlib.sha256()
        with gzip.open(folder/name, 'rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                h.update(block)
        result[name] = h.hexdigest()
    result['child_inputs.npz'] = digest(folder/'child_inputs.npz')
    return result


def progress(out, state, dataset, status, job_key=None):
    atomic_json(out/'training_progress.json', dict(status=status, controller_pid=os.getpid(),
        job_key=job_key, optimizer_steps=state['optimizer_steps'], processed_jobs=len(state['job_visits']),
        processed_reference_versions=len(state['processed_versions']),
        updated_reference_versions=len(state['updated_versions']), passes_completed=state['passes_completed'],
        completed_job_visits=len(state['completed']), required_job_visits=state['passes']*len(dataset.jobs),
        registered_jobs=len(dataset.jobs), registered_versions=len(dataset.versions),
        ready_jobs=len(dataset.ready_jobs()), independent_model_evaluations=0,
        query_full_training_coverage_not_claimed=True, full_model_quality_not_established=True,
        unix_time=time.time()))


def fit_job(model, optimizer, row, steps, state, out, dataset, pass_index):
    history = []
    for local_step in range(steps):
        model.eval()
        unary, penalty = forward_scores(model, row)
        predicted = greedy_nullable_set(unary, row['pairs'], penalty, row['pc'])
        candidates = [predicted]
        for view in row['views']:
            teacher, _ = reference_comparator(view['table'], row['pc'], unary)
            candidates.append(teacher)
        teacher = max(candidates, key=lambda selected: expected_utility(row, selected))
        before, after = expected_utility(row, predicted), expected_utility(row, teacher)
        advantage = after-before
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if advantage > 0:
            row.update(row['lazy_competition'].relations_for_selections(predicted, teacher))
            mass = sum(v['weight'] for v in row['views'])
            normalizer = sum(v['weight']/max(1, v['G']) for v in row['views'])/mass
            gradient = backward_complete_hinge(model, row, predicted, teacher, advantage,
                normalizer, mass*len(dataset.jobs), packed_batch)
            if gradient['active']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                state['optimizer_steps'] += 1
                state['updated_versions'] = sorted(set(state['updated_versions']) | {v['key'] for v in row['views']})
        else:
            gradient = dict(loss=0., active=False, gradient_terms=0, affected_parents=0,
                            affected_relations=0, all_complete_set_terms_used=True)
        event = dict(pass_index=pass_index, job_key=row['job_key'], local_step=local_step+1,
            actual_optimizer_steps=state['optimizer_steps'], prediction_parents=len(predicted),
            comparator_parents=len(teacher), expected_prediction_utility=before,
            expected_comparator_utility=after, complete_gradient=gradient)
        history.append(event)
        progress(out, state, dataset, 'ACTUAL_COMPLETE_REFERENCE_PARENT_OPTIMIZATION', row['job_key'])
        print('ALL_TRAIN_PARENT_FIT', event, flush=True)
    return history


def post_job_result(model, row, checkpoint, folder):
    model.eval()
    unary, penalty = forward_scores(model, row)
    selected = greedy_nullable_set(unary, row['pairs'], penalty, row['pc'])
    save_gzip(folder/'selected_parents.json.gz', [row['pool']['parents'][i] for i in selected])
    freeze = dict(selected_parent_indices=selected, checkpoint_sha256=digest(checkpoint),
        query_pool_receipt_sha256=row['pool_receipt_sha256'], runtime_uses_reference_geometry=False,
        selected_geometry_sha256=digest(folder/'selected_parents.json.gz'),
        unary_sha256=hashlib.sha256(unary.tobytes()).hexdigest(),
        pairs_sha256=hashlib.sha256(row['pairs'].tobytes()).hexdigest(),
        penalty_sha256=hashlib.sha256(penalty.tobytes()).hexdigest(),
        scope='Training output after this job update, not final model or independent evaluation')
    write(folder/'prediction_freeze.json', freeze)
    evaluations = []
    for view in row['views']:
        result = evaluate_reference_subset(view['table'], selected)
        direct = score_native_parents(decoded_subset(row['pool']['parents'], selected), view['target'],
            [False]*len(view['target']['parents']))['categories']['overall']
        if direct['TP'] != result['TP'] or direct['FP'] != result['FP']:
            raise ValueError('Complete prediction subset disagrees with original direct D3')
        evaluations.append(dict(version_key=view['key'], G=view['G'], **result,
            pool_capacity_reference_evaluation=view['pool_capacity_reference_evaluation']))
    return evaluations


def main(name, passes, steps_per_job, selector_checkpoint, selector_sha256):
    if min(passes, steps_per_job) < 1 or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('A local experiment name and complete positive visit budgets are required')
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    dataset = FullShardDataset(ROOT)
    out, storage, query_checkpoint = initialize(name, 'anchored_v2')
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {s: digest(ROOT/s) for s in SOURCE_NAMES}
        query = torch.load(query_checkpoint, map_location='cpu', weights_only=False)
        if 'legacy.trunk.0.weight' not in query['state_dict']:
            raise ValueError('This complete dataset fitter requires the declared anchored query architecture')
        priors = FrozenRun446Priors(ROOT)
        cached = CachedRun446RelationPrior(priors)
        initial = Path(selector_checkpoint).resolve() if selector_checkpoint else None
        if initial is not None:
            initial.relative_to(ROOT)
            if not selector_sha256 or digest(initial) != selector_sha256:
                raise ValueError('Frozen selector initialization fingerprint differs')
        elif selector_sha256:
            raise ValueError('A selector fingerprint requires a checkpoint')
        source_revision = (read(out/'protocol.json')['source_commit_at_initialization']
                           if (out/'protocol.json').exists() else git_head(ROOT))
        protocol = dict(schema='ALL_REGISTERED_TRAIN_COMPLETE_PARENT_STREAM_V1',
            source_commit_at_initialization=source_revision,
            jobs=list(dataset.jobs), versions=sorted(dataset.versions), registry_sha256=dataset.registry_sha256,
            query_checkpoint_sha256=digest(query_checkpoint), query_optimizer_steps=query['optimizer_steps'],
            original_run446_prior_sha256=priors.checkpoint_sha256,
            selector_initialization=str(initial.relative_to(ROOT)) if initial else None,
            selector_initialization_sha256=selector_sha256 if initial else None,
            selector_initialization_is_not_an_accepted_best_model=True,
            optimizer_initialization='Fresh AdamW; resume restores full saved optimizer state',
            passes=passes, steps_per_job=steps_per_job, seed=1031, learning_rate=.0003, adam_epsilon=1e-6,
            fixed_query_fractions=list(FIXED_QUERY_FRACTIONS), query_mode='fixed_and_grid',
            competition_mode='lazy', resource_mode='input_scaled_v1', sources=sources,
            original_reference_geometry_compilation_chunk=1024,
            ordering='Ready inputs by all-profile span count, then key; every registered job required each pass',
            objective='Separate full expected D3 reference utility, family-equal version weights, no physical negatives',
            host_jobs_loaded_simultaneously=1, complete_set_gradient_subsampling=False,
            cache_eviction='Only own frozen pool after durable receipt, result and recoverable checkpoint',
            query_full_training_coverage_not_claimed=True, independent_model_evaluations=0,
            held_or_diagnostic_fit_calls=0, full_model_promotion_not_claimed=True)
        if (out/'protocol.json').exists():
            if read(out/'protocol.json') != protocol:
                raise ValueError('Recorded complete training recipe changed; use a new declared experiment')
        else:
            write(out/'protocol.json', protocol)
        (out/'checkpoints').mkdir(exist_ok=True)
        (out/'job_records').mkdir(exist_ok=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        parent_state = {k.removeprefix('parent_encoder.'): v for k, v in query['state_dict'].items()
                        if k.startswith('parent_encoder.')}
        model = OrderedParentSetModel(parent_state).to(device)
        if initial:
            model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=False)['state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001, eps=1e-6)
        state = dict(passes=passes, optimizer_steps=0, completed=[], job_visits={}, processed_versions=[],
                     updated_versions=[], passes_completed=0, protocol_sha256=digest(out/'protocol.json'))
        if (out/'resume.json').exists():
            binding = read(out/'resume.json')
            path = out/binding['checkpoint']
            if digest(path) != binding['sha256']:
                raise ValueError('Recoverable checkpoint fingerprint changed')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['protocol_sha256'] != state['protocol_sha256']:
                raise ValueError('Recovered state belongs to a different source or dataset recipe')
            model.load_state_dict(saved.pop('state_dict'))
            optimizer.load_state_dict(saved.pop('optimizer'))
            torch.set_rng_state(saved.pop('torch_rng'))
            state = saved
        query_model = LegacyAnchoredFullWeldModel()
        query_model.load_state_dict(query['state_dict'])
        size_cache = {}
        for pass_index in range(passes):
            done = {v.split('/', 1)[1] for v in state['completed'] if v.startswith(str(pass_index)+'/')}
            while len(done) < len(dataset.jobs):
                if any(digest(ROOT/s) != value for s, value in sources.items()):
                    raise ValueError('Live complete training source changed')
                ready = [k for k in dataset.ready_jobs() if k not in done]
                if not ready:
                    progress(out, state, dataset, 'WAITING_REGISTERED_INPUTS_AFTER_ACTUAL_PARENT_FIT')
                    time.sleep(5)
                    continue
                for key in ready:
                    if key not in size_cache:
                        with np.load(dataset.shards/key/'fragment_inputs.npz', allow_pickle=False) as arrays:
                            size_cache[key] = int(arrays['offsets'][-1])
                key = min(ready, key=lambda k: (size_cache[k], k))
                identity = str(pass_index)+'/'+key
                record = out/'job_records'/str(pass_index)/key
                record.mkdir(parents=True, exist_ok=True)
                if (record/'receipt.json').exists():
                    raise ValueError('Unrecovered completed job result exists; inspect its checkpoint before replay')
                try:
                    progress(out, state, dataset, 'PREPARING_COMPLETE_ACTUAL_INPUT_PARENT_POOL', key)
                    prepare(dataset, key, query_model, query, priors, out, 'fixed_and_grid',
                            FIXED_QUERY_FRACTIONS, 'lazy', 'input_scaled_v1')
                    pool_folder = out/'pools'/key
                    receipt = read(pool_folder/'receipt.json')
                    semantic = semantic_fingerprints(pool_folder)
                    previous = out/'job_records/0'/key/'pool_receipt.json'
                    if pass_index and read(previous)['semantic_fingerprints'] != semantic:
                        raise ValueError('Regenerated complete input prediction geometry differs across passes')
                    write(record/'pool_receipt.json', dict(receipt, semantic_fingerprints=semantic))
                    print('ALL_TRAIN_POOL_INPUT_FROZEN', key, 'parents', receipt['parents'], 'children', receipt['children'], flush=True)
                    with threadpool_limits(limits=2):
                        row = load_job(dataset, key, out, cached, reference_chunk_size=1024)
                        state['processed_versions'] = sorted(set(state['processed_versions']) | {v['key'] for v in row['views']})
                        history = fit_job(model, optimizer, row, steps_per_job, state, out, dataset, pass_index)
                        state['completed'].append(identity)
                        state['job_visits'][key] = state['job_visits'].get(key, 0)+1
                        state['passes_completed'] = pass_index + int(len(done)+1 == len(dataset.jobs))
                        checkpoint = out/'checkpoints'/(str(pass_index)+'_'+key+'.pt')
                        save_checkpoint(checkpoint, model, optimizer, state)
                        evaluations = post_job_result(model, row, checkpoint, record)
                    result = dict(status='COMPLETE_REGISTERED_JOB_FIT_RECORDED_NOT_MODEL_PROMOTION',
                        pass_index=pass_index, job_key=key, history=history, training_views=evaluations,
                        checkpoint=str(checkpoint.relative_to(out)), checkpoint_sha256=digest(checkpoint),
                        protocol_sha256=state['protocol_sha256'], pool_receipt_sha256=digest(record/'pool_receipt.json'),
                        reference_versions_processed=[v['key'] for v in row['views']], independent_evaluations=0,
                        original_direct_D3_agreement=True, parent_geometry_builds=cached.parent_geometry_builds,
                        genuine_relation_feature_rows=cached.relation_rows)
                    atomic_json(record/'receipt.json', result)
                    evidence_names = ['protocol.json', str((record/'pool_receipt.json').relative_to(out)),
                                      str((record/'prediction_freeze.json').relative_to(out)),
                                      str((record/'receipt.json').relative_to(out))]
                    record_id = 'all_parent_'+hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+str(pass_index)+'_'+key
                    record_path = ROOT/'experiment_records'/(record_id+'.json')
                    if record_path.exists():
                        old = read(record_path)
                        if old['metadata']['checkpoint_sha256'] != result['checkpoint_sha256']:
                            raise ValueError('Immutable job experiment record differs')
                    else:
                        record_experiment(record_id, result['status'],
                            [str((out/p).relative_to(ROOT)) for p in evidence_names],
                            [str(checkpoint.relative_to(ROOT)), str(query_checkpoint.relative_to(ROOT))],
                            dict(result, protocol_source_commit=source_revision,
                                 per_job_output_not_final_model_accuracy=True))
                    atomic_json(out/'resume.json', dict(checkpoint=result['checkpoint'], sha256=result['checkpoint_sha256']))
                    if pool_folder.resolve().parent != storage or pool_folder.name != key:
                        raise ValueError('Refuse cleanup outside this controller-owned workpiece pool')
                    for filename, value in receipt['files'].items():
                        if digest(pool_folder/filename) != value:
                            raise ValueError('Candidate input changed before own-cache cleanup')
                    shutil.rmtree(pool_folder)
                    cached.clear()
                    del row
                    gc.collect()
                    done.add(key)
                    progress(out, state, dataset, 'ACTUAL_REGISTERED_JOB_FIT_DURABLE_CACHE_EVICTED')
                    print('ALL_TRAIN_PARENT_JOB_COMPLETE', identity, 'updates', state['optimizer_steps'],
                          'versions', len(state['processed_versions']), '/', len(dataset.versions), flush=True)
                except Exception:
                    atomic_json(out/'failure.json', dict(status='INCOMPLETE_NO_TRUNCATED_OUTPUT_OR_PROMOTION',
                        job_key=key, pass_index=pass_index, traceback=traceback.format_exc(),
                        completed_job_visits=len(state['completed']), protocol_sha256=state['protocol_sha256']))
                    raise
        assert set(state['processed_versions']) == set(dataset.versions)
        assert len(state['completed']) == passes*len(dataset.jobs)
        assert all(state['job_visits'][k] == passes for k in dataset.jobs)
        progress(out, state, dataset, 'ALL_REGISTERED_PARENT_FIT_PASSES_COMPLETE_EVALUATION_REMAINING')
        atomic_json(out/'receipt.json', dict(status='ALL_REGISTERED_TRAIN_PARENT_FIT_COMPLETE_NOT_PROMOTED',
            **state, all_registered_jobs_visited_every_pass=True, independent_model_evaluations=0,
            per_job_training_scores_not_aggregated_as_final_model_accuracy=True,
            query_full_training_coverage_not_claimed=True, full_query_checkpoint_refit_and_evaluation_remaining=True,
            original_run446_unchanged=True, held_or_diagnostic_fit_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--steps-per-job', type=int, default=8)
    parser.add_argument('--selector-checkpoint')
    parser.add_argument('--selector-sha256')
    args = parser.parse_args()
    main(args.experiment_name, args.passes, args.steps_per_job, args.selector_checkpoint, args.selector_sha256)
