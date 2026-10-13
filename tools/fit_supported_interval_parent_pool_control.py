"""Fit and export a complete actual supported-input parent set on TRAIN only.

Uses the verified unchanged complete D3 comparator and nullable set objective.
Every registered label version of the chosen workpiece is retained. This is a
control of the whole output path, not full-dataset fitting or held accuracy.
"""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from tools.export_full_dataset_split_view import local_name
from tools.reciprocal_relation_training_backend import configure
from tools.train_full_predicted_parent_selector import forward_scores, packed_batch, expected_utility
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel
from weld_pipeline.server_reciprocal_cached_relation_prior import ReciprocalCachedRun446RelationPrior
from weld_pipeline.server_offline_parent_set_comparator import reference_comparator
from weld_pipeline.server_offline_parent_set_reference import evaluate_reference_subset
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set
from weld_pipeline.server_streaming_parent_set_fit import backward_complete_hinge
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.server_full_weld_csv_export import export_decoded, verify_roundtrip
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import ROOT, read, write, digest


def run(args):
    torch.set_num_threads(2); torch.manual_seed(1031)
    base = ROOT / 'server_rebuild/full_dataset_v2'; old = ROOT / 'server_rebuild/full_dataset_v1'
    pools = base / local_name(args.pool_name); out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, base)
    key = args.job_key
    if key not in dataset.jobs: raise ValueError('Only a registered TRAIN workpiece can be fitted')
    receipt = read(pools / 'pools' / key / 'receipt.json')
    if not receipt['complete_first_TRAIN_pass'] and not args.partial_training_control:
        raise ValueError('Partial interval optimization cannot be admitted as complete TRAIN fitting')
    loader, guards, backend = configure(
        [old/'compact_endpoint_actual_full_pool_20261011_v2', old/'compact_endpoint_actual_snapshots_20261011_v2'],
        old/'compact_endpoint_reference_controls_20261011_v2.json',
        old/'bounded_relation_geometry_controls_20261011_v1/receipt.json',
        old/'bounded_relation_competition_controls_20261011_v1/receipt.json',
        old/'bounded_relation_actual_fit_controls_20261011_v1/receipt.json',
        old/'reciprocal_relation_controls_20261011_v1/receipt.json',
        old/'reciprocal_relation_actual_fit_controls_20261011_v1/receipt.json', out)
    protocol = dict(query_checkpoint_sha256=receipt['query_checkpoint_sha256'], job_key=key,
        registry_sha256=dataset.registry_sha256, complete_input_pool_receipt_sha256=digest(pools/'pools'/key/'receipt.json'),
        source_sha256=digest(__file__), backend=backend, guards={g:digest(ROOT/g) for g in guards},
        steps=args.steps, device=args.device, learning_rate=.0003, seed=1031,
        held_or_diagnostic_fit_calls=0, full_dataset_fit_claimed=False,
        complete_selected_output_and_csv_required=True, training_case_accuracy_is_not_independent=True)
    write(out/'protocol.json', protocol)
    checkpoint = pools/'query_checkpoint.pt'
    if digest(checkpoint) != protocol['query_checkpoint_sha256']: raise ValueError('Frozen interval model changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    parent_state = {k.removeprefix('query.parent_encoder.'):v for k,v in state['state_dict'].items()
                    if k.startswith('query.parent_encoder.')}
    if not parent_state: raise ValueError('Complete interval parent encoder missing; no random replacement')
    model = OrderedParentSetModel(parent_state).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001, eps=1e-6)
    priors = ReciprocalCachedRun446RelationPrior(ROOT)
    with threadpool_limits(limits=2): data = loader(dataset, key, pools, priors, reference_chunk_size=1024)
    if {v['key'] for v in data['views']} != {v['key'] for v in dataset.jobs[key]['versions']}:
        raise ValueError('A complete label version disappeared')
    history = []; updates = 0
    for step in range(args.steps):
        model.eval()
        with threadpool_limits(limits=2): unary, penalties = forward_scores(model, data)
        predicted = greedy_nullable_set(unary, data['pairs'], penalties, data['pc'])
        alternatives = [predicted]
        for view in data['views']:
            alternatives.append(reference_comparator(view['table'], data['pc'], unary)[0])
        teacher = max(alternatives, key=lambda ids: expected_utility(data, ids))
        advantage = expected_utility(data, teacher) - expected_utility(data, predicted)
        model.train(); optimizer.zero_grad(set_to_none=True)
        if advantage > 0:
            with threadpool_limits(limits=2):
                data.update(data['lazy_competition'].relations_for_selections(predicted, teacher))
            mass = sum(v['weight'] for v in data['views'])
            normalizer = sum(v['weight']/max(1,v['G']) for v in data['views'])/mass
            gradient = backward_complete_hinge(model, data, predicted, teacher, advantage, normalizer, 1., packed_batch)
            if gradient['active']:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();updates += 1
        else:
            gradient = dict(loss=0.,active=False,all_complete_set_terms_used=True)
        row = dict(step=step+1, parameter_updates=updates, predicted_parents=len(predicted),
            comparator_parents=len(teacher), prediction_utility=expected_utility(data,predicted),
            comparator_utility=expected_utility(data,teacher), gradient=gradient)
        history.append(row);write(out/'progress.json',row);print('SUPPORTED_COMPLETE_PARENT_FIT',row,flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    protocol_sha256=digest(out/'protocol.json')),out/'model.pt')
    model.eval()
    with threadpool_limits(limits=2): unary, penalty = forward_scores(model,data)
    selected = list(map(int,greedy_nullable_set(unary,data['pairs'],penalty,data['pc'])))
    decoded = decoded_subset(data['pool']['parents'],selected)
    # Freeze actual prediction before its training-case score is computed.
    write(out/'prediction_freeze.json',dict(selected_parent_indices=selected,checkpoint_sha256=digest(out/'model.pt'),
        query_checkpoint_sha256=protocol['query_checkpoint_sha256'],pool_receipt_sha256=data['pool_receipt_sha256'],
        selection_uses_reference=False,total_prediction_children=decoded['total_prediction_children']))
    write(out/'decoded.json',decoded)
    scores = []
    for view in data['views']:
        compact = evaluate_reference_subset(view['table'],selected)
        direct = score_native_parents(decoded,view['target'],[False]*len(view['target']['parents']))['categories']['overall']
        if direct['TP'] != compact['TP'] or direct['FP'] != compact['FP']:
            raise ValueError('Complete selection differs from original direct geometry scorer')
        scores.append(dict(version_key=view['key'],G=view['G'],**compact))
    # Fail before starting an export that would exhaust this task's storage.
    if shutil.disk_usage(out).free < 1024**3 + decoded['total_prediction_children'] * 65536:
        raise RuntimeError('Insufficient space for complete curve/frame CSV export; no truncated export accepted')
    schema = export_decoded(out/'csv',key,decoded);roundtrip=verify_roundtrip(out/'csv',decoded)
    write(out/'receipt.json',dict(status='PARTIAL_TRAIN_COMPLETE_INPUT_SET_FIT_OUTPUT_AND_CSV_CONTROL_NOT_PROMOTED',
        input_children=receipt['children'],input_parents=receipt['parents'],selected_parents=len(selected),
        selected_children=decoded['total_prediction_children'],parameter_updates=updates,history=history,
        all_registered_workpiece_versions_retained=True,training_case_scores=scores,
        csv=schema,csv_roundtrip=roundtrip,checkpoint_sha256=digest(out/'model.pt'),
        full_dataset_fit=False,independent_evaluation=False,held_or_test_reads=0,full_contract_pass=False))
    print('SUPPORTED_COMPLETE_PARENT_OUTPUT_CONTROL',len(selected),decoded['total_prediction_children'],scores,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool-name',required=True);p.add_argument('--experiment-name',required=True)
    p.add_argument('--job-key',required=True);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--partial-training-control',action='store_true')
    a=p.parse_args()
    if a.steps < 1:p.error('Positive actual optimization steps required')
    run(a)
