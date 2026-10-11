"""Verify the pure runtime on a complete actual frozen TRAIN pool.

No label is opened. Real fitted checkpoint scores, all input tokens and the
complete selection are compared with the original consumer and saved control.
These numerical controls are not a held evaluation or extraction accuracy.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset
from weld_pipeline.server_parent_prediction_runtime import load_input_pool, load_selector, pack_tokens, forward_scores, predict
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.train_full_predicted_parent_selector import pack_tokens as original_pack, forward_scores as original_forward
from tools.evaluate_full_weld_predictions import complete_train_fit

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


class InputOnlyProxy:
    def __init__(self, dataset):
        self.root, self.shards = dataset.root, dataset.shards
        self.jobs, self.registry_sha256 = dataset.jobs, dataset.registry_sha256
        self.reference_calls = 0

    def supervision(self, *args, **kwargs):
        self.reference_calls += 1
        raise AssertionError('A pure runtime cannot call training supervision')

    def reference(self, *args, **kwargs):
        self.reference_calls += 1
        raise AssertionError('A pure runtime cannot call held references')


def main(pool_name, selector_name, job_key, output_name):
    start = time.perf_counter()
    torch.set_num_threads(2)
    dataset = FullShardDataset(ROOT)
    held = EvaluationShardDataset(ROOT)
    proxy = InputOnlyProxy(dataset)
    pool_folder, selector_folder = BASE / pool_name, BASE / selector_name
    checkpoint = selector_folder / 'model.pt'
    receipt = read(selector_folder / 'receipt.json')
    query_sha = digest(pool_folder / 'query_checkpoint.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, state, protocol = load_selector(checkpoint, receipt['checkpoint_sha256'],
        selector_folder / 'protocol.json', query_sha, device)
    # Deny actual qualification reference directories while making predictions.
    # Metadata registration is already loaded; neither algorithm needs labels.
    labels = {str((ROOT / v['label_directory']).resolve()) for j in dataset.jobs.values() for v in j['versions']}
    labels.update(str((ROOT / v['label_directory']).resolve()) for j in held.jobs.values() for v in j['versions'])
    denied = []

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            name = str(Path(args[0]).resolve())
            if any(name == p or name.startswith(p + '/') for p in labels):
                denied.append(name)
                raise AssertionError('Pure prediction attempted to open a qualified reference')

    sys.addaudithook(audit)
    data = load_input_pool(proxy, job_key, pool_folder)
    proposals_file = pool_folder / 'pools' / job_key / 'child_inputs.npz'
    with np.load(proposals_file, allow_pickle=False) as arrays:
        proposals = {name: arrays[name] for name in arrays.files}
    original_tokens = original_pack(data['pool'], proposals)
    for name in ('flat', 'offsets', 'counts'):
        if not np.array_equal(data[name], original_tokens[name]):
            raise AssertionError('Complete parent tokens differ from the TRAIN consumer')
    old_unary, old_penalty = original_forward(model, data)
    unary, penalty = forward_scores(model, data)
    if not np.array_equal(unary, old_unary) or not np.array_equal(penalty, old_penalty):
        raise AssertionError('Complete runtime score arrays differ from the existing consumer')
    with np.load(selector_folder / (job_key + '_scores.npz'), allow_pickle=False) as saved:
        if not np.array_equal(unary, saved['unary']) or not np.array_equal(penalty, saved['penalty']):
            raise AssertionError('Reloaded fitted model differs from its actual saved full scores')
    parents, decoded, freeze = predict(model, data)
    selected = greedy_nullable_set(old_unary, data['pairs'], old_penalty, data['pc'])
    if freeze['selected_parent_indices'] != list(map(int, selected)):
        raise AssertionError('Complete nullable whole-parent selection changed')
    original_freeze = read(selector_folder / (job_key + '_prediction_freeze.json'))
    if original_freeze['selected_parent_indices'] != freeze['selected_parent_indices']:
        raise AssertionError('Actual trained control selection changed')
    if decoded['total_prediction_children'] != sum(p['child_count'] for p in parents):
        raise AssertionError('Decoded output lost a selected child')
    rejected_mismatch = False
    try:
        load_selector(checkpoint, receipt['checkpoint_sha256'], selector_folder / 'protocol.json', '0' * 64, device)
    except ValueError:
        rejected_mismatch = True
    if not rejected_mismatch or denied or proxy.reference_calls:
        raise AssertionError('Query binding or reference separation was not enforced')
    # The actual partial control cannot pass a full-TRAIN fitting requirement.
    partial_rejected = False
    try:
        query = torch.load(pool_folder / 'query_checkpoint.pt', map_location='cpu', weights_only=False)
        complete_train_fit(query, state, query_sha, receipt['checkpoint_sha256'], selector_folder / 'receipt.json',
                           selector_folder / 'receipt.json', protocol)
    except (ValueError, KeyError):
        partial_rejected = True
    if not partial_rejected:
        raise AssertionError('A partial trained control was accepted as full TRAIN coverage')
    result = dict(status='PASS_ACTUAL_COMPLETE_POOL_RUNTIME_PARITY_NOT_MODEL_ACCURACY',
        pool_name=pool_name, selector_name=selector_name, job_key=job_key, device=device,
        input_parents=len(data['counts']), input_children=data['receipt']['children'],
        complete_parent_token_rows=len(data['flat']), competition_pairs=len(data['pairs']),
        selected_parents=len(parents), prediction_children=decoded['total_prediction_children'],
        all_parent_tokens_bitwise_equal=True, all_unary_and_penalty_scores_bitwise_equal=True,
        saved_actual_fitted_scores_bitwise_equal=True, whole_selected_parent_indices_equal=True,
        actual_control_output_empty=len(parents) == 0, nonempty_output_accuracy_not_claimed=True,
        mismatched_query_checkpoint_rejected=True, partial_fit_not_accepted_as_full_fit=True,
        qualified_reference_file_opens=0, reference_API_calls=proxy.reference_calls, optimizer_updates=0,
        query_checkpoint_sha256=query_sha, selector_checkpoint_sha256=digest(checkpoint),
        pool_receipt_sha256=data['pool_receipt_sha256'],
        unary_sha256=hashlib.sha256(unary.tobytes()).hexdigest(),
        penalty_sha256=hashlib.sha256(penalty.tobytes()).hexdigest(),
        sources={name: digest(ROOT / name) for name in ['tools/verify_full_parent_prediction_runtime.py',
            'weld_pipeline/server_parent_prediction_runtime.py', 'tools/evaluate_full_weld_predictions.py']},
        independent_model_evaluations=0, full_contract_pass=False, seconds=time.perf_counter() - start)
    destination = BASE / output_name
    if destination.exists():
        raise ValueError('Numerical control evidence is immutable; choose a new output name')
    write(destination, result)
    print('PURE_FULL_PARENT_PREDICTION_CONTROL_PASS', len(data['counts']), len(data['pairs']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-name', default='full_predicted_parent_anchored_initial_v2')
    parser.add_argument('--selector-name', default='full_predicted_parent_selector_anchored_control_v2')
    parser.add_argument('--job-key', default='cloud_cddce4b885b57063')
    parser.add_argument('--output-name', default='full_parent_prediction_runtime_validation_v1.json')
    args = parser.parse_args()
    main(args.pool_name, args.selector_name, args.job_key, args.output_name)
