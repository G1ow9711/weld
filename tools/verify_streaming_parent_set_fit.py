"""Compare dense and streamed full-set gradients on real frozen input tokens.

Selections and small deterministic parameter perturbations are mathematical
controls, not reference labels or trained checkpoints. No geometry target is
opened and no extraction accuracy is reported.
"""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel
from weld_pipeline.server_streaming_parent_set_fit import backward_complete_hinge
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_predicted_parent_pool import load_gzip
from tools.train_full_predicted_parent_selector import pack_tokens, packed_batch, utility_difference

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(job, pool_name, output_name):
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(1081)
    folder = BASE / pool_name / 'pools' / job
    receipt = read(folder / 'receipt.json')
    assert receipt['geometry_GT_reads'] == 0
    for name, sha in receipt['files'].items():
        assert digest(folder / name) == sha
    pool = dict(parents=load_gzip(folder / 'parent_pool.json.gz'))
    with np.load(folder / 'child_inputs.npz', allow_pickle=False) as arrays:
        proposals = {name: arrays[name] for name in arrays.files}
    data = pack_tokens(pool, proposals)
    with np.load(folder / 'relations.npz', allow_pickle=False) as arrays:
        for name in ('pairs', 'pair_features', 'base_penalties'):
            data[name] = arrays[name]
    pairs, counts = data['pairs'], data['counts']
    data['pair_scale'] = 2 * np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]])
    assert len(counts) >= 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrderedParentSetModel().to(device).train()
    with torch.no_grad():
        # Nonzero heads exercise gradients through the complete encoder,
        # including the geometry columns, not only the initial unary layer.
        model.unary.weight.normal_(0, .001)
        model.unary.bias.fill_(.002)
        model.pair[-1].weight.normal_(0, .001)
        model.pair[-1].bias.fill_(.002)
    predicted = np.arange(320).tolist()
    teacher = np.unique(np.r_[np.arange(160, 512), np.arange(len(counts)-16, len(counts))]).tolist()
    dense, stream = copy.deepcopy(model), copy.deepcopy(model)
    advantage, normalizer, mass = 10000., .3, .37
    loss = torch.relu(utility_difference(dense, data, predicted, teacher)*normalizer + advantage)*mass
    loss.backward()
    control = backward_complete_hinge(stream, data, predicted, teacher,
        advantage, normalizer, mass, packed_batch, parent_batch_size=16, relation_batch_size=128)
    assert control['active'] and control['gradient_terms'] > 3
    assert abs(float(loss.detach())-control['loss']) < .002
    error, checked = 0., 0
    parameters = dict(stream.named_parameters())
    for name, parameter in dense.named_parameters():
        actual = parameters[name].grad
        if parameter.grad is None:
            assert actual is None
            continue
        assert actual is not None, name
        error = max(error, float(abs(parameter.grad-actual).max()))
        torch.testing.assert_close(actual, parameter.grad, atol=3e-6, rtol=3e-5)
        checked += 1
    assert checked > 10 and float(dense.child.weight.grad[:, 128:].abs().sum()) > 0
    first = torch.optim.AdamW(dense.parameters(), lr=.0003, eps=1e-6)
    second = torch.optim.AdamW(stream.parameters(), lr=.0003, eps=1e-6)
    torch.nn.utils.clip_grad_norm_(dense.parameters(), 5., error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(stream.parameters(), 5., error_if_nonfinite=True)
    first.step()
    second.step()
    optimizer_difference = 0.
    for name, parameter in dense.named_parameters():
        optimizer_difference = max(optimizer_difference, float(abs(parameter.detach()-dict(stream.named_parameters())[name].detach()).max()))
        torch.testing.assert_close(parameter, dict(stream.named_parameters())[name], atol=5e-7, rtol=3e-5)
    empty = copy.deepcopy(model)
    empty.zero_grad(set_to_none=True)
    result = backward_complete_hinge(empty, data, [], [], 0., normalizer, mass, packed_batch)
    assert not result['active'] and result['loss'] == 0 and result['gradient_terms'] == 0
    assert all(p.grad is None for p in empty.parameters())
    inactive = copy.deepcopy(model)
    inactive.zero_grad(set_to_none=True)
    with torch.no_grad():
        difference = float(utility_difference(inactive, data, predicted, teacher))
    a, b = (teacher, predicted) if difference > 0 else (predicted, teacher)
    result = backward_complete_hinge(inactive, data, a, b, 0., normalizer, mass,
        packed_batch, parent_batch_size=16, relation_batch_size=128)
    assert not result['active'] and result['loss'] == 0
    assert all(p.grad is None for p in inactive.parameters())
    output = BASE / output_name
    if output.exists():
        raise ValueError('Control records are immutable; choose a new evidence name')
    write(output, dict(status='PASS_DENSE_AND_STREAMED_COMPLETE_SET_GRADIENT_CONTROL_NOT_MODEL_EVALUATION',
        job_key=job, pool_name=pool_name, pool_receipt_sha256=digest(folder/'receipt.json'),
        candidate_parents=len(counts), real_input_token_rows=len(data['flat']),
        device=device, adam_epsilon=1e-6,
        loss_absolute_difference=abs(float(loss.detach())-control['loss']), gradient_max_absolute_difference=error,
        optimizer_max_parameter_difference=optimizer_difference, optimizer_absolute_tolerance=5e-7,
        gradient_parameter_tensors_checked=checked, active_control=control,
        actual_optimizer_steps_per_control_model=1, dense_and_stream_optimizer_updates_match=True,
        empty_and_inactive_hinges_have_no_gradients=True, reference_geometry_reads=0,
        mathematical_control_selections_not_reference_teacher=True,
        source_sha256=digest(__file__), helper_sha256=digest(ROOT/'weld_pipeline/server_streaming_parent_set_fit.py'),
        selector_source_sha256=digest(ROOT/'tools/train_full_predicted_parent_selector.py'),
        seconds=time.perf_counter()-start))
    print('STREAMED_COMPLETE_SET_GRADIENT_CONTROL_PASS', checked, error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--pool-name', default='full_predicted_parent_initial_v1')
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    main(args.job, args.pool_name, args.output_name)
