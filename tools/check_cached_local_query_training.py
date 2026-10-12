"""Actual complete TRAIN input parity before changing workpiece update order."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from tools.train_full_local_query_geometry import LiveLocalTrainingTensors, SOURCES as ORIGINAL_SOURCES
from tools.train_full_weld_reference_recall import fit_step
from weld_pipeline.server_cached_local_query_training import (
    CachedLocalQueryInput, CachedLocalTrainingTensors)
from weld_pipeline.server_deployed_query_assignments import DeployedGridTrainingAssignments
from weld_pipeline.server_local_query_weld_model import LocalQueryWeldModel
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_query_point_neighborhood import RawQueryNeighborhood
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/check_cached_local_query_training.py',
    'weld_pipeline/server_cached_local_query_training.py',
    *ORIGINAL_SOURCES]))


def main(args):
    torch.set_num_threads(2); torch.manual_seed(1031)
    base = ROOT / 'server_rebuild/full_dataset_v2'
    if Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('An independent local control name is required')
    dataset = PartitionedFullShardDataset(ROOT, base)
    key = args.version_key
    job_key, version = dataset.versions[key]
    job = dataset.jobs[job_key]
    out = base / args.experiment_name
    if out.exists():
        raise ValueError('Use a new independent parity-control directory')
    out.mkdir()
    source_hashes = {name:digest(ROOT / name) for name in SOURCES}
    raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
    if digest(raw) != job['workpiece_sha256']:
        raise ValueError('The registered full XYZ input changed')
    xyz = ply_points(raw)
    if len(xyz) != job['original_points']:
        raise ValueError('Full raw point count differs')
    inp = dataset.input_case(job_key)
    view = dataset.supervision(key)
    folder = base / 'full_task_supervision_v1' / job_key / key
    verify(folder, dataset, version)
    with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
        witnesses = {name:arrays[name] for name in arrays.files}
    sample = DeployedGridTrainingAssignments(inp).build(view, witnesses, 1031)
    neighborhoods = RawQueryNeighborhood(xyz)
    initial_receipt = read(base / 'local_query_xyz_initialization_control_20261012_v1/receipt.json')
    initial_path = ROOT / initial_receipt['local_initialization_path']
    if digest(initial_path) != initial_receipt['local_initialization_sha256']:
        raise ValueError('Original local-query initialization changed')
    state = torch.load(initial_path, map_location='cpu', weights_only=False)
    cached = CachedLocalQueryInput(inp, sample, neighborhoods)
    assert cached.verify(sample)
    live = LiveLocalTrainingTensors(state['mean'], state['std'], 'cpu', neighborhoods)
    frozen = CachedLocalTrainingTensors(state['mean'], state['std'], 'cpu')
    a = live.batch(inp, sample); b = frozen.batch(cached, sample)
    for name in ('features', 'profile_ids', 'geometry', 'query_local_features'):
        assert torch.equal(a[name], b[name]), name
    assert set(a['token_sequences']) == set(b['token_sequences'])
    for pid in a['token_sequences']:
        assert torch.equal(a['token_sequences'][pid], b['token_sequences'][pid])
    model_a = LocalQueryWeldModel(); model_a.load_state_dict(state['state_dict'], strict=True)
    model_b = copy.deepcopy(model_a)
    with torch.no_grad():
        pa = model_a.predict(model_a.encode_queries(**a), live.tensor(sample['features'][:, 456]))
        pb = model_b.predict(model_b.encode_queries(**b), frozen.tensor(sample['features'][:, 456]))
    assert set(pa) == set(pb)
    for name in pa:
        assert torch.equal(pa[name], pb[name]), name
    oa = torch.optim.AdamW([p for p in model_a.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
    ob = torch.optim.AdamW([p for p in model_b.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
    results = []
    mass = version['family_equal_version_weight'] * len(dataset.versions)
    for step in range(2):
        with threadpool_limits(limits=2):
            ra = fit_step(model_a, live, inp, sample, oa, mass)
            rb = fit_step(model_b, frozen, cached, sample, ob, mass)
        assert ra == rb
        assert all(torch.equal(value, model_b.state_dict()[name])
                   for name,value in model_a.state_dict().items())
        results.append(dict(step=step+1, live=ra, cached=rb))
    for name in ('features', 'geometry', 'input_query_positions_mm', 'profile_ids'):
        changed = dict(sample); changed[name] = np.array(sample[name], copy=True)
        changed[name].flat[0] += 1
        try:
            cached.verify(changed)
        except ValueError:
            pass
        else:
            raise ValueError('Changed cached input was accepted: ' + name)
    invalid = dict(sample)
    invalid['input_query_positions_mm'] = np.array(sample['input_query_positions_mm'], copy=True)
    invalid['input_query_positions_mm'][0] += np.sqrt(2)
    try:
        CachedLocalQueryInput(inp, invalid, neighborhoods)
    except ValueError:
        pass
    else:
        raise ValueError('A non-deployed input query was cached')
    shared = {pid:np.array(tokens, copy=True) for pid,tokens in cached.tokens_by_profile.items()}
    shared[next(iter(shared))].flat[0] += 1
    try:
        CachedLocalQueryInput(inp, sample, neighborhoods, shared)
    except ValueError:
        pass
    else:
        raise ValueError('Changed shared input tokens were accepted')
    assert source_hashes == {name:digest(ROOT / name) for name in SOURCES}
    write(out / 'receipt.json', dict(status='PASS_ACTUAL_COMPLETE_TRAIN_CACHED_LOCAL_INPUT_AND_TWO_UPDATE_PARITY',
        version_key=key, job_key=job_key, training_registry_sha256=dataset.registry_sha256,
        actual_raw_xyz_sha256=digest(raw), complete_raw_points=len(xyz), reference_children=view.target['original_children'],
        input_arrays_and_tokens_bitwise_equal=True, all_named_predictions_bitwise_equal=True,
        all_complete_task_losses_and_two_parameter_updates_bitwise_equal=True,
        changed_inputs_rejected=True, non_deployed_queries_rejected=True,
        changed_shared_tokens_rejected=True, cached_input_bindings=cached.bindings,
        original_initialization_sha256=digest(initial_path), sources=source_hashes,
        actual_updates_per_control_arm=2, fit_results=results, held_reference_reads=0,
        controls_are_not_full_training_or_model_output_accuracy=True, unix_time=time.time()))
    print('ACTUAL_COMPLETE_TRAIN_INPUT_CACHE_PARITY_PASS', key, view.target['original_children'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--version-key', required=True)
    main(parser.parse_args())
