"""Verify every cached preparation batch against a real completed TRAIN epoch.

The original first epoch contains immutable input-array fingerprints for all
all registered versions and all 64 original seeds. These are the numerical baseline; no
old inference or preparation run is repeated wholesale. One actual full view
also checks live versus memoized model losses and two optimizer updates.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.diagnose_full_query_geometry import atomic
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from tools.train_full_local_query_geometry import LiveLocalTrainingTensors
from tools.train_full_local_query_geometry_mixed import SOURCES as ORIGINAL_SOURCES
from tools.train_full_weld_reference_recall import fit_step
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors
from weld_pipeline.server_local_query_weld_model import LocalQueryWeldModel
from weld_pipeline.server_memoized_training_queries import MemoizedRawQueryNeighborhood, MemoizedTrainingAssignments
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_query_point_neighborhood import RawQueryNeighborhood
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/check_full_train_query_memoization.py',
    'weld_pipeline/server_memoized_training_queries.py', *ORIGINAL_SOURCES]))


def equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a,b in zip(left,right))
    return left == right


def update_parity(inp, xyz, sample, cached, initialization):
    state = torch.load(initialization, map_location='cpu', weights_only=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    old, new = [LocalQueryWeldModel().to(device) for _ in range(2)]
    for model in (old, new):
        model.load_state_dict(state['state_dict']); model.train()
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                   lr=.0001, weight_decay=.0001) for m in (old,new)]
    direct = LiveLocalTrainingTensors(state['mean'], state['std'], device, RawQueryNeighborhood(xyz))
    memo = CachedLocalTrainingTensors(state['mean'], state['std'], device)
    for _ in range(2):
        a = fit_step(old, direct, inp, sample, optimizers[0], 1.)
        b = fit_step(new, memo, cached, sample, optimizers[1], 1.)
        if (a != b or not equal(old.state_dict(), new.state_dict())
                or not equal(optimizers[0].state_dict(), optimizers[1].state_dict())):
            raise ValueError('Actual full-view losses or two parameter updates differ')
    if not all(torch.isfinite(v).all() for v in new.state_dict().values()):
        raise ValueError('Actual control parameters are nonfinite')
    return dict(actual_complete_children=sample['report']['children'], updates_per_model=2,
        actual_control_optimizer_updates=4, all_losses_parameters_and_optimizer_states_bitwise_equal=True)


def main(args):
    torch.set_num_threads(2); torch.manual_seed(1031)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base); dataset.require_complete_inputs()
    original = base / 'full_local_query_geometry_mixed_v4_round02'
    checkpoint = original / 'epoch_000.pt'; checkpoint_sha = digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protocol = read(original / 'protocol.json'); settings = protocol['settings']
    expected = {f'0:{key}' for key in dataset.versions}
    if (set(state['completed_views']) != expected or len(state['completed_views']) != len(expected)
            or state['optimizer_steps'] != len(expected) * 64 or settings['steps_per_view'] != 64
            or state['sources'] != {name:digest(ROOT / name) for name in state['sources']}
            or state['protocol_sha256'] != digest(original / 'protocol.json')):
        raise ValueError('Authentic complete first epoch is required as the numerical baseline')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {name:digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, baseline_checkpoint_sha256=checkpoint_sha,
        training_registry_sha256=dataset.registry_sha256, seed=settings['seed'], original_seeds_per_view=64,
        every_original_cached_array_binding_required=True, scope='TRAIN_PREPARATION_PARITY_AND_CONTROL_ONLY',
        production_optimizer_updates=0, planned_control_optimizer_updates=4, held_reference_reads=0))
    started=time.perf_counter(); totals=Counter(); records=[]; parity=None
    admission=read(base/'local_query_xyz_initialization_control_20261012_v1/receipt.json')
    initialization=ROOT/admission['local_initialization_path']
    if digest(initialization) != admission['local_initialization_sha256']:
        raise ValueError('Actual preserved control initialization changed')
    try:
        with threadpool_limits(limits=2):
            for job_key,job in dataset.jobs.items():
                if sources != {name:digest(ROOT/name) for name in sources}:
                    raise ValueError('Memoization control source changed')
                raw=ROOT.parent/'data/G_Data'/job['workpiece_source']
                if digest(raw) != job['workpiece_sha256']:
                    raise ValueError('Original complete raw input changed')
                xyz=ply_points(raw)
                if len(xyz) != job['original_points']:
                    raise ValueError('Complete input point count changed')
                inp=dataset.input_case(job_key)
                neighborhoods=MemoizedRawQueryNeighborhood(xyz,inp)
                builder=MemoizedTrainingAssignments(inp); shared_tokens={}
                for version in job['versions']:
                    key=version['key']; view=dataset.supervision(key)
                    folder=base/'full_task_supervision_v1'/job_key/key
                    verify(folder,dataset,version)
                    with np.load(folder/'owner_witnesses.npz',allow_pickle=False) as z:
                        witnesses={name:z[name] for name in z.files}
                    old=state['view_evidence']['0:'+key]
                    if (old['reference_children']!=version['children']
                            or old['input_receipt_sha256']!=digest(dataset.shards/job_key/'receipt.json')
                            or old['owner_witness_receipt_sha256']!=digest(folder/'receipt.json')
                            or old['label_receipt_sha256']!=version['label_receipt_sha256']):
                        raise ValueError('Original complete TRAIN view binding changed')
                    for step in range(64):
                        sample=builder.build(view,witnesses,settings['seed']+step)
                        actual=CachedLocalQueryInput(inp,sample,neighborhoods,shared_tokens)
                        actual.verify(sample)
                        if (actual.bindings!=old['cached_input_bindings'][step]
                                or sample['report']!=old['reports'][step]
                                or actual.context_report!=old['local_context_reports'][step]):
                            raise ValueError(f'Actual memoized preparation differs: {key} step {step}')
                        if parity is None:
                            parity=update_parity(inp,xyz,sample,actual,initialization)
                        totals['complete_batches']+=1
                    totals['versions']+=1; totals['children_per_update_round']+=version['children']
                    records.append(dict(version_key=key,verified_complete_original_seed_batches=64,
                        original_children=version['children'],every_input_and_context_binding_bitwise_equal=True))
                    atomic(out/'progress.json',dict(status='VERIFYING_ALL_ACTUAL_TRAIN_MEMOIZED_BATCHES',
                        totals=dict(totals),held_reference_reads=0,elapsed_seconds=time.perf_counter()-started))
                    print('ACTUAL_TRAIN_MEMO_PARITY',key,totals['versions'],'/',len(dataset.versions),flush=True)
                totals.update(input_jobs=1,neighborhood_cache_hits=neighborhoods.cache_hits,
                    actual_neighborhood_computations=neighborhoods.actual_computations,
                    closest_cache_hits=builder.closest_cache_hits,
                    actual_closest_computations=builder.actual_closest_computations)
                del actual,sample,inp,xyz,neighborhoods,builder,shared_tokens,view,witnesses
                gc.collect()
        if (totals['complete_batches']!=len(dataset.versions)*64 or totals['versions']!=len(dataset.versions)
                or totals['children_per_update_round']!=sum(v['children'] for _,v in dataset.versions.values()) or totals['input_jobs']!=len(dataset.jobs)
                or sources!={name:digest(ROOT/name) for name in sources}
                or digest(checkpoint)!=checkpoint_sha):
            raise ValueError('A full original TRAIN batch, raw input or source is missing')
        atomic(out/'receipt.json',dict(status='PASS_ALL_ACTUAL_TRAIN_QUERY_MEMOIZATION_BITWISE_PARITY',
            recorded_utc=datetime.now(timezone.utc).isoformat(),totals=dict(totals),records=records,
            actual_full_view_update_parity=parity,elapsed_seconds=time.perf_counter()-started,
            baseline_checkpoint_sha256=checkpoint_sha,held_reference_reads=0,
            production_optimizer_updates=0,actual_control_optimizer_updates=4,
            model_output_quality_not_evaluated=True,completed_optimization_rounds=0))
    except BaseException:
        atomic(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-name',default='full_dataset_v2')
    parser.add_argument('--experiment-name',required=True)
    main(parser.parse_args())
