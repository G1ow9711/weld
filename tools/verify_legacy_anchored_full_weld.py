"""Actual-input initialization and training/runtime-query admission controls.

The control checks the concrete geometric regression identified by the initial
actual-pool diagnostic. It is not extraction accuracy or independent validation.
"""
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_prediction_runtime import carrier_segment
from weld_pipeline.server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from weld_pipeline.server_runtime_aligned_training_samples import RuntimeAlignedTrainingAssignments, fixed_query_positions
from weld_pipeline.sjg_local_runtime_904 import digest, write
from tools.train_full_weld_anchored import load_query_initialization, load_warm_initialization, TrainingTensors, fit_step, SOURCES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    dataset = FullShardDataset(ROOT)
    out = BASE / 'full_weld_anchored_v2'
    out.mkdir(exist_ok=True)
    initial = load_query_initialization()
    warm, path = load_warm_initialization(out, initial)
    model = LegacyAnchoredFullWeldModel().eval()
    model.initialize(warm['state_dict'], initial['state_dict'])
    converter = TrainingTensors(initial['mean'], initial['std'], 'cpu')
    rows = []
    update_checked = False
    source = {name: digest(ROOT / name) for name in SOURCES}
    with threadpool_limits(limits=2):
        for job_key in dataset.ready_jobs():
            if not all((BASE / 'full_task_supervision_v1' / job_key / v['key'] / 'receipt.json').exists()
                       for v in dataset.jobs[job_key]['versions']):
                continue
            inp = dataset.input_case(job_key)
            builder = RuntimeAlignedTrainingAssignments(inp)
            for version in dataset.jobs[job_key]['versions']:
                key = version['key']
                supervision = dataset.supervision(key)
                with np.load(BASE / 'full_task_supervision_v1' / job_key / key / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                    witnesses = {name: arrays[name] for name in arrays.files}
                sample = builder.build(supervision, witnesses, 1031)
                assert len(sample['features']) == supervision.target['original_children']
                for pid, q in zip(sample['profile_ids'], sample['input_query_positions_mm']):
                    assert q in fixed_query_positions(inp, int(pid))
                assert all(np.isfinite(sample[name]).all() for name in ['features', 'geometry'])
                with torch.inference_mode():
                    packed = model.encode_queries(**converter.batch(inp, sample))
                    q = converter.tensor(sample['features'][:, 456])
                    prediction = model.predict(packed, q)
                    previous = model.legacy(converter.features(sample['features']), q)
                    assert torch.equal(prediction['interval_ratios'], previous[0])
                    assert torch.equal(prediction['weld_type'], previous[1])
                    assert torch.equal(prediction['reference_score'], previous[2])
                    world = sample['input_query_xyz'][:, None] + np.einsum('nsi,nji->nsj',
                        prediction['line_points'].numpy().astype(float)*1000, sample['input_query_bases'])
                    errors = []
                    for owner, pid in enumerate(sample['profile_ids']):
                        curve = inp.curve(int(pid))
                        segment = carrier_segment(curve, prediction['interval_ratios'][owner].numpy(),
                                                  bool(prediction['direction'][owner].argmax()))
                        errors.append(float(np.linalg.norm(world[owner]-np.asarray(segment['points'])[[0,-1]],axis=1).max()))
                    assert max(errors, default=0) < .01
                rows.append(dict(version_key=key, children=supervision.target['original_children'],
                                 zero_residual_anchor_max_error_mm=max(errors, default=0),
                                 all_assigned_queries_are_deployed_fixed_queries=True))
                if not update_checked:
                    model.train()
                    before = {name: value.detach().clone() for name, value in model.legacy.state_dict().items()}
                    point_before = model.line_points.weight.detach().clone()
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001)
                    fit = fit_step(model, converter, inp, sample, optimizer, 1.)
                    assert np.isfinite(fit['loss']) and not torch.equal(point_before, model.line_points.weight)
                    assert all(torch.equal(before[name], value) for name, value in model.legacy.state_dict().items())
                    with torch.inference_mode():
                        after = model.predict(model.encode_queries(**converter.batch(inp, sample)), q)
                        assert torch.equal(after['reference_score'], previous[2])
                    # Restore zero residuals for initialization checks on the
                    # remaining actual inputs; this update is not a checkpoint.
                    model.initialize(warm['state_dict'], initial['state_dict'])
                    model.eval()
                    update_checked = True
            print('ANCHORED_QUERY_ADMISSION', job_key, len(rows), flush=True)
    assert rows and update_checked
    write(BASE / 'legacy_anchored_full_weld_validation.json', dict(
        status='PASS_ACTUAL_INPUT_ANCHOR_AND_RUNTIME_QUERY_ADMISSION_NOT_OUTPUT_ACCURACY',
        sources=source, verification_source_sha256=digest(__file__), registry_sha256=dataset.registry_sha256,
        warm_initialization_sha256=digest(path), warm_initialization_optimizer_steps=warm['optimizer_steps'],
        checked_versions=len(rows), checked_children=sum(row['children'] for row in rows), rows=rows,
        legacy_bounds_types_and_scores_exact_initial_parity=True,
        legacy_parameters_and_scores_unchanged_after_actual_update=True,
        zero_residual_line_and_arc_endpoints_preserve_finite_carrier_geometry=True,
        runtime_and_training_fixed_query_assignments_identical=True,
        all_reference_children_retained=True, in_memory_optimizer_updates=1,
        independent_or_full_prediction_evaluations=0, held_or_protected_fit_calls=0,
        seconds=time.perf_counter()-start))
    print('ANCHORED_FULL_WELD_CONTROL_PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()
