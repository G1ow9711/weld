"""Real TRAIN admission, exact warm initialization and unknown-mask controls."""
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_training_samples import FullTrainingAssignments
from weld_pipeline.server_full_weld_model import FullWeldModel, complete_child_losses
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import digest, write
from tools.prepare_full_task_supervision import verify as verify_supervision
from tools.train_full_weld_warmstart import load_query_initialization, TrainingTensors, fit_step

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    dataset = FullShardDataset(ROOT)
    ready = list(dataset.ready_jobs())
    checked, totals = [], dict(children=0, parents=0, strong=0, weak=0, known_attributes=0, known_frames=0)
    initial = load_query_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FullWeldModel().to(device)
    model.inherit_query(initial['state_dict'])
    old = JointReferenceModel().to(device)
    old.load_state_dict(initial['state_dict'])
    converter = TrainingTensors(initial['mean'], initial['std'], device)
    parity = False
    mask_control = False
    update_control = False
    for job_key in ready:
        versions = dataset.jobs[job_key]['versions']
        if not all((BASE / 'full_task_supervision_v1' / job_key / v['key'] / 'receipt.json').exists() for v in versions):
            continue
        inp = dataset.input_case(job_key)
        builder = FullTrainingAssignments(inp)
        for version in versions:
            folder = BASE / 'full_task_supervision_v1' / job_key / version['key']
            verify_supervision(folder, dataset, version)
            supervision = dataset.supervision(version['key'])
            with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                witnesses = {name: arrays[name] for name in arrays.files}
            with threadpool_limits(limits=2):
                sample = builder.build(supervision, witnesses, 1031)
            targets = sample['targets']
            assert len(sample['features']) == supervision.target['original_children']
            assert all(np.isfinite(sample[name]).all() for name in ['features', 'geometry', 'input_query_positions_mm', 'input_query_xyz', 'input_query_bases'])
            assert np.all(np.isin(targets['attribute_state'][targets['state_known']], [0, 1, 2, 3]))
            assert np.isnan(targets['attribute_mm'][~targets['numeric_known']]).all()
            assert np.isfinite(targets['attribute_mm'][targets['numeric_known']]).all()
            assert np.array_equal(np.concatenate([np.asarray(v) for v in sample['parent_children']]),
                                  np.arange(supervision.target['original_children']))
            # Free endpoints include every owner, including missed finite carriers.
            world = sample['input_query_xyz'][:, None, :] + np.einsum('nsi,nji->nsj',
                        targets['line_points'].astype(float) * 1000, sample['input_query_bases'])
            originals = []
            for parent in supervision.parents:
                for child in parent['children']:
                    if child['original_kind'] == 'line':
                        originals.append(child['representation']['endpoints_mm'])
                    else:
                        from weld_pipeline.contract_curve_geometry import CircularArc
                        arc = CircularArc.from_parameters(child['representation']['arc_parameters'])
                        originals.append(arc.point_at(np.array([0., arc.length_mm])))
            point_error = float(np.abs(world - np.asarray(originals)).max())
            assert point_error < .01
            axes_world = targets['frame_axes'].astype(float) @ sample['input_query_bases'][:, None].transpose(0, 1, 3, 2)
            known = targets['frame_known']
            assert np.allclose(axes_world[known], supervision.raw_export_axes[known], atol=1e-6, rtol=0)
            if not parity:
                model.eval()
                with torch.no_grad():
                    batch = converter.batch(inp, sample)
                    hidden = model.encode_queries(**batch)
                    q = converter.tensor(sample['features'][:, 456])
                    prediction = model.predict(hidden, q)
                    previous = old(batch['features'], q)
                    assert torch.equal(prediction['interval_ratios'], previous[0])
                    assert torch.equal(prediction['weld_type'], previous[1])
                    assert torch.equal(prediction['reference_score'], previous[2])
                    safe = {name: converter.tensor(v) for name, v in targets.items()}
                    safe['state_known'] = torch.zeros_like(safe['state_known'])
                    safe['numeric_known'] = torch.zeros_like(safe['numeric_known'])
                    safe['frame_known'] = torch.zeros_like(safe['frame_known'])
                    safe['attribute_state'].fill_(-1)
                    safe['attribute_mm'].fill_(float('nan'))
                    safe['frame_axes'].fill_(float('nan'))
                    safe['frame_offsets'].fill_(float('nan'))
                    loss = complete_child_losses(prediction, safe)
                    for name in ['attribute_state', 'attribute_mm', 'frame_axes', 'frame_offsets']:
                        assert torch.isfinite(loss[name]) and float(loss[name]) == 0
                parity = mask_control = True
                model.train()
                before = model.line_points.weight.detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
                result = fit_step(model, converter, inp, sample, optimizer, 1.)
                assert np.isfinite(result['loss']) and not torch.equal(before, model.line_points.weight)
                update_control = True
            report = sample['report']
            totals['children'] += report['children']
            totals['parents'] += report['parents']
            totals['strong'] += report['strong_carrier_assignments']
            totals['weak'] += report['weak_free_geometry_assignments']
            totals['known_attributes'] += int((targets['state_known'] & targets['attribute_applicable']).sum())
            totals['known_frames'] += int(known.sum())
            checked.append(dict(version_key=version['key'], job_key=job_key,
                                geometry_target_roundtrip_max_mm=point_error, **report))
        print('FULL_WARMSTART_ADMISSION_CHECKED', job_key, len(checked), flush=True)
        del inp, builder
    assert checked and parity and mask_control and update_control
    write(BASE / 'full_weld_warmstart_validation.json', dict(
          status='PASS_CURRENT_READY_TRAIN_ADMISSION_AND_IN_MEMORY_FIT_NOT_MODEL_ACCURACY',
          registered_versions=len(dataset.versions), checked_versions=len(checked), totals=totals,
          rows=checked, legacy_query_initialization_exact_parity=True,
          entirely_unknown_process_and_frame_labels_have_exact_zero_loss=True,
          real_in_memory_optimizer_update_control=True, in_memory_fit_calls=1,
          held_or_protected_fit_calls=0, full_prediction_evaluations=0,
          source_sha256=digest(__file__), model_sha256=digest(ROOT / 'weld_pipeline/server_full_weld_model.py'),
          assignment_sha256=digest(ROOT / 'weld_pipeline/server_full_training_samples.py'),
          trainer_sha256=digest(ROOT / 'tools/train_full_weld_warmstart.py'),
          registry_sha256=dataset.registry_sha256, seconds=time.perf_counter() - start))
    print('FULL_WELD_WARMSTART_CHECK_PASS', totals, flush=True)


if __name__ == '__main__':
    main()
