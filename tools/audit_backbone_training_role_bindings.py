"""Bind checkpoint-declared training sources to current registered raw inputs.

The original checkpoint metadata, complete original cohort arrays and complete
registered PLY inputs are read. No fitting or extraction scoring occurs. This
audit does not certify absent external exposure, arbitrary transformed identity
or independent generalization. Missing or ambiguous identities remain explicit.
"""
import argparse
from pathlib import Path
import traceback

import numpy as np
import torch

from tools.qualify_full_dataset_labels import ply_points
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/audit_backbone_training_role_bindings.py',
           'tools/qualify_full_dataset_labels.py',
           'weld_pipeline/sjg_local_runtime_904.py']


def owned(path):
    result = Path(path).resolve(strict=True)
    result.relative_to(ROOT)
    return result


def plain(value):
    if not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('A plain independent name is required')
    return value


def main(args):
    base = ROOT / 'server_rebuild' / plain(args.dataset_name)
    data = Path(args.data_root).resolve(strict=True)
    cohort_path = owned(args.cohort)
    cohort = {str(row['source_key']):row for row in read(cohort_path)['cases']}
    train_path, held_path = [base / name for name in ('training_registry.json', 'evaluation_registry.json')]
    train, held = read(train_path), read(held_path)
    split = base / 'split_manifest.json'
    if train['split_manifest_sha256'] != digest(split):
        raise ValueError('Current training registration changed')
    jobs = [('TRAIN', row) for row in train['cases']] + [('HELD', row) for row in held['cases']]
    checkpoint_paths = [owned(path) for path in args.checkpoint]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError('Checkpoint inputs must be distinct')
    out = ROOT / 'server_rebuild' / plain(args.experiment_name)
    out.mkdir(exist_ok=False)
    bindings = {str(path.relative_to(ROOT)):digest(path) for path in
                [cohort_path, train_path, held_path, split, *checkpoint_paths,
                 *[ROOT / source for source in SOURCES]]}
    write(out / 'protocol.json', dict(bindings=bindings, registered_jobs=len(jobs),
        full_original_and_current_point_arrays_required=True, coordinate_transform_applied=False,
        fit_calls=0, model_predictions_scored=0, independent_generalization_certified=False))
    torch.set_num_threads(1)
    checkpoints, keys = [], set()
    for path in checkpoint_paths:
        state = torch.load(path, map_location='cpu', weights_only=False)
        protocol = state['protocol']
        declared = protocol.get('train', protocol.get('sources'))
        if not isinstance(declared, list) or not declared:
            raise ValueError('Checkpoint must explicitly declare its training sources')
        source_keys = [str(key) for key in declared]
        if len(set(source_keys)) != len(source_keys) or not set(source_keys) <= cohort.keys():
            raise ValueError('Every declared training source needs original cohort metadata')
        keys.update(source_keys)
        checkpoints.append(dict(checkpoint=str(path.relative_to(ROOT)), sha256=digest(path),
            embedded_protocol=protocol, declared_train_sources=source_keys,
            metadata_is_not_complete_execution_or_external_exposure_history=True))
        del state
    rows = []
    for key in sorted(keys):
        original = cohort[key]
        path = owned(cohort_path.parent.parent / original['input_file'].replace('\\', '/'))
        if digest(path) != original['input_sha256']:
            raise ValueError('Original full point array changed')
        xyz = np.load(path, mmap_mode='r', allow_pickle=False)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError('Original cohort must have complete XYZ arrays')
        comparisons = []
        for role, job in jobs:
            if job['original_points'] != len(xyz):
                continue
            source = (data / job['workpiece_source']).resolve(strict=True)
            source.relative_to(data)
            if digest(source) != job['workpiece_sha256']:
                raise ValueError('Registered raw PLY changed')
            current = ply_points(source)
            if current.shape != xyz.shape:
                raise ValueError('Registered original point count differs from raw PLY')
            exact, maximum = True, 0.0
            for start in range(0, len(xyz), 100000):
                a, b = xyz[start:start+100000], current[start:start+100000]
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Original or registered XYZ has nonfinite values')
                exact = exact and np.array_equal(a, b)
                maximum = max(maximum, float(np.max(np.abs(a-b))))
            if digest(source) != job['workpiece_sha256']:
                raise ValueError('Raw PLY changed during complete comparison')
            comparisons.append(dict(job_key=job['job_key'], current_role=role,
                workpiece_source=job['workpiece_source'], workpiece_sha256=job['workpiece_sha256'],
                versions=[dict(key=v['key'], family=v['family_key'], role=v.get('role', 'TRAIN'))
                          for v in job['versions']], complete_same_order_numeric_xyz_exact=bool(exact),
                complete_same_order_maximum_coordinate_difference_mm=maximum))
            del current
        matches = [row for row in comparisons if row['complete_same_order_numeric_xyz_exact']]
        row = dict(original_source=key, original_input_sha256=original['input_sha256'],
            complete_original_points=len(xyz), every_same_count_registered_input_compared=True,
            comparisons=comparisons, exact_matches=matches,
            mapping_status='EXACT_UNIQUE_CURRENT_TRAIN_SOURCE' if len(matches)==1 and matches[0]['current_role']=='TRAIN'
                else 'KNOWN_HELD_EXPOSURE' if any(r['current_role']=='HELD' for r in matches)
                else 'UNRESOLVED_OR_AMBIGUOUS_INPUT_IDENTITY',
            no_match_does_not_prove_absence_of_rotated_cropped_or_reordered_derivatives=True)
        rows.append(row)
        write(out / (plain(key) + '_assessment.json'), row)
        print('ACTUAL_CHECKPOINT_SOURCE_ROLE', key, row['mapping_status'], flush=True)
        xyz._mmap.close()
    if bindings != {name:digest(ROOT / name) for name in bindings}:
        raise ValueError('Registered checkpoint role audit bindings changed')
    write(out / 'receipt.json', dict(status='DECLARED_BACKBONE_TRAINING_SOURCE_ROLE_AUDIT_COMPLETE',
        protocol_sha256=digest(out / 'protocol.json'), checkpoints=checkpoints, sources=rows,
        declared_train_sources=len(keys), full_original_points=sum(r['complete_original_points'] for r in rows),
        exact_unique_current_train_sources=sum(r['mapping_status']=='EXACT_UNIQUE_CURRENT_TRAIN_SOURCE' for r in rows),
        known_held_exposure_sources=sum(r['mapping_status']=='KNOWN_HELD_EXPOSURE' for r in rows),
        unresolved_or_ambiguous_sources=sum(r['mapping_status']=='UNRESOLVED_OR_AMBIGUOUS_INPUT_IDENTITY' for r in rows),
        missing_older_training_data_manifests_or_external_history_not_reconstructed=True,
        fit_calls=0, model_predictions_scored=0, labels_or_registered_roles_modified=False,
        independent_generalization_certified=False, full_contract_pass=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException:
        out = ROOT / 'server_rebuild' / plain(args.experiment_name)
        if out.is_dir():
            write(out / 'failure.json', dict(traceback=traceback.format_exc(), fit_calls=0))
        raise
