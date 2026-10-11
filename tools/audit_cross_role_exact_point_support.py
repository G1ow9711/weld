"""Find complete XYZ-support containment across every model-use role.

Every original target cloud supplies a complete nearest-point tree. A bounded
source sample is only a necessary-condition screen; every source point is
checked before declaring exact support containment. No transform is fitted.
Coordinates and multiplicities remain unchanged. This does not exclude all
rotated or resampled derivatives or certify physical workpiece identity.
"""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from tools.qualify_full_dataset_labels import BASE, DATA, ROOT, ply_points
from tools.audit_full_point_alignment_followup import directed_distances
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def exact_support(points, tree):
    distances = directed_distances(points, tree, np.eye(4))
    return dict(source_points_checked=len(points), maximum_nearest_distance_mm=float(distances.max()),
                all_original_points_have_exact_coordinate_support_in_target=bool(np.all(distances == 0.)),
                exact_match_fraction=float(np.mean(distances == 0.)),
                point_multiplicities_scored=True, multiset_injection_not_claimed=True)


def controls():
    rng = np.random.default_rng(1031)
    source = rng.normal(size=(4097, 3))
    target = np.vstack([source, source[:37], [[100., 100., 100.]]])
    positive = exact_support(source, cKDTree(target))
    assert positive['all_original_points_have_exact_coordinate_support_in_target']
    assert not exact_support(target, cKDTree(source))['all_original_points_have_exact_coordinate_support_in_target']
    indices = np.unique(np.linspace(0, len(source)-1, 2048, dtype=np.int64))
    missing = next(index for index in range(len(source)) if index not in set(indices.tolist()))
    changed = source.copy(); changed[missing] = [-100., -100., -100.]
    assert np.all(cKDTree(target).query(changed[indices])[0] == 0.)
    assert not exact_support(changed, cKDTree(target))['all_original_points_have_exact_coordinate_support_in_target']
    shifted = source + [1.e-10, 0., 0.]
    assert not exact_support(shifted, cKDTree(target))['all_original_points_have_exact_coordinate_support_in_target']
    return dict(unequal_count_complete_support_detected=True, directed_extra_points_rejected=True,
                unseen_sample_counterexample_rejected_by_full_check=True, near_match_not_exact=True)


def atomic(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def main(args):
    for name in [args.experiment_name, args.moments_audit]:
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Independent local audit names required')
    out, prior_dir = BASE / args.experiment_name, BASE / args.moments_audit
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    sources = {name: digest(ROOT / name) for name in ['tools/audit_cross_role_exact_point_support.py',
        'tools/qualify_full_dataset_labels.py', 'tools/audit_full_point_alignment_followup.py']}
    try:
        prior = read(prior_dir / 'receipt.json')
        bindings = prior['bindings']
        if bindings != {name: digest(BASE / name) for name in bindings}:
            raise ValueError('Original model-use roles changed')
        metadata = []
        for relative, sha in sorted(prior['individual_input_records'].items()):
            path = prior_dir / relative
            if digest(path) != sha:
                raise ValueError('Input metadata changed')
            metadata.append(read(path))
        write(out / 'protocol.json', dict(sources=sources, controls=controls(), role_bindings=bindings,
            prior_receipt_sha256=digest(prior_dir / 'receipt.json'),
            every_directed_cross_role_pair_required=2*prior['checked_pairs'],
            sample_capacity_per_source=2048, complete_target_point_tree=True,
            exact_zero_distance_necessary_screen_only=True, all_source_points_checked_for_candidates=True,
            input_units='millimeter', ML_fit_calls=0, weld_reference_reads=0,
            original_input_roles_models_or_signed_frames_changed=False,
            arbitrary_rotated_resampled_identity_or_independence_not_certified=True))

        def load(row):
            path = DATA / row['source_path']
            if digest(path) != row['workpiece_sha256']:
                raise ValueError('Input bytes changed before reading')
            points = ply_points(path)
            if len(points) != row['points'] or digest(path) != row['workpiece_sha256']:
                raise ValueError('Complete unchanged original point cloud required')
            return points

        samples = {}
        with threadpool_limits(limits=2):
            for index, row in enumerate(metadata):
                points = load(row)
                indices = np.unique(np.linspace(0, len(points)-1, min(2048, len(points)), dtype=np.int64))
                samples[row['workpiece_sha256']] = points[indices].copy()
                del points; gc.collect()
                atomic(out / 'progress.json', dict(status='READING_BOUNDED_SOURCE_SAMPLES',
                    controller_pid=os.getpid(), completed_inputs=index+1, required_inputs=len(metadata),
                    ML_fit_calls=0, weld_reference_reads=0, unix_time=time.time()))
            (out / 'targets').mkdir()
            directed_count = confirmed_count = 0
            results = []
            for index, target in enumerate(metadata):
                target_points = load(target); tree = cKDTree(target_points)
                rows = []
                for source in metadata:
                    if source['role'] == target['role']:
                        continue
                    distance = tree.query(samples[source['workpiece_sha256']])[0]
                    candidate = bool(np.all(distance == 0.))
                    row = dict(source_sha256=source['workpiece_sha256'], source_path=source['source_path'],
                        source_role=source['role'], source_points=source['points'],
                        source_sample_points=len(distance), sample_exact_match_fraction=float(np.mean(distance == 0.)),
                        sample_maximum_distance_mm=float(distance.max()), full_check_required=candidate)
                    if candidate:
                        source_points = load(source)
                        row['full_point_check'] = exact_support(source_points, tree)
                        del source_points; gc.collect()
                        confirmed_count += int(row['full_point_check']['all_original_points_have_exact_coordinate_support_in_target'])
                    rows.append(row); directed_count += 1
                path = out / 'targets' / (target['workpiece_sha256'] + '.json')
                write(path, dict(target=target, complete_target_points=len(target_points), directed_sources=rows))
                results.append(dict(path=str(path.relative_to(out)), sha256=digest(path)))
                del tree, target_points; gc.collect()
                atomic(out / 'progress.json', dict(status='CHECKING_EVERY_DIRECTED_CROSS_ROLE_EXACT_SUPPORT',
                    controller_pid=os.getpid(), completed_targets=index+1, required_targets=len(metadata),
                    directed_pairs_checked=directed_count, directed_pairs_required=2*prior['checked_pairs'],
                    confirmed_directed_containments=confirmed_count,
                    ML_fit_calls=0, weld_reference_reads=0, unix_time=time.time()))
                print('EXACT_CROSS_ROLE_SUPPORT', index+1, '/', len(metadata),
                      'PAIRS', directed_count, 'CONFIRMED', confirmed_count, flush=True)
        if directed_count != 2*prior['checked_pairs'] or sources != {name: digest(ROOT / name) for name in sources}:
            raise ValueError('Complete fixed directed pair scope and unchanged sources required')
        if bindings != {name: digest(BASE / name) for name in bindings}:
            raise ValueError('Input-role bindings changed during audit')
        write(out / 'receipt.json', dict(status='ALL_DIRECTED_CROSS_ROLE_EXACT_SUPPORT_CHECKS_COMPLETE',
            sources=sources, protocol_sha256=digest(out / 'protocol.json'), results=results,
            all_inputs=len(metadata), directed_pairs_checked=directed_count,
            confirmed_directed_containments=confirmed_count, ML_fit_calls=0, weld_reference_reads=0,
            arbitrary_rotated_resampled_derivatives_not_excluded=True,
            original_input_roles_models_or_frames_changed=False, seconds=time.perf_counter()-started))
        print('ALL_DIRECTED_CROSS_ROLE_EXACT_SUPPORT_COMPLETE', directed_count, confirmed_count, flush=True)
    except Exception:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--moments-audit', required=True)
    main(parser.parse_args())
