"""Score every original point at the closest cross-role sample hypotheses.

This input-only follow-up keeps the proposed proper transforms fixed. It does
not certify physical identity, arbitrary-crop independence, or registration
optimality. Neither model fitting nor weld-reference reading is permitted.
"""
import argparse
import gc
from itertools import combinations
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from tools.qualify_full_dataset_labels import BASE, DATA, ROOT, ply_points
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def directed_distances(points, tree, transform, chunk=65536):
    rotation, translation = transform[:3, :3], transform[:3, 3]
    distances = np.empty(len(points), np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        distances[start:stop] = tree.query(points[start:stop] @ rotation.T + translation)[0]
    return distances


def score(first, second, transform):
    transform = np.asarray(transform, np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('A finite fixed rigid hypothesis is required')
    np.testing.assert_allclose(transform[3], [0., 0., 0., 1.], rtol=0, atol=1.e-10)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1.e-8)
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1., rtol=0, atol=1.e-8):
        raise ValueError('Reflections and scale changes are not proper rigid hypotheses')
    forward = directed_distances(first, cKDTree(second), transform)
    backward = directed_distances(second, cKDTree(first), np.linalg.inv(transform))
    quantiles = [0., .5, .9, .95, .99, 1.]
    thresholds = [.01, .1, .5, 1., 2., 3.]
    return dict(first_points_scored=len(first), second_points_scored=len(second),
        point_multiplicity_preserved=True, transform_first_to_second=transform.tolist(),
        quantile_probabilities=quantiles,
        forward_quantiles_input_units=np.quantile(forward, quantiles).tolist(),
        backward_quantiles_input_units=np.quantile(backward, quantiles).tolist(),
        overlap_distance_thresholds_input_units=thresholds,
        forward_overlap_fractions=[float(np.mean(forward <= value)) for value in thresholds],
        backward_overlap_fractions=[float(np.mean(backward <= value)) for value in thresholds],
        thresholds_are_diagnostics_not_business_acceptance_or_identity_rules=True)


def controls():
    rng = np.random.default_rng(1031)
    first = rng.normal(size=(1024, 3)) * [3., 7., 11.]
    theta = np.deg2rad(37.)
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]]
    transform[:3, 3] = [17., -21., 9.]
    second = np.vstack([first, first[:137]]) @ transform[:3, :3].T + transform[:3, 3]
    exact = score(first, second, transform)
    assert max(exact['forward_quantiles_input_units'] + exact['backward_quantiles_input_units']) < 1.e-10
    cropped = score(first, second[:311], transform)
    assert cropped['backward_overlap_fractions'][0] == 1.
    assert cropped['forward_overlap_fractions'][0] < 1.
    direct = cKDTree(second).query(first @ transform[:3, :3].T + transform[:3, 3])[0]
    np.testing.assert_array_equal(directed_distances(first, cKDTree(second), transform, chunk=17), direct)
    invalid = transform.copy(); invalid[:3, 0] *= -1
    try:
        score(first, second, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Reflection accepted')
    return dict(all_points_and_multiplicity_checked=True, unequal_count_rigid_pair_exact=True,
                crop_overlap_is_directional=True, chunk_boundaries_exact=True, reflection_rejected=True)


def atomic(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def main(args):
    for name in [args.experiment_name, args.sample_audit, args.moments_audit]:
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Independent local audit names are required')
    out, samples, moments = BASE / args.experiment_name, BASE / args.sample_audit, BASE / args.moments_audit
    out.mkdir(exist_ok=False)
    source_sha = digest(__file__)
    started = time.perf_counter()
    try:
        receipt, prior = read(samples / 'receipt.json'), read(moments / 'receipt.json')
        if (receipt['status'] != 'ALL_CROSS_ROLE_COUNT_AGNOSTIC_SAMPLE_REGISTRATIONS_COMPLETE'
                or receipt['checked_pairs'] != prior['checked_pairs']
                or receipt['source_sha256'] != digest(ROOT / 'tools/audit_full_cross_role_input_alignment.py')
                or receipt['protocol_sha256'] != digest(samples / 'protocol.json')):
            raise ValueError('A current complete cross-role sample audit is required')
        bindings = prior['bindings']
        if bindings != {name: digest(BASE / name) for name in bindings}:
            raise ValueError('Original input-role bindings changed')
        metadata = {}
        for relative, sha in prior['individual_input_records'].items():
            path = moments / relative
            if digest(path) != sha:
                raise ValueError('Input provenance changed')
            row = read(path); metadata[row['workpiece_sha256']] = row
        rows = []
        for record in receipt['results']:
            path = samples / record['path']
            if digest(path) != record['sha256']:
                raise ValueError('A sample hypothesis changed')
            rows.append((record['path'], record['sha256'], read(path)))
        plan = []
        roles = sorted({row['role'] for row in metadata.values()})
        for first_role, second_role in combinations(roles, 2):
            candidates = [row for row in rows if set([row[2]['first_role'], row[2]['second_role']]) == {first_role, second_role}]
            for kind, field in [('full_shape', 'bidirectional_p95_input_units'),
                                ('partial_shape', 'smaller_direction_p95_input_units')]:
                chosen = min(candidates, key=lambda row: (row[2][kind][field], row[0]))
                plan.append(dict(role_pair=[first_role, second_role], hypothesis_kind=kind,
                    sample_pair_path=chosen[0], sample_pair_sha256=chosen[1],
                    first=chosen[2]['first'], second=chosen[2]['second'],
                    sample_distance_input_units=chosen[2][kind][field],
                    transform_first_to_second=chosen[2][kind]['transform_first_to_second']))
        protocol = dict(source_sha256=source_sha, controls=controls(),
            sample_receipt_sha256=digest(samples / 'receipt.json'),
            moments_receipt_sha256=digest(moments / 'receipt.json'), role_bindings=bindings,
            fixed_plan=plan, full_points_in_each_selected_direction=True,
            no_transform_refitting=True, ML_fit_calls=0, weld_reference_reads=0,
            only_closest_full_and_partial_sample_hypothesis_per_role_pair=True,
            whole_dataset_identity_or_independence_not_certified=True)
        write(out / 'protocol.json', protocol)
        (out / 'pairs').mkdir()
        results = []
        with threadpool_limits(limits=2):
            for index, item in enumerate(plan):
                points = []
                for key in ['first', 'second']:
                    row = metadata[item[key]]; path = DATA / row['source_path']
                    if digest(path) != row['workpiece_sha256']:
                        raise ValueError('Raw input changed before follow-up')
                    cloud = ply_points(path)
                    if len(cloud) != row['points'] or not np.isfinite(cloud).all():
                        raise ValueError('Every finite original XYZ point is required')
                    if digest(path) != row['workpiece_sha256']:
                        raise ValueError('Raw input changed during reading')
                    points.append(cloud)
                result = {**item, **score(*points, item['transform_first_to_second'])}
                path = out / 'pairs' / f'{index:02d}.json'
                write(path, result)
                results.append(dict(path=str(path.relative_to(out)), sha256=digest(path)))
                del points, cloud
                gc.collect()
                atomic(out / 'progress.json', dict(status='SCORING_ALL_POINTS_AT_FIXED_SELECTED_HYPOTHESES',
                    controller_pid=os.getpid(), completed_hypotheses=len(results), required_hypotheses=len(plan),
                    ML_fit_calls=0, weld_reference_reads=0, unix_time=time.time()))
                print('FULL_POINT_ALIGNMENT_FOLLOWUP', len(results), '/', len(plan), flush=True)
        if source_sha != digest(__file__) or bindings != {name: digest(BASE / name) for name in bindings}:
            raise ValueError('Source or original roles changed')
        write(out / 'receipt.json', dict(status='SELECTED_CROSS_ROLE_FIXED_HYPOTHESES_ALL_POINTS_SCORED',
            source_sha256=source_sha, protocol_sha256=digest(out / 'protocol.json'), results=results,
            hypotheses_scored=len(results), cross_role_sample_pairs_considered=len(rows),
            all_points_in_selected_hypotheses_only=True,
            arbitrary_transforms_crops_resampling_and_dataset_independence_not_certified=True,
            ML_fit_calls=0, weld_reference_reads=0, inference_input_roles_or_frames_changed=False,
            seconds=time.perf_counter()-started))
        print('SELECTED_CROSS_ROLE_FULL_POINT_FOLLOWUP_COMPLETE', len(results), flush=True)
    except Exception:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--sample-audit', required=True)
    parser.add_argument('--moments-audit', required=True)
    main(parser.parse_args())
