"""Register every cross-role input pair without a point-count exclusion.

Input-only proper rigid hypotheses use full-input moments and bounded samples.
All sampled points contribute to final bidirectional distances. Trimmed ICP
only proposes transforms, including partial overlaps. Proximity is a diagnostic
for full-point follow-up, never physical identity or independent-test proof.
The model, input bytes, roles and signed business frames are unchanged.
"""
import argparse
import gc
from itertools import combinations, permutations, product
from pathlib import Path
import os
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from tools.qualify_full_dataset_labels import DATA, ROOT, BASE, ply_points
from weld_pipeline.sjg_local_runtime_904 import read, write, digest


def proper_fit(first, second):
    ac, bc = first.mean(0), second.mean(0)
    u, _, vt = np.linalg.svd((first - ac).T @ (second - bc))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, bc - rotation @ ac


def sampled_row(points, statistics, capacity=2048):
    indices = np.unique(np.linspace(0, len(points) - 1, min(capacity, len(points)), dtype=np.int64))
    sample = np.asarray(points[indices], np.float64)
    center = np.asarray(statistics['center'], np.float64)
    centered = sample - center
    tree = cKDTree(centered)
    distances = tree.query(centered, k=min(2, len(centered)))[0]
    spacing = float(np.median(distances[:, -1])) if len(centered) > 1 else 0.
    return dict(points=centered, center=center, axes=np.asarray(statistics['principal_axes']),
                tree=tree, spacing=spacing, statistics=statistics, sample_indices=indices)


def align(first, second, iterations=24):
    a, b = first['points'], second['points']
    tree = second['tree']
    small = a[np.linspace(0, len(a) - 1, min(256, len(a)), dtype=np.int64)]
    seeds = []
    for permutation in permutations(range(3)):
        for signs in product([-1., 1.], repeat=3):
            rotation = second['axes'][:, permutation] @ np.diag(signs) @ first['axes'].T
            if np.linalg.det(rotation) <= 0:
                continue
            d = tree.query(small @ rotation.T)[0]
            seeds.append((float(np.mean(np.minimum(d, np.quantile(d, .9)))), rotation))
    best_full = best_partial = None

    def assess(rotation, translation, trim, steps):
        nonlocal best_full, best_partial
        moved = a @ rotation.T + translation
        forward = tree.query(moved)[0]
        backward = cKDTree(moved).query(b)[0]
        fq, bq = np.quantile(forward, [0, .5, .9, .95, .99, 1]), np.quantile(backward, [0, .5, .9, .95, .99, 1])
        full, partial = max(fq[3], bq[3]), min(fq[3], bq[3])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = second['center'] + translation - rotation @ first['center']
        result = dict(transform_first_to_second=transform.tolist(),
            forward_quantiles_input_units=fq.tolist(), backward_quantiles_input_units=bq.tolist(),
            bidirectional_p95_input_units=float(full), smaller_direction_p95_input_units=float(partial),
            fit_trim_quantile=trim, fit_iterations=steps,
            all_sample_points_used_in_both_final_directions=True)
        if best_full is None or full < best_full['bidirectional_p95_input_units']:
            best_full = result
        if best_partial is None or partial < best_partial['smaller_direction_p95_input_units']:
            best_partial = result

    for _, seed in sorted(seeds, key=lambda value: value[0])[:2]:
        assess(seed, np.zeros(3), 1., 0)
        for trim in [.9, .5]:
            rotation, translation = seed.copy(), np.zeros(3)
            for step in range(iterations):
                moved = a @ rotation.T + translation
                distances, nearest = tree.query(moved)
                keep = distances <= np.quantile(distances, trim)
                if np.count_nonzero(keep) < 3:
                    break
                change, shift = proper_fit(moved[keep], b[nearest[keep]])
                rotation, translation = change @ rotation, change @ translation + shift
                if np.linalg.norm(change - np.eye(3)) < 1.e-10 and np.linalg.norm(shift) < 1.e-10:
                    break
            assess(rotation, translation, trim, step + 1)
    return dict(full_shape=best_full, partial_shape=best_partial,
        first_sample_points=len(a), second_sample_points=len(b),
        point_counts_not_required_equal=True, scale_or_reflection_not_fitted=True,
        input_only_registration_not_model_fit=True,
        full_point_identity_and_cropped_independence_not_certified=True)


def controls():
    rng = np.random.default_rng(1031)
    original = rng.normal(size=(1024, 3)) * [2., 5., 9.]
    theta = np.deg2rad(37.)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    resampled = np.vstack([original, original[:137]]) @ rotation.T + [17., -21., 9.]

    def stat(points):
        center = points.mean(0)
        axes = np.linalg.eigh((points-center).T @ (points-center) / len(points))[1]
        return dict(center=center.tolist(), principal_axes=axes.tolist())

    first, second = sampled_row(original, stat(original)), sampled_row(resampled, stat(resampled))
    result = align(first, second)
    assert result['first_sample_points'] != result['second_sample_points']
    assert result['full_shape']['bidirectional_p95_input_units'] < 1.e-8
    transform = np.asarray(result['full_shape']['transform_first_to_second'])
    np.testing.assert_allclose(transform[:3, :3], rotation, rtol=0, atol=1.e-8)
    np.testing.assert_allclose(transform[:3, 3], [17., -21., 9.], rtol=0, atol=1.e-8)
    assert np.linalg.det(transform[:3, :3]) > 0
    return dict(different_count_resampled_rotated_input_registered=True,
        proper_rotation_translation_recovered=True, all_sample_points_scored_both_directions=True)


def atomic(path, value):
    pending = path.with_name(path.name + '.incomplete')
    write(pending, value)
    os.replace(pending, path)


def main(args):
    for name in [args.experiment_name, args.moments_audit]:
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Independent local audit names required')
    out, audit = BASE / args.experiment_name, BASE / args.moments_audit
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    source_sha = digest(__file__)
    prior = read(audit / 'receipt.json')
    try:
        if (prior['status'] != 'ALL_FULL_XYZ_CROSS_ROLE_RIGID_NECESSARY_CONDITIONS_CHECKED'
                or prior['sources'] != {n: digest(ROOT / n) for n in prior['sources']}
                or prior['bindings'] != {n: digest(BASE / n) for n in prior['bindings']}):
            raise ValueError('Current complete-input moments and roles required')
        check = controls()
        write(out / 'protocol.json', dict(source_sha256=source_sha,
            prior_moments_audit=args.moments_audit, prior_receipt_sha256=digest(audit / 'receipt.json'),
            role_bindings=prior['bindings'], sample_capacity_per_input=2048,
            all_cross_role_pairs_required=prior['checked_pairs'], point_count_filter=False,
            seed_mode='24_PROPER_SIGNED_PCA_HYPOTHESES_BEST_TWO',
            ICP_trim_quantiles=[.9, .5], ICP_max_steps_per_seed_and_trim=24,
            final_distances_use_all_samples=True, ML_fit_calls=0, weld_reference_reads=0,
            inference_and_business_frames_unchanged=True, controls=check))
        (out / 'pairs').mkdir()
        rows = []
        with threadpool_limits(limits=2):
            for relative, sha in sorted(prior['individual_input_records'].items()):
                path = audit / relative
                if digest(path) != sha:
                    raise ValueError('Full-input moments changed')
                statistics = read(path)
                source = DATA / statistics['source_path']
                if digest(source) != statistics['workpiece_sha256']:
                    raise ValueError('Raw input changed before registration')
                points = ply_points(source)
                if len(points) != statistics['points']:
                    raise ValueError('All original point counts must be retained')
                rows.append(sampled_row(points, statistics))
                if digest(source) != statistics['workpiece_sha256']:
                    raise ValueError('Input changed while its bounded sample was read')
                del points
                gc.collect()
                atomic(out / 'progress.json', dict(status='READING_COMPLETE_INPUT_FOR_BOUNDED_SAMPLES',
                    controller_pid=os.getpid(), completed_inputs=len(rows), required_inputs=prior['all_raw_inputs'],
                    ML_fit_calls=0, weld_reference_reads=0, unix_time=time.time()))
            results = []
            for first, second in combinations(rows, 2):
                a, b = first['statistics'], second['statistics']
                if a['role'] == b['role']:
                    continue
                value = dict(first=a['workpiece_sha256'], second=b['workpiece_sha256'],
                    first_role=a['role'], second_role=b['role'],
                    first_input_points=a['points'], second_input_points=b['points'],
                    first_sample_spacing=first['spacing'], second_sample_spacing=second['spacing'],
                    **align(first, second))
                name = a['workpiece_sha256'] + '_' + b['workpiece_sha256'] + '.json'
                path = out / 'pairs' / name
                write(path, value)
                results.append(dict(path=str(path.relative_to(out)), sha256=digest(path)))
                atomic(out / 'progress.json', dict(status='REGISTERING_EVERY_CROSS_ROLE_INPUT_PAIR_NO_COUNT_FILTER',
                    controller_pid=os.getpid(), completed_pairs=len(results), required_pairs=prior['checked_pairs'],
                    ML_fit_calls=0, weld_reference_reads=0, unix_time=time.time()))
                if len(results) % 32 == 0:
                    print('CROSS_ROLE_INPUT_ALIGNMENT', len(results), '/', prior['checked_pairs'], flush=True)
        if len(results) != prior['checked_pairs'] or source_sha != digest(__file__):
            raise ValueError('All fixed pairs and unchanged algorithm are required')
        if prior['bindings'] != {n: digest(BASE / n) for n in prior['bindings']}:
            raise ValueError('Roles changed during the read-only audit')
        write(out / 'receipt.json', dict(status='ALL_CROSS_ROLE_COUNT_AGNOSTIC_SAMPLE_REGISTRATIONS_COMPLETE',
            all_qualified_inputs=len(rows), qualified_versions=prior['qualified_versions'],
            checked_pairs=len(results), results=results, protocol_sha256=digest(out / 'protocol.json'),
            source_sha256=source_sha, all_model_use_roles_and_counts_kept=True,
            original_input_or_roles_or_model_or_business_frames_changed=False,
            ML_fit_calls=0, weld_reference_reads=0, model_output_quality_scores=0,
            samples_are_not_full_point_identity_proof=True, cropped_or_resampled_independence_not_certified=True,
            candidate_full_point_followup_required=True, seconds=time.perf_counter()-started))
        print('ALL_CROSS_ROLE_COUNT_AGNOSTIC_ALIGNMENT_COMPLETE', len(results), flush=True)
    except Exception:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--moments-audit', required=True)
    main(parser.parse_args())
