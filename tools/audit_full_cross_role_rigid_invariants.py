"""Exhaustive input-only necessary-condition screening for rigid duplicates.

Every qualified workpiece contributes all XYZ points and their multiplicity.
Rigid near-bijections preserve count and covariance eigenvalues within the
declared perturbation bound. Passing this screen is only a candidate: it does
not prove a registration, physical identity, crop independence or no exposure.
No weld references, model predictions or business frames are read or changed.
"""
import argparse
from collections import defaultdict
import gc
from itertools import combinations
from pathlib import Path
import os
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from tools.qualify_full_dataset_labels import DATA, ROOT, BASE, ply_points
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def moments(points):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Every finite XYZ point is required')
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / len(points)
    eigenvalues, axes = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all():
        raise ValueError('Nonfinite full-input covariance')
    radius = float(np.sqrt(np.maximum(eigenvalues, 0).sum()))
    return dict(points=len(points), center=center.tolist(), covariance=covariance.tolist(),
        covariance_eigenvalues=eigenvalues.tolist(), principal_axes=axes.tolist(),
        rms_radius=radius, max_absolute_coordinates=np.abs(points).max(axis=0).tolist())


def necessary_condition(first, second, relative_tolerance=1.e-6):
    radius = max(first['rms_radius'], second['rms_radius'])
    tolerance = max(1.e-6, relative_tolerance * radius)
    # Under a near-bijection with point error <= tolerance, the centered error
    # is at most 2*tolerance. Cauchy-Schwarz bounds covariance perturbation by
    # 4*RMS_radius*tolerance + 4*tolerance**2; Weyl then bounds each eigenvalue.
    perturbation = 4 * radius * tolerance + 4 * tolerance**2
    n = max(first['points'], second['points'])
    coordinate_scale = max(sum(x*x for x in row['max_absolute_coordinates'])
                           for row in [first, second])
    rounding = 64 * np.finfo(np.float64).eps * n * max(coordinate_scale, radius**2)
    difference = float(np.max(np.abs(np.asarray(first['covariance_eigenvalues'])
                                    - np.asarray(second['covariance_eigenvalues']))))
    same_count = first['points'] == second['points']
    return dict(candidate=bool(same_count and difference <= perturbation + rounding),
        point_counts_equal=same_count, eigenvalue_difference=difference,
        covariance_perturbation_bound=perturbation, conservative_float64_allowance=rounding,
        input_coordinate_tolerance=tolerance, tolerance_is_not_weld_acceptance=True,
        candidate_is_not_identity_proof=True)


def controls():
    generator = np.random.default_rng(1031)
    points = generator.normal(size=(257, 3)) * [2., 4., 9.]
    angle = np.deg2rad(37.)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    original = moments(points)
    rotated = points @ rotation.T + [13000., -27000., 7000.]
    assert necessary_condition(original, moments(rotated))['candidate']
    tolerance = necessary_condition(original, moments(rotated))['input_coordinate_tolerance']
    jitter = generator.normal(size=points.shape)
    jitter *= (tolerance * .9 / np.linalg.norm(jitter, axis=1))[:, None]
    assert necessary_condition(original, moments(rotated + jitter))['candidate']
    assert necessary_condition(original, moments(points[::-1]))['candidate']
    assert not necessary_condition(original, moments(points[:-1]))['candidate']
    assert not necessary_condition(original, moments(points * 1.2))['candidate']
    # Opposite coordinates share moments without asserting proper-rotation
    # identity. This deliberately remains a candidate requiring registration.
    assert necessary_condition(original, moments(-points))['candidate']
    try:
        moments([[np.nan, 0, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite input accepted')
    return dict(proper_rotation_translation_and_bounded_jitter_retained=True,
        order_invariance_checked=True, multiplicity_and_count_retained=True,
        different_scale_rejected=True, equal_moments_not_asserted_identity=True,
        nonfinite_input_rejected=True)


def atomic(path, value):
    pending = path.with_name(path.name + '.incomplete')
    write(pending, value)
    os.replace(pending, path)


def main(args):
    if (not args.experiment_name or Path(args.experiment_name).name != args.experiment_name
            or args.experiment_name in ('.', '..')):
        raise ValueError('An independent local audit name required')
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    files = ['qualification_summary.json', 'training_registry.json', 'evaluation_registry.json',
             'split_manifest.json']
    bindings = {name: digest(BASE / name) for name in files}
    sources = {str(Path(__file__).relative_to(ROOT)): digest(__file__),
               'tools/qualify_full_dataset_labels.py': digest(ROOT / 'tools/qualify_full_dataset_labels.py')}
    try:
        check = controls()
        qualification = read(BASE / files[0])
        train, evaluation = read(BASE / files[1]), read(BASE / files[2])
        role_by_key = {}
        for job in train['cases']:
            for version in job['versions']:
                if version['key'] in role_by_key:
                    raise ValueError('A registered version appears more than once')
                role_by_key[version['key']] = 'TRAIN'
        for job in evaluation['cases']:
            for version in job['versions']:
                if version['key'] in role_by_key:
                    raise ValueError('A held version overlaps another role')
                role_by_key[version['key']] = version['role']
        if set(role_by_key) != {v['key'] for v in qualification['qualified']}:
            raise ValueError('All qualified versions must have exactly one actual model-use role')
        grouped = defaultdict(list)
        for version in qualification['qualified']:
            grouped[version['source_files'][version['workpiece_source']]].append(version)
        write(out / 'protocol.json', dict(sources=sources, bindings=bindings,
            all_qualified_versions_required=len(role_by_key), unique_raw_inputs_required=len(grouped),
            all_actual_model_use_roles=True, same_count_near_bijection_relative_tolerance=1.e-6,
            candidate_registration_not_performed=True, GT_geometry_reads=0,
            model_predictions_scored=0, fit_calls=0, input_and_split_bytes_changed=False))
        rows = []
        (out / 'inputs').mkdir()
        with threadpool_limits(limits=2):
            for sha, versions in sorted(grouped.items()):
                first = versions[0]
                source = DATA / first['workpiece_source']
                if digest(source) != sha:
                    raise ValueError('Qualified input fingerprint changed')
                points = ply_points(source)
                if any(v['workpiece_points'] != len(points) for v in versions):
                    raise ValueError('Same-input versions disagree on point count')
                row = dict(workpiece_sha256=sha, source_path=first['workpiece_source'],
                    versions=[dict(key=v['key'], family_id=v['family_id'], split=v['split'],
                                   role=role_by_key[v['key']]) for v in versions], **moments(points))
                if digest(source) != sha:
                    raise ValueError('Input changed while all its moments were calculated')
                roles = {v['role'] for v in row['versions']}
                if len(roles) != 1:
                    raise ValueError('Byte-identical input crosses model-use roles')
                row['role'] = roles.pop()
                write(out / 'inputs' / (sha + '.json'), row)
                rows.append(row)
                del points
                gc.collect()
                atomic(out / 'progress.json', dict(status='READING_ALL_QUALIFIED_XYZ_INPUTS',
                    controller_pid=os.getpid(), completed_inputs=len(rows), required_inputs=len(grouped),
                    fit_calls=0, GT_geometry_reads=0, unix_time=time.time()))
                print('RIGID_INVARIANT_INPUT', len(rows), '/', len(grouped), flush=True)
        pairs, candidates = [], []
        for first, second in combinations(rows, 2):
            if first['role'] == second['role']:
                continue
            row = dict(first=first['workpiece_sha256'], second=second['workpiece_sha256'],
                first_role=first['role'], second_role=second['role'],
                **necessary_condition(first, second))
            pairs.append(row)
            if row['candidate']:
                candidates.append(row)
        if bindings != {name: digest(BASE / name) for name in files} or sources != {
                name: digest(ROOT / name) for name in sources}:
            raise ValueError('Frozen audit bindings changed')
        write(out / 'all_cross_role_pairs.json', dict(pairs=pairs, candidates=candidates,
            all_cross_role_input_pairs_checked=True, covariance_match_is_not_identity=True))
        write(out / 'receipt.json', dict(status='ALL_FULL_XYZ_CROSS_ROLE_RIGID_NECESSARY_CONDITIONS_CHECKED',
            qualified_versions=len(role_by_key), all_raw_inputs=len(rows), checked_pairs=len(pairs),
            unresolved_registration_candidates=len(candidates), controls=check, bindings=bindings,
            sources=sources, all_input_coordinates_and_multiplicities_used=True,
            individual_input_records={str(p.relative_to(out)): digest(p) for p in sorted((out / 'inputs').glob('*.json'))},
            pair_results_sha256=digest(out / 'all_cross_role_pairs.json'),
            GT_geometry_reads=0, fit_calls=0, model_predictions_scored=0,
            role_or_model_or_input_changed=False, registration_identity_not_certified=True,
            cropped_or_resampled_derivatives_not_certified=True,
            historical_exposure_not_certified=True, independent_generalization_certified=False,
            protocol_sha256=digest(out / 'protocol.json'), seconds=time.perf_counter() - started))
        print('ALL_CROSS_ROLE_RIGID_INVARIANTS_CHECKED', len(pairs), 'CANDIDATES', len(candidates), flush=True)
    except Exception:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
