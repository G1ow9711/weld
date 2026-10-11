"""Control missed regions, thickness changes and independent face support.

Synthetic geometry checks exercise recovery semantics, not weld accuracy.
Original plate-evidence controls and genuine point-support edge cases remain.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from weld_pipeline import server_plate_pair_evidence as check
from weld_pipeline.server_supported_joint_regions import collect_regions, recover_region, supported_distance, POLICY
from weld_pipeline.sjg_local_runtime_904 import digest, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/verify_supported_joint_regions.py', 'tools/recover_all_finite_joint_regions.py',
    'weld_pipeline/server_supported_joint_regions.py', 'weld_pipeline/server_plate_pair_evidence.py']


def cloud(step=False, gap=False):
    t = np.arange(0., 1201., 2.)
    if gap:
        t = t[(t <= 400) | (t >= 600)]
    transverse = np.arange(-10., 81., 2.)
    z = np.arange(0., 81., 2.)
    tt, yy = np.meshgrid(t, transverse)
    floor = np.vstack([np.c_[tt.ravel(), yy.ravel(), np.full(tt.size, height)] for height in (0., -9.)])
    tt, zz = np.meshgrid(t, z)
    opposite = np.where(tt.ravel() < 600, -10., -9.) if step else np.full(tt.size, -10.)
    stem = np.vstack([np.c_[tt.ravel(), np.zeros(tt.size), zz.ravel()],
                      np.c_[tt.ravel(), opposite, zz.ravel()]])
    return np.vstack([floor, stem])


def profile(a, b, pair=(0, 1)):
    return dict(proposal_branch='global_observed_plane_pair', face_indices=list(pair),
        representation=dict(kind='line', endpoints_mm=[[a, 0, 0], [b, 0, 0]]))


def geometric_value(results):
    return [dict(observed_face_regions=r['observed_face_regions'], region_index=r['region_index'],
        children=r['children'], unresolved_intervals=r['unresolved_intervals']) for r in results]


def main(args):
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in ('', '.', '..'):
        raise ValueError('Independent semantic-control evidence is required')
    out = ROOT / 'server_rebuild' / args.experiment_name
    out.mkdir(exist_ok=False)
    with threadpool_limits(limits=2):
        check.self_test()
        faces = [dict(normal=[0, 0, 1], center=[600, 40, 0]),
                 dict(normal=[0, 1, 0], center=[600, 0, 40])]
        profiles = [profile(-100, -80), profile(0, 700), profile(600, 1200)]
        xyz = cloud(step=True)
        tree = cKDTree(xyz)
        original = check.segment_attributes(xyz, tree,
            dict(endpoints_mm=[[0, 0, 0], [1200, 0, 0]]), [f['normal'] for f in faces])
        assert original['plate_pair_supported'] and not original['uniform_thickness']
        regions, counts = collect_regions(profiles, faces)
        assert len(regions) == 2 and counts['all_finite_candidate_intervals_collected'] == 3
        expected = [recover_region(xyz, tree, region) for region in regions]
        assert not expected[0]['children'] and expected[1]['children']
        thicknesses = [s['plate_attributes'][1]['thickness_mm'] for s in expected[1]['children'] if s['plate_attributes']]
        assert any(abs(value-9) < 1.e-6 for value in thicknesses)
        assert any(abs(value-10) < 1.e-6 for value in thicknesses)
        assert all(min(abs(value-9), abs(value-10)) < 1.e-6 for value in thicknesses)
        assert any(s['plate_attributes'] is None and s['attributes_status'] == 'UNKNOWN' for s in expected[1]['children'])
        for permutation in ([2, 1, 0], [1, 0, 2], [0, 2, 1]):
            alternate, _ = collect_regions([profiles[index] for index in permutation], faces)
            actual = [recover_region(xyz, tree, region) for region in alternate]
            assert geometric_value(actual) == geometric_value(expected)
        uniform = cloud()
        utree = cKDTree(uniform)
        # Original local fitting uses normals and endpoints, not cached centers.
        stale_centers = [dict(normal=[0, 0, 1], center=[600, 40, 10]), faces[1]]
        retained, retained_counts = collect_regions([profile(0, 1200)], stale_centers)
        assert retained_counts['all_finite_candidate_intervals_collected'] == 1
        local = recover_region(uniform, utree, retained[0])
        assert local['children'] and all(child['plate_attributes'] for child in local['children'])
        transition, _ = collect_regions([profile(0, 1200)], stale_centers)
        transition_result = recover_region(xyz, tree, transition[0])
        assert any(child['plate_attributes'] is None for child in transition_result['children'])
        assert transition_result['children'] == expected[1]['children']
        assert transition_result['unresolved_intervals'] == expected[1]['unresolved_intervals']
        reversed_line = profile(1200, 0)
        retained_reverse, _ = collect_regions([reversed_line], stale_centers)
        assert geometric_value([recover_region(uniform, utree, retained_reverse[0])]) == geometric_value([local])
        # One face pair may contain distinct original finite lines.
        displaced = profile(0, 1200)
        displaced['representation']['endpoints_mm'] = [[0, 0, .1], [1200, 0, .1]]
        distinct_lines, _ = collect_regions([profile(0, 1200), displaced], faces)
        assert len(distinct_lines) == 2
        short, _ = collect_regions([profile(20, 22)], faces)
        short_result = recover_region(uniform, utree, short[0])
        assert abs(sum(s['length_mm'] for s in short_result['children']) - 2.) < 1.e-6
        gaps, _ = collect_regions([profile(0, 1200)], faces)
        hole = cloud(gap=True)
        hole_result = recover_region(hole, cKDTree(hole), gaps[0])
        assert hole_result['children'] and not any(
            np.asarray(s['endpoints_mm'])[:, 0].min() < 450 and np.asarray(s['endpoints_mm'])[:, 0].max() > 550
            for s in hole_result['children'])
        literal_gaps, _ = collect_regions([profile(0, 400), profile(403, 500)], faces)
        assert len(literal_gaps) == 2
        separate, _ = collect_regions([profile(0, 400), profile(600, 1000, (2, 1))],
            faces + [dict(normal=[0, 0, 1], center=[1100, 40, 0])])
        assert {tuple(r['face_indices']) for r in separate} == {(0, 1), (1, 2)}
        normals = np.asarray([f['normal'] for f in faces], float)
        dense = np.vstack([np.tile([[0., 1., 0.]], (100, 1)), [[0., 0., 2.]]])
        supported = supported_distance(dense, cKDTree(dense), np.array([[0., 1., 0.]]), normals, np.zeros(2))
        assert abs(supported[0] - np.sqrt(5.)) < 1.e-12
        single = uniform[uniform[:, 2] == 0.]
        distance = supported_distance(single, cKDTree(single), np.array([[20., 0., 0.]]), normals, np.zeros(2))
        assert distance[0] > 4.
        # One flat plate's end face still cannot become a recovered two-plate joint.
        a = np.arange(-50., 51., 2.)
        xx, yy = np.meshgrid(a, a)
        plate = np.vstack([np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z)] for z in (-10., 0.)])
        yy, zz = np.meshgrid(a, np.arange(-10., .1, 2.))
        plate = np.vstack([plate, *[np.c_[np.full(yy.size, x), yy.ravel(), zz.ravel()] for x in (-50., 50.)]])
        edge_faces = [dict(normal=[1, 0, 0], center=[50, 0, -5]), dict(normal=[0, 0, 1], center=[0, 0, 0])]
        edge = dict(proposal_branch='global_observed_plane_pair', face_indices=[0, 1],
            representation=dict(kind='line', endpoints_mm=[[50, -40, 0], [50, 40, 0]]))
        er, _ = collect_regions([edge], edge_faces)
        assert not recover_region(plate, cKDTree(plate), er[0])['children']
    write(out / 'receipt.json', dict(status='PASS_ALL_FINITE_REGION_AND_LOCAL_THICKNESS_RECOVERY_CONTROLS',
        sources={name: digest(ROOT / name) for name in SOURCES},
        original_plate_support_and_rigid_transform_controls_passed=True,
        invalid_first_interval_does_not_block_later_valid_region=True,
        all_input_candidate_orders_same_geometry_and_attributes=True,
        varying_thickness_recovers_both_stable_regions_without_median_blending=True,
        ambiguous_transition_attributes_stay_unknown=True, real_observation_gap_not_bridged=True,
        explicit_three_mm_candidate_gap_kept=True, actual_finite_face_regions_not_aliased=True,
        two_mm_supported_region_kept=True, original_dense_64_neighbor_truncation_rechecked=True,
        shared_single_face_or_single_plate_end_not_accepted=True,
        original_projection_and_support_and_thickness_thresholds_preserved=True,
        stale_cached_face_center_cannot_reject_locally_supported_candidate=True,
        unknown_thickness_transition_geometry_uses_validated_neighbor_own_faces=True,
        distinct_original_lines_for_same_face_pair_retained=True,
        policy=POLICY, model_fit_calls=0, reference_geometry_reads=0,
        accuracy_or_customer_fields_certified=False))
    print('SUPPORTED_JOINT_REGION_SEMANTIC_CONTROLS_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
