"""Physical counterexamples for the plane-coordinate finite base proposal."""
import json
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.finite_joint_base_support import FiniteJointBases, VERSION


def grid(xs, ys):
    return np.array([[x, y, 0.] for x in xs for y in ys], float)


def main():
    started = time.perf_counter()
    out = Path('weld_pipeline/reports/finite_base_uv_20261001')
    if out.exists():
        raise ValueError('Frozen physical receipt exists')
    out.mkdir(parents=True)
    ordinary = grid(np.arange(0., 52., 2.), np.arange(-32., 34., 2.))
    angle = np.deg2rad(1.)
    rotation = np.array([[np.cos(angle), 0., -np.sin(angle)], [0., 1., 0.],
                         [np.sin(angle), 0., np.cos(angle)]])
    holed = ordinary[~((ordinary[:, 0] > 0.) & (ordinary[:, 0] < 26.) & (np.abs(ordinary[:, 1]) < 24.))]
    disconnected = grid(np.r_[np.arange(-50., -8., 2.), np.arange(10., 52., 2.)], np.arange(-32., 34., 2.))
    cases = [
        ('flat_finite_endpoint', ordinary, np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), True),
        ('one_degree_tilted_endpoint', ordinary @ rotation.T, np.zeros(3), rotation @ [0.,0.,1.], rotation @ [1.,0.,0.], True),
        ('translated_tilted_endpoint', ordinary @ rotation.T + [31.,-12.,103.], np.array([31.,-12.,103.]), rotation @ [0.,0.,1.], rotation @ [1.,0.,0.], True),
        ('normal_sign_equivalence', ordinary, np.zeros(3), np.array([0.,0.,-1.]), np.array([1.,0.,0.]), True),
        ('narrow_eight_mm_finite_board', grid(np.arange(0.,52.,2.), np.arange(0.,10.,2.)), np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), True),
        ('toward_outside_finite_endpoint', ordinary, np.zeros(3), np.array([0.,0.,1.]), np.array([-1.,0.,0.]), False),
        ('single_intersection_line', grid(np.arange(0.,52.,2.), [0.]), np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), False),
        ('two_outline_rows', grid(np.arange(0.,52.,2.), [0.,8.]), np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), False),
        ('local_opening_both_sides', holed, np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), False),
        ('disconnected_coplanar_islands', disconnected, np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), False),
        ('remote_material', ordinary + [40.,0.,0.], np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), False),
        ('parallel_remote_base', ordinary + [0.,0.,8.], np.zeros(3), np.array([0.,0.,1.]), np.array([1.,0.,0.]), False),
    ]
    rows = []
    for name, points, q, normal, inward, expected in cases:
        value, diagnostic = FiniteJointBases(points).support(q, normal, inward)
        supported = value is not None
        row = dict(name=name, expected_support=expected, actual_support=supported,
                   passed=supported == expected, value=value, diagnostic=diagnostic)
        rows.append(row)
        print('BASE_PHYSICS', name, 'EXPECTED', expected, 'ACTUAL', supported, flush=True)
    result = dict(version=VERSION, cases=rows, passed=all(row['passed'] for row in rows),
                  source_sha256=sha256('weld_pipeline/finite_joint_base_support.py'),
                  verifier_sha256=sha256(__file__), elapsed_seconds=time.perf_counter()-started,
                  synthetic_physics_only=True, real_workpiece_accuracy_not_measured=True,
                  full_contract_pass=False)
    (out / 'synthetic_receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    if not result['passed']:
        raise ValueError('Finite base physical counterexample failed; preserve failure receipt')
    print('FINITE_BASE_PHYSICS_OK', len(rows), flush=True)


if __name__ == '__main__':
    main()
