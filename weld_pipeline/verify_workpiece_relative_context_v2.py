"""Physical checks for relative context, fixed gravity, and equivalent contact frames."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from training.labels import sha256
from weld_pipeline.signed_material_support import SignedMaterialSupport
from weld_pipeline.workpiece_relative_context import (
    WorkpieceRelativeContext, frame_views, transform_context, PARAMETERS)


def main():
    out = Path('weld_pipeline/reports/workpiece_context_physical_153v2_20261001.json')
    if out.exists():
        raise ValueError('Preserve relative-context verification receipt')
    xy = np.array([[x, y] for x in np.arange(-40., 41., 4.)
                   for y in np.arange(-32., 33., 4.)])
    # Mirror contact surfaces sit within a cloud symmetric about its mid-height.
    xyz = np.concatenate([np.c_[xy, np.full(len(xy), z)] for z in (-200., -5.5, 5.5, 200.)])
    normals = np.array([[0., 1., 0.], [0., 0., 1.]])
    tangent = np.array([1., 0., 0.])
    query = np.array([8., 0., -5.5])
    local = SignedMaterialSupport(xyz)
    context = WorkpieceRelativeContext(xyz)
    f = context.describe(query, normals, tangent)
    combined = np.r_[local.describe(query, normals, tangent), f]
    cases = []
    if not np.isclose(f[18], .48625) or not np.isclose(f[0], .008):
        raise ValueError('Actual relative location/height not represented')
    cases.append(dict(case='known_workpiece_relative_location', passed=True))
    for exchange in (False, True):
        order = [1, 0] if exchange else [0, 1]
        for flip_a, flip_b in ((False, False), (True, False), (False, True), (True, True)):
            signs = np.array([-1. if flip_a else 1., -1. if flip_b else 1.])
            ns = normals[order]*signs[:, None]
            for reverse in (False, True):
                ts = -tangent if reverse else tangent
                got = context.describe(query, ns, ts)
                expected = transform_context(f, exchange, flip_a, flip_b, reverse)
                error = float(np.abs(got-expected).max())
                if error > 2e-6:
                    raise ValueError('Recomputed equivalent contact context differs')
                physical = np.r_[local.describe(query, ns, ts), got]
                orbit = frame_views(combined)
                if min(float(np.abs(physical-v).max()) for v in orbit) > 2e-6:
                    raise ValueError('Combined physical frame differs from augmented frame')
                got_orbit = frame_views(physical)
                if any(min(float(np.abs(v-w).max()) for w in got_orbit) > 2e-6 for v in orbit):
                    raise ValueError('Combined frame orbit not closed')
                cases.append(dict(case='physical_contact_frame', exchange=exchange,
                    flip_a=flip_a, flip_b=flip_b, reverse=reverse, max_error=error, passed=True))
    rotation = Rotation.from_euler('z', .37).as_matrix()
    for matrix, move in ((np.eye(3), np.array([17841., -65092., 3721.])),
                         (rotation, np.array([13431., -2492., 840.]))):
        moved = xyz @ matrix.T + move
        got = WorkpieceRelativeContext(moved).describe(query @ matrix.T+move,
            normals @ matrix.T, tangent @ matrix.T)
        error = float(np.abs(got-f).max())
        if error > 2e-6:
            raise ValueError('Translation/fixed-gravity yaw invariance failed')
        cases.append(dict(case='translation_or_fixed_gravity_rotation', max_error=error, passed=True))
    upper_query = query.copy(); upper_query[2] = 5.5
    upper_ns = normals.copy(); upper_ns[1] *= -1.
    lower_local = combined[:166]
    upper_local = local.describe(upper_query, upper_ns, tangent)
    if min(float(np.abs(upper_local-v[:166]).max()) for v in frame_views(combined)) > 2e-6:
        raise ValueError('Mirror example does not reproduce old-local ambiguity')
    upper_context = context.describe(upper_query, upper_ns, tangent)
    upper_whole = np.r_[upper_local, upper_context]
    separation = min(float(np.abs(upper_whole-v).max()) for v in frame_views(combined))
    if separation <= 1e-6 or not np.isclose(upper_context[18], .51375):
        raise ValueError('New context still collapses vertically distinct mirror contacts')
    # Measure gravity-only separation independently instead of assuming it is insufficient.
    gravity_lower = np.r_[lower_local, f[19:22]]
    gravity_upper = np.r_[upper_local, upper_context[19:22]]
    gravity_separation = min(float(np.abs(gravity_upper-v).max())
        for v in frame_views(gravity_lower, 'gravity'))
    cases.append(dict(case='mirror_contact_whole_context_distinction',
        whole_orbit_separation=separation, gravity_only_orbit_separation=gravity_separation, passed=True))
    for name, points in (('single_point', np.zeros((1, 3))),
                         ('line', np.c_[np.arange(6.), np.zeros((6, 2))]),
                         ('flat', np.c_[xy, np.zeros(len(xy))])):
        value = WorkpieceRelativeContext(points).describe(np.zeros(3), normals, tangent)
        if not np.isfinite(value).all() or value[18] != 0:
            raise ValueError('Degenerate workpiece context invalid')
        cases.append(dict(case='finite_'+name, passed=True))
    # Gravity is physical: a roll changes vertical height and normal roles.
    roll = Rotation.from_euler('x', .5).as_matrix()
    rolled = WorkpieceRelativeContext(xyz @ roll.T).describe(query @ roll.T,
        normals @ roll.T, tangent @ roll.T)
    if np.allclose(rolled, f, atol=1e-6):
        raise ValueError('Gravity direction erased by unrestricted rigid invariance')
    cases.append(dict(case='gravity_changes_under_roll', passed=True))
    out.write_text(json.dumps(dict(all_passed=True, cases=cases, parameters=PARAMETERS,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/workpiece_relative_context.py',
                                      'weld_pipeline/signed_material_support.py']},
        real_reference_read=False, material_identity_proven=False,
        whole_height_is_not_contract_plate_height=True, full_contract_pass=False), indent=2), encoding='utf8')
    print('WORKPIECE_CONTEXT_PHYSICAL', len(cases), 'PASSED', flush=True)


if __name__ == '__main__':
    main()
