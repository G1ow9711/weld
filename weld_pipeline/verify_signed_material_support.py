"""Physical counterexamples and frame invariance for the signed PCD descriptor."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from training.labels import sha256
from weld_pipeline.signed_material_support import SignedMaterialSupport, frame_views, FEATURE_COUNT, PARAMETERS


def face(axis, offset, transverse, along):
    a, b = np.meshgrid(transverse, along)
    points = np.zeros((a.size, 3))
    points[:, axis] = offset
    points[:, 1-axis] = a.ravel()
    points[:, 2] = b.ravel()
    return points


def main():
    out = Path('weld_pipeline/reports/signed_material_support_physical_20261001.json')
    if out.exists():
        raise ValueError('Preserve physical receipt')
    along = np.arange(-80., 80.1, 2.)
    negative = np.arange(-32., .1, 2.)
    both = np.arange(-32., 32.1, 2.)
    positive = np.arange(0., 40.1, 2.)
    convex = np.vstack([face(0, 0., negative, along), face(1, 0., negative, along),
                        face(0, -4., negative, along), face(1, -4., negative, along)])
    tee = np.vstack([face(0, 0., positive, along), face(0, 4., positive, along),
                     face(1, 0., both, along), face(1, -4., both, along)])
    normals = np.array([[1., 0., 0.], [0., 1., 0.]])
    tangent = np.array([0., 0., 1.])
    query = np.zeros(3)
    fc = SignedMaterialSupport(convex).describe(query, normals, tangent)
    ft = SignedMaterialSupport(tee).describe(query, normals, tangent)
    if np.allclose(fc, ft, atol=1e-6):
        raise ValueError('Convex same-body edge and T support indistinguishable')
    rays_c, rays_t = fc[:84].reshape(2, 3, 2, 7), ft[:84].reshape(2, 3, 2, 7)
    if np.any(rays_c[:, :, 1, :] > 0) or not np.any(rays_t[1, :, 1, :] > 0):
        raise ValueError('Coupled signed support sides lost')
    if not np.any(ft[108:156].reshape(2, 2, 4, 3)[..., 2]):
        raise ValueError('Nearby parallel backface evidence missing')
    cases = [dict(case='convex_vs_tee', passed=True), dict(case='signed_surface_sides', passed=True),
             dict(case='parallel_backfaces', passed=True)]
    rotation = Rotation.from_rotvec([.31, -.42, .18]).as_matrix()
    for name, xyz, expected in [('convex', convex, fc), ('tee', tee, ft)]:
        extractor = SignedMaterialSupport(xyz)
        views = frame_views(expected)
        for exchange in (False, True):
            ns = normals[::-1] if exchange else normals.copy()
            for a, b in ((1., 1.), (-1., 1.), (1., -1.), (-1., -1.)):
                got = extractor.describe(query, ns * np.array([a, b])[:, None], tangent)
                error = min(float(np.abs(got-v).max()) for v in views)
                if error > 1e-6:
                    raise ValueError('Signed/order permutation differs: ' + name)
                cases.append(dict(case=name+'_equivalent_normals', exchange=exchange, signs=[a, b],
                                  max_error=error, passed=True))
                orbit = np.stack(frame_views(got))
                if any(min(float(np.abs(v-w).max()) for w in orbit) > 1e-6 for v in views):
                    raise ValueError('Equivalent-frame orbit not closed')
        for move, matrix in [(np.array([1000., -2000., 317.]), np.eye(3)),
                             (np.array([-641., 923., 1107.]), rotation)]:
            got = SignedMaterialSupport(xyz @ matrix.T + move).describe(query @ matrix.T + move,
                                                                       normals @ matrix.T, tangent @ matrix.T)
            error = float(np.abs(got-expected).max())
            if error > 1e-6:
                raise ValueError('PCD relative-frame rigid invariance failed')
            cases.append(dict(case=name+'_rigid_transform', max_error=error, passed=True))
        reversed_t = extractor.describe(query, normals, -tangent)
        if not np.array_equal(reversed_t, expected):
            raise ValueError('Material descriptor changes when seam traversal reverses')
        cases.append(dict(case=name+'_reverse_traversal', passed=True))
    absent = tee[np.abs(tee[:, 2]) > 6.]
    missing = SignedMaterialSupport(absent).describe(query, normals, tangent)
    if missing[157] <= ft[157] or np.array_equal(missing, ft):
        raise ValueError('Removed local support not observable')
    cases.append(dict(case='removed_contact_support', passed=True))
    out.write_text(json.dumps(dict(cases=cases, all_passed=True, feature_count=FEATURE_COUNT,
        parameters=PARAMETERS, extractor_sha256=sha256('weld_pipeline/signed_material_support.py'),
        verifier_sha256=sha256(__file__), no_real_truth_read=True,
        hidden_material_identity_proven=False, full_contract_pass=False), indent=2), encoding='utf8')
    print('SIGNED_MATERIAL_PHYSICAL_CASES', len(cases), 'PASSED', flush=True)


if __name__ == '__main__':
    main()
