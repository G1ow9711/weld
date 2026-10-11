"""Local height candidates from shared point-only physics and a frozen head."""
import numpy as np

from weld_pipeline.contact_plate_extent_model import physical_span_baseline

VERSION = 'point_local_height_datum_candidates_v1'


def endpoint_inputs(profile, point, normals):
    q = np.asarray(point, float); normals = np.asarray(normals, float).reshape(2, 3).copy()
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths < 1e-8):
        return None, {'state': 'unknown', 'value': None, 'reason': 'INVALID_PCD_CONTACT_NORMALS'}
    normals /= lengths[:, None]
    standing = normals[np.abs(normals[:, 2]) < .4]
    bases = normals[np.abs(normals[:, 2]) >= np.cos(np.deg2rad(3.))]
    if len(standing) != 1 or len(bases) != 1:
        return None, {'state': 'unknown', 'value': None, 'reason': 'LOCAL_HEIGHT_BOARD_ROLE_UNRESOLVED'}
    feature, observation = profile.observe(q, standing[0])
    if feature[0] == 0:
        return None, {'state': 'unknown', 'value': None, 'reason': 'NO_FINITE_PCD_CONTACT_FACE'}
    across = np.cross([0., 0., 1.], standing[0]); across /= np.linalg.norm(across)
    normal_xy = standing[0][:2] / np.linalg.norm(standing[0][:2])
    levels = profile.levels[np.abs(profile.levels - q[2]) <= 3.]
    local = [float(z) for z in levels if profile.horizontal_support(q, across, normal_xy, z) == 1.]
    if not local:
        return None, {'state': 'unknown', 'value': None, 'reason': 'JOINT_LOCAL_BASE_AREA_NOT_SUPPORTED'}
    local.sort(key=lambda z: abs(z - q[2]))
    return feature, {'state': 'candidate_not_reviewed', 'value': None,
        'locally_supported_joint_base_z_mm': local[0],
        'source': 'PCD_physical_profiles_trained_on_fixed_FIT_contact_board_and_own_local_base_candidates',
        'reason': 'CUSTOMER_TRUTH_REVIEW_AND_COMPLETE_GEOMETRY_ACCEPTANCE_PENDING',
        'height_definition': '实际接触板及局部底板基准', 'reviewed_customer_truth': False,
        'base_material_identity_independently_verified': False, 'full_contract_pass': False}


def estimate_children(bundle, profile, sequences, predictions):
    results, features, pending = [], [], []
    for weld in predictions:
        for index, segment in enumerate(weld['segments']):
            result = {'parent_id': weld['id'], 'child_index': index, 'fields': {}}
            if weld['type'] != 'horizontal' or segment['kind'] != 'line':
                unknown = {'state': 'unknown', 'value': None, 'reason': 'VERTICAL_OR_CURVE_HEIGHT_ROLE_NOT_FROZEN'}
                result['fields'] = {field: unknown.copy() for field in ('start_height', 'end_height')}
                results.append(result); continue
            points = np.array(segment['points'], float); options = []
            for si, sequence in enumerate(sequences):
                if sequence['type'] != weld['type'] or sequence.get('curve_kind') == 'arc':
                    continue
                origin = sequence['xyz'][0].astype(float); direction = sequence['direction']
                t = (points - origin) @ direction
                error = np.linalg.norm(np.cross(points - origin, direction), axis=1).max()
                if (error <= .25 and t.min() >= -.01 and
                    t.max() <= sequence['sample_step_mm'] * (len(sequence['xyz']) - 1) + .01):
                    options.append((float(error), si))
            if options:
                sequence = sequences[min(options)[1]]
                normals = sequence['features'][len(sequence['features']) // 2, 60:66].reshape(2, 3)
                for field, position in (('start_height', points[0]), ('end_height', points[-1])):
                    feature, record = endpoint_inputs(profile, position, normals)
                    result['fields'][field] = record
                    if feature is not None:
                        pending.append(record); features.append(feature)
            else:
                result['fields'] = {field: {'state': 'unknown', 'value': None,
                    'reason': 'PREDICTED_LINE_NOT_ASSOCIATED_WITH_PCD_CONTACT'}
                    for field in ('start_height', 'end_height')}
            results.append(result)
    if features:
        x = np.asarray(features, np.float32)
        values = physical_span_baseline(x) + bundle['span_residual'].predict(x)
        for record, value in zip(pending, values):
            if value > 0 and np.isfinite(value):
                record['value'] = float(value)
            else:
                record.update(state='unknown', value=None, reason='INVALID_LEARNED_LOCAL_HEIGHT')
    return results
