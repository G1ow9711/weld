"""Own-plane finite base hypotheses and independently bound contact profiles."""
import numpy as np

from weld_pipeline.contact_plate_extent_model import physical_span_baseline

VERSION = 'point_local_height_datum_candidates_v4'


def project_contact(profile, bases, point, normal):
    q = np.asarray(point, float)
    choices = []
    for plane in bases.candidates(q, normal):
        n, offset = plane['normal'], plane['offset']
        projected = q+n*(offset-q@n)
        d2 = float(plane['tree'].query(projected@plane['basis'].T)[0])
        distance = float(np.hypot(d2, offset-q@n))
        if distance <= 3.:
            choices.append((distance, projected))
    if not choices:
        return None, dict(reason='NO_FINITE_CONTACT_MATERIAL_WITHIN_GEOMETRY_TOLERANCE')
    choices.sort(key=lambda item: item[0])
    if len(choices)>1 and choices[1][0]-choices[0][0] <= .15:
        return None, dict(reason='MULTIPLE_FINITE_CONTACT_PLANES_NEAR_ENDPOINT')
    projected = choices[0][1]
    features, observation = profile.observe(projected, normal)
    if features[0] == 0 or observation['component'] is None:
        return None, dict(reason='FINITE_CONTACT_COMPONENT_UNRESOLVED')
    return (features, observation, projected), dict(point_projection_mm=projected.tolist(),
        joint_sample_distance_mm=choices[0][0], point_correction_mm=float(np.linalg.norm(projected-q)))


def physical_height(profile, point, standing_normal, inward, base_z):
    plane = profile.plane(point, standing_normal)
    q = np.asarray(point, float)
    uvq = np.array([q@plane['across'], q@plane['vertical']])
    candidates = [component for component in plane['components']
                  if component['u_min']-12. <= uvq[0] <= component['u_max']+12.
                  and component['v_min']-12. <= uvq[1] <= component['v_max']+12.]
    if len(candidates) != 1:
        return None, dict(reason='CONTACT_INSTANCE_UNRESOLVED')
    ids = candidates[0]['ids']
    uv, xyz = plane['uv'][ids], plane['points'][ids]
    direction = float(np.asarray(inward)@plane['across'])
    if abs(direction) < 1e-8:
        return None, dict(reason='CONTACT_INWARD_DIRECTION_UNRESOLVED')
    bins = np.round(uv[:, 0]/.25).astype(np.int64)
    occupied, inverse = np.unique(bins, return_inverse=True)
    low, high = np.full(len(occupied), np.inf), np.full(len(occupied), -np.inf)
    np.minimum.at(low, inverse, xyz[:, 2]); np.maximum.at(high, inverse, xyz[:, 2])
    positions = occupied*.25
    delta = np.sign(direction)*(positions-uvq[0])
    valid = np.flatnonzero((delta >= -.15) & (delta <= 8.) & (high-low > 5.))
    if not len(valid):
        return None, dict(reason='NO_INWARD_FINITE_CONTACT_BOUNDARY_COLUMN')
    selected = valid[np.argmin(np.maximum(0., delta[valid]))]
    bottom, top = float(low[selected]), float(high[selected])
    if abs(base_z-bottom) <= 3. and abs(base_z-top) > 3.:
        value, direction_name = top-base_z, 'board_above_local_base'
    elif abs(base_z-top) <= 3. and abs(base_z-bottom) > 3.:
        value, direction_name = base_z-bottom, 'board_below_local_base'
    else:
        return None, dict(reason='BASE_AT_CONTACT_INTERIOR_HEIGHT_DIRECTION_UNRESOLVED',
                          bottom_z_mm=bottom, top_z_mm=top, base_z_mm=base_z)
    return float(value), dict(height_direction=direction_name, bottom_z_mm=bottom, top_z_mm=top,
                            column_offset_inward_mm=float(delta[selected]),
                            column_width_mm=.25, material_identity_verified=False)


def endpoint_inputs(profile, bases, point, normals, toward_interior):
    q, normals = np.asarray(point, float), np.asarray(normals, float).reshape(2,3).copy()
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths < 1e-8):
        return None, dict(state='unknown', value=None, reason='INVALID_PCD_CONTACT_NORMALS')
    normals /= lengths[:,None]
    standing = normals[np.abs(normals[:,2]) < .4]
    bottom = normals[np.abs(normals[:,2]) >= np.cos(np.deg2rad(3.))]
    if len(standing) != 1 or len(bottom) != 1:
        return None, dict(state='unknown', value=None, reason='LOCAL_HEIGHT_BOARD_ROLE_UNRESOLVED')
    bound, contact_diagnostic = project_contact(profile, bases, q, standing[0])
    if bound is None:
        return None, dict(state='unknown', value=None, **contact_diagnostic)
    features, observation, projected = bound
    base, diagnostic = bases.support(q, bottom[0], toward_interior)
    if base is None:
        return None, dict(state='unknown', value=None, reason='FINITE_JOINT_BASE_NOT_RESOLVED',
                          base_diagnostic=diagnostic, contact_diagnostic=contact_diagnostic)
    direct, direct_diagnostic = physical_height(profile, projected, standing[0], toward_interior, base['local_base_z_mm'])
    return features, dict(state='candidate_not_reviewed', value=None, version=VERSION,
        source='fresh_PCD_contact_component_and_same_joint_base_plane_plus_frozen140_head',
        base=base, contact_diagnostic=contact_diagnostic,
        physical_height_mm=direct, physical_diagnostic=direct_diagnostic,
        height_definition='实际接触板及局部底板基准', reason='CUSTOMER_TRUTH_REVIEW_AND_COMPLETE_GEOMETRY_ACCEPTANCE_PENDING',
        reviewed_customer_truth=False, base_material_identity_independently_verified=False,
        full_contract_pass=False)


def estimate_children(bundle, profile, bases, sequences, predictions):
    results, features, pending = [], [], []
    for weld in predictions:
        for index, segment in enumerate(weld['segments']):
            result = dict(parent_id=weld['id'], child_index=index, fields={})
            if weld['type'] != 'horizontal' or segment['kind'] != 'line':
                unknown = dict(state='unknown', value=None, reason='VERTICAL_OR_CURVE_HEIGHT_ROLE_NOT_FROZEN')
                result['fields'] = {key: unknown.copy() for key in ('start_height', 'end_height')}
                results.append(result); continue
            points = np.array(segment['points'], float); options = []
            for si, sequence in enumerate(sequences):
                if sequence['type'] != 'horizontal' or sequence.get('curve_kind') == 'arc':
                    continue
                origin, direction = sequence['xyz'][0].astype(float), sequence['direction']
                projection = (points-origin)@direction
                error = float(np.linalg.norm(np.cross(points-origin, direction), axis=1).max())
                if error <= .25 and projection.min() >= -.01 and projection.max() <= sequence['sample_step_mm']*(len(sequence['xyz'])-1)+.01:
                    options.append((error, si))
            if not options:
                result['fields'] = {key: dict(state='unknown', value=None, reason='PREDICTED_LINE_NOT_ASSOCIATED_WITH_PCD_CONTACT')
                                    for key in ('start_height', 'end_height')}
                results.append(result); continue
            sequence = sequences[min(options)[1]]
            tangent = points[-1]-points[0]
            for field, position, inward in [('start_height', points[0], tangent), ('end_height', points[-1], -tangent)]:
                along = float((position-sequence['xyz'][0])@sequence['direction'])/sequence['sample_step_mm']
                sample = int(np.clip(round(along), 0, len(sequence['features'])-1))
                normals = sequence['features'][sample,60:66].reshape(2,3)
                feature, record = endpoint_inputs(profile, bases, position, normals, inward)
                result['fields'][field] = record
                if feature is not None:
                    features.append(feature); pending.append(record)
            results.append(result)
    if features:
        x = np.asarray(features, np.float32)
        baseline = physical_span_baseline(x)
        residual = bundle['span_residual'].predict(x)
        for record, base, correction in zip(pending, baseline, residual):
            value = float(base+correction)
            record.update(physical_span_baseline_mm=float(base), learned_residual_mm=float(correction))
            if value > 0 and np.isfinite(value):
                record['value'] = value
            else:
                record.update(state='unknown', value=None, reason='INVALID_LEARNED_LOCAL_HEIGHT')
    return results
