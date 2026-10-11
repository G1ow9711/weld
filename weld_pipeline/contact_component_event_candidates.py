"""Add observed coplanar contact-component endpoints, without process cutbacks."""
import numpy as np

from weld_pipeline.physical_event_candidates import candidate_positions

VERSION = 'pcd_contact_component_event_candidates_v1'


def component_positions(sequence, geometry):
    planes = geometry['vertical_planes']
    base, base_counts = candidate_positions(sequence, planes, finite=True)
    q = np.asarray(sequence['xyz'], float); u = np.asarray(sequence['direction'], float)
    origin = q[0]; span = float((q[-1]-origin)@u)
    positions = base.tolist(); components = 0
    for record in geometry['horizontal_lines']:
        plane = planes[int(record['plane_id'])]
        n = np.asarray(plane['normal_xy'], float); d = np.asarray(plane['direction_xy'], float)
        if (abs(n@u[:2]) > .00447213595499958
                or abs(n@origin[:2]-plane['offset_mm']) > .25
                or abs(record['z_mm']-origin[2]) > .015):
            continue
        components += 1
        for value in record['range_mm']:
            point = np.r_[n*plane['offset_mm']+d*value, record['z_mm']]
            position = float((point-origin)@u)
            if 0. <= position <= span and np.linalg.norm(np.cross(point-origin, u)) <= .25:
                positions.append(position)
    result = np.unique(np.asarray(positions, float))
    return result, dict(base_counts, matching_components=components,
                        component_endpoints_added=len(result)-len(base), unique_positions=len(result))


def physical_checks():
    sequence = dict(xyz=np.array([[-20., 0., 0.], [120., 0., 0.]]),
                    direction=np.array([1., 0., 0.]), type='horizontal')
    plane = dict(normal_xy=[0., 1.], direction_xy=[1., 0.], offset_mm=0.,
                 range_mm=[0., 100.], z_range_mm=[0., 40.])
    geometry = dict(vertical_planes=[plane, dict(plane, offset_mm=3.)], horizontal_lines=[
        dict(plane_id=0, z_mm=0., range_mm=[0., 6.2680269]),
        dict(plane_id=0, z_mm=0., range_mm=[6.274, 100.]),
        dict(plane_id=0, z_mm=10., range_mm=[20., 30.]),
        dict(plane_id=1, z_mm=0., range_mm=[40., 50.])])
    positions, stats = component_positions(sequence, geometry)
    if not np.allclose(positions, [20., 26.2680269, 26.274, 120.], atol=1e-12, rtol=0):
        raise ValueError('Contact components erased a short tail or used another layer/plane')
    reverse = dict(sequence, xyz=sequence['xyz'][::-1], direction=-sequence['direction'])
    rev, _ = component_positions(reverse, geometry)
    if not np.allclose(rev, np.sort(140.-positions), atol=1e-12, rtol=0):
        raise ValueError('Component proposals changed under reversed traversal')
    angle = .731; c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    translation = np.array([1034.5, -842.25, 87.])
    moved_planes = []
    for p in geometry['vertical_planes']:
        n = rotation[:2, :2]@p['normal_xy']; d = rotation[:2, :2]@p['direction_xy']
        moved_planes.append(dict(p, normal_xy=n.tolist(), direction_xy=d.tolist(),
            offset_mm=p['offset_mm']+float(n@translation[:2]),
            range_mm=(np.asarray(p['range_mm'])+d@translation[:2]).tolist(),
            z_range_mm=(np.asarray(p['z_range_mm'])+translation[2]).tolist()))
    moved_records = [dict(r, z_mm=r['z_mm']+translation[2],
        range_mm=(np.asarray(r['range_mm'])+np.asarray(moved_planes[r['plane_id']]['direction_xy'])@translation[:2]).tolist())
        for r in geometry['horizontal_lines']]
    moved, _ = component_positions(dict(sequence, xyz=sequence['xyz']@rotation.T+translation,
        direction=rotation@sequence['direction']), dict(vertical_planes=moved_planes, horizontal_lines=moved_records))
    # Exact duplicate removal may retain roundoff copies after a transform.
    # Compare geometric sets, preserving all genuinely distinct close events.
    distances = np.abs(moved[:, None]-positions[None, :])
    error = float(max(distances.min(0).max(), distances.min(1).max()))
    if error > 1e-9:
        raise ValueError('Component proposal positions changed under translation/yaw')
    flips = [dict(p, normal_xy=(-np.asarray(p['normal_xy'])).tolist(), offset_mm=-p['offset_mm'],
        direction_xy=(-np.asarray(p['direction_xy'])).tolist(),
        range_mm=[-p['range_mm'][1], -p['range_mm'][0]]) for p in geometry['vertical_planes']]
    records = [dict(r, range_mm=[-r['range_mm'][1], -r['range_mm'][0]]) for r in geometry['horizontal_lines']]
    flip, _ = component_positions(sequence, dict(vertical_planes=flips, horizontal_lines=records))
    if not np.array_equal(positions, flip):
        raise ValueError('Component normal/direction sign changed physical points')
    return dict(short_tail_and_close_boundaries_pass=True, wrong_layer_and_plane_rejected=True,
        reverse_and_plane_sign_pass=True, translation_yaw_max_error_mm=error,
        configuration_margin_not_invented=True, diagnostics_not_geometry_acceptance=True)
