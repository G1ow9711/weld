"""Offline finite-face measurements in a fixed contacting CAD identity pair."""
import numpy as np

VERSION = 'fixed_contact_face_local_height_datum_v1'


class FixedContactDatum:
    def __init__(self, vertices, triangles, owners, face_ids):
        self.triangles = np.asarray(vertices, float)[np.asarray(triangles)]
        self.owners = np.asarray(owners); self.face_ids = np.asarray(face_ids)
        self.faces = {}

    def face(self, board):
        key = (int(board['volume']), int(board['face']))
        if key not in self.faces:
            self.faces[key] = self.triangles[(self.owners == key[0]) & (self.face_ids == key[1])]
        return self.faces[key]

    def project(self, board, point):
        q = np.asarray(point, float); triangles = self.face(board)
        if not len(triangles):
            return {'state': 'FIXED_CONTACT_FACE_HAS_NO_MESH'}
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        e1, e2, aq = b - a, c - a, q - a
        d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
        determinant = d11 * d22 - d12 * d12
        valid = determinant > 1e-14
        safe = np.maximum(determinant, 1e-14)
        d1q, d2q = (e1 * aq).sum(1), (e2 * aq).sum(1)
        u = (d22 * d1q - d12 * d2q) / safe
        v = (d11 * d2q - d12 * d1q) / safe
        inside = valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1. + 1e-8)
        candidates = [a[inside] + u[inside, None] * e1[inside] + v[inside, None] * e2[inside]]
        for index in range(3):
            start, finish = triangles[:, index], triangles[:, (index + 1) % 3]
            delta = finish - start
            t = np.clip(((q - start) * delta).sum(1) / np.maximum((delta * delta).sum(1), 1e-14), 0., 1.)
            candidates.append(start + t[:, None] * delta)
        all_points = np.concatenate(candidates)
        distance = np.linalg.norm(all_points - q, axis=1)
        best = int(distance.argmin())
        return {'state': 'FIXED_FACE_PROJECTION', 'xyz_mm': all_points[best].tolist(),
                'distance_mm': float(distance[best]), 'volume': int(board['volume']), 'face': int(board['face'])}

    def standing_section(self, board, point, toward_interior):
        projection = self.project(board, point)
        if projection['state'] != 'FIXED_FACE_PROJECTION':
            return projection
        if projection['distance_mm'] > 3.:
            return {'state': 'FIXED_FACE_ENDPOINT_DISTANCE_EXCEEDS_3', 'projection': projection}
        n = np.asarray(board['outward_normal_world'], float); n /= np.linalg.norm(n)
        vertical = np.array([0., 0., 1.]) - n * n[2]
        factor = np.linalg.norm(vertical)
        if factor < .1:
            return {'state': 'HEIGHT_FACE_IS_HORIZONTAL'}
        vertical /= factor; across = np.cross(vertical, n)
        q = np.asarray(projection['xyz_mm'])
        direction = float(np.asarray(toward_interior) @ across)
        if abs(direction) > 1e-6:
            # A fixed 0.01 mm one-sided query inside this child at a height corner.
            q = q + np.sign(direction) * .01 * across
        if np.linalg.norm(q - np.asarray(point)) > 3.:
            return {'state': 'DIRECTED_QUERY_DISTANCE_EXCEEDS_3', 'projection': projection}
        triangles = self.face(board)
        if np.max(np.abs(triangles @ n - q @ n)) > .15:
            return {'state': 'NONPLANAR_FIXED_CONTACT_FACE', 'projection': projection}
        uv = triangles @ np.stack((across, vertical), axis=1)
        uq, vq = float(q @ across), float(q @ vertical)
        intervals = []
        select = (uv[:, :, 0].min(1) <= uq + 1e-6) & (uv[:, :, 0].max(1) >= uq - 1e-6)
        for triangle in uv[select]:
            values = []
            for index in range(3):
                a, b = triangle[index], triangle[(index + 1) % 3]
                if abs(b[0] - a[0]) < 1e-10:
                    if abs(a[0] - uq) <= 1e-6:
                        values.extend((a[1], b[1]))
                else:
                    t = (uq - a[0]) / (b[0] - a[0])
                    if -1e-6 <= t <= 1. + 1e-6:
                        values.append(a[1] + t * (b[1] - a[1]))
            if values:
                intervals.append((float(min(values)), float(max(values))))
        merged = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1] + .05:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        if not any(low - .15 <= vq <= high + .15 and high - low > 1. for low, high in merged):
            return {'state': 'NO_DIRECTED_CONTACT_MATERIAL_INTERVAL', 'projection': projection}
        low, high = merged[0][0], merged[-1][1]
        return {'state': 'LOCAL_FIXED_FACE_SECTION', 'projection': projection,
            'measurement_query_xyz_mm': q.tolist(), 'up_outer_span_mm': high - low,
            'vertical_outer_span_mm': (high - low) * factor,
            'bottom_xyz_mm': (q + (low - vq) * vertical).tolist(),
            'top_xyz_mm': (q + (high - vq) * vertical).tolist(),
            'separated_material_intervals': merged,
            'weld_material_continuity_implied': False}

    def height(self, standing, base, point, toward_interior):
        n = np.asarray(base['outward_normal_world'], float)
        if abs(n[2]) < np.cos(np.deg2rad(3.)):
            return {'state': 'LOCAL_BASE_CONTACT_FACE_IS_NOT_HORIZONTAL'}
        local_base = self.project(base, point)
        if local_base['state'] != 'FIXED_FACE_PROJECTION' or local_base['distance_mm'] > 3.:
            return {'state': 'LOCAL_BASE_FIXED_FACE_UNRESOLVED', 'local_base': local_base}
        section = self.standing_section(standing, point, toward_interior)
        if section['state'] != 'LOCAL_FIXED_FACE_SECTION':
            return {'state': section['state'], 'section': section, 'local_base': local_base}
        base_z = local_base['xyz_mm'][2]
        bottom, top = section['bottom_xyz_mm'][2], section['top_xyz_mm'][2]
        if abs(base_z - bottom) <= 3. and abs(base_z - top) > 3.:
            height, direction = top - base_z, 'board_above_local_base'
        elif abs(base_z - top) <= 3. and abs(base_z - bottom) > 3.:
            height, direction = base_z - bottom, 'board_below_local_base'
        else:
            return {'state': 'LOCAL_BASE_AT_BOARD_INTERIOR_HEIGHT_DIRECTION_UNRESOLVED',
                    'section': section, 'local_base': local_base}
        return {'state': 'LOCAL_CONTACT_HEIGHT_CANDIDATE', 'height_mm': float(height),
            'height_direction': direction, 'section': section, 'local_base': local_base,
            'definition': 'actual_contact_board_boundary_relative_to_its_same_joint_base_face',
            'user_definition': '实际接触板及局部底板基准',
            'reviewed_customer_truth': False, 'full_contract_pass': False}
