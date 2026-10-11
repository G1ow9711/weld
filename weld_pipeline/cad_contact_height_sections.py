"""Offline mesh measurements of finite contacting faces, never runtime inputs."""
import numpy as np


class ContactFaceSections:
    def __init__(self, vertices, triangles, owners):
        self.triangles = np.asarray(vertices, float)[np.asarray(triangles)]
        self.owners = np.asarray(owners)
        cross = np.cross(self.triangles[:, 1] - self.triangles[:, 0],
                         self.triangles[:, 2] - self.triangles[:, 0])
        norms = np.linalg.norm(cross, axis=1)
        self.normals = cross / np.maximum(norms[:, None], 1e-12)
        self.cache = {}

    def section(self, volume, point, normal):
        q = np.asarray(point, float); n = np.asarray(normal, float).copy()
        n /= np.linalg.norm(n)
        if n[np.argmax(np.abs(n))] < 0:
            n *= -1
        vertical = np.array([0., 0., 1.]) - n * n[2]
        factor = np.linalg.norm(vertical)
        if factor < .1:
            return {'state': 'horizontal_face_not_a_height_board'}
        vertical /= factor; across = np.cross(vertical, n)
        offset = float(q @ n); key = (int(volume), *n.round(7), round(offset, 3))
        if key not in self.cache:
            owner = self.owners == volume
            plane = np.max(np.abs(self.triangles @ n - offset), axis=1) <= .15
            parallel = np.abs(self.normals @ n) >= np.cos(np.deg2rad(3.))
            triangles = self.triangles[owner & plane & parallel]
            self.cache[key] = triangles @ np.stack((across, vertical), axis=1)
        uv = self.cache[key]
        uq, vq = float(q @ across), float(q @ vertical)
        heights = []; details = []
        for epsilon in (-.05, 0., .05):
            intervals = []
            for triangle in uv[(uv[:, :, 0].min(axis=1) <= uq + epsilon + 1e-7) &
                               (uv[:, :, 0].max(axis=1) >= uq + epsilon - 1e-7)]:
                values = []
                for index in range(3):
                    a, b = triangle[index], triangle[(index + 1) % 3]
                    du = b[0] - a[0]
                    if abs(du) < 1e-10:
                        if abs(a[0] - uq - epsilon) <= 1e-7:
                            values.extend((a[1], b[1]))
                    else:
                        t = (uq + epsilon - a[0]) / du
                        if -1e-7 <= t <= 1. + 1e-7:
                            values.append(a[1] + t * (b[1] - a[1]))
                if values:
                    intervals.append((float(min(values)), float(max(values))))
            merged = []
            for low, high in sorted(intervals):
                if merged and low <= merged[-1][1] + .05:
                    merged[-1][1] = max(merged[-1][1], high)
                else:
                    merged.append([low, high])
            containing = [(a, b) for a, b in merged if a - .15 <= vq <= b + .15 and b - a > 1.]
            if len(containing) == 1:
                contact_low, contact_high = containing[0]
                # Height spans the outer limits of this same finite CAD face;
                # a hole stays explicit and is not filled as weld material.
                low, high = merged[0][0], merged[-1][1]
                heights.append((high - low) * factor)
                bottom = q + (low - vq) * vertical
                top = q + (high - vq) * vertical
                details.append({'query_across_offset_mm': epsilon, 'local_up_span_mm': high - low,
                    'vertical_span_mm': (high - low) * factor, 'bottom_xyz_mm': bottom.tolist(),
                    'top_xyz_mm': top.tolist(), 'separated_material_intervals': len(merged),
                    'contact_interval_vertical_span_mm': (contact_high - contact_low) * factor,
                    'material_intervals_up_mm': merged})
        if not heights:
            return {'state': 'no_unique_contacting_material_interval', 'triangles_in_plane': len(uv)}
        if max(heights) - min(heights) > .2:
            return {'state': 'one_sided_height_discontinuity', 'candidates': details}
        return {'state': 'finite_contact_face_section_candidate', 'vertical_span_mm': float(np.median(heights)),
                'candidates': details, 'reviewed_customer_truth': False,
                'base_datum_applied': False}
