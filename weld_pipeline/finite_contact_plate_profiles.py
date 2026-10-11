"""Finite plate extent candidates and supported local datum levels from XYZ.

These are measurement features. Sparse or ambiguous surfaces remain explicit;
finite bounding intervals are not proof of material continuity through holes.
"""
import numpy as np
from scipy.spatial import cKDTree

VERSION = 'finite_contact_plate_profiles_v1'
WINDOWS_MM = (8., 24., 80.)
FEATURE_COUNT = 39


class FiniteContactPlates:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, float)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError('Finite millimeter XYZ required')
        self.axes = {}; self.planes = {}; self.level_trees = {}
        if not len(self.xyz):
            self.levels = np.empty(0)
        else:
            z, count = np.unique(np.round(self.xyz[:, 2], 2), return_counts=True)
            self.levels = z[count >= 20]

    def plane(self, point, normal):
        normal = np.asarray(normal, float).copy()
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length < 1e-9:
            raise ValueError('Nondegenerate finite plane normal required')
        normal /= length
        if normal[np.argmax(np.abs(normal))] < 0:
            normal *= -1
        up = np.array([0., 0., 1.]); projected_up = up - normal * normal[2]
        vertical_factor = np.linalg.norm(projected_up)
        if vertical_factor < .1:
            return None
        vertical = projected_up / vertical_factor
        across = np.cross(vertical, normal)
        axis = tuple(normal.round(7)); offset = float(np.asarray(point) @ normal)
        key = (*axis, round(offset, 3))
        if key not in self.planes:
            if axis not in self.axes:
                self.axes[axis] = self.xyz @ normal
            p = self.xyz[np.abs(self.axes[axis] - offset) <= .15]
            uv = p @ np.stack((across, vertical), axis=1)
            components = []; columns = None
            if len(p):
                anchor = float(uv[:, 0].min())
                bins = np.floor((uv[:, 0] - anchor) / 8.).astype(np.int64)
                occupied, inverse = np.unique(bins, return_inverse=True)
                low = np.full(len(occupied), np.inf); high = np.full(len(occupied), -np.inf)
                np.minimum.at(low, inverse, uv[:, 1]); np.maximum.at(high, inverse, uv[:, 1])
                count = np.bincount(inverse)
                # A shared horizontal boundary row alone must not join tall faces.
                tall = np.flatnonzero((high - low > 5.) & (count >= 2))
                groups = []
                for index in tall:
                    if (not groups or occupied[index] - occupied[groups[-1][-1]] > 1 or
                        min(high[index], high[groups[-1][-1]]) -
                        max(low[index], low[groups[-1][-1]]) <= 1.):
                        groups.append([index])
                    else:
                        groups[-1].append(index)
                for group in groups:
                    ids = np.flatnonzero(np.isin(inverse, group))
                    u, v = uv[ids].T
                    components.append({'u_min': float(u.min()), 'u_max': float(u.max()),
                        'v_min': float(v.min()), 'v_max': float(v.max()), 'ids': ids,
                        'columns': len(group), 'support': len(ids)})
                columns = {'anchor': anchor, 'occupied': occupied, 'low': low,
                           'high': high, 'count': count}
            self.planes[key] = {'points': p, 'uv': uv, 'components': components,
                'normal': normal, 'vertical': vertical, 'across': across,
                'vertical_factor': vertical_factor, 'columns': columns}
        return self.planes[key]

    def horizontal_support(self, point, across, normal_xy, level):
        key = round(float(level), 2)
        if key not in self.level_trees:
            p = self.xyz[np.abs(self.xyz[:, 2] - level) <= .08, :2]
            self.level_trees[key] = cKDTree(p) if len(p) else None
        tree = self.level_trees[key]
        if tree is None:
            return 0.
        # Require a two-dimensional patch on at least one material side.
        # A densely sampled vertical face intersection is only a line.
        scores = []
        for side in (-1., 1.):
            probes = np.array([np.asarray(point)[:2] + u * across[:2] + side * v * normal_xy
                               for u in (-12., 12.) for v in (6., 18.)])
            distances = tree.query(probes, k=1)[0]
            scores.append(float(np.mean(distances <= 6.)))
        return max(scores)

    def observe(self, point, normal):
        q = np.asarray(point, float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('Finite endpoint required')
        plane = self.plane(q, normal)
        features = []; diagnostic = {'version': VERSION, 'reasons': [], 'component': None,
            'horizontal_base_candidates': [], 'complete_material_continuity_verified': False}
        if plane is None or not len(plane['points']):
            diagnostic['reasons'].append('NO_SUPPORTED_NON_HORIZONTAL_CONTACT_PLANE')
            return np.zeros(FEATURE_COUNT, np.float32), diagnostic
        uvq = np.array([q @ plane['across'], q @ plane['vertical']])
        p, uv = plane['points'], plane['uv']
        support = len(p); factor = plane['vertical_factor']
        features.extend((1., abs(plane['normal'][2]), np.log1p(support) / 10.,
                         np.ptp(uv[:, 0]) / 1000., np.ptp(uv[:, 1]) / 1000.))
        candidates = [c for c in plane['components'] if c['u_min'] - 12. <= uvq[0] <= c['u_max'] + 12.
                      and c['v_min'] - 12. <= uvq[1] <= c['v_max'] + 12.]
        features.append(min(len(candidates), 5) / 5.)
        component = candidates[0] if len(candidates) == 1 else None
        if component is not None:
            diagnostic['component'] = {k: v for k, v in component.items() if k != 'ids'}
            diagnostic['component']['vertical_span_mm'] = (component['v_max'] - component['v_min']) * factor
            features.extend(((component['v_max'] - component['v_min']) / 1000.,
                (component['v_max'] - uvq[1]) / 1000., (uvq[1] - component['v_min']) / 1000.,
                (uvq[0] - component['u_min']) / 1000., (component['u_max'] - uvq[0]) / 1000.,
                np.log1p(component['support']) / 10.))
        else:
            features.extend([0.] * 6)
            diagnostic['reasons'].append('FINITE_CONTACT_COMPONENT_UNRESOLVED')
        diagnostic['windows'] = {}
        for radius in WINDOWS_MM:
            ids = np.flatnonzero(np.abs(uv[:, 0] - uvq[0]) <= radius)
            v = uv[ids, 1]; z = p[ids, 2]
            if len(v):
                low, high = float(v.min()), float(v.max())
                interior = int(((v > low + 5.) & (v < high - 5.)).sum())
                features.extend((np.log1p(len(v)) / 10., (high - low) / 1000.,
                    (high - uvq[1]) / 1000., (uvq[1] - low) / 1000.,
                    (np.quantile(v, .95) - np.quantile(v, .05)) / 1000., interior / len(v)))
                diagnostic['windows'][str(radius)] = {'support': len(v), 'up_span_mm': high - low,
                    'vertical_span_mm': float(np.ptp(z)), 'top_z_mm': float(z.max()),
                    'bottom_z_mm': float(z.min()), 'interior_support': interior}
            else:
                features.extend([0.] * 6)
                diagnostic['windows'][str(radius)] = {'support': 0}
        if component is not None:
            ids = component['ids']; bottom = float(p[ids, 2].min()); top = float(p[ids, 2].max())
            levels = self.levels[np.abs(self.levels - bottom) <= 50.]
            levels = levels[np.argsort(np.abs(levels - bottom))][:12]
            normal_xy = plane['normal'][:2].copy(); normal_xy /= np.linalg.norm(normal_xy)
            for level in levels:
                fraction = self.horizontal_support(q, plane['across'], normal_xy, level)
                if fraction == 1.:
                    diagnostic['horizontal_base_candidates'].append({'level_z_mm': float(level),
                        'top_from_base_mm': top - float(level), 'bottom_to_base_mm': bottom - float(level),
                        'two_dimensional_probe_support': fraction,
                        'connected_material_identity_verified': False})
            diagnostic['horizontal_base_candidates'].sort(key=lambda r: abs(r['bottom_to_base_mm']))
        for index in range(3):
            bases = diagnostic['horizontal_base_candidates']
            if index < len(bases):
                features.extend((1., bases[index]['top_from_base_mm'] / 1000.,
                                 bases[index]['bottom_to_base_mm'] / 1000.))
            else:
                features.extend((0., 0., 0.))
        if not diagnostic['horizontal_base_candidates']:
            diagnostic['reasons'].append('LOCAL_HORIZONTAL_BASE_NOT_RESOLVED')
        if any(w.get('support', 0) and not w.get('interior_support', 0)
               for w in diagnostic['windows'].values()):
            diagnostic['reasons'].append('SPARSE_CONTACT_FACE_INTERIOR')
        result = np.asarray(features, np.float32)
        if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
            raise ValueError('Invalid finite plate profile')
        return result, diagnostic
