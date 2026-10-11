"""Finite original CAD support of offline line/contact attribute candidates.

Query and output coordinates describe CAD references; original GT stays intact.
Missing intervals are retained exactly as returned, including numerical tails.
"""
from collections import defaultdict
import math

import gmsh
import numpy as np

VERSION = 'FINITE_OFFLINE_STEP_LINE_AND_NORMAL_STRIP_SUPPORT_273_v1'


class CadQueryError(RuntimeError):
    pass


def native(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception as error:
        raise CadQueryError(str(error)) from error


def union_and_missing(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        lo, hi = max(0., float(lo)), min(1., float(hi))
        if hi < lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    missing, cursor = [], 0.
    for lo, hi in merged:
        if lo > cursor:
            missing.append([cursor, lo])
        cursor = max(cursor, hi)
    if cursor < 1.:
        missing.append([cursor, 1.])
    return merged, missing


def material_normal(volume, tag):
    lower, upper = native(gmsh.model.getParametrizationBounds, 2, tag)
    for fraction in [.5, .3, .7, .15, .85]:
        uv = np.asarray(lower) + fraction * (np.asarray(upper) - lower)
        if not native(gmsh.model.isInside, 2, tag, uv.tolist(), parametric=True):
            continue
        point = np.asarray(native(gmsh.model.getValue, 2, tag, uv.tolist()))
        normal = np.asarray(native(gmsh.model.getNormal, tag, uv.tolist()))
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or abs(norm - 1.) > 1e-3:
            continue
        normal /= norm
        plus = native(gmsh.model.isInside, 3, volume, (point + .01 * normal).tolist())
        minus = native(gmsh.model.isInside, 3, volume, (point - .01 * normal).tolist())
        if plus != minus:
            return {'normal_CAD': (normal if minus else -normal).tolist(), 'verified': True,
                    'interior_point_mm': point.tolist(), 'plus_inside': int(plus), 'minus_inside': int(minus),
                    'normal_membership_step_mm': .01, 'method': 'FRESH_SURFACE_NORMAL_AND_SAME_SOLID_INTERIOR_MEMBERSHIP'}
    return {'normal_CAD': None, 'verified': False, 'method': 'NO_VERIFIED_FACE_INTERIOR_NORMAL'}


class FiniteStepSupport:
    def __init__(self, cad, transform, tolerances):
        self.cad, self.tolerances = cad, tolerances
        transform = np.asarray(transform, dtype=float)
        self.rotation, self.translation = transform[:3, :3], transform[:3, 3]
        self.original_entities = set(native(gmsh.model.occ.getEntities))
        counts = {str(d): len(native(gmsh.model.getEntities, d)) for d in range(4)}
        if counts != cad['entity_counts']:
            raise ValueError('Fresh native STEP entity counts disagree with bound CAD cache')
        self.volume_mass = {}
        self.faces = {}
        self.by_volume = defaultdict(list)
        for volume in cad['volumes']:
            tag = int(volume['volume'])
            mass = native(gmsh.model.occ.getMass, 3, tag)
            if abs(mass - volume['volume_mm3']) > max(1e-4, abs(mass) * 1e-9):
                raise ValueError('Fresh native solid tag/mass does not match bound cache')
            self.volume_mass[tag] = mass
            boundary = {int(t) for d, t in native(gmsh.model.getBoundary, [(3, tag)], combined=False, oriented=False) if d == 2}
            for face in volume['planar_faces']:
                face_tag = int(face['face'])
                if face_tag not in boundary:
                    raise ValueError('Cached contact face is not in its actual native solid')
                center = np.asarray(native(gmsh.model.occ.getCenterOfMass, 2, face_tag))
                area = native(gmsh.model.occ.getMass, 2, face_tag)
                if np.linalg.norm(center - face['center_mm']) > tolerances['face_center_binding_mm']:
                    raise ValueError('Fresh native face center disagrees with bound CAD tag')
                if abs(area - face['area_mm2']) > max(1e-4, abs(area) * 1e-9):
                    raise ValueError('Fresh native face area disagrees with bound CAD tag')
                signed = material_normal(tag, face_tag)
                descriptor = {'volume': tag, 'face': face_tag, 'board_name': volume['name'],
                              'center_mm': center.tolist(), 'area_mm2': area,
                              'bbox_mm': list(native(gmsh.model.getBoundingBox, 2, face_tag)),
                              'fresh_material_normal': signed,
                              'cached_planarity_residual_mm': face['planarity_residual_mm'],
                              'cached_normal_spread_degrees': face['normal_spread_degrees']}
                self.faces[(tag, face_tag)] = descriptor
                self.by_volume[tag].append(descriptor)

    def cleanup_queries(self):
        # Only new tags in this private native model are removed. No original tag
        # or file is an owned temporary, even when geometry coordinates coincide.
        for dim in [3, 2, 1, 0]:
            created = [(d, t) for d, t in native(gmsh.model.occ.getEntities, dim)
                       if (d, t) not in self.original_entities]
            if created:
                native(gmsh.model.occ.remove, created, recursive=False)
        native(gmsh.model.occ.synchronize)
        if not self.original_entities.issubset(set(native(gmsh.model.occ.getEntities))):
            raise ValueError('Query cleanup changed original native STEP identities')

    def verify_original_solids(self):
        for volume, original in self.volume_mass.items():
            if abs(native(gmsh.model.occ.getMass, 3, volume) - original) > max(1e-4, abs(original) * 1e-10):
                raise ValueError('Original STEP solid changed during offline query')

    def _line_support(self, a, b, faces):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if not math.isfinite(length) or length <= 0.:
            raise ValueError('A positive whole line query is required')
        all_intervals, witnesses, bbox_exclusions = [], [], []
        for face in faces:
            bounds = np.asarray(face['bbox_mm'])
            pad = self.tolerances['bbox_exclusion_padding_mm']
            if np.any(np.maximum(a, b) < bounds[:3] - pad) or np.any(np.minimum(a, b) > bounds[3:] + pad):
                bbox_exclusions.append({'face': face['face'], 'bbox_mm': face['bbox_mm']})
                continue
            pa = native(gmsh.model.occ.addPoint, *map(float, a))
            pb = native(gmsh.model.occ.addPoint, *map(float, b))
            query = native(gmsh.model.occ.addLine, pa, pb)
            copied = native(gmsh.model.occ.copy, [(2, face['face'])])
            common, _ = native(gmsh.model.occ.intersect, [(1, query)], copied)
            native(gmsh.model.occ.synchronize)
            face_record = {'face': face['face'], 'common_dimensions': [int(d) for d, _ in common], 'curve_parts': []}
            for dim, tag in common:
                if dim != 1:
                    continue
                lower, upper = native(gmsh.model.getParametrizationBounds, 1, tag)
                points = np.asarray(native(gmsh.model.getValue, 1, tag, [float(lower[0]), float(upper[0])])).reshape(2, 3)
                parameters = sorted(map(float, (points - a) @ delta / (length * length)))
                mass = native(gmsh.model.occ.getMass, 1, tag)
                if abs(mass - float(np.linalg.norm(points[1] - points[0]))) > self.tolerances['native_line_length_consistency_mm']:
                    raise CadQueryError('Native common curve does not have the query line length')
                off_axis = np.linalg.norm(np.cross(points - a, delta / length), axis=1)
                if np.max(off_axis) > self.tolerances['native_line_axis_consistency_mm']:
                    raise CadQueryError('Native common curve leaves original query line')
                all_intervals.append(parameters)
                face_record['curve_parts'].append({'interval': parameters, 'length_mm': mass,
                                                 'endpoints_CAD_mm': points.tolist(),
                                                 'native_type': native(gmsh.model.getType, 1, tag)})
            witnesses.append(face_record)
        covered, missing = union_and_missing(all_intervals)
        return {'query_length_mm': length, 'query_endpoints_CAD_mm': [a.tolist(), b.tolist()],
                'candidate_faces': [f['face'] for f in faces], 'bbox_exclusions': bbox_exclusions,
                'native_intersection_witnesses': witnesses, 'covered_intervals': covered,
                'missing_intervals': missing, 'missing_total_length_mm': sum(hi - lo for lo, hi in missing) * length,
                'strict_complete_joint_CAD_reference_support': not missing,
                'internal_missing_intervals': [x for x in missing if x[0] > 0. and x[1] < 1.]}

    def _normal_strip(self, a, b, displacement, volume):
        vertices = [a, b, b + displacement, a + displacement]
        points = [native(gmsh.model.occ.addPoint, *map(float, p)) for p in vertices]
        edges = [native(gmsh.model.occ.addLine, points[i], points[(i + 1) % 4]) for i in range(4)]
        loop = native(gmsh.model.occ.addWire, edges)
        strip = native(gmsh.model.occ.addPlaneSurface, [loop])
        solid = native(gmsh.model.occ.copy, [(3, int(volume))])
        outside, _ = native(gmsh.model.occ.cut, [(2, strip)], solid)
        native(gmsh.model.occ.synchronize)
        return {'vertices_CAD_mm': [p.tolist() for p in vertices],
                'query_area_mm2': float(np.linalg.norm(np.cross(b - a, displacement))),
                'outside_dimensions': [int(d) for d, _ in outside],
                'outside_surface_areas_mm2': [native(gmsh.model.occ.getMass, d, tag) for d, tag in outside if d == 2],
                'strict_complete_inside_same_solid_closure': not any(d == 2 for d, _ in outside)}

    def child(self, target, pair):
        points = np.asarray(target['original_reference_points'], dtype=float)
        if target['kind'] != 'line' or points.shape != (2, 3):
            return {'status': 'ORIGINAL_WORLD_CURVE_NOT_IMPLEMENTED_FOR_THIS_CAD_LINE_AUDIT',
                    'original_world_kind': target['kind'], 'original_point_count': len(points)}
        if len(pair) != 2 or pair[0]['volume'] == pair[1]['volume']:
            return {'status': 'NO_DISTINCT_ACTUAL_CONTACT_BOARD_PAIR'}
        selected = [self.faces[(int(b['volume']), int(b['face']))] for b in pair]
        if any(not f['fresh_material_normal']['verified'] for f in selected):
            return {'status': 'FRESH_MATERIAL_OUTWARD_DIRECTION_UNVERIFIED'}
        normals = np.asarray([f['fresh_material_normal']['normal_CAD'] for f in selected])
        expected = np.asarray([self.rotation.T @ np.asarray(b['outward_normal_world']) for b in pair])
        if np.max(np.linalg.norm(normals - expected, axis=1)) > self.tolerances['signed_normal_vector_binding']:
            return {'status': 'FRESH_MATERIAL_DIRECTION_DISAGREES_WITH_BOUND_OFFLINE_CANDIDATE',
                    'fresh_normals_CAD': normals.tolist(), 'candidate_normals_CAD': expected.tolist()}
        gram = normals @ normals.T
        if np.linalg.det(gram) < .1:
            return {'status': 'CONTACT_PLANES_DO_NOT_DEFINE_A_STABLE_DISTINCT_BOARD_LINE'}
        centers = np.asarray([f['center_mm'] for f in selected])
        constants = np.einsum('ij,ij->i', centers, normals)
        local = (points - self.translation) @ self.rotation
        joint = local + (constants - local @ normals.T) @ np.linalg.inv(gram) @ normals
        projection = np.linalg.norm(joint - local, axis=1)
        result = {'status': 'FINITE_JOINT_CAD_REFERENCE_CHECK_EXECUTED',
                  'original_world_endpoints_mm': points.tolist(), 'original_geometry_preserved': True,
                  'joint_reference_endpoints_CAD_mm': joint.tolist(),
                  'joint_reference_endpoints_world_mm': (joint @ self.rotation.T + self.translation).tolist(),
                  'endpoint_reference_projection_distances_mm': projection.tolist(),
                  'projection_within_frozen_reference_contact_band': bool(np.max(projection) <= self.tolerances['reference_contact_projection_band_mm']),
                  'boards': [], 'customer_side_order_frozen': False,
                  'whole_curve_material_normal_certified': False}
        cosine = math.cos(math.radians(self.tolerances['coplanar_normal_group_degrees']))
        for i, board in enumerate(pair):
            volume, normal = int(board['volume']), normals[i]
            center = centers[i]
            front, possible_back = [], []
            for face in self.by_volume[volume]:
                sign = face['fresh_material_normal']
                if not sign['verified']:
                    continue
                own_normal = np.asarray(sign['normal_CAD'])
                offset = float((np.asarray(face['center_mm']) - center) @ normal)
                if own_normal @ normal >= cosine and abs(offset) <= self.tolerances['coplanar_plane_offset_mm']:
                    front.append(face)
                elif own_normal @ normal <= -cosine and offset < 0.:
                    if abs(-offset - board['thickness_candidate_mm']) <= self.tolerances['back_plane_candidate_match_mm']:
                        possible_back.append((offset, face))
            rec = {'volume': volume, 'selected_contact_face': int(board['face']), 'board_name_metadata_only': board['board_name'],
                   'fresh_outward_normal_world': (self.rotation @ normal).tolist(),
                   'fresh_material_normal': selected[i]['fresh_material_normal'],
                   'front_support': self._line_support(joint[0], joint[1], front)}
            if not possible_back:
                rec['status'] = 'NO_SAME_SOLID_OPPOSITE_FACE_AT_CANDIDATE_THICKNESS'
            elif max(x[0] for x in possible_back) - min(x[0] for x in possible_back) > self.tolerances['coplanar_plane_offset_mm']:
                rec['status'] = 'MULTIPLE_DISTINCT_OPPOSITE_PLANE_DISTANCES_UNRESOLVED'
            else:
                offset = sum(x[0] for x in possible_back) / len(possible_back)
                back = [x[1] for x in possible_back]
                rec.update(status='DIRECT_SAME_SOLID_BACK_PLANE_AND_NORMAL_STRIP_CHECK_EXECUTED',
                           thickness_direct_mm=-offset, thickness_original_candidate_mm=board['thickness_candidate_mm'],
                           back_support=self._line_support(joint[0] + offset * normal, joint[1] + offset * normal, back),
                           material_strip=self._normal_strip(joint[0], joint[1], offset * normal, volume))
            result['boards'].append(rec)
        result['strict_complete_geometric_candidate_support'] = bool(result['projection_within_frozen_reference_contact_band']
            and all(b['front_support']['strict_complete_joint_CAD_reference_support']
                    and b.get('back_support', {}).get('strict_complete_joint_CAD_reference_support', False)
                    and b.get('material_strip', {}).get('strict_complete_inside_same_solid_closure', False)
                    for b in result['boards']))
        result['any_internal_front_support_gap'] = any(b['front_support']['internal_missing_intervals'] for b in result['boards'])
        return result
