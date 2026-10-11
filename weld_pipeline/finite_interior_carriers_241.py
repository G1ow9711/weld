"""Finite sampled-face interior hypotheses from current XYZ observations.

No source-name, reference, CAD, checkpoint or weld labels enter this API.
Midplanes are nominal geometric hypotheses, never proof of filled material.
"""
from dataclasses import dataclass

import numpy as np

from .finite_surface_sections_240 import (
    UnresolvedSection, intersect_interval_unions, plane_intersection,
    projected_face_sections)

VERSION = 'PURE_XYZ_FINITE_INTERIOR_CARRIERS_241_v1'
BOX_NUMERICAL_ENCLOSURE_MM = 1e-7


@dataclass
class FiniteSampledPatch:
    face_id: int
    normal: np.ndarray
    offset_mm: float
    origin_mm: np.ndarray
    basis_uv: np.ndarray
    uv_mm: np.ndarray
    triangles: np.ndarray
    raw_representative_ids: np.ndarray

    def __post_init__(self):
        if not isinstance(self.face_id, (int, np.integer)) or self.face_id < 0:
            raise ValueError('Original nonnegative finite-face ordinal required')
        for key in ('normal', 'origin_mm', 'basis_uv', 'uv_mm'):
            setattr(self, key, np.asarray(getattr(self, key), np.float64))
        self.triangles = np.asarray(self.triangles)
        self.raw_representative_ids = np.asarray(self.raw_representative_ids)
        if (self.normal.shape != (3,) or self.origin_mm.shape != (3,)
                or self.basis_uv.shape != (2, 3) or self.uv_mm.ndim != 2
                or self.uv_mm.shape[1] != 2 or self.triangles.ndim != 2
                or self.triangles.shape[1] != 3
                or not np.issubdtype(self.triangles.dtype, np.integer)
                or not np.issubdtype(self.raw_representative_ids.dtype, np.integer)
                or self.raw_representative_ids.shape != (len(self.uv_mm),)
                or (self.raw_representative_ids < 0).any()
                or (self.triangles < 0).any() or (self.triangles >= len(self.uv_mm)).any()
                or not all(np.isfinite(getattr(self, key)).all() for key in
                    ('normal', 'origin_mm', 'basis_uv', 'uv_mm'))
                or not np.isfinite(self.offset_mm)
                or abs(np.linalg.norm(self.normal)-1.) > 1e-9
                or not np.allclose(self.basis_uv@self.basis_uv.T, np.eye(2), rtol=0, atol=1e-9)
                or np.max(np.abs(self.basis_uv@self.normal)) > 1e-9
                or abs(self.origin_mm@self.normal-self.offset_mm) > 1e-7):
            raise ValueError('Complete valid finite sampled patch required')

    @classmethod
    def from_current_raw(cls, face_record, raw_xyz_mm, raw_ids, triangles):
        raw_ids = np.asarray(raw_ids)
        raw = np.asarray(raw_xyz_mm)
        if (raw.ndim != 2 or raw.shape[1] != 3
                or not np.issubdtype(raw_ids.dtype, np.integer)
                or raw_ids.ndim != 1 or (raw_ids < 0).any() or (raw_ids >= len(raw)).any()):
            raise ValueError('Valid original raw XYZ and original row indices required')
        basis = np.asarray(face_record['basis_uv'], float)
        origin = np.asarray(face_record['origin_mm'], float)
        return cls(int(face_record['face_id']), face_record['normal'], face_record['offset_mm'],
            origin, basis, (raw[raw_ids]-origin)@basis.T, triangles, raw_ids)

    def box(self):
        if not len(self.triangles):
            return None
        points = self.uv_mm[np.unique(self.triangles)]@self.basis_uv + self.origin_mm
        return points.min(axis=0), points.max(axis=0)

    def sections(self, line):
        result = projected_face_sections(line, self.uv_mm, self.triangles,
            self.basis_uv, self.origin_mm)
        result['face_id'] = int(self.face_id)
        return result


def nominal_midplane(normal_a, offset_a, normal_b, offset_b, parallel_cosine):
    a, b = np.asarray(normal_a, float), np.asarray(normal_b, float)
    if (a.shape != (3,) or b.shape != (3,) or not np.isfinite([offset_a, offset_b, parallel_cosine]).all()
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not 0 < parallel_cosine <= 1
            or abs(np.linalg.norm(a)-1.) > 1e-9 or abs(np.linalg.norm(b)-1.) > 1e-9):
        raise ValueError('Unit computational normals, finite offsets and frozen cosine required')
    dot = float(a@b)
    if abs(dot) < parallel_cosine:
        raise UnresolvedSection('Pair is outside the declared approximate-parallel family')
    sign = 1. if dot >= 0 else -1.
    if np.array_equal(a, sign*b) and float(offset_a) == sign*float(offset_b):
        raise UnresolvedSection('Coincident side planes do not define a distinct middle surface')
    total = a + sign*b
    magnitude = float(np.linalg.norm(total))
    return total/magnitude, float((offset_a+sign*offset_b)/magnitude)


def projected_box(box, normal, offset):
    if box is None:
        return None
    low, high = (np.asarray(value, float) for value in box)
    n = np.asarray(normal, float)
    projection = np.eye(3)-np.outer(n, n)
    positive, negative = np.maximum(projection, 0), np.minimum(projection, 0)
    shift = float(offset)*n
    return positive@low+negative@high+shift, positive@high+negative@low+shift


def boxes_have_common_point(boxes):
    if any(box is None for box in boxes):
        return False
    low = np.max([box[0] for box in boxes], axis=0)
    high = np.min([box[1] for box in boxes], axis=0)
    # Enclosure protects the broad phase from numerical false exclusions.
    # It does not enlarge any sampled triangles or final interval.
    return bool(np.all(low <= high+BOX_NUMERICAL_ENCLOSURE_MM))


def _empty(kind, faces, status, pair_id=None):
    return dict(version=VERSION, kind=kind, face_ids=[int(face.face_id) for face in faces],
        material_pair_ordinal=pair_id, status=status, line=None, intervals_mm=[],
        supports=[], material_identity_verified=False, filled_material_verified=False,
        reviewed_customer_truth=False, signed_customer_normal_verified=False,
        weld_identity_verified=False)


def _clip(line, faces, record):
    records = [face.sections(line) for face in faces]
    intervals = records[0]['union_intervals_mm']
    for support in records[1:]:
        intervals = intersect_interval_unions(intervals, support['union_intervals_mm'])
    record['line'] = dict(origin_mm=line.origin_mm.tolist(), direction=line.direction.tolist())
    record['intervals_mm'] = intervals.tolist()
    record['supports'] = [dict(face_id=row['face_id'],
        triangle_ordinals=row['triangle_ordinals'].tolist(),
        triangle_intervals_mm=row['intervals'].tolist()) for row in records]
    record['status'] = 'FINITE_GEOMETRIC_HYPOTHESIS' if len(intervals) else 'NO_COMMON_FINITE_TRIANGLE_DOMAIN'
    record['point_contact_intervals'] = int(np.sum(intervals[:, 0] == intervals[:, 1])) if len(intervals) else 0
    record['positive_intervals'] = int(np.sum(intervals[:, 0] < intervals[:, 1])) if len(intervals) else 0
    record['no_positive_gap_or_short_interval_removed'] = True
    return record


def finite_face_intersection(left, right, *, use_provable_box_exclusion=True):
    faces = (left, right)
    result = _empty('SAMPLED_FACE_INTERSECTION_HYPOTHESIS', faces, 'UNRESOLVED')
    if use_provable_box_exclusion and not boxes_have_common_point([face.box() for face in faces]):
        result['status'] = 'PROVABLY_DISJOINT_FINITE_BOUNDING_BOXES'
        return result
    try:
        line = plane_intersection(left.normal, left.offset_mm, right.normal, right.offset_mm)
        return _clip(line, faces, result)
    except UnresolvedSection as error:
        result['status'], result['reason'] = 'UNRESOLVED_GEOMETRY', str(error)
        return result


def finite_nominal_mid_intersection(left, right, support, *, material_pair_ordinal,
        parallel_cosine, use_provable_box_exclusion=True):
    faces = (left, right, support)
    result = _empty('NOMINAL_MID_HYPOTHESIS', faces, 'UNRESOLVED', material_pair_ordinal)
    result['material_interpretations'] = ['SAME_MATERIAL_UNVERIFIED',
        'INTER_OBJECT_AIR_GAP_UNVERIFIED', 'SAMPLING_PLANE_ARTIFACT_UNVERIFIED']
    if len(left.triangles):
        vertices = left.uv_mm[np.unique(left.triangles)]@left.basis_uv + left.origin_mm
        separation = right.offset_mm - vertices@right.normal
        result['signed_separation_at_left_support_range_mm'] = [float(separation.min()), float(separation.max())]
        result['side_planes_cross_within_left_support'] = bool(separation.min() <= 0 <= separation.max())
    try:
        normal, offset = nominal_midplane(left.normal, left.offset_mm,
            right.normal, right.offset_mm, parallel_cosine)
        result['midplane'] = dict(normal=normal.tolist(), offset_mm=offset,
            construction='ORIENTED_UNIT_PLANE_FUNCTION_A_PLUS_B_EQUALS_ZERO')
        boxes = [projected_box(left.box(), normal, offset),
                 projected_box(right.box(), normal, offset), support.box()]
        if use_provable_box_exclusion and not boxes_have_common_point(boxes):
            result['status'] = 'PROVABLY_DISJOINT_PROJECTED_FINITE_BOUNDING_BOXES'
            return result
        line = plane_intersection(normal, offset, support.normal, support.offset_mm)
        return _clip(line, faces, result)
    except UnresolvedSection as error:
        result['status'], result['reason'] = 'UNRESOLVED_GEOMETRY', str(error)
        return result


def lift_local_curve_to_world(face, geometry):
    """Lift a future decoder's continuous UV geometry, preserving its phase.

    This is a location/decode interface, not a trained curve prediction. A face
    domain alone supplies no curve parameters. Ordered children call this API
    independently; nearby endpoints and positive gaps are never joined here.
    """
    from .contract_curve_geometry import CircularArc
    from .primitive_incidence_209 import parse_primitive

    def point(value):
        uv = np.asarray(value, float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError('Finite continuous UV curve coordinates required')
        return uv@face.basis_uv+face.origin_mm

    kind = geometry.get('kind')
    if kind == 'line':
        result = dict(kind='line', units='mm', start_mm=point(geometry['start_uv_mm']).tolist(),
                      end_mm=point(geometry['end_uv_mm']).tolist())
    elif kind == 'arc':
        angle = float(geometry['start_angle_rad'])
        radial = np.cos(angle)*face.basis_uv[0]+np.sin(angle)*face.basis_uv[1]
        arc = CircularArc(point(geometry['center_uv_mm']), float(geometry['radius_mm']),
            np.cross(face.basis_uv[0], face.basis_uv[1]), radial,
            float(geometry['sweep_rad']), bool(geometry.get('is_closed', False)))
        result = dict(kind='arc', units='mm', arc_parameters=arc.to_parameters())
    elif kind == 'polyline':
        result = dict(kind='polyline', units='mm',
            knots_mm=[point(q).tolist() for q in geometry['knots_uv_mm']],
            is_closed=bool(geometry.get('is_closed', False)))
    else:
        raise ValueError('A decoded line, analytic arc or ordered polyline is required')
    return dict(domain_face_id=int(face.face_id), geometry_parameters=parse_primitive(result).to_parameters(),
        semantic_status='UNTRAINED_CONTINUOUS_SURFACE_LOCATION_INTERFACE',
        curve_inside_domain_verified=False, weld_identity_verified=False,
        material_identity_verified=False, positive_gap_merging_performed=False)
