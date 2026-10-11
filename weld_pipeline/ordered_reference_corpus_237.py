"""Offline ordered geometry supervision; no runtime proposals or material truth."""
from hashlib import sha256
import math

import numpy as np

from .contract_curve_geometry import fit_ordered_circle, sample_arc, verify_sampled_arc

VERSION = 'ALL_TYPED_ORDERED_ORIGINAL_REFERENCE_GEOMETRY_237_v1'


def array_digest(value):
    array = np.ascontiguousarray(value)
    return sha256(str(array.dtype).encode('ascii') + str(array.shape).encode('ascii')
                  + array.tobytes()).hexdigest()


def prepare_reference(child):
    """Retain originals and derive physical reference probes, never a prediction."""
    original = np.asarray(child['points'], dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or len(original) < 2 or not np.isfinite(original).all():
        raise ValueError('Every original reference point must be finite and ordered')
    common = dict(original_reference_array_sha256=array_digest(original),
                  original_point_count=len(original), original_shape_preserved=True,
                  reference_curve_plane_normal_is_material_normal=False)
    if child['kind'] == 'line':
        if original.shape != (2, 3):
            raise ValueError('Line reference must contain exactly two original endpoints')
        length = float(np.linalg.norm(original[1] - original[0]))
        if not math.isfinite(length) or length <= 0.:
            raise ValueError('Degenerate line reference remains unknown in full denominator')
        count = max(1, int(math.ceil(length)))
        q = np.linspace(0., length, count + 1)
        points = original[0] + (q / length)[:, None] * (original[1] - original[0])
        analytic = dict(kind='line', start_mm=original[0].tolist(), end_mm=original[1].tolist(),
                        analytic_length_mm=length, maximum_chord_error_mm=0.)
    elif child['kind'] == 'arc':
        arc = fit_ordered_circle(original, tolerance_mm=1e-6)
        samples = sample_arc(arc, max_step_mm=1., max_chord_error_mm=.1)
        audit = verify_sampled_arc(samples)
        length = arc.length_mm
        if abs(float(child['length_mm']) - length) > 1e-6:
            raise ValueError('Ordered analytic arc disagrees with original labelled arc length')
        q = np.asarray(samples['sample_positions_mm'], dtype=np.float64)
        points = np.asarray(samples['points'], dtype=np.float64)
        analytic = dict(kind='arc', parameters=arc.to_parameters(), analytic_length_mm=length,
                        sampling_audit=audit, reference_geometry_uncertainty_allowance_mm=1e-6)
    else:
        raise ValueError('Unsupported reference kind remains unknown: ' + str(child['kind']))
    if not np.isfinite(points).all() or not (np.diff(q) > 0.).all() or np.diff(q).max() > 1. + 1e-12:
        raise ValueError('Physical reference sampling does not satisfy the frozen contract')
    common.update(analytic=analytic, probes=len(q), maximum_physical_step_mm=float(np.diff(q).max()),
                  original_labelled_length_mm=float(child['length_mm']),
                  labelled_minus_analytic_length_mm=float(child['length_mm']) - length)
    return original, q, points, common


def adjacent_reference_evidence(previous, following):
    """Report exact original adjacency without swallowing any positive gap."""
    left = np.asarray(previous['points'], dtype=np.float64)[-1]
    right = np.asarray(following['points'], dtype=np.float64)[0]
    if left.shape != (3,) or right.shape != (3,) or not np.isfinite([left, right]).all():
        return dict(status='UNKNOWN_ORIGINAL_ADJACENCY', gap_mm=None,
                    exact_coordinate_equal=False, actual_material_or_break_reason_verified=False)
    gap = float(np.linalg.norm(right - left))
    equal = bool(np.array_equal(left, right))
    return dict(status='EXACT_REFERENCE_ENDPOINT_MATCH' if equal else 'POSITIVE_REFERENCE_ENDPOINT_GAP',
                previous_end_mm=left.tolist(), following_start_mm=right.tolist(), gap_mm=gap,
                exact_coordinate_equal=equal, positive_gap_preserved=not equal,
                actual_material_or_break_reason_verified=False,
                automatic_material_node_merge=False)
