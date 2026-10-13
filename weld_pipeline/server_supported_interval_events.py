"""Shared input-only finite line/arc event construction for interval selection.

Grid events retain their grid provenance. Observed finite support and point
projections are evidence, not certified contacts or ground-truth weld ends.
No reference geometry, model prediction or source identity enters this API.
"""
import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree

from .server_finite_carrier_grid import finite_carrier_grid
from .sjg_curve_queries_907 import CurveCarrier


SCHEMA = 'FINITE_INPUT_INTERVAL_EVENT_BANK_V1'
COLUMNS = ('carrier_start', 'carrier_end', 'observed_support_boundary',
    'input_fragment_boundary', 'observed_point_projection', 'is_grid',
    'closed_carrier', 'arc_carrier', 'nearest_raw_distance_over_3mm',
    'projection_point_known', 'finite_support_known', 'inside_finite_support',
    'arclength_ratio')


class FiniteInputIntervalEvents:
    def __init__(self, xyz, input_receipt_sha256, spacing_mm=1., max_events=1_000_001):
        self.xyz = np.asarray(xyz, np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1:] != (3,) or not len(self.xyz) or not np.isfinite(self.xyz).all():
            raise ValueError('A complete finite raw XYZ input is required')
        if (len(input_receipt_sha256) != 64
                or any(c not in '0123456789abcdef' for c in input_receipt_sha256)):
            raise ValueError('The frozen input receipt fingerprint is required')
        if not math.isfinite(spacing_mm) or spacing_mm <= 0 or max_events < 2:
            raise ValueError('Positive fixed event spacing and explicit resources required')
        self.input_receipt_sha256 = input_receipt_sha256
        self.spacing_mm, self.max_events = float(spacing_mm), int(max_events)
        self.tree = cKDTree(self.xyz)

    def build(self, profile, fragment_spans):
        curve = CurveCarrier(profile['representation']); length = curve.length
        spans = np.asarray(fragment_spans, np.float64)
        if (spans.ndim != 2 or spans.shape[1] != 2 or not len(spans)
                or not np.isfinite(spans).all() or np.any(spans[:, 1] <= spans[:, 0])
                or np.any(spans < -1e-6) or np.any(spans > length + 1e-6)):
            raise ValueError('Every input fragment must lie on the same finite carrier')
        fragments = np.unique(np.clip(spans.ravel(), 0., length))
        support = np.asarray(profile.get('support_intervals_mm', []), np.float64).reshape(-1, 2)
        if (not np.isfinite(support).all() or np.any(support[:, 1] <= support[:, 0])
                or np.any(support < -1e-6) or np.any(support > length + 1e-6)):
            raise ValueError('Observed finite support intervals are malformed')
        support = np.clip(support, 0., length)
        support_events = np.unique(support.ravel())
        ids = np.asarray(profile.get('raw_support_point_ids', []))
        if (ids.ndim != 1 or (len(ids) and (ids.dtype.kind not in 'iu'
                or np.any(ids < 0) or np.any(ids >= len(self.xyz))))):
            raise ValueError('Observed support point indices must belong to this full raw input')
        ids = ids.astype(np.int64)
        if len(ids):
            projected, distance = curve.project(self.xyz[ids])
            retained = distance <= 3.
            projected, ids = projected[retained], ids[retained]
            projected, first = np.unique(projected, return_index=True); ids = ids[first]
            rejected_projections = int((~retained).sum())
        else:
            projected = np.empty(0, np.float64); rejected_projections = 0
        anchors = np.unique(np.r_[0., length, fragments, support_events, projected])
        grid = finite_carrier_grid(length, anchors, self.spacing_mm, self.max_events)
        positions = grid['positions_mm']
        if len(positions) > self.max_events:
            raise ValueError('Grid plus original observed events exceeds declared resources; no events dropped')
        points = curve.at(positions)
        nearest, _ = self.tree.query(points, workers=2)
        projection_ids = np.full(len(positions), -1, np.int64)
        projection_ids[np.searchsorted(positions, projected)] = ids
        inside = np.zeros(len(positions), bool)
        # Range marking preserves disjoint observation gaps without N*M arrays.
        for left, right in support:
            begin = np.searchsorted(positions, left, side='left')
            end = np.searchsorted(positions, right, side='right')
            inside[begin:end] = True
        provenance = np.column_stack((positions == 0., positions == length,
            np.isin(positions, support_events), np.isin(positions, fragments),
            np.isin(positions, projected), grid['is_grid'],
            np.full(len(positions), curve.kind == 'arc' and curve.arc.is_closed),
            np.full(len(positions), curve.kind == 'arc'), nearest / 3., projection_ids >= 0,
            np.full(len(positions), bool(len(support))), inside, positions / length)).astype(np.float32)
        if provenance.shape != (len(positions), len(COLUMNS)) or not np.isfinite(provenance).all():
            raise ValueError('Finite event evidence contract changed')
        return dict(schema=SCHEMA, positions_mm=positions, xyz_mm=points, provenance=provenance,
            projection_point_ids=projection_ids, input_receipt_sha256=self.input_receipt_sha256,
            curve_representation=profile['representation'], length_mm=length,
            audit=dict(events=len(positions), columns=list(COLUMNS), raw_points=len(self.xyz),
                original_fragment_boundaries=len(fragments), observed_support_boundaries=len(support_events),
                observed_point_projections=len(projected), source_point_projections_farther_than_3mm=rejected_projections,
                pure_grid_only=int((grid['is_grid'] & ~grid['is_existing_event']).sum()),
                grid_spacing_mm=grid['grid_spacing_mm'], grid_is_observed_contact=False,
                support_is_certified_weld_contact=False, GT_or_STEP_reads=0))


def event_identity(bank, index):
    point = np.asarray(bank['xyz_mm'][int(index)], '<f8').copy()
    point[point == 0.] = 0.  # Exact +0/-0 equality, no coordinate rounding.
    return hashlib.sha256(bytes.fromhex(bank['input_receipt_sha256']) + point.tobytes()).hexdigest()


def interval_child(bank, left, right):
    """Construct the selected whole interval without fitted or averaged ends."""
    left, right = int(left), int(right)
    positions = bank['positions_mm']
    if not (0 <= left < len(positions) and 0 <= right < len(positions) and positions[left] < positions[right]):
        raise ValueError('A complete strictly ordered finite event pair is required')
    curve = CurveCarrier(bank['curve_representation'])
    lo, hi = float(positions[left]), float(positions[right])
    ends = bank['xyz_mm'][[left, right]].tolist()
    if curve.kind == 'line':
        child = dict(kind='line', endpoints_mm=ends, length_mm=float(np.linalg.norm(np.asarray(ends[1]) - ends[0])),
                     is_closed=False)
        if child['length_mm'] <= 0:
            raise ValueError('Distinct arclength events collapsed at world-coordinate precision')
    else:
        arc = curve.arc.subarc(lo, hi)
        child = dict(kind='arc', endpoints_mm=ends, length_mm=arc.length_mm,
                     is_closed=arc.is_closed, arc_parameters=arc.to_parameters())
    child.update(boundary_event_ids=[event_identity(bank, left), event_identity(bank, right)],
        input_interval_mm=[lo, hi], input_event_pair=[left, right],
        input_receipt_sha256=bank['input_receipt_sha256'], finite_input_geometry_preserved=True,
        contact_or_weld_validity_certified=False)
    return child


def shared_event_records(bank, indices):
    """Grid-only endpoints cannot supply an observed common-boundary claim."""
    result = {}
    for index in indices:
        index = int(index); row = bank['provenance'][index]
        if row[4]:
            construction = 'OBSERVED_PROJECTION'
        elif row[2]:
            construction = 'OBSERVED_BOUNDARY'
        elif row[0] or row[1]:
            construction = 'FINITE_CARRIER_BOUNDARY'
        else:
            continue
        result[event_identity(bank, index)] = dict(xyz_mm=bank['xyz_mm'][index].tolist(),
            input_receipt_sha256=bank['input_receipt_sha256'], construction=construction)
    return result
