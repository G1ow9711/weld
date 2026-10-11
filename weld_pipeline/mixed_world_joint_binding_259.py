"""Exact source world joins and occurrence-aware physical traversal.

Source permission, logical cuts, primitive kind and weld/material identity are
separate. Frozen247 capacity and254--258 artifacts are not changed here.
"""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction as F
import math

from .concrete_curve_certificate_250 import ConcreteCurve, UnprovenCurve, _sqrt_fraction_bounds
from .decoder_output_slots_247 import CAPACITIES, local_region_triangles
from .mixed_surface_curve_binding_253 import SurfaceCompiler
from .original_curve_carriers_256 import ExactCarrier, read_fraction
from .raw_triangle_world_charts_258 import (RawTriangleChart, exact, certify_world_line,
                                           certify_exact_affine_circle_image)


VERSION = 'EXPLICIT_MIXED_RAW_WORLD_JOINS_AND_COMPLETE_PARENT_TRAVERSAL_259_v1'
PROTOCOL = dict(
    version=VERSION,
    source_domains=['ORIGINAL_RAW_TRIANGLE', 'ORIGINAL_RAW_CARRIER', 'LEGACY_AFFINE_SURFACE'],
    local_role_capacity='UNCHANGED_247_CANONICAL_ANCHOR_ROLE_LOCAL_ORDINAL',
    carrier_endpoint='EXPLICIT_ORIGINAL_VERTEX_OR_STRICT_EDGE_INTERIOR_OCCURRENCE',
    zero_plateau='OCCURRENCE_ID_RETAINED_Q_OR_WORLD_EQUALITY_DOES_NOT_MERGE_OCCURRENCES',
    world_join='EXPLICIT_PHYSICAL_REFERENCE_AND_EXACT_SOURCE_WORLD_EQUALITY_AND_BOTH_OWN_PERMISSIONS',
    portal_alias='ORIGINAL_RAW_VERTEX_OR_SAME_RAW_EDGE_WEIGHTS_NOT_NEAR_COORDINATES',
    projection_connectors=False,
    intrinsic_corners_add_role_capacity=False,
    business_cuts='EXPLICIT_LOGICAL_BOUNDARY_IDENTITIES_ONLY',
    parent_turn_limit='CANONICAL_SOURCE_ANCHORS_TIMES_247_GEOMETRY_SPAN_CAPACITY',
    parent_total_turns='SUM_ALL_CHILD_AND_EXPLICIT_GAP_NATIVE_TRAVERSALS_OVER_THE_SAME_PERIOD',
    open_parent_turns='FLOOR_TOTAL_TRAVERSAL_COMPLETE_PERIODS_WITH_REMAINDER_COUNTED_IN_CAPACITY',
    closed_parent='EXPLICIT_PHASE_OCCURRENCE_AND_EXACT_DECLARED_COMPLETE_PERIODS',
    physical_length='ORIGINAL_PER_EDGE_DERIVED_LOWER_UPPER_INTEGRALS_NOT_NATIVE_Q_LENGTH',
    positive_gap='POSITIVE_COMPLETE_PHYSICAL_INTERVAL_AND_EVERY_POSITIVE_ORIGINAL_EXPORT_EDGE_REMAINS_POSITIVE',
    evaluation='STREAMED_SUBDIVISION_PRESERVES_EVERY_CORNER_ZERO_EDGE_AND_PERIOD_OCCURRENCE',
    evaluation_limits=dict(step_mm=1., chord_mm=.1, finite_point_budget=200001),
    whole_child_world_kind='SEPARATE_CERTIFICATE_FROM_SPAN_STORAGE_KIND',
    general_parent_graph_binding_complete=False,
    material_or_weld_identity_verified=False, full_contract_pass=False)


def _r(value):
    return read_fraction(value) if isinstance(value, dict) else exact(value)


def _world_key(raw_ids, weights):
    coefficients = {}
    for raw_id, weight in zip(raw_ids, weights):
        coefficients[raw_id] = coefficients.get(raw_id, F(0)) + weight
    return ('RAW_WEIGHTED_POINT', tuple(sorted((key, value) for key, value in coefficients.items() if value)))


def _export(point):
    exported = tuple(float(value) for value in point)
    if not all(map(math.isfinite, exported)):
        raise UnprovenCurve('Exact source point is not finitely exportable')
    squared = sum((exact(value) - original)**2 for value, original in zip(exported, point))
    return exported, _sqrt_fraction_bounds(squared)[1]


@dataclass(frozen=True)
class CarrierOccurrence:
    """Vertices remain distinct even when several have the same native q."""
    kind: str
    ordinal: int
    alpha: F = F(0)

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or record.get('kind') not in ('VERTEX', 'EDGE_INTERIOR'):
            raise ValueError('Explicit original carrier occurrence required')
        keys = {'kind', 'ordinal'} if record['kind'] == 'VERTEX' else {'kind', 'ordinal', 'alpha'}
        if set(record) != keys or type(record['ordinal']) is not int:
            raise ValueError('Exact occurrence fields and integer ordinal required')
        return cls(record['kind'], record['ordinal'], _r(record.get('alpha', 0)))

    def validate(self, carrier):
        if type(self.ordinal) is not int or self.kind not in ('VERTEX', 'EDGE_INTERIOR') or not isinstance(self.alpha, F):
            raise ValueError('Original occurrence type and ordinal required')
        if self.kind == 'VERTEX':
            if not 0 <= self.ordinal <= len(carrier.edges) or self.alpha != 0:
                raise ValueError('Original vertex occurrence outside carrier')
        elif not 0 <= self.ordinal < len(carrier.edges) or not 0 < self.alpha < 1:
            raise ValueError('Strict original edge-interior occurrence required; endpoints are vertices')
        return self

    def topological_parameter(self, carrier):
        self.validate(carrier)
        value = F(self.ordinal) + (self.alpha if self.kind == 'EDGE_INTERIOR' else 0)
        return value % len(carrier.edges) if carrier.is_closed else value

    def canonical_identity(self, carrier):
        self.validate(carrier)
        if self.kind == 'VERTEX' and carrier.is_closed and self.ordinal == len(carrier.edges):
            return 'VERTEX', 0, F(0)
        return self.kind, self.ordinal, self.alpha

    def exact_point(self, carrier):
        self.validate(carrier)
        if self.kind == 'VERTEX':
            return carrier.points_exact[self.ordinal]
        a, b = carrier.points_exact[self.ordinal:self.ordinal + 2]
        return tuple(x + self.alpha * (y - x) for x, y in zip(a, b))

    def native_q(self, carrier):
        self.validate(carrier)
        if self.kind == 'VERTEX':
            value = carrier.prefix[self.ordinal]
        else:
            edge = carrier.edges[self.ordinal]
            value = edge.native_start + self.alpha * (edge.native_end - edge.native_start)
        return value % carrier.native_length if carrier.is_closed and carrier.native_length else value

    def incident_edges(self, carrier):
        self.validate(carrier)
        if self.kind == 'EDGE_INTERIOR':
            return (self.ordinal,)
        count = len(carrier.edges)
        if carrier.is_closed:
            return tuple(sorted({(self.ordinal - 1) % count, self.ordinal % count}))
        return tuple(index for index in (self.ordinal - 1, self.ordinal) if 0 <= index < count)


class OccurrenceSlice:
    """A complete lazy source walk. No original corner or zero edge is dropped."""
    def __init__(self, carrier, start, end, direction, extra_turns=0, *, maximum_turns):
        if (type(direction) is not int or direction not in (-1, 1)
                or type(extra_turns) is not int or extra_turns < 0
                or type(maximum_turns) is not int or maximum_turns < 1):
            raise ValueError('Explicit direction, nonnegative complete turns and frozen capacity required')
        self.carrier, self.start, self.end = carrier, start.validate(carrier), end.validate(carrier)
        self.direction, self.extra_turns = direction, extra_turns
        count = len(carrier.edges)
        first, last = start.topological_parameter(carrier), end.topological_parameter(carrier)
        if carrier.is_closed:
            delta = (direction * (last - first)) % count + extra_turns * count
        else:
            if extra_turns:
                raise ValueError('Original open carrier cannot add periodic cycles')
            delta = direction * (last - first)
        if delta <= 0 or carrier.native_length <= 0:
            raise ValueError('Strict positive source traversal required; zero support is retained separately')
        self.first = first
        self.last = first + direction * delta
        self.maximum_turns = maximum_turns
        # Topological turns also bound walks over arbitrarily long zero plateaux.
        if carrier.is_closed and delta > maximum_turns * count:
            raise UnprovenCurve('Complete plus partial source traversal exceeds frozen global span capacity')

    def segments(self):
        carrier, count = self.carrier, len(self.carrier.edges)
        lower, upper = sorted((self.first, self.last))
        first, last = math.floor(lower), math.ceil(upper)
        cells = range(first, last) if self.direction == 1 else range(last - 1, first - 1, -1)
        for cell in cells:
            ordinal = cell % count if carrier.is_closed else cell
            if not 0 <= ordinal < count:
                raise ValueError('Occurrence interval leaves the original open carrier')
            lo, hi = max(lower, F(cell)) - cell, min(upper, F(cell + 1)) - cell
            if not lo < hi:
                continue
            edge = carrier.edges[ordinal]
            a, b = carrier.points_exact[ordinal:ordinal + 2]
            start = tuple(x + lo * (y - x) for x, y in zip(a, b))
            end = tuple(x + hi * (y - x) for x, y in zip(a, b))
            if self.direction == -1:
                start, end = end, start
            width = hi - lo
            yield dict(original_edge_ordinal=ordinal, original_period_ordinal=cell // count if carrier.is_closed else 0,
                       start_exact=start, end_exact=end, original_zero_edge=edge.squared_length == 0,
                       edge_fraction=width, physical_lower=edge.physical_lower * width,
                       physical_upper=edge.physical_upper * width,
                       native_length=(edge.native_end - edge.native_start) * width,
                       direction=self.direction)

    def metrics(self, *, check_export=False):
        physical_lo = physical_hi = native = F(0)
        count = zeros = 0
        previous = self.start.exact_point(self.carrier)
        for row in self.segments():
            if row['start_exact'] != previous:
                raise UnprovenCurve('Original occurrences do not form one exact directed world chain')
            previous = row['end_exact']
            if check_export:
                a, _ = _export(row['start_exact'])
                b, _ = _export(row['end_exact'])
                if row['start_exact'] != row['end_exact'] and a == b:
                    raise UnprovenCurve('Positive source geometry or gap collapsed in world export')
            physical_lo += row['physical_lower']
            physical_hi += row['physical_upper']
            native += row['native_length']
            count += 1
            zeros += row['original_zero_edge']
        if previous != self.end.exact_point(self.carrier) or count == 0 or physical_lo <= 0:
            raise UnprovenCurve('Complete positive physical source walk is not proved')
        if self.carrier.is_closed and native > self.maximum_turns * self.carrier.native_length:
            raise UnprovenCurve('Whole native metric traversal exceeds frozen global span capacity')
        return dict(status='PASS_COMPLETE_OCCURRENCE_AWARE_PHYSICAL_WALK', original_edge_occurrences=count,
                    original_zero_edges_retained=zeros, physical_length_lower_mm=physical_lo,
                    physical_length_upper_mm=physical_hi, native_traversal_length_mm=native,
                    source_corners_add_role_capacity=False, geometry_or_material_GT_verified=False)

    def evaluation_points(self, *, step_mm=1., chord_mm=.1, point_budget=200001):
        if (not 0 < step_mm <= 1. or not 0 < chord_mm <= .1
                or type(point_budget) is not int or point_budget < 2):
            raise ValueError('Frozen physical step/chord limits and finite evaluation budget required')
        emitted = 0
        for row in self.segments():
            a, b = row['start_exact'], row['end_exact']
            # Conservative float spacing over every convex interpolation point.
            ulps = []
            for x, y in zip(a, b):
                magnitude = float(max(abs(x), abs(y)))
                if not math.isfinite(magnitude):
                    raise UnprovenCurve('Original world interval not finitely exportable')
                ulps.append(exact(math.ulp(math.nextafter(magnitude, math.inf))))
            numeric_upper = _sqrt_fraction_bounds(sum(value * value for value in ulps))[1]
            margin = exact(step_mm) - 2 * exact(numeric_upper)
            if margin <= 0 or numeric_upper > chord_mm:
                raise UnprovenCurve('World float spacing cannot certify the frozen step/chord limits')
            subdivisions = max(1, math.ceil(row['physical_upper'] / margin))
            if emitted + subdivisions + 1 > point_budget:
                raise UnprovenCurve('Evaluation budget exhausted; complete original walk remains unproven')
            previous = None
            for index in range(subdivisions + 1):
                fraction = F(index, subdivisions)
                point = tuple(x + fraction * (y - x) for x, y in zip(a, b))
                exported, error = _export(point)
                if previous is not None and previous[0] != point and previous[1] == exported:
                    raise UnprovenCurve('Positive evaluation subinterval collapsed in world export')
                previous = point, exported
                emitted += 1
                yield dict(point_mm=exported, exact_point_mm=point, world_export_error_upper_mm=error,
                           maximum_physical_step_upper_mm=row['physical_upper'] / subdivisions + 2 * exact(numeric_upper),
                           chord_error_upper_mm=numeric_upper, original_edge_ordinal=row['original_edge_ordinal'],
                           original_period_ordinal=row['original_period_ordinal'], original_zero_edge=row['original_zero_edge'],
                           intrinsic_corner_or_slice_endpoint=index in (0, subdivisions), evaluation_index=emitted - 1)


def certify_complete_carrier_parent(slices, *, anchor_count, direction, is_closed, turn_count):
    if (not slices or type(anchor_count) is not int or anchor_count < 1
            or type(direction) is not int or direction not in (-1, 1)
            or type(is_closed) is not bool or type(turn_count) is not int or turn_count < 0):
        raise ValueError('Complete child/gap walk and explicit frozen parent declarations required')
    carrier = slices[0].carrier
    capacity = anchor_count * CAPACITIES['geometry_span']
    total = low = high = F(0)
    count = zeros = 0
    for index, walk in enumerate(slices):
        same_source = (walk.carrier is carrier or (walk.carrier.carrier_id == carrier.carrier_id
                       and walk.carrier.raw_row_ids == carrier.raw_row_ids
                       and walk.carrier.points_exact == carrier.points_exact and walk.carrier.is_closed == carrier.is_closed))
        if not same_source or walk.direction != direction or walk.maximum_turns != capacity:
            raise ValueError('One declared original carrier period and source-derived capacity required')
        if index and slices[index - 1].end.canonical_identity(carrier) != walk.start.canonical_identity(carrier):
            raise ValueError('Parent occurrence chain changed phase or skipped a zero-plateau occurrence')
        metrics = walk.metrics(check_export=True)
        total += metrics['native_traversal_length_mm']
        low += metrics['physical_length_lower_mm']
        high += metrics['physical_length_upper_mm']
        count += metrics['original_edge_occurrences']
        zeros += metrics['original_zero_edges_retained']
    if carrier.is_closed:
        ratio = total / carrier.native_length
        if ratio > capacity:
            raise UnprovenCurve('Accumulated whole-parent complete plus partial traversal exceeds global capacity')
        same_phase = slices[0].start.canonical_identity(carrier) == slices[-1].end.canonical_identity(carrier)
        if is_closed:
            if not same_phase or turn_count < 1 or ratio != turn_count:
                raise ValueError('Closed parent requires its original phase and exact declared complete periods')
        elif same_phase or math.floor(ratio) != turn_count:
            raise ValueError('Open parent phase or declared complete turns disagrees with the whole walk')
    elif is_closed or turn_count:
        raise ValueError('Original open carrier cannot declare a periodic parent')
    return dict(status='PASS_COMPLETE_PARENT_PHASE_DIRECTION_AND_PHYSICAL_TRAVERSAL',
                source_global_span_capacity=capacity, total_native_traversal_mm=total,
                physical_length_lower_mm=low, physical_length_upper_mm=high,
                original_edge_occurrences=count, original_zero_edges_retained=zeros,
                all_children_and_explicit_gaps_must_be_supplied=True, material_or_weld_identity_verified=False)


def certify_positive_world_gap(walk):
    value = walk.metrics(check_export=True)
    return dict(status='PASS_STRICT_POSITIVE_COMPLETE_WORLD_GAP_AND_EXPORT', **{
        key: item for key, item in value.items() if key != 'status'}, business_boundaries_must_be_distinct=True,
        no_positive_length_cutoff_or_gap_merge=True)


def certify_streamed_world_line(segment_iterator):
    axis = previous = first = None
    count = zeros = 0
    for row in segment_iterator:
        a, b = row['start_exact'], row['end_exact']
        if previous is not None and a != previous:
            raise ValueError('Whole child world span chain is discontinuous')
        first = a if first is None else first
        delta = tuple(y - x for x, y in zip(a, b))
        if any(delta):
            if axis is None:
                axis = delta
            # The finite three-point certificate checks direction and collinearity.
            certify_world_line((first, a, b))
            if sum(x * y for x, y in zip(axis, delta)) <= 0:
                raise ValueError('Whole child world line backtracks')
        else:
            zeros += 1
        previous, count = b, count + 1
    if axis is None:
        raise ValueError('Whole line child has no positive world geometry')
    return dict(status='PASS_COMPLETE_STREAMED_COLLINEAR_SAME_DIRECTION_WORLD_LINE',
                original_span_edge_occurrences=count, original_zero_edges_retained=zeros,
                stored_polyline_was_not_automatically_labelled_arc=True, GT_geometry_verified=False)


def certify_world_arc_parameters(record):
    curve = ConcreteCurve(record)
    if curve.kind != 'arc' or curve.record['dimensions'] != 3:
        raise ValueError('One declared analytic world-space arc authority required')
    image = certify_exact_affine_circle_image(curve.record['u_axis'], curve.record['v_axis'])
    if image['status'] != 'PASS_EXACT_RECORDED_AFFINE_CIRCLE_SIMILARITY':
        raise ValueError('Unequal or nonorthogonal world basis is an ellipse, not an arc authority')
    norm_squared = sum(exact(value)**2 for value in curve.record['u_axis'])
    return dict(status='PASS_ANALYTIC_WORLD_CIRCLE_AUTHORITY',
                world_radius_squared_mm=exact(curve.record['radius_mm'])**2 * norm_squared,
                exact_equal_basis_norm_and_orthogonality_verified=True,
                source_own_curve_coverage_verified=False, complete_child_body_verified=False)


def certify_world_arc_restrictions(authority_record, ordered_restrictions, *, authority_ref):
    """A kind/phase gate only. Source-domain coverage is checked separately."""
    kind = certify_world_arc_parameters(authority_record)
    if not ordered_restrictions or not isinstance(authority_ref, tuple) or len(authority_ref) != 3:
        raise ValueError('Existing geometry-span authority slot and complete ordered restrictions required')
    if authority_ref[1] != 'geometry_span' or type(authority_ref[0]) is not int or authority_ref[2] != 0:
        raise ValueError('Arc authority must use one existing247 geometry-span slot')
    previous = F(0)
    for restriction in ordered_restrictions:
        if set(restriction) != {'authority_ref', 'parameter_interval'} or restriction['authority_ref'] != authority_ref:
            raise ValueError('Every world arc span must be a restriction of the same declared authority')
        interval = restriction['parameter_interval']
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError('Explicit complete normalized analytic parameter interval required')
        first, last = map(_r, interval)
        if not 0 <= first < last <= 1 or first != previous:
            raise ValueError('Arc restrictions skip, overlap or backtrack in authoritative phase')
        previous = last
    if previous != 1:
        raise ValueError('Complete child world arc body was not covered')
    return dict(status='PASS_COMPLETE_ORDERED_WORLD_ARC_RESTRICTIONS_KIND_AND_PHASE',
                authority_kind=kind, restriction_count=len(ordered_restrictions),
                source_own_curve_coverage_verified=False, storage_polyline_accepted_by_nearness=False,
                world_sampling_certificate_required=True, GT_geometry_verified=False)


class FrozenMixedSourceProvider:
    def __init__(self, slots, carriers, raw_xyz_mm, *, maximum_cached_domains=3):
        if any(connection.execute('PRAGMA query_only').fetchone() != (1,) for connection in (slots, carriers)):
            raise ValueError('Read-only frozen source manifests required')
        if type(maximum_cached_domains) is not int or maximum_cached_domains < 1:
            raise ValueError('Positive bounded source-context cache required')
        self.slots, self.carriers, self.raw = slots, carriers, raw_xyz_mm
        self.anchor_count = slots.execute('SELECT count(*) FROM anchors').fetchone()[0]
        self.maximum_turns = self.anchor_count * CAPACITIES['geometry_span']
        self.maximum_cached_domains = maximum_cached_domains
        self.carrier_cache, self.surface_cache = OrderedDict(), OrderedDict()
        self.compiler = SurfaceCompiler(slots)

    def carrier(self, carrier_id):
        import json
        if carrier_id in self.carrier_cache:
            self.carrier_cache.move_to_end(carrier_id)
            return self.carrier_cache[carrier_id]
        row = self.carriers.execute('SELECT is_closed,raw_row_ids_json FROM carriers WHERE carrier_id=?', (carrier_id,)).fetchone()
        if row is None:
            raise ValueError('Carrier identity absent from the complete original manifest')
        ids = json.loads(row[1])
        result = ExactCarrier(self.raw[ids], ids, bool(row[0]), carrier_id=carrier_id)
        self.carrier_cache[carrier_id] = result
        while len(self.carrier_cache) > self.maximum_cached_domains:
            self.carrier_cache.popitem(last=False)
        return result

    def raw_position(self, anchor, face_id, triangle_ordinal, lambda1, lambda2):
        row = self.slots.execute('SELECT raw_a,raw_b,raw_c FROM triangles WHERE face_id=? AND triangle_ordinal=?',
                                 (face_id, triangle_ordinal)).fetchone()
        if row is None or self.slots.execute('SELECT 1 FROM regions WHERE anchor_id=? AND face_id=? AND triangle_ordinal=? LIMIT 1',
                                             (anchor, face_id, triangle_ordinal)).fetchone() is None:
            raise ValueError('Original raw-triangle position outside its defining anchor local permission')
        chart = RawTriangleChart(self.raw[list(row)], row, face_id=face_id, triangle_ordinal=triangle_ordinal)
        first, second = _r(lambda1), _r(lambda2)
        point = chart.point_exact(first, second)
        exported, uncertainty = _export(point)
        return dict(domain=('RAW_TRIANGLE', face_id, triangle_ordinal), root_expression=_world_key(row, (1 - first - second, first, second)),
                    point_exact=point, point_mm=exported, world_export_upper_mm=uncertainty,
                    own_local_permission_verified=True, raw_support_rank=chart.rank)

    def carrier_position(self, anchor, carrier_id, occurrence):
        carrier = self.carrier(carrier_id)
        permitted = {row[0] for row in self.carriers.execute('SELECT edge_ordinal FROM carrier_regions WHERE anchor_id=? AND carrier_id=?',
                                                          (anchor, carrier_id))}
        if not permitted.intersection(occurrence.incident_edges(carrier)):
            raise ValueError('Original carrier occurrence outside its defining anchor local edge permission')
        if occurrence.kind == 'VERTEX':
            key = _world_key((carrier.raw_row_ids[occurrence.ordinal],), (F(1),))
        else:
            key = _world_key(carrier.raw_row_ids[occurrence.ordinal:occurrence.ordinal + 2], (1 - occurrence.alpha, occurrence.alpha))
        point = occurrence.exact_point(carrier)
        exported, uncertainty = _export(point)
        return dict(domain=('RAW_CARRIER', carrier_id), root_expression=key, occurrence=occurrence,
                    point_exact=point, point_mm=exported, world_export_upper_mm=uncertainty,
                    own_local_permission_verified=True, native_q_mm=occurrence.native_q(carrier))

    def affine_position(self, anchor, face_id, uv):
        if len(uv) != 2:
            raise ValueError('Two declared legacy affine parameters required')
        if face_id not in self.surface_cache:
            self.surface_cache[face_id] = self.compiler.face(face_id)
            while len(self.surface_cache) > self.maximum_cached_domains:
                self.surface_cache.popitem(last=False)
        self.surface_cache.move_to_end(face_id)
        context = self.surface_cache[face_id]
        local = tuple(map(_r, uv))
        regions = local_region_triangles(self.slots, anchor, face_id)
        if not any(context['domain'].original_triangle_member(ordinal, local) for ordinal in regions):
            raise ValueError('Legacy affine position outside its defining anchor original local permission')
        point = tuple(exact(float(context['origin'][axis])) + sum(local[index] * exact(float(context['basis'][index, axis]))
                      for index in range(2)) for axis in range(3))
        exported, uncertainty = _export(point)
        return dict(domain=('LEGACY_AFFINE', face_id), root_expression=('AFFINE_POINT', face_id, local),
                    point_exact=point, point_mm=exported, world_export_upper_mm=uncertainty,
                    own_local_permission_verified=True)


def prove_explicit_world_join(left, right, *, physical_ref_declared):
    if physical_ref_declared is not True:
        raise ValueError('Source world equality does not create an implicit physical reference')
    if not left.get('own_local_permission_verified') or not right.get('own_local_permission_verified'):
        raise ValueError('Both defining roles require their own original source local permission')
    if left['point_exact'] != right['point_exact']:
        raise UnprovenCurve('Different exact source world points; no approximate alias or default projection bridge')
    return dict(status='PASS_EXPLICIT_EXACT_WORLD_JOIN_WITH_BOTH_OWN_PERMISSIONS',
                same_original_raw_expression=left['root_expression'] == right['root_expression'],
                proven_exact_world_point_equality=True, occurrence_identity_remains_separate=True,
                material_or_business_identity_inferred=False)
