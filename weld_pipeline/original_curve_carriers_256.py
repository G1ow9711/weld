"""Complete source-defined 1D carriers over original raw XYZ rows.

Native q is the exact rational sum of outward edge-length bounds. It is in
millimetres but is not claimed to be exact physical arclength. Raw corners
belong to the source path, not to additional free decoder node slots.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
import math
import struct

import numpy as np

from weld_pipeline.concrete_curve_certificate_250 import (
    ConcreteCurve, UnprovenCurve, _fraction_bounds, _sqrt_fraction_bounds)

VERSION = 'COMPLETE_ORIGINAL_RAW_XYZ_CURVE_CARRIER_DOMAINS_256_v1'
POINT_BUDGET = 200001


def rational(value):
    if isinstance(value, F):
        return value
    if type(value) is int:
        return F(value)
    if type(value) in (float, np.float64, np.float32) and math.isfinite(float(value)):
        return F.from_float(float(value))
    raise ValueError('Finite numeric native q required; booleans and implicit strings forbidden')


def fraction_record(value):
    value = rational(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def read_fraction(value):
    if set(value) != {'numerator', 'denominator'}:
        raise ValueError('Explicit numerator and denominator required')
    answer = F(int(value['numerator']), int(value['denominator']))
    if fraction_record(answer) != value:
        raise ValueError('Canonical positive-denominator rational encoding required')
    return answer


def _least_rotation(sequence):
    """Booth's linear-time minimum rotation, retaining repeated row IDs."""
    size = len(sequence)
    if size < 2:
        return 0
    doubled = sequence + sequence
    a, b, matched = 0, 1, 0
    while a < size and b < size and matched < size:
        x, y = doubled[a + matched], doubled[b + matched]
        if x == y:
            matched += 1
        else:
            if x > y:
                a += matched + 1
                if a == b:
                    a += 1
            else:
                b += matched + 1
                if a == b:
                    b += 1
            matched = 0
    return min(a, b)


def canonical_raw_path(source_sha256, plane_key, raw_row_ids, is_closed):
    ids = tuple(raw_row_ids)
    if (type(is_closed) is not bool or len(ids) < 2
            or any(type(value) is not int or value < 0 for value in ids)
            or not isinstance(plane_key, bytes) or len(plane_key) != 32
            or not isinstance(source_sha256, str) or len(bytes.fromhex(source_sha256)) != 32):
        raise ValueError('Frozen source SHA, canonical plane, original row sequence and closure required')
    if is_closed and ids[0] != ids[-1]:
        raise ValueError('Declared closure requires the same original first/last row ID')
    edges = len(ids) - 1
    if is_closed:
        core = ids[:-1]
        forward, reverse = core, tuple(reversed(core))
        a, b = _least_rotation(forward), _least_rotation(reverse)
        fa = forward[a:] + forward[:a]
        rb = reverse[b:] + reverse[:b]
        if fa <= rb:
            canonical, orientation = fa, 1
            canonical_to_original = tuple((a + i) % edges for i in range(edges))
        else:
            canonical, orientation = rb, -1
            canonical_to_original = tuple(edges - 1 - (b + i) % edges for i in range(edges))
        canonical += (canonical[0],)
        original_to_canonical = [0] * edges
        for i, original in enumerate(canonical_to_original):
            original_to_canonical[original] = i
        original_to_canonical += [original_to_canonical[0]]
    else:
        if ids <= ids[::-1]:
            canonical, orientation = ids, 1
            original_to_canonical = list(range(edges + 1))
        else:
            canonical, orientation = ids[::-1], -1
            original_to_canonical = list(range(edges, -1, -1))
    key = (VERSION.encode('ascii') + b'\0' + bytes.fromhex(source_sha256) + plane_key
           + struct.pack('<?Q', is_closed, len(canonical))
           + struct.pack('<' + str(len(canonical)) + 'q', *canonical))
    original_edge_to_canonical = [
        original_to_canonical[i] if orientation == 1 else original_to_canonical[i + 1]
        for i in range(edges)]
    return {
        'carrier_id': sha256(key).hexdigest(), 'canonical_key_hex': key.hex(),
        'canonical_raw_row_ids': list(canonical), 'is_closed': is_closed,
        'original_direction_in_canonical_q': orientation,
        'original_vertex_to_canonical_occurrence': original_to_canonical,
        'original_edge_to_canonical_occurrence': original_edge_to_canonical,
        'duplicate_serialization_adds_capacity': False,
    }


@dataclass(frozen=True)
class ExactEdge:
    ordinal: int
    raw_a: int
    raw_b: int
    squared_length: F
    physical_lower: F
    physical_upper: F
    native_start: F
    native_end: F


class ExactCarrier:
    def __init__(self, points_mm, raw_row_ids, is_closed, *, carrier_id=None):
        points = np.asarray(points_mm, dtype=np.float64)
        ids = tuple(raw_row_ids)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or len(points) != len(ids) or not np.isfinite(points).all()
                or any(type(value) is not int or value < 0 for value in ids)
                or type(is_closed) is not bool):
            raise ValueError('Complete finite original raw XYZ and occurrence IDs required')
        if is_closed and (ids[0] != ids[-1] or not np.array_equal(points[0], points[-1])):
            raise ValueError('Closed carrier must preserve the exact source closing row and XYZ')
        self.points_mm = points.copy()
        self.raw_row_ids = ids
        self.is_closed = is_closed
        self.carrier_id = carrier_id
        self.points_exact = tuple(tuple(F.from_float(float(x)) for x in point) for point in points)
        self.prefix = [F(0)]
        self.edges = []
        lower_total = F(0)
        upper_total = F(0)
        for index, (a, b) in enumerate(zip(self.points_exact, self.points_exact[1:])):
            squared = sum((x - y) ** 2 for x, y in zip(a, b))
            lo, hi = _sqrt_fraction_bounds(squared)
            low, high = F.from_float(lo), F.from_float(hi)
            if low ** 2 > squared or high ** 2 < squared or (squared > 0 and high <= 0):
                raise UnprovenCurve('Original edge metric could not be enclosed')
            end = self.prefix[-1] + high
            self.edges.append(ExactEdge(index, ids[index], ids[index + 1], squared,
                                        low, high, self.prefix[-1], end))
            self.prefix.append(end)
            lower_total += low
            upper_total += high
        self.prefix = tuple(self.prefix)
        self.edges = tuple(self.edges)
        self.native_length = self.prefix[-1]
        self.physical_lower = lower_total
        self.physical_upper = upper_total
        self.native_excess_upper = upper_total - lower_total

    def metric_record(self):
        return {
            'native_length_mm': fraction_record(self.native_length),
            'physical_length_lower_mm': fraction_record(self.physical_lower),
            'physical_length_upper_mm': fraction_record(self.physical_upper),
            'native_q_excess_upper_mm': fraction_record(self.native_excess_upper),
            'native_q_is_exact_physical_arclength': False,
            'edge_count': len(self.edges),
            'zero_original_edges': sum(edge.squared_length == 0 for edge in self.edges),
            'positive_original_geometry': self.native_length > 0,
        }

    def point_exact(self, native_q):
        q = rational(native_q)
        if self.native_length <= 0:
            raise UnprovenCurve('Zero-length source carrier retained, positive curve not proved')
        if self.is_closed:
            q %= self.native_length
        elif not 0 <= q <= self.native_length:
            raise ValueError('Native q outside the complete original open carrier')
        if q == self.native_length:
            return self.points_exact[-1]
        index = min(len(self.edges) - 1, bisect_right(self.prefix, q) - 1)
        edge = self.edges[index]
        if edge.native_end == edge.native_start:
            return self.points_exact[index]
        t = (q - edge.native_start) / (edge.native_end - edge.native_start)
        return tuple(a + t * (b - a) for a, b in zip(self.points_exact[index], self.points_exact[index + 1]))

    def point_record(self, native_q):
        exact = self.point_exact(native_q)
        exported = tuple(float(x) for x in exact)
        error_squared = sum((F.from_float(value) - x) ** 2 for value, x in zip(exported, exact))
        return {'point_mm': list(exported), 'exact_point_mm': [fraction_record(x) for x in exact],
                'export_error_upper_mm': _sqrt_fraction_bounds(error_squared)[1],
                'native_q_mm': fraction_record(rational(native_q))}

    def slice(self, start_q, end_q, direction, turn_count=0, *,
              point_budget=POINT_BUDGET, winding_limit=None):
        if (type(direction) is not int or direction not in (-1, 1)
                or type(turn_count) is not int or turn_count < 0
                or type(point_budget) is not int or not 2 <= point_budget <= POINT_BUDGET):
            raise ValueError('Explicit direction, complete cycles and finite point budget required')
        if winding_limit is not None and (type(winding_limit) is not int or winding_limit < 1
                                          or turn_count > winding_limit):
            raise UnprovenCurve('Declared turn count exceeds the frozen finite source slot budget')
        length = self.native_length
        if length <= 0:
            raise UnprovenCurve('Complete zero-length source retained; positive traversal unproved')
        start, end = rational(start_q), rational(end_q)
        if self.is_closed:
            if not (0 <= start < length and 0 <= end < length):
                raise ValueError('Closed phase must be explicit in the canonical half-open q period')
            delta = (direction * (end - start)) % length + turn_count * length
            unwrapped_end = start + direction * delta
        else:
            if turn_count or not 0 <= start <= length or not 0 <= end <= length:
                raise ValueError('Open carrier forbids turns or out-of-domain phases')
            delta = direction * (end - start)
            unwrapped_end = end
        if delta <= 0:
            raise ValueError('Strictly positive directed native interval required')
        lower, upper = min(start, unwrapped_end), max(start, unwrapped_end)
        # This lower bound prevents huge cycle enumeration before the resource check.
        if self.is_closed and turn_count * len(self.edges) + 1 > point_budget:
            raise UnprovenCurve('Every original cycle occurrence exceeds the finite query budget')
        cycles = range(lower // length - 1, upper // length + 1) if self.is_closed else (0,)
        records = []
        physical_low = F(0)
        physical_high = F(0)
        for cycle in cycles:
            shift = cycle * length
            first = max(0, bisect_left(self.prefix, lower - shift) - 1)
            last = min(len(self.edges), bisect_right(self.prefix, upper - shift))
            for edge in self.edges[first:last]:
                a, b = edge.native_start + shift, edge.native_end + shift
                if a == b:
                    # Periodic phase plateaux have deterministic half-open ownership;
                    # an open source retains all zero occurrences at both endpoints.
                    retained = (lower <= a < upper if self.is_closed else lower <= a <= upper)
                    if not retained:
                        continue
                    lo = hi = a
                    point_a = self.points_exact[edge.ordinal]
                    point_b = self.points_exact[edge.ordinal + 1]
                    ratio = F(0)
                else:
                    lo, hi = max(a, lower), min(b, upper)
                    if lo >= hi:
                        continue
                    ta, tb = (lo - a) / (b - a), (hi - a) / (b - a)
                    pa, pb = self.points_exact[edge.ordinal], self.points_exact[edge.ordinal + 1]
                    point_a = tuple(x + ta * (y - x) for x, y in zip(pa, pb))
                    point_b = tuple(x + tb * (y - x) for x, y in zip(pa, pb))
                    ratio = (hi - lo) / (b - a)
                records.append((lo, cycle, edge.ordinal, hi, point_a, point_b, ratio))
                physical_low += edge.physical_lower * ratio
                physical_high += edge.physical_upper * ratio
                if len(records) + 1 > point_budget:
                    raise UnprovenCurve('Complete original corner sequence exceeds the finite query budget')
        records.sort(key=lambda value: value[:3])
        if direction == -1:
            records.reverse()
        points = [self.point_exact(start)]
        lineage = []
        for lo, cycle, ordinal, hi, a, b, ratio in records:
            if direction == -1:
                a, b = b, a
            if points[-1] != a:
                raise UnprovenCurve('Original directed source occurrences do not form an exact chain')
            points.append(b)
            lineage.append({'canonical_edge_ordinal': ordinal, 'period_ordinal': cycle,
                            'native_interval_lower_mm': fraction_record(lo),
                            'native_interval_upper_mm': fraction_record(hi),
                            'original_zero_edge': self.edges[ordinal].squared_length == 0,
                            'direction': direction})
        if points[-1] != self.point_exact(end) or not records:
            raise UnprovenCurve('Directed carrier endpoint expression disagrees with the complete chain')
        exported = [[float(value) for value in point] for point in points]
        max_error_squared = max(sum((F.from_float(value) - exact) ** 2
                                    for value, exact in zip(row, point))
                                for row, point in zip(exported, points))
        for a, b, exported_a, exported_b in zip(points, points[1:], exported, exported[1:]):
            if a != b and exported_a == exported_b:
                raise UnprovenCurve('Positive original geometry collapsed in the float64 export')
        export_error = _sqrt_fraction_bounds(max_error_squared)[1]
        closed = self.is_closed and start == end and turn_count > 0
        concrete = ConcreteCurve({'kind': 'polyline', 'points': exported,
                                  'is_closed': closed, 'geometry_uncertainty_mm': export_error})
        return {
            'version': VERSION, 'status': 'PASS_COMPLETE_ORIGINAL_DIRECTED_CARRIER_SLICE',
            'carrier_id': self.carrier_id, 'start_q_mm': fraction_record(start),
            'end_q_mm': fraction_record(end), 'unwrapped_end_q_mm': fraction_record(unwrapped_end),
            'direction': direction, 'turn_count': turn_count,
            'is_closed': closed, 'native_traversal_length_mm': fraction_record(delta),
            'physical_length_lower_mm': fraction_record(physical_low),
            'physical_length_upper_mm': fraction_record(physical_high),
            'original_edge_occurrences': lineage, 'world_curve': concrete.record,
            'source_corners_add_free_geometry_node_capacity': False,
            'all_source_corners_and_zero_edges_retained': True,
            'joint_GT_geometry_verified': False, 'material_identity_verified': False,
            'actual_model_prediction': False, 'full_contract_pass': False,
        }


def permitted_native_point(db, carrier, anchor_id, native_q):
    if type(anchor_id) is not int or anchor_id < 1 or carrier.carrier_id is None:
        raise ValueError('Existing canonical anchor and frozen carrier identity required')
    q = rational(native_q)
    if not 0 <= q <= carrier.native_length:
        raise ValueError('Position must lie within the original canonical carrier period')
    ordinals = [row[0] for row in db.execute(
        'SELECT edge_ordinal FROM carrier_regions WHERE anchor_id=? AND carrier_id=?',
        (anchor_id, carrier.carrier_id))]
    if not any(carrier.edges[index].native_start <= q <= carrier.edges[index].native_end
               for index in ordinals):
        raise ValueError('Native q outside every explicitly inherited original local edge')
    return dict(carrier.point_record(q), anchor_id=anchor_id,
                permitted_original_edge_ordinals=ordinals, own_local_permission_verified=True)
