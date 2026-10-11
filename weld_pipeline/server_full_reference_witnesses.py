"""Offline owner-bound line/arc labels on all frozen finite input profiles.

Projected endpoints are supervision, never input features or inserted events.
Every accepted interval passes the existing D3 child test. Complete reference
parents remain in SupervisionView even when this carrier representation cannot
express a child. Missing witnesses are unknown, not physical negative labels.
"""
from collections import Counter, OrderedDict
import math

import numpy as np

from .contract_curve_geometry import CircularArc, sample_arc
from .contract_geometry_d3_409 import child_error
from .legacy_incremental_event_runtime_393 import reverse_segment
from .server_circle_capacity_prefilter import possible_circle_indices
from .server_finite_carrier_grid import finite_carrier_grid
from .sjg_curve_queries_907 import CurveCarrier
from .world_set_evaluation_285 import reference_segment
from .world_set_runtime_285 import line_segment


class ReferenceWitnessBuilder:
    def __init__(self, inp):
        self.inp = inp
        self.curves = [CurveCarrier(profile['representation']) for profile in inp.profiles]
        self.line_ids = np.array([i for i, curve in enumerate(self.curves) if curve.kind == 'line'], dtype=np.int64)
        self.arc_ids = np.array([i for i, curve in enumerate(self.curves) if curve.kind == 'arc'], dtype=np.int64)
        self.origins = np.array([self.curves[i].ends[0] for i in self.line_ids]).reshape(-1, 3)
        self.delta = np.array([self.curves[i].ends[1] - self.curves[i].ends[0] for i in self.line_ids]).reshape(-1, 3)
        self.length2 = np.square(self.delta).sum(1)
        self.centers = np.array([self.curves[i].arc.center_mm for i in self.arc_ids]).reshape(-1, 3)
        self.normals = np.array([self.curves[i].arc.plane_normal for i in self.arc_ids]).reshape(-1, 3)
        self.radii = np.array([self.curves[i].arc.radius_mm for i in self.arc_ids])
        self.grids = OrderedDict()

    def grid(self, pid):
        if pid not in self.grids:
            self.grids[pid] = finite_carrier_grid(self.curves[pid].length, [])['positions_mm']
            if len(self.grids) > 16:
                self.grids.popitem(last=False)
        self.grids.move_to_end(pid)
        return self.grids[pid]

    def possible_profiles(self, child, reference):
        points = np.asarray(reference['points'], dtype=np.float64)
        if child['original_kind'] == 'line':
            distances = []
            for point in points[[0, -1]]:
                delta = point - self.origins
                parameter = np.clip(np.einsum('ij,ij->i', delta, self.delta) / self.length2, 0, 1)
                distances.append(np.linalg.norm(delta - parameter[:, None] * self.delta, axis=1))
            # Numerical guard only retains candidates; exact D3 still decides.
            scale = max(1., float(np.abs(points).max()))
            return self.line_ids[np.maximum(*distances) <= 3. + 128 * np.finfo(float).eps * scale]
        possible = possible_circle_indices(self.centers, self.normals, self.radii, points)
        return self.arc_ids[np.asarray(possible, dtype=np.int64)]

    def segment(self, pid, lo, hi):
        curve = self.curves[pid]
        if hi <= curve.length:
            representation = curve.segment(lo, hi)
        else:
            if curve.kind != 'arc' or not curve.arc.is_closed or hi - lo >= curve.length:
                raise ValueError('Wrapped intervals require an explicit open subarc of a closed input circle')
            arc = curve.arc
            start = (curve.at(np.array([lo]))[0] - arc.center_mm) / arc.radius_mm
            represented = CircularArc(arc.center_mm, arc.radius_mm, arc.plane_normal, start,
                                      math.copysign((hi - lo) / arc.radius_mm, arc.sweep_rad), False)
            representation = dict(kind='arc', arc_parameters=represented.to_parameters())
        return (line_segment(*representation['endpoints_mm']) if curve.kind == 'line'
                else sample_arc(CircularArc.from_parameters(representation['arc_parameters'])))

    def projected_interval(self, pid, reference, full_closed):
        curve = self.curves[pid]
        if full_closed and curve.kind == 'arc' and curve.arc.is_closed:
            return 0., curve.length
        q, _ = curve.project(np.asarray(reference['points']))
        if curve.kind == 'arc' and curve.arc.is_closed:
            q = np.unwrap(q / curve.length * (2 * np.pi)) * curve.length / (2 * np.pi)
            lo, hi = float(q.min()), float(q.max())
            width = hi - lo
            if width >= curve.length:
                raise ValueError('Whole-loop extent lacks a matching closed reference contract')
            lo %= curve.length
            return lo, lo + width
        return float(min(q[0], q[-1])), float(max(q[0], q[-1]))

    def query_ranges(self, pid, lo, hi):
        grid = self.grid(pid)
        length = self.curves[pid].length
        pieces = [(lo, hi)] if hi <= length else [(lo, length), (0., hi - length)]
        ranges = [[int(np.searchsorted(grid, a, side='left')), int(np.searchsorted(grid, b, side='right'))]
                  for a, b in pieces]
        ranges.extend([[0, 0]] * (2 - len(ranges)))
        return np.asarray(ranges, dtype=np.int64)

    def input_grid_point_limit(self, pid, lo, hi):
        # A tiny reference may contain no fixed query. Try an existing grid
        # extent and accept it only if the original D3 geometry also accepts it.
        grid = self.grid(pid)
        curve = self.curves[pid]
        center = (lo + hi) / 2
        center = center % curve.length if curve.kind == 'arc' and curve.arc.is_closed else np.clip(center, 0., curve.length)
        index = int(np.argmin(abs(grid - center)))
        return float(grid[max(0, index - 1)]), float(grid[min(len(grid) - 1, index + 1)])

    def build(self, supervision, progress=None):
        records = []
        missing = []
        rejections = Counter()
        for parent_index, parent in enumerate(supervision.parents):
            full_closed = parent['is_closed'] and len(parent['children']) == 1
            for child_index, child in enumerate(parent['children']):
                native = dict(child, world_representation=child['representation'])
                reference = reference_segment(native)
                candidates = self.possible_profiles(child, reference)
                accepted = 0
                for profile_id in candidates:
                    pid = int(profile_id)
                    try:
                        lo, hi = self.projected_interval(pid, reference, full_closed)
                    except ValueError:
                        rejections['UNSUPPORTED_PROJECTED_LOOP_EXTENT'] += 1
                        continue
                    grid_limit = False
                    if hi <= lo or not np.any(np.diff(self.query_ranges(pid, lo, hi), axis=1) > 0):
                        lo, hi = self.input_grid_point_limit(pid, lo, hi)
                        grid_limit = True
                    if not hi > lo:
                        rejections['NO_POSITIVE_INPUT_GRID_EXTENT'] += 1
                        continue
                    segment = self.segment(pid, lo, hi)
                    segment['weld_type'] = child['weld_type']
                    checked = None
                    reversed_direction = False
                    for reverse in (False, True):
                        represented = dict(segment, **reverse_segment(segment)) if reverse else segment
                        checked = child_error(represented, child, reference, full_closed, False)
                        if checked is not None:
                            reversed_direction = reverse
                            break
                    if checked is None:
                        rejections['ORIGINAL_D3_REJECTED'] += 1
                        continue
                    ranges = self.query_ranges(pid, lo, hi)
                    if not np.any(ranges[:, 1] > ranges[:, 0]):
                        rejections['NO_FIXED_INPUT_QUERY'] += 1
                        continue
                    records.append(dict(profile=pid, parent=parent_index, child=child_index,
                                        attribute_row=int(supervision.parent_offsets[parent_index] + child_index),
                                        kind=0 if child['original_kind'] == 'line' else 1,
                                        weld_type=0 if child['weld_type'] == 'horizontal' else 1,
                                        lo=lo, hi=hi, query_ranges=ranges,
                                        direction_reversed=reversed_direction, full_closed=full_closed,
                                        wraps=hi > self.curves[pid].length,
                                        input_grid_point_limit=grid_limit))
                    accepted += 1
                if not accepted:
                    missing.append(dict(parent=parent_index, child=child_index, kind=child['original_kind'],
                                        same_kind_prefilter_profiles=len(candidates),
                                        reason='NO_D3_ACCEPTED_INTERVAL_WITH_FIXED_INPUT_QUERY'))
                if progress:
                    progress(parent_index, child_index, len(records))
        counts = Counter((row['parent'], row['child']) for row in records)
        G = supervision.target['original_children']
        def column(name, dtype):
            return np.asarray([row[name] for row in records], dtype=dtype)
        arrays = dict(profile=column('profile', np.int64), parent=column('parent', np.int64),
                      child=column('child', np.int64), attribute_row=column('attribute_row', np.int64),
                      kind=column('kind', np.int8), weld_type=column('weld_type', np.int8),
                      interval_mm=np.asarray([[row['lo'], row['hi']] for row in records], dtype=np.float64).reshape(-1, 2),
                      query_ranges=np.asarray([row['query_ranges'] for row in records], dtype=np.int64).reshape(-1, 2, 2),
                      direction_reversed=column('direction_reversed', bool), full_closed=column('full_closed', bool),
                      wraps=column('wraps', bool), input_grid_point_limit=column('input_grid_point_limit', bool),
                      owner_weight=np.asarray([1 / (G * counts[row['parent'], row['child']]) for row in records]),
                      parent_offsets=supervision.parent_offsets)
        return arrays, dict(original_children=G, original_parents=len(supervision.parents),
                            represented_children=len(counts), missing_children=missing,
                            witnesses=len(records), geometry_denominator_retained=G,
                            positive_owner_mass=float(arrays['owner_weight'].sum()),
                            rejections=dict(rejections), physical_negative_labels=False,
                            wrapped_arc_witnesses=int(arrays['wraps'].sum()),
                            numerical_or_short_extent_input_grid_witnesses=int(arrays['input_grid_point_limit'].sum()),
                            original_D3_checked_every_witness=True,
                            actual_input_grid_no_reference_coordinate_insertion=True)
