"""Complete owner targets for native and visually added finite input curves.

Retains the established responsibility and full target construction. The only
feature table contains genuine native rows, with explicit global row indices;
new curves receive no fabricated legacy columns. References choose assignment
to an existing one-mm query and never create query positions or input events.
Free geometric target arrays remain supervision, not a regression output route.
"""
from collections import defaultdict
import math
import numpy as np
from .contract_curve_geometry import CircularArc
from .server_finite_carrier_grid import finite_carrier_grid
from .server_full_training_samples import FullTrainingAssignments, query_frame, geometry_descriptor


class VisualAugmentedTrainingAssignments(FullTrainingAssignments):
    def build(self, supervision, witnesses, seed):
        by_owner = defaultdict(list)
        for i, row in enumerate(witnesses['attribute_row']):
            by_owner[int(row)].append(i)
        rng = np.random.default_rng(seed)
        rows = []
        children = [c for p in supervision.parents for c in p['children']]
        for owner, child in enumerate(children):
            alternatives = by_owner[owner]
            strong = bool(alternatives)
            if strong:
                wi = int(rng.choice(alternatives))
                pid = int(witnesses['profile'][wi])
                lo, hi = map(float, witnesses['interval_mm'][wi])
            else:
                pid, lo, hi = self.closest(child)
            curve = self.curves[pid]
            mid = (lo + hi) / 2
            if curve.kind == 'arc' and curve.arc.is_closed:
                mid %= curve.length
            mid = float(np.clip(mid, 0, curve.length))
            grid = finite_carrier_grid(curve.length, [])['positions_mm']
            query = float(grid[int(np.argmin(abs(grid - mid)))])
            point, basis, observed_support = query_frame(self.inp, pid, query)
            if child['original_kind'] == 'line':
                endpoints = np.asarray(child['representation']['endpoints_mm'], float)
                arc_shape = np.zeros(11, float)
            else:
                arc = CircularArc.from_parameters(child['representation']['arc_parameters'])
                endpoints = arc.point_at(np.array([0., arc.length_mm]))
                arc_shape = np.r_[(arc.center_mm - point) @ basis / 1000,
                                  math.log(arc.radius_mm / 1000), arc.plane_normal @ basis,
                                  arc.start_direction @ basis, arc.sweep_rad / math.pi]
            projected, _ = curve.project(endpoints)
            direction = int(projected[1] < projected[0])
            if strong:
                direction = int(witnesses['direction_reversed'][wi])
            frame_axes = supervision.raw_export_axes[owner] @ basis
            frame_offsets = (supervision.frame_translations[owner] - endpoints) @ basis / 1000
            rows.append(dict(profile=pid, query=query, point=point, basis=basis,
                             geometry=geometry_descriptor(curve, point, basis, observed_support),
                             strong=strong, interval=np.array([lo, hi]) / curve.length,
                             interval_known=strong and hi <= curve.length,
                             direction=direction,
                             points=(endpoints - point) @ basis / 1000,
                             arc_shape=arc_shape, frame_axes=frame_axes,
                             frame_offsets=frame_offsets))
        if len(rows) != supervision.target['original_children']:
            raise ValueError('A missing carrier removed a complete child target')
        def stack(name, dtype=np.float32):
            return np.asarray([r[name] for r in rows], dtype=dtype)
        profile = stack('profile', np.int64)
        native_rows = np.flatnonzero(profile < self.inp.original_profile_count)
        visual_rows = np.flatnonzero(profile >= self.inp.original_profile_count)
        features = np.empty((len(native_rows), 457), dtype=np.float32)
        for pid in np.unique(profile[native_rows]):
            local = np.flatnonzero(profile[native_rows] == pid)
            ids = native_rows[local]
            features[local] = self.inp.features_at(int(pid), stack('query', np.float64)[ids])
        targets = dict(kind=np.array([c['original_kind'] == 'arc' for c in children], np.int64),
                       weld_type=np.array([c['weld_type'] == 'vertical' for c in children], np.int64),
                       direction=stack('direction', np.int64),
                       closed=np.array([c['original_kind'] == 'arc' and
                            c['representation']['arc_parameters']['is_closed'] for c in children], np.int64),
                       line_points=stack('points'), arc_shape=stack('arc_shape'),
                       interval_ratios=stack('interval'), interval_known=stack('interval_known', bool),
                       attribute_state=supervision.attribute_arrays['state'].astype(np.int64),
                       attribute_mm=supervision.numeric_mm.astype(np.float32),
                       state_known=supervision.label_known, numeric_known=supervision.numeric_known,
                       attribute_applicable=supervision.task_applicable,
                       frame_known=supervision.frame_algebraically_known,
                       frame_axes=stack('frame_axes'), frame_offsets=stack('frame_offsets'))
        parents = [list(range(int(a), int(b))) for a, b in zip(supervision.parent_offsets[:-1], supervision.parent_offsets[1:])]
        parent_ids = np.repeat(np.arange(len(parents)), np.diff(supervision.parent_offsets))
        positive_links = [(indices[i], indices[i + 1]) for indices in parents
                           for i in range(len(indices) - 1)]
        positive_links += [(indices[-1], indices[0]) for indices, parent in zip(parents, supervision.parents)
                           if parent['is_closed'] and len(indices) > 1]
        negatives = []
        # Different labelled parents are reference-link negatives only. This
        # does not assert physical non-weld or an exhaustive physical count.
        for i in range(len(rows)):
            possible = np.flatnonzero(parent_ids != parent_ids[i])
            if len(possible):
                negatives.append((i, int(rng.choice(possible))))
        pairs = positive_links + negatives
        relative = []
        for left, right in pairs:
            delta = (rows[right]['point'] - rows[left]['point']) @ rows[left]['basis'] / 1000
            tangent = rows[right]['basis'][:, 0] @ rows[left]['basis']
            relative.append(np.r_[delta, tangent, np.linalg.norm(delta),
                                  float(rows[left]['profile'] == rows[right]['profile'])])
        return dict(native_features=features, native_row_indices=native_rows, visual_row_indices=visual_rows,
                    query_ratios=stack('query', np.float64) / self.inp.lengths[profile],
                    profile_ids=profile, geometry=stack('geometry'), targets=targets,
                    input_query_positions_mm=stack('query', np.float64),
                    input_query_xyz=stack('point', np.float64), input_query_bases=stack('basis', np.float64),
                    parent_children=parents,
                    parent_closed=np.array([p['is_closed'] for p in supervision.parents], np.int64),
                    link_left=np.array([a for a, _ in pairs], np.int64),
                    link_right=np.array([b for _, b in pairs], np.int64),
                    link_relative=np.asarray(relative, np.float32).reshape(-1, 8),
                    link_labels=np.r_[np.ones(len(positive_links)), np.zeros(len(negatives))].astype(np.int64),
                    report=dict(children=len(rows), strong_carrier_assignments=sum(r['strong'] for r in rows),
                                unrepresented_reference_assignments=sum(not r['strong'] for r in rows),
                                parents=len(parents), reference_link_positives=len(positive_links),
                                reference_link_negatives=len(negatives),
                                input_queries_are_frozen_grid_points=True, physical_negative_labels=False,
                                complete_reference_denominator_retained=True,
                                parent_order_used_only_as_TRAIN_teacher_supervision=True))
