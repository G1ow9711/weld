"""Pure-input finite fragment/port graph specification (no training imports).

Objects partition the ORIGINAL analytic profile at changes in observed support.
Each has a continuous start and end domain; the full fragment is not a weld.
Numerical duplicate boundaries are collapsed only at floating-point roundoff.
Two unresolved side hypotheses are retained; neither is a material truth label.

This module constructs the compact geometric graph. Connection membership is
an input-only predicate. A capacity receipt is NOT a receipt that all feature
rows and runtime edges have been materialized or scored.
"""
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier

VERSION = 'FINITE_ORIGINAL_CURVE_SUPPORT_FRAGMENT_GRAPH_v2'
SIDE_HYPOTHESES = ('UNCONFIRMED_A', 'UNCONFIRMED_B')
LINK_CLASSES = ('NO_LINK', 'SAME_CHILD', 'SHARED', 'POSITIVE_GAP')
GEOMETRY_FIELDS = (
    'length_log1p', 'profile_length_log1p', 'domain_lo_ratio', 'domain_hi_ratio',
    'start_x_m', 'start_y_m', 'start_z_m', 'end_x_m', 'end_y_m', 'end_z_m',
    'mid_x_m', 'mid_y_m', 'mid_z_m', 'start_tx', 'start_ty', 'start_tz',
    'end_tx', 'end_ty', 'end_tz', 'face_a_nx', 'face_a_ny', 'face_a_nz',
    'face_b_nx', 'face_b_ny', 'face_b_nz', 'normal_dot', 'curvature_per_mm',
    'arc_sweep', 'support_fraction', 'sampling_void_fraction', 'side_sign',
    'profile_domain_fraction')
MASK_FIELDS = (
    'line', 'arc', 'closed_carrier', 'observed_support', 'sampling_void',
    'face_a_known', 'face_b_known', 'pair_known', 'material_known',
    'side_confirmed', 'unconfirmed_side_a', 'unconfirmed_side_b',
    'at_profile_start', 'at_profile_end', 'endpoint_continuous', 'finite_domain')
PAIR_FIELDS = ('distance_a_log1p_10mm', 'distance_b_log1p_10mm',
               'joint_support_exp_10mm', 'nearest_points_separation_log1p_10mm',
               'pair_known')
LINK_FIELDS = (
    'aabb_min_distance_mm', 'exit_entry_distance_mm', 'exit_tx', 'exit_ty',
    'exit_tz', 'entry_tx', 'entry_ty', 'entry_tz', 'tangent_dot',
    'a_line', 'a_arc', 'b_line', 'b_arc', 'same_original_curve',
    'shared_face_a_a', 'shared_face_a_b', 'shared_face_b_a', 'shared_face_b_b',
    'a_side_a', 'a_side_b', 'b_side_a', 'b_side_b',
    'a_supported', 'b_supported', 'a_void', 'b_void',
    'material_identity_known', 'side_identity_known', 'corner_allowed',
    'same_child_allowed', 'shared_allowed', 'gap_allowed')
assert len(GEOMETRY_FIELDS) == 32 and len(MASK_FIELDS) == 16 and len(LINK_FIELDS) == 32


def profile_spans(profile):
    curve = CurveCarrier(profile['representation']); length = curve.length
    ends = curve.at(np.array([0., length]))
    tol = 64 * np.finfo(float).eps * max(1., length, float(np.abs(ends).max()))
    raw = [0., length]
    for name in ('support_intervals_mm', 'unobserved_gaps_mm'):
        for lo, hi in profile.get(name, []):
            raw.extend([min(length, max(0., float(lo))), min(length, max(0., float(hi)))])
    anchors = []
    for q in sorted(raw):
        if not anchors or q - anchors[-1] > tol: anchors.append(q)
    anchors[-1] = length
    rows = []
    for lo, hi in zip(anchors[:-1], anchors[1:]):
        mid = (lo + hi) / 2
        supported = any(a <= mid <= b for a, b in profile.get('support_intervals_mm', []))
        void = any(a < mid < b for a, b in profile.get('unobserved_gaps_mm', []))
        # Adjacent equal support states have no physical boundary between them.
        if rows and rows[-1][2:] == [int(supported), int(void)]: rows[-1][1] = hi
        else: rows.append([lo, hi, int(supported), int(void)])
    if not rows or rows[0][0] != 0 or rows[-1][1] != length:
        raise ValueError('Incomplete original finite domain')
    return np.asarray(rows, np.float64), tol


def curve_bounds(curve, lo, hi):
    """Exact coordinate extrema of a finite line or circular arc, not sampling."""
    q = [float(lo), float(hi)]
    if curve.kind == 'arc':
        a = curve.arc; u = a.start_direction; v = np.cross(a.plane_normal, u)
        sign = np.sign(a.sweep_rad)
        for axis in range(3):
            base = np.arctan2(sign * v[axis], u[axis])
            for k in range(-2, 4):
                x = (base + k * np.pi) * a.radius_mm
                if lo < x < hi: q.append(float(x))
    points = curve.at(np.asarray(q))
    return np.stack((points.min(0), points.max(0)))


class JointSegmentGraph:
    def __init__(self, profiles, offsets, spans):
        self.profiles = profiles; self.offsets = np.asarray(offsets, np.int64)
        self.spans = np.asarray(spans, np.float64); self._curves = {}; self._bounds = {}; self._gap_points = {}

    @classmethod
    def build(cls, profiles):
        arrays = []; offsets = [0]
        for profile in profiles:
            rows, _ = profile_spans(profile); arrays.append(rows)
            offsets.append(offsets[-1] + len(rows))
        return cls(profiles, offsets, np.concatenate(arrays))

    def curve(self, pid):
        pid = int(pid)
        if pid not in self._curves: self._curves[pid] = CurveCarrier(self.profiles[pid]['representation'])
        return self._curves[pid]

    def profile(self, fragment):
        return int(np.searchsorted(self.offsets, int(fragment), side='right') - 1)

    def bounds(self, fragment):
        fragment = int(fragment)
        if fragment not in self._bounds:
            self._bounds[fragment] = curve_bounds(self.curve(self.profile(fragment)), *self.spans[fragment, :2])
        return self._bounds[fragment]

    def witness_fragments(self, alternative):
        pid = int(alternative['profile']); lo, hi = float(alternative['lo']), float(alternative['hi'])
        start, stop = self.offsets[pid:pid+2]; rows = self.spans[start:stop]
        if alternative['closed']: selected = np.arange(start, stop)
        elif alternative['wrap']:
            selected = np.r_[np.flatnonzero(rows[:, 1] > hi) + start,
                             np.flatnonzero(rows[:, 0] < lo) + start]
        else: selected = np.flatnonzero((rows[:, 1] > lo) & (rows[:, 0] < hi)) + start
        if not len(selected): raise ValueError('Witness lost every finite fragment')
        if len(set(selected.tolist())) != len(selected):
            raise ValueError('Wrapped witness revisits the same unbroken fragment')
        if not alternative['forward']: selected = selected[::-1]
        return [int(x) for x in selected]

    def connection(self, a, b, connection, a_forward=True, b_forward=True):
        """Pure-input legal edge predicate; never receives labels or GT points.

        SHARED uses a conservative 6 mm AABB broad phase (two 3 mm endpoints).
        Exact shared coordinate and curve consistency remain decoder constraints.
        GAP allows a local port pair, one original carrier, or a finite observed
        void with support-role continuation. Sharing a face alone is NOT a gap.
        No maximum physical gap length is imposed on a supported void bridge.
        """
        a, b = int(a), int(b); pa, pb = self.profile(a), self.profile(b)
        aa, bb = self.bounds(a), self.bounds(b)
        distance = float(np.linalg.norm(np.maximum(0., np.maximum(aa[0]-bb[1], bb[0]-aa[1]))))
        faces_a = set(self.profiles[pa].get('face_indices', []))
        faces_b = set(self.profiles[pb].get('face_indices', []))
        shared_face = bool(faces_a & faces_b)
        if connection == 'SAME_CHILD':
            if pa != pb or a_forward != b_forward: return False
            next_fragment = a + (1 if a_forward else -1)
            if next_fragment == b: return True
            curve = self.curve(pa)
            return bool(curve.kind == 'arc' and curve.arc.is_closed and
                        {a, b} == {int(self.offsets[pa]), int(self.offsets[pa+1]-1)})
        if connection == 'SHARED': return distance <= 6.
        if connection == 'POSITIVE_GAP':
            if pa == pb or distance <= 6.: return True
            if not shared_face: return False
            # Finite void endpoints, rather than an unbounded all-to-all face
            # relation. This is observed sampling evidence, not a true hole label.
            for source_pid in (pa,pb):
                carrier=self.curve(source_pid)
                if source_pid not in self._gap_points:
                    gaps=np.asarray(self.profiles[source_pid].get('unobserved_gaps_mm',[]),np.float64).reshape(-1,2)
                    self._gap_points[source_pid]=carrier.at(np.clip(gaps,0,carrier.length))
                all_points=self._gap_points[source_pid]
                if not len(all_points):continue
                # Necessary point-to-box bound only; the original exact curve
                # projection check below still decides membership. Roundoff is
                # padded in the broad phase, never in the exact 3 mm decision.
                pad=64*np.finfo(float).eps*max(1.,float(np.abs(all_points).max()))
                box_near=[]
                for box in (aa,bb):
                    delta=np.maximum(0.,np.maximum(box[0]-all_points,all_points-box[1]))
                    box_near.append(np.linalg.norm(delta,axis=-1)<=3.+pad)
                possible=(box_near[0][:,0]&box_near[1][:,1])|(box_near[0][:,1]&box_near[1][:,0])
                for points in all_points[possible]:
                    near=[]
                    for fragment,pid in ((a,pa),(b,pb)):
                        curve=self.curve(pid);q,_=curve.project(points)
                        q=np.clip(q,*self.spans[fragment,:2])
                        near.append(np.linalg.norm(curve.at(q)-points,axis=1)<=3.)
                    if (near[0][0] and near[1][1]) or (near[0][1] and near[1][0]):return True
            return False
        if connection == 'NO_LINK': return True
        raise ValueError('Unknown graph connection class')


def specification():
    return dict(version=VERSION, object_features=[f'bin_{b:02d}_{v}' for b in range(16) for v in PAIR_FIELDS]
                + list(GEOMETRY_FIELDS) + list(MASK_FIELDS), link_features=list(LINK_FIELDS),
                endpoint_domain='both endpoint parameters continuous in the original fragment interval',
                side_hypotheses=SIDE_HYPOTHESES, directed_states=('forward', 'reverse'),
                link_classes=LINK_CLASSES, support_identity='original profile + support fragment + side hypothesis',
                no_exclusivity_across_distinct_unconfirmed_profiles=True,
                normal_sign_or_material_truth_inferred=False, GT_reads=0,
                numeric_boundary_rule='64*float64_epsilon*max(1,length,coordinate_scale)',
                full_curve_geometry_preserved=True, all_features_materialized=False,
                gap_rule='local finite ports OR same original curve OR explicit finite sampling-void bridge with shared observed support',
                sampling_void_is_not_process_gap_truth=True,
                all_runtime_links_materialized=False)
