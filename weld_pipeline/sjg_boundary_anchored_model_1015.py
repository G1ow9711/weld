"""Millimetre-scale distances to observed finite-support boundaries.

All anchors come from the pure-input carrier's finite surface support. They
are features, not declarations that a boundary is a weld or an endpoint.
"""
import numpy as np
import torch
from torch import nn
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_surface_material_joint_model_985 import SurfaceMaterialJointModel
from weld_pipeline.sjg_local_graph_data_916 import queries as base_queries


SCALES_MM = np.array([1., 2., 4., 8., 16., 32., 64., 128.])


def boundary_features(profiles, nodes, q):
    nodes = np.asarray(nodes, np.int64);q = np.asarray(q, np.float64)
    result = np.zeros((len(q), 32), np.float32)
    for node in np.unique(nodes):
        ids = np.flatnonzero(nodes == node);profile = profiles[int(node)]
        curve = CurveCarrier(profile['representation']);local = q[ids]
        closed = curve.kind == 'arc' and curve.arc.is_closed
        intervals = sorted([list(map(float, v)) for v in profile['support_intervals_mm']])
        merged = []
        for lo, hi in intervals:
            if not 0 <= lo <= hi <= curve.length + 1e-5:
                raise ValueError('Support interval outside input carrier')
            if merged and lo <= merged[-1][1] + 1e-5:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        starts = [v[0] for v in merged];ends = [v[1] for v in merged]
        if closed and merged and merged[0][0] <= 1e-5 and merged[-1][1] >= curve.length - 1e-5:
            # The circle's parameter origin is not an observed boundary.
            starts = starts[1:];ends = ends[:-1]
        def nearest(anchors):
            if not anchors:
                return np.full(len(local), np.inf)
            delta = local[:, None] - np.asarray(anchors)[None, :]
            if closed:
                delta = (delta + curve.length/2) % curve.length - curve.length/2
            return np.abs(delta).min(1)
        distances = [np.full(len(local), np.inf) if closed else local,
                     np.full(len(local), np.inf) if closed else curve.length-local,
                     nearest(starts), nearest(ends)]
        result[ids] = np.concatenate([np.exp(-.5*(v[:, None]/SCALES_MM[None, :])**2) for v in distances], axis=1)
    assert result.shape == (len(q), 32) and np.isfinite(result).all()
    return result


def queries(profiles, nodes, q, xyz, point, tree):
    patch, query = base_queries(profiles, nodes, q, xyz, point, tree)
    return patch, np.concatenate([query, boundary_features(profiles, nodes, q)], axis=1)


class BoundaryAnchoredModel(SurfaceMaterialJointModel):
    def __init__(self):
        super().__init__()
        original = self.event[0]
        self.event[0] = nn.Linear(original.in_features + 32, original.out_features)


def adapt_initial_state(state):
    result = dict(state)
    weight = result['event.0.weight']
    assert weight.shape == (128, 170)
    result['event.0.weight'] = torch.cat([weight, weight.new_zeros((128, 32))], dim=1)
    return result


def controls():
    from weld_pipeline.contract_curve_geometry import CircularArc
    profile = dict(representation=dict(kind='line', endpoints_mm=[[0, 0, 0], [100, 0, 0]]),
                   support_intervals_mm=[[0, 40], [60, 100]])
    q = np.array([0., 2., 40., 60., 100.]);nodes = np.zeros(len(q), np.int64)
    values = boundary_features([profile], nodes, q)
    assert values[0, 0] == 1 and abs(values[1, 1]-np.exp(-.5)) < 1e-7
    assert values[2, 24] == 1 and values[3, 16] == 1
    shifted = dict(profile, representation=dict(kind='line', endpoints_mm=[[100, 80, 4], [100, 180, 4]]))
    assert np.array_equal(values, boundary_features([shifted], nodes, q))
    circle = CircularArc([0, 0, 0], 20, [0, 0, 1], [1, 0, 0], 2*np.pi, True)
    closed = dict(representation=dict(kind='arc', arc_parameters=circle.to_parameters()),
                  support_intervals_mm=[[0, circle.length_mm]])
    assert not boundary_features([closed], np.array([0]), np.array([0.])).any()
    assert np.array_equal(values, boundary_features([dict(profile, fake_target='ignored')], nodes, q))
    model = BoundaryAnchoredModel()
    joint = torch.zeros(2, 96);patch = torch.zeros(2, 16, 23);query = torch.zeros(2, 42)
    query[:, 10:] = torch.from_numpy(values[:2])
    output, offset = model.query_events(joint, patch, query)
    output.sum().backward()
    assert model.event[0].weight.grad[:, 170:].abs().sum() > 0
    return dict(millimetre_boundary_width_correct=True, finite_gap_caps_preserved=True,
                rigid_transform_invariant_anchor_features=True, full_circle_has_no_fake_origin_boundary=True,
                extra_target_field_unused=True, new_anchor_features_receive_gradients=True,
                anchors_do_not_classify_welds=True, optimizer_updates=0)
