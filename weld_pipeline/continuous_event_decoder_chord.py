"""Opt-in continuous-event decoder with independent C1/D4 arc output.

The original decoder remains untouched and remains the default for existing
models. Event interval selection runs its exact original function source.
Only arc representation changes; the straight-line branch is identical.
"""
import ast
import bisect
from pathlib import Path

import numpy as np

from weld_pipeline.contract_curve_geometry import CircularArc, CurveGeometryError, sample_arc

VERSION = 'continuous_typed_event_intervals_chord_v1'


def _original_interval_selector():
    # The legacy module imports torch indirectly through curved_sequences.
    # Execute only these two pure-CPU definitions from the original source.
    path = Path(__file__).with_name('continuous_event_decoder.py')
    source = ast.parse(path.read_text(encoding='utf8'), filename=str(path))
    names = {'validate_events', 'select_event_intervals'}
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise RuntimeError('LEGACY_SELECTOR_UNAVAILABLE: original pure-CPU interval functions required')
    namespace = {'np': np, 'bisect': bisect}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['select_event_intervals']


select_event_intervals = _original_interval_selector()


def _arc_interval(seq, u, v):
    if 'sequence_arc_parameters' in seq:
        complete = CircularArc.from_parameters(seq['sequence_arc_parameters'])
        arc = complete.subarc(u, v)
        exact_endpoints = complete.point_at(np.asarray([u, v], dtype=float))
    else:
        curve = seq['curve']
        radius = float(curve['radius_mm'])
        if not np.isfinite(radius) or radius <= 0.:
            raise CurveGeometryError('INVALID_SEQUENCE_RADIUS: positive finite radius required')
        sign = float(curve.get('sequence_direction_sign', 1.))
        if sign not in (-1., 1.):
            raise CurveGeometryError('INVALID_SEQUENCE_DIRECTION: +1 or -1 required')
        if 'center_mm' in curve:
            center = np.asarray(curve['center_mm'], float)
            normal = np.asarray(curve['plane_normal'], float)
            initial = np.asarray(curve['sequence_start_direction'], float)
        else:
            center = np.r_[np.asarray(curve['center_xy_mm'], float), float(curve['height_mm'])]
            normal = np.asarray([0., 0., 1.])
            theta = float(curve['sequence_start_angle_rad'])
            initial = np.asarray([np.cos(theta), np.sin(theta), 0.])
        # Validate basis and selected sweep before generating points.
        reference = CircularArc(center, radius, normal, initial, sign * (v - u) / radius,
                                bool((v - u) / radius == 2. * np.pi))
        basis = reference.start_direction
        second = np.cross(reference.plane_normal, basis)
        angles = sign * np.asarray([u, v], dtype=float) / radius
        directions = np.cos(angles)[:, None] * basis + np.sin(angles)[:, None] * second
        arc = CircularArc(center, radius, reference.plane_normal, directions[0], reference.sweep_rad,
                          reference.is_closed)
        exact_endpoints = np.asarray(center) + radius * directions
        if arc.is_closed:
            exact_endpoints[1] = exact_endpoints[0]
    result = sample_arc(arc)
    # Shared event coordinates use the same candidate calculation on both sides
    # of a cut. This preserves an exact common breakpoint, within roundoff of
    # the independently delivered analytic subarc parameters.
    result['points'][0] = exact_endpoints[0].tolist()
    result['points'][-1] = exact_endpoints[1].tolist()
    result['source_interval_mm'] = [float(u), float(v)]
    return result


def decode_event_sets(sequences, foregrounds, event_sets, threshold=.5, weld_penalty=2., mask_weight=.25):
    if len(sequences) != len(foregrounds) or len(sequences) != len(event_sets):
        raise ValueError('Each candidate needs foreground and typed continuous events')
    result = []
    for seq, foreground, events in zip(sequences, foregrounds, event_sets):
        step = float(seq['sample_step_mm'])
        intervals, internal = select_event_intervals(foreground, step, events, threshold, weld_penalty, mask_weight)
        for a, b, pa, pb in intervals:
            divisions = np.unique(np.r_[a, internal[(internal[:, 0] > a) & (internal[:, 0] < b), 0], b])
            segments = []
            for u, v in zip(divisions[:-1], divisions[1:]):
                if seq.get('curve_kind') == 'arc':
                    segments.append(_arc_interval(seq, u, v))
                else:
                    q = np.asarray(seq['xyz'][0], float); direction = np.asarray(seq['direction'], float)
                    segments.append(dict(kind='line', points=[(q+u*direction).tolist(), (q+v*direction).tolist()]))
            result.append(dict(id=f'weld_{len(result)+1:04d}', type=seq['type'], segments=segments,
                               confidence=float(min(pa, pb))))
    return result
