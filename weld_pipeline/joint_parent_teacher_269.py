"""Offline joint-parent teacher poses using frozen source grammar and capacity.

Reference geometry is used only for supervision and its complete C1/C2 audit.
It is never an initial locator feature. Parent membership and shared cuts come
from original registration, not names, lengths or coordinate-nearness rules.
"""
from fractions import Fraction as F
import numpy as np

from .raw_triangle_world_charts_258 import exact
from .concrete_curve_certificate_250 import ConcreteCurve, UnprovenCurve, _sqrt_fraction_bounds
from .mixed_parent_source_binding_260 import _legacy_segment_covered
from .mixed_parent_graph_260 import VERSION


def slot_ref(anchor, role, ordinal):
    return dict(anchor_id=int(anchor), role=role, local_slot_ordinal=int(ordinal))


def exact_world(context, uv):
    return tuple(exact(float(context['origin'][axis])) + sum(
        F(uv[k]) * exact(float(context['basis'][k, axis])) for k in range(2)) for axis in range(3))


def concrete_exact_line(first, last):
    if first == last:
        raise UnprovenCurve('Joint source child must retain positive length')
    exported = [list(map(float, point)) for point in (first, last)]
    error = max(_sqrt_fraction_bounds(sum((exact(value) - coordinate) ** 2
        for value, coordinate in zip(row, point)))[1] for row, point in zip(exported, (first, last)))
    return ConcreteCurve(dict(kind='line', points=exported, is_closed=False, geometry_uncertainty_mm=error))


def common_line_parent(group, context, offset_mm):
    """One exact affine line and one coordinate for each registered shared cut.

    Full parent support proves support of all ordered restrictions. Each child
    still needs its own original reference C1/C2 and endpoint-slot checks.
    """
    rows = sorted(group, key=lambda row: row['truth_segment_index'])
    if not rows or any(row['kind'] != 'line' for row in rows):
        raise UnprovenCurve('Common-line teacher requires declared line children')
    if len({row['truth_parent_id'] for row in rows}) != 1:
        raise ValueError('Different registered parents cannot be merged')
    if len({row['truth_segment_index'] for row in rows}) != len(rows):
        raise ValueError('Duplicate original child ordinal')
    points = [tuple(exact(float(value)) for value in rows[0]['original_reference_points'][0])]
    for previous, following in zip(rows, rows[1:]):
        end = tuple(exact(float(value)) for value in previous['original_reference_points'][-1])
        start = tuple(exact(float(value)) for value in following['original_reference_points'][0])
        if end != start:
            raise UnprovenCurve('Nonshared original join requires an explicit positive gap teacher')
        points.append(end)
    points.append(tuple(exact(float(value)) for value in rows[-1]['original_reference_points'][-1]))
    delta = tuple(last - first for first, last in zip(points[0], points[-1]))
    squared = sum(value * value for value in delta)
    if squared == 0:
        raise UnprovenCurve('Positive common parent reference axis required')
    parameters = [sum((value - start) * axis for value, start, axis in zip(point, points[0], delta)) / squared
                  for point in points]
    if parameters[0] != 0 or parameters[-1] != 1 or any(a >= b for a, b in zip(parameters, parameters[1:])):
        raise UnprovenCurve('Original child order is not a positive common line walk')
    basis, origin = context['basis'], context['origin']
    uv = (np.asarray([list(map(float, points[0])), list(map(float, points[-1]))]) - origin) @ basis.T
    tangent = uv[1] - uv[0]
    if not np.linalg.norm(tangent):
        raise UnprovenCurve('Source plane cannot contain the common parent direction')
    sideways = np.asarray([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    uv += float(offset_mm) * sideways
    uv = np.maximum(context['descriptor']['uv_bounds_mm'][0],
                    np.minimum(context['descriptor']['uv_bounds_mm'][1], uv))
    first, last = [tuple(exact(float(value)) for value in row) for row in uv]
    _legacy_segment_covered(context, first, last)
    positions = [tuple(first[k] + parameter * (last[k] - first[k]) for k in range(2)) for parameter in parameters]
    world = [exact_world(context, point) for point in positions]
    children = [dict(kind='line', uv=[positions[index], positions[index + 1]],
        world_points=[world[index], world[index + 1]], endpoints=[(positions[index], positions[index]),
        (positions[index + 1], positions[index + 1])], world_curve=concrete_exact_line(world[index], world[index + 1]),
        is_closed=False, turn_count=0, direction=1) for index in range(len(rows))]
    return dict(kind='line', references=rows, children=children, boundary_positions=positions,
        boundary_bounds=[(point, point) for point in positions], master_axis_world=tuple(b - a for a, b in zip(world[0], world[-1])),
        original_common_parameters=parameters, shared_registered_cuts=len(rows) - 1,
        is_closed=False, turn_count=0, direction=1)


def assembly_requests(selected):
    """Finite global requests, with one boundary request per shared cut."""
    requests = dict(parent=[], child=[], geometry_span=[], boundary=[])
    for index, item in enumerate(selected):
        requests['parent'].append(((index, 0), item['boundary_anchors'][0]))
        for child_index in range(len(item['proposal']['children'])):
            for role in ('child', 'geometry_span'):
                requests[role].append(((index, child_index), item['boundary_anchors'][child_index]))
        for boundary_index, anchors in enumerate(item['boundary_anchors']):
            requests['boundary'].append(((index, boundary_index), anchors))
    return requests


def assemble_joint_parents(selected, matches):
    graph = dict(version=VERSION, coordinates=[], geometry_spans=[], children=[], parents=[])
    mapping = []
    for index, item in enumerate(selected):
        proposal, face = item['proposal'], item['face_id']
        domain = dict(type='LEGACY_SURFACE', face_id=face)
        parent_anchor, parent_ordinal = matches['parent'][(index, 0)]
        p = slot_ref(parent_anchor, 'parent', parent_ordinal)
        boundaries = [slot_ref(matches['boundary'][(index, ordinal)][0], 'boundary',
                     matches['boundary'][(index, ordinal)][1]) for ordinal in range(len(item['boundary_anchors']))]
        spans = [slot_ref(matches['geometry_span'][(index, ordinal)][0], 'geometry_span',
                 matches['geometry_span'][(index, ordinal)][1]) for ordinal in range(len(proposal['children']))]
        child_refs = [slot_ref(matches['child'][(index, ordinal)][0], 'child',
                      matches['child'][(index, ordinal)][1]) for ordinal in range(len(proposal['children']))]
        if proposal['kind'] == 'line':
            for boundary, point in zip(boundaries, proposal['boundary_positions']):
                graph['coordinates'].append(dict(slot=boundary, domain=domain, position=dict(uv_mm=list(point))))
            frame = dict(kind='line', axis_world=list(proposal['master_axis_world']))
        elif proposal['kind'] == 'arc' and len(proposal['children']) == 1:
            graph['coordinates'].append(dict(slot=boundaries[0], domain=domain,
                curve_endpoint=dict(span=spans[0], endpoint_index=0)))
            endpoint = dict(slot=boundaries[1], domain=domain, curve_endpoint=dict(span=spans[0], endpoint_index=1))
            if proposal['is_closed']:
                endpoint.pop('curve_endpoint')
                endpoint.update(physical_ref=boundaries[0], local_position=dict(curve_endpoint=dict(span=spans[0], endpoint_index=1)))
            graph['coordinates'].append(endpoint)
            frame = dict(kind='arc', authority_span=spans[0])
        else:
            raise UnprovenCurve('General mixed analytic parent teacher not yet proved')
        for ordinal, (row, child_proposal, child_ref, span) in enumerate(zip(proposal['references'], proposal['children'], child_refs, spans)):
            span_row = dict(slot=span, domain=domain, kind=child_proposal['kind'], start_boundary=boundaries[ordinal],
                end_boundary=boundaries[ordinal + 1], internal_geometry_nodes=[], is_closed=child_proposal['is_closed'])
            if child_proposal['kind'] == 'arc':
                span_row.update(parameters=child_proposal['parameters'], parameter_interval=[0, 1])
            graph['geometry_spans'].append(span_row)
            graph['children'].append(dict(slot=child_ref, parent=p, order_index=ordinal, start_boundary=boundaries[ordinal],
                end_boundary=boundaries[ordinal + 1], geometry_spans=[span], geometry_kind='line' if child_proposal['kind'] == 'line' else 'curve',
                world_kind=child_proposal['kind'], weld_type=row['type']))
            mapping.append(dict(truth_child_id=row['truth_child_id'], truth_parent_id=row['truth_parent_id'],
                original_truth_segment_index=row['truth_segment_index'], child_slot=child_ref, parent_slot=p,
                geometry_span_slot=span, ordered_start_slot=boundaries[ordinal], ordered_end_slot=boundaries[ordinal + 1],
                source_face_id=face, geometry_certificate=item['child_certificates'][ordinal]))
        graph['parents'].append(dict(slot=p, children=child_refs, start_boundary=boundaries[0], end_boundary=boundaries[-1],
            allowed_domains=[domain], direction=proposal['direction'], is_closed=proposal['is_closed'],
            phase_boundary=boundaries[0], turn_count=proposal['turn_count'],
            gaps=[dict(relation='SHARED_BOUNDARY', left_boundary=boundary,
                       right_boundary=boundary, geometry_spans=[]) for boundary in boundaries[1:-1]], frame=frame))
    return graph, mapping
