"""Continuous source binding for a finite graph; never a weld GT oracle.

Coordinate aliases share one symbolic root. Arc endpoints must be that same
expression, not just close floating numbers. Each selected role has its own
original local permission, and every whole curve stays in its declared face.
"""
from collections import OrderedDict
from fractions import Fraction as F
import math

import numpy as np

from .concrete_curve_certificate_250 import ConcreteCurve, UnprovenCurve, _fraction_bounds, _sqrt_fraction_bounds
from .decoder_graph_structure_254 import FrozenSurfaceGraphValidator, slot
from .decoder_output_slots_247 import local_region_triangles
from .mixed_curve_domain_253 import certify_curve_domain
from .mixed_surface_curve_binding_253 import SurfaceCompiler, _endpoint_intervals, _local_endpoint_proven, _lift_exact


VERSION = 'FROZEN_FINITE_GRAPH_CONTINUOUS_SURFACE_BINDING_255_v1'
PROTOCOL = dict(version=VERSION, graph_schema='COMPLETE_FINITE_SOURCE_DECODER_GRAPH_STRUCTURE_254_v1',
    coordinate_identity='EXPLICIT_ACYCLIC_SYMBOLIC_ROOT_NOT_COORDINATE_CLUSTERING',
    analytic_endpoint_coupling='SAME_DECLARED_EXPRESSION_OR_PROVEN_EXACT_POINT_EQUALITY',
    coordinate_permissions='OWN_ORIGINAL_LOCAL_CLOSED_TRIANGLE_SUPPORT',
    parent_child_span_permissions='ORDERED_START_IN_OWN_ANCHOR_LOCAL_REGION',
    whole_curve_permissions='OWN_DECLARED_FINITE_SURFACE_ONLY',
    geometry_node_is_business_cut=False, numeric_export_uncertainty='COMPUTED_FROM_SOURCE_EXPRESSIONS_ONLY',
    carrier_and_cross_domain_permissions='PENDING_COMPLETE_SOURCE_PROVIDER',
    general_parent_direction_and_positive_gap_metric_verified=False,
    material_identity_or_GT_assignment_verified=False, full_contract_pass=False)


def _f(value):
    return F.from_float(float(value))


def _point_expression(curve, endpoint):
    value = curve.record
    if endpoint == 0 or value['is_closed']:
        point = tuple(_f(c)+_f(value['radius_mm'])*_f(u) for c, u in zip(value['center_mm'], value['u_axis']))
        return ('POINT', point)
    return ('ARC_END', tuple(map(_f, value['center_mm'])), _f(value['radius_mm']),
            tuple(map(_f, value['u_axis'])), tuple(map(_f, value['v_axis'])), _f(value['sweep_rad']))


def _same_expression(a, b):
    return a['expression'] == b['expression'] or (a['bounds'][0] == a['bounds'][1]
        == b['bounds'][0] == b['bounds'][1])


def _corners(bounds):
    lo, hi = bounds
    return ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1]))


def _midpoint(bounds):
    point = [float((lo+hi)/2) for lo, hi in zip(*bounds)]
    if not all(map(math.isfinite, point)):
        raise UnprovenCurve('Finite symbolic endpoint cannot be exported')
    error = sum(max(abs(_f(x)-lo), abs(_f(x)-hi))**2 for x, lo, hi in zip(point, *bounds))
    return point, _sqrt_fraction_bounds(error)[1]


def _world_root(context, root):
    uv, _ = _midpoint(root['bounds'])
    exported, _ = _lift_exact(uv, context['basis'], context['origin'])
    lower, upper = [], []
    for k in range(3):
        lo = hi = _f(context['origin'][k])
        for j in range(2):
            coefficient = _f(context['basis'][j, k])
            ends = sorted((coefficient*root['bounds'][0][j], coefficient*root['bounds'][1][j]))
            lo += ends[0]
            hi += ends[1]
        lower.append(lo)
        upper.append(hi)
    error = sum(max(abs(_f(x)-lo), abs(_f(x)-hi))**2 for x, lo, hi in zip(exported, lower, upper))
    return dict(point_mm=exported, numerical_enclosure_upper_mm=_sqrt_fraction_bounds(error)[1],
                source_expression_world_lower_mm=[_fraction_bounds(x)[0] for x in lower],
                source_expression_world_upper_mm=[_fraction_bounds(x)[1] for x in upper])


def _polygon_own_domain(context, roots, local, start_ordinals):
    results = []
    for a, b in zip(roots, roots[1:]):
        corners = (*_corners(a['bounds']), *_corners(b['bounds']))
        support = next((int(i) for i in start_ordinals
                        if all(context['domain'].original_triangle_member(int(i), point) for point in corners)), None)
        if support is None:
            low = np.array([_fraction_bounds(min(point[j] for point in corners))[0] for j in range(2)])
            high = np.array([_fraction_bounds(max(point[j] for point in corners))[1] for j in range(2)])
            candidates = np.flatnonzero(((context['domain'].box_lo <= low) & (high <= context['domain'].box_hi)).all(axis=1))
            support = next((int(i) for i in candidates
                            if all(context['domain'].original_triangle_member(int(i), point) for point in corners)), None)
        if support is None:
            # The zero-uncertainty path can cross multiple triangles; its exact
            # closed interval union must cover the complete edge, including holes.
            if local.record['geometry_uncertainty_mm'] == 0:
                return certify_curve_domain(local, context['uv'], context['triangles'],
                    domain_id=context['domain'].domain_id, compiled_domain=context['domain'])
            return dict(status='UNPROVEN_SYMBOLIC_POLYGON_ACROSS_ORIGINAL_SUPPORTS',
                        unknown_curve_retained=True, approximate_midpoint_is_not_a_domain_proof=True)
        results.append(dict(original_support_triangle_ordinal=support,
                            original_support_rank=context['domain'].support_rank[support]))
    return dict(status='PASS_COMPLETE_SYMBOLIC_POLYGON_IN_ORIGINAL_CLOSED_SUPPORTS',
                original_edge_count=len(results), edge_supports=results,
                all_endpoint_enclosures_and_complete_convex_interpolations_proven=True,
                original_zero_edges_retained=sum(a['bounds'] == b['bounds'] for a, b in zip(roots, roots[1:])),
                domain_filling_or_positive_gap_merging_used=False)


class FrozenSurfaceGraphBinder:
    def __init__(self, connection, maximum_cached_faces=3):
        if type(maximum_cached_faces) is not int or maximum_cached_faces < 1:
            raise ValueError('Positive source-context cache size required')
        self.db = connection
        self.validator = FrozenSurfaceGraphValidator(connection)
        self.compiler = SurfaceCompiler(connection)
        self.contexts = OrderedDict()
        self.maximum_cached_faces = maximum_cached_faces

    def context(self, face_id):
        if face_id in self.contexts:
            self.contexts.move_to_end(face_id)
            return self.contexts[face_id]
        context = self.compiler.face(face_id)
        self.contexts[face_id] = context
        while len(self.contexts) > self.maximum_cached_faces:
            self.contexts.popitem(last=False)
        return context

    def bind(self, graph):
        structure = self.validator.validate(graph)
        result = dict(version=VERSION, structure=structure, selected_coordinate_count=len(graph['coordinates']),
            selected_span_count=len(graph['geometry_spans']), selected_child_count=len(graph['children']),
            selected_parent_count=len(graph['parents']), coordinates=[], spans=[], children=[], parents=[],
            structure_and_global_slot_capacity_verified=True, all_own_source_local_permissions_verified=False,
            all_complete_own_surface_curves_verified=False, coordinate_arc_parameter_coupling_verified=False,
            complete_ordered_child_bodies_compiled=False, general_parent_direction_and_gap_metric_verified=False,
            material_or_weld_identity_verified=False, actual_all111_or_all1605_joint_expression_run=False,
            model_inference_calls=0, actual_weld_optimizer_updates=0, full_contract_pass=False)
        if structure['status'] != 'PASS_COMPLETE_FROZEN_DECODER_GRAPH_STRUCTURE':
            return {**result, 'status': 'UNPROVEN_GRAPH_CARRIER_OR_CROSS_DOMAIN_BINDING',
                    'issues': structure['unbound_domain_or_transition_issues']}
        coordinates = {slot(row['slot']): row for row in graph['coordinates']}
        spans = {slot(row['slot']): row for row in graph['geometry_spans']}
        children = {slot(row['slot']): row for row in graph['children']}
        parents = {slot(row['slot']): row for row in graph['parents']}
        arc_curves = {key: ConcreteCurve({'kind': 'arc', **row['parameters']})
                      for key, row in spans.items() if row['kind'] == 'arc'}
        resolved = {}
        for initial in coordinates:
            current, chain = initial, []
            while current not in resolved:
                row = coordinates[current]
                chain.append(current)
                if 'physical_ref' in row:
                    current = slot(row['physical_ref'])
                    continue
                if 'position' in row:
                    point = tuple(map(_f, row['position']['uv_mm']))
                    root = dict(root=current, face_id=row['domain']['id'], expression=('POINT', point), bounds=(point, point))
                else:
                    endpoint = row['curve_endpoint']
                    curve = arc_curves[slot(endpoint['span'])]
                    index = endpoint['endpoint_index']
                    root = dict(root=current, face_id=row['domain']['id'], expression=_point_expression(curve, index),
                                bounds=_endpoint_intervals(curve)[index])
                resolved[current] = root
                break
            root = resolved[current]
            for key in chain:
                resolved[key] = root
        groups = {}
        for key, row in coordinates.items():
            groups.setdefault(row['domain']['id'], dict(coordinates=[], spans=[], located=[]))['coordinates'].append(key)
        for key, row in spans.items():
            groups.setdefault(row['domain']['id'], dict(coordinates=[], spans=[], located=[]))['spans'].append(key)
        for key, row in (*parents.items(), *children.items()):
            start = slot(row['start_boundary'])
            face_id = resolved[start]['face_id']
            groups.setdefault(face_id, dict(coordinates=[], spans=[], located=[]))['located'].append((key, start))
        issues, world_roots, bound_spans = [], {}, {}
        local_checks = located_checks = triangles = 0
        for face_id, group in sorted(groups.items()):
            context = self.context(face_id)
            triangles += len(context['triangles'])
            for key in group['coordinates']:
                root = resolved[key]
                regions = local_region_triangles(self.db, key[0], face_id)
                allowed = bool(regions) and _local_endpoint_proven(context, regions, root['bounds'])
                local_checks += 1
                if not allowed:
                    issues.append(dict(reason='UNPROVEN_DEFINING_COORDINATE_ORIGINAL_LOCAL_PERMISSION', slot=list(key)))
                if root['root'] not in world_roots:
                    world_roots[root['root']] = _world_root(context, root)
                result['coordinates'].append(dict(slot=coordinates[key]['slot'], physical_root_slot=coordinates[root['root']]['slot'],
                    face_id=face_id, own_original_local_permission_verified=allowed, **world_roots[root['root']]))
            for key, start in group['located']:
                regions = local_region_triangles(self.db, key[0], face_id)
                allowed = bool(regions) and _local_endpoint_proven(context, regions, resolved[start]['bounds'])
                located_checks += 1
                if not allowed:
                    issues.append(dict(reason='UNPROVEN_LOCATED_PARENT_OR_CHILD_LOCAL_START_PERMISSION', slot=list(key)))
            for key in group['spans']:
                row = spans[key]
                ends = [slot(row[field]) for field in ('start_boundary', 'end_boundary')]
                regions = local_region_triangles(self.db, key[0], face_id)
                if not regions or not _local_endpoint_proven(context, regions, resolved[ends[0]]['bounds']):
                    issues.append(dict(reason='UNPROVEN_SPAN_ORIGINAL_LOCAL_START_PERMISSION', slot=list(key)))
                try:
                    if row['kind'] == 'arc':
                        local = arc_curves[key]
                        endpoints = _endpoint_intervals(local)
                        coupled = []
                        for index, coordinate_key in enumerate(ends):
                            expected = dict(expression=_point_expression(local, index), bounds=endpoints[index])
                            coupled.append(_same_expression(resolved[coordinate_key], expected))
                        if not all(coupled):
                            issues.append(dict(reason='UNPROVEN_ARC_PARAMETERS_AND_EXPLICIT_ENDPOINT_EXPRESSION_COUPLING',
                                               slot=list(key), ordered_endpoint_coupling=coupled))
                            result['spans'].append(dict(slot=row['slot'], status='UNPROVEN_ARC_ENDPOINT_COUPLING'))
                            continue
                        own = certify_curve_domain(local, context['uv'], context['triangles'],
                            domain_id=context['domain'].domain_id, compiled_domain=context['domain'])
                        record = dict(local.record)
                        center, ce = _lift_exact(record['center_mm'], context['basis'], context['origin'])
                        u, ue = _lift_exact(record['u_axis'], context['basis'])
                        v, ve = _lift_exact(record['v_axis'], context['basis'])
                        numeric_error = _fraction_bounds(_f(ce)+_f(record['radius_mm'])*(_f(ue)+_f(ve)))[1]
                        record.update(center_mm=center, u_axis=u, v_axis=v, geometry_uncertainty_mm=numeric_error,
                                      orientation_frame_id=context['domain'].domain_id,
                                      orientation_frame_basis=context['basis'].tolist())
                    else:
                        sequence = [ends[0], *(slot(ref) for ref in row['internal_geometry_nodes']), ends[1]]
                        points_and_errors = [_midpoint(resolved[value]['bounds']) for value in sequence]
                        local = ConcreteCurve(dict(kind=row['kind'], points=[p for p, _ in points_and_errors],
                            is_closed=row['is_closed'], geometry_uncertainty_mm=max(error for _, error in points_and_errors)))
                        own = _polygon_own_domain(context, [resolved[value] for value in sequence], local, regions)
                        record = dict(local.record)
                        record['points'] = [world_roots[resolved[value]['root']]['point_mm'] for value in sequence]
                        numeric_error = max(world_roots[resolved[value]['root']]['numerical_enclosure_upper_mm'] for value in sequence)
                        record['geometry_uncertainty_mm'] = numeric_error
                    if not own['status'].startswith('PASS'):
                        issues.append(dict(reason='UNPROVEN_OR_OUTSIDE_COMPLETE_OWN_SOURCE_SURFACE_CURVE',
                                           slot=list(key), own_domain_status=own['status']))
                    world = ConcreteCurve(record)
                    bound = dict(slot=row['slot'], face_id=face_id, kind=row['kind'], start_boundary=row['start_boundary'],
                        end_boundary=row['end_boundary'], status='PASS_CONTINUOUS_OWN_SURFACE_SPAN_AND_SYMBOLIC_ENDPOINT_BINDING'
                        if own['status'].startswith('PASS') else 'UNPROVEN_OR_OUTSIDE_COMPLETE_OWN_SURFACE_SPAN',
                        local_curve=local.record, world_curve=world.record, own_domain_certificate=own,
                        world_numerical_geometry_uncertainty_upper_mm=numeric_error,
                        source_arc_endpoint_expression_or_vertex_coupling_verified=True)
                    result['spans'].append(bound)
                    bound_spans[key] = bound
                except (UnprovenCurve, ValueError) as error:
                    issues.append(dict(reason='UNPROVEN_OR_INVALID_CONCRETE_SOURCE_SPAN', slot=list(key), detail=str(error)))
                    result['spans'].append(dict(slot=row['slot'], status='UNPROVEN_OR_INVALID_CONCRETE_SOURCE_SPAN', detail=str(error)))
        for key, row in children.items():
            selected = [slot(ref) for ref in row['geometry_spans']]
            body = [bound_spans[value]['world_curve'] for value in selected if value in bound_spans]
            complete = len(body) == len(selected) and all(bound_spans[value]['status'].startswith('PASS') for value in selected)
            result['children'].append(dict(slot=row['slot'], parent=row['parent'], order_index=row['order_index'],
                geometry_kind=row['geometry_kind'], weld_type=row['weld_type'], start_boundary=row['start_boundary'],
                end_boundary=row['end_boundary'], ordered_world_geometry_spans=body,
                selected_span_count=len(selected), complete_span_count=len(body),
                status='PASS_COMPLETE_ORDERED_CHILD_SURFACE_BODY' if complete else 'UNPROVEN_COMPLETE_CHILD_SURFACE_BODY',
                geometric_corners_added_as_business_children=False))
        for key, row in parents.items():
            result['parents'].append(dict(slot=row['slot'], ordered_children=row['children'], start_boundary=row['start_boundary'],
                end_boundary=row['end_boundary'], phase_boundary=row['phase_boundary'], is_closed=row['is_closed'],
                direction=row['direction'], turn_count=row['turn_count'], explicit_gaps=row['gaps'],
                closed_shared_physical_root_verified=not row['is_closed'] or
                    resolved[slot(row['start_boundary'])]['root'] == resolved[slot(row['end_boundary'])]['root'],
                general_gap_metric_and_parent_direction_verified=False))
        result.update(status='PASS_COMPLETE_GRAPH_OWN_SURFACE_CURVES_AND_SYMBOLIC_BINDING' if not issues
                      else 'UNPROVEN_COMPLETE_GRAPH_SOURCE_GEOMETRY_BINDING', issues=issues,
            physical_coordinate_root_count=len(world_roots), original_coordinate_local_permission_checks=local_checks,
            located_parent_child_local_permission_checks=located_checks, complete_used_original_face_triangle_count=triangles,
            all_own_source_local_permissions_verified=not any('PERMISSION' in issue['reason'] for issue in issues),
            all_complete_own_surface_curves_verified=not issues,
            coordinate_arc_parameter_coupling_verified=not any('COUPLING' in issue['reason'] for issue in issues),
            complete_ordered_child_bodies_compiled=not issues,
            maximum_world_numerical_geometry_uncertainty_upper_mm=max(
                [row['numerical_enclosure_upper_mm'] for row in result['coordinates']]
                + [row.get('world_numerical_geometry_uncertainty_upper_mm', 0.) for row in result['spans']] + [0.]))
        return result
