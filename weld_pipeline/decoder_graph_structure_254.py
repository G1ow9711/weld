"""Finite source-slot graph syntax and ownership, before geometric feasibility.

One coordinate definition is shared by explicit references. Business children
are separate from geometric spans/corners. This gate never claims that a curve
is in its permitted domain, matches weld truth, or identifies material sides.
"""
from collections import Counter
import math

from .concrete_curve_certificate_250 import ConcreteCurve
from .decoder_output_slots_247 import CAPACITIES, validate_slot_capacity


VERSION = 'COMPLETE_FINITE_SOURCE_DECODER_GRAPH_STRUCTURE_254_v1'
SCHEMA = dict(
    version=VERSION,
    source_slot_identity='CANONICAL_ANCHOR_ROLE_LOCAL_ORDINAL',
    per_anchor_capacities=dict(CAPACITIES),
    definitions=['coordinates', 'geometry_spans', 'children', 'parents'],
    coordinate_roles=['boundary', 'geometry_node'],
    exactly_one_coordinate_form=['position', 'physical_ref', 'curve_endpoint'],
    coordinate_aliases='EXPLICIT_ACYCLIC_REFS_ONLY_NO_COORDINATE_CLUSTERING',
    domains=['surface', 'carrier'],
    geometry_spans=['line', 'arc', 'polyline'],
    internal_geometry_nodes='ORDERED_FROZEN_GEOMETRY_NODE_REFS',
    geometry_corners_are_business_cuts=False,
    geometry_span_ownership='EXACTLY_ONE_CHILD_OR_EXPLICIT_PARENT_GAP',
    child_count_rule='SELECT_GLOBAL_FINITE_CHILD_SLOTS',
    maximum_children_per_parent=None,
    maximum_internal_node_reference_occurrences='GLOBAL_FROZEN_GEOMETRY_NODE_CAPACITY',
    shared_cut='SAME_LOGICAL_BOUNDARY_SLOT_REF',
    positive_gap='DISTINCT_BOUNDARY_REFS_EXPLICIT_DIRECTED_INTERVAL_AND_SPAN_CHAIN',
    closure='EXPLICIT_PHASE_WINDING_DISTINCT_LOGICAL_START_END_SHARED_PHYSICAL_ROOT',
    current_actual_cross_domain_policy='SAME_FACE_ONLY',
    current_actual_carrier_binding='PENDING_COMPLETE_SOURCE_PROVIDER',
    structure_gate_implies_joint_geometry=False,
    structure_gate_implies_model_accuracy=False,
    full_contract_pass=False,
)


def _keys(record, required, optional=()):
    if not isinstance(record, dict) or not set(required) <= set(record) or set(record) - set(required) - set(optional):
        raise ValueError('MISSING_OR_UNDECLARED_DECODER_FIELD')


def slot(record, expected=None):
    _keys(record, ('anchor_id', 'role', 'local_slot_ordinal'))
    anchor, role, ordinal = record['anchor_id'], record['role'], record['local_slot_ordinal']
    if (type(anchor) is not int or role not in CAPACITIES or type(ordinal) is not int
            or not 0 <= ordinal < CAPACITIES[role] or (expected is not None and role != expected)):
        raise ValueError('INVALID_SOURCE_SLOT_REF_OR_ROLE')
    return anchor, role, ordinal


def domain(record):
    _keys(record, ('type', 'id'))
    if record['type'] not in ('surface', 'carrier') or type(record['id']) is not int or record['id'] < 0:
        raise ValueError('INVALID_TYPED_SOURCE_DOMAIN_REF')
    return record['type'], record['id']


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('FINITE_LITERAL_PARAMETER_REQUIRED')
    return float(value)


class FrozenSurfaceGraphValidator:
    def __init__(self, connection):
        if connection.execute('PRAGMA query_only').fetchone() != (1,):
            raise ValueError('READ_ONLY_FROZEN_SOURCE_MANIFEST_REQUIRED')
        self.db = connection
        self.anchor_count = connection.execute('SELECT count(*) FROM anchors').fetchone()[0]
        self.surface_ids = frozenset(row[0] for row in connection.execute('SELECT face_id FROM domains'))

    def validate(self, graph):
        _keys(graph, ('version', 'coordinates', 'geometry_spans', 'children', 'parents'))
        if graph['version'] != VERSION or any(not isinstance(graph[key], list) for key in graph if key != 'version'):
            raise ValueError('FROZEN_GRAPH_VERSION_AND_EXPLICIT_DEFINITION_LISTS_REQUIRED')
        coordinates, spans, children, parents, definitions = {}, {}, {}, {}, []
        issues = []

        def declare_domain(record):
            identity = domain(record)
            if identity[0] == 'surface' and identity[1] not in self.surface_ids:
                raise ValueError('SURFACE_DOMAIN_NOT_IN_FROZEN_SOURCE_MANIFEST')
            if identity[0] == 'carrier':
                issues.append(dict(reason='UNBOUND_SOURCE_CARRIER_DOMAIN', domain=list(identity)))
            return identity

        for key, table, role in (('coordinates', coordinates, None), ('geometry_spans', spans, 'geometry_span'),
                                 ('children', children, 'child'), ('parents', parents, 'parent')):
            for record in graph[key]:
                if not isinstance(record, dict) or 'slot' not in record:
                    raise ValueError('EXPLICIT_FINITE_SLOT_DEFINITION_REQUIRED')
                identity = slot(record['slot'], role)
                if key == 'coordinates' and identity[1] not in ('boundary', 'geometry_node'):
                    raise ValueError('COORDINATE_MUST_DEFINE_BOUNDARY_OR_GEOMETRY_NODE_SLOT')
                if identity in table:
                    raise ValueError('DUPLICATE_DEFINING_SLOT_OUTPUT')
                table[identity] = record
                definitions.append(record['slot'])
        capacity = validate_slot_capacity(self.db, definitions)
        coordinate_domains, coordinate_dependencies = {}, {}
        for identity, record in coordinates.items():
            _keys(record, ('slot', 'domain'), ('position', 'physical_ref', 'curve_endpoint'))
            selected = [key for key in ('position', 'physical_ref', 'curve_endpoint') if key in record]
            if len(selected) != 1:
                raise ValueError('ONE_COORDINATE_DEFINITION_OR_EXPLICIT_REF_REQUIRED')
            coordinate_domains[identity] = declare_domain(record['domain'])
            if selected[0] == 'position':
                position = record['position']
                if coordinate_domains[identity][0] == 'surface':
                    _keys(position, ('uv_mm',))
                    if not isinstance(position['uv_mm'], list) or len(position['uv_mm']) != 2:
                        raise ValueError('TWO_CONTINUOUS_UV_PARAMETERS_REQUIRED')
                    for value in position['uv_mm']:
                        _finite(value)
                else:
                    _keys(position, ('q_mm',))
                    _finite(position['q_mm'])
                coordinate_dependencies[identity] = None
            elif selected[0] == 'physical_ref':
                reference = slot(record['physical_ref'])
                if reference not in coordinates:
                    raise ValueError('COORDINATE_REF_NOT_DEFINED')
                coordinate_dependencies[identity] = reference
            else:
                endpoint = record['curve_endpoint']
                _keys(endpoint, ('span', 'endpoint_index'))
                reference = slot(endpoint['span'], 'geometry_span')
                if reference not in spans or spans[reference].get('kind') != 'arc':
                    raise ValueError('DERIVED_ENDPOINT_REQUIRES_DECLARED_ARC_PARAMETERS')
                if type(endpoint['endpoint_index']) is not int or endpoint['endpoint_index'] not in (0, 1):
                    raise ValueError('ORDERED_ARC_ENDPOINT_INDEX_REQUIRED')
                coordinate_dependencies[identity] = None

        roots = {}
        for initial in coordinates:
            chain, seen, current = [], set(), initial
            while current not in roots:
                if current in seen:
                    raise ValueError('PHYSICAL_COORDINATE_REFERENCE_CYCLE')
                seen.add(current)
                chain.append(current)
                dependency = coordinate_dependencies[current]
                if dependency is None:
                    roots[current] = current
                    break
                current = dependency
            root = roots[current]
            for identity in chain:
                roots[identity] = root
                if coordinate_domains[identity] != coordinate_domains[root]:
                    issues.append(dict(reason='UNBOUND_CROSS_DOMAIN_COORDINATE_PERMISSION',
                                       slot=list(identity), root=list(root)))

        span_domains, span_ends, span_closed = {}, {}, {}
        internal_references = 0
        for identity, record in spans.items():
            _keys(record, ('slot', 'domain', 'kind', 'start_boundary', 'end_boundary',
                           'internal_geometry_nodes', 'is_closed'), ('parameters',))
            selected_domain = declare_domain(record['domain'])
            span_domains[identity] = selected_domain
            if record['kind'] not in ('line', 'arc', 'polyline') or type(record['is_closed']) is not bool:
                raise ValueError('DECLARED_PRIMITIVE_KIND_AND_CLOSURE_REQUIRED')
            if record['kind'] == 'line' and record['is_closed']:
                raise ValueError('POSITIVE_LINE_CANNOT_DECLARE_FULL_CLOSURE')
            ends = tuple(slot(record[key], 'boundary') for key in ('start_boundary', 'end_boundary'))
            if any(end not in coordinates for end in ends):
                raise ValueError('SPAN_BOUNDARY_REF_NOT_DEFINED')
            nodes = record['internal_geometry_nodes']
            if not isinstance(nodes, list):
                raise ValueError('EXPLICIT_ORDERED_GEOMETRY_NODE_REFS_REQUIRED')
            if record['kind'] != 'polyline' and nodes:
                raise ValueError('INTERNAL_NODE_LIST_BELONGS_TO_POLYLINE_ONLY')
            if record['kind'] == 'polyline' and not nodes and selected_domain[0] == 'surface':
                raise ValueError('SURFACE_POLYLINE_REQUIRES_INTERNAL_SELECTED_GEOMETRY_NODES')
            node_ids = [slot(row, 'geometry_node') for row in nodes]
            if any(node not in coordinates for node in node_ids):
                raise ValueError('INTERNAL_GEOMETRY_NODE_REF_NOT_DEFINED')
            internal_references += len(node_ids)
            for point in (*ends, *node_ids):
                if coordinate_domains[point] != selected_domain:
                    issues.append(dict(reason='UNBOUND_CROSS_DOMAIN_SPAN_COORDINATE_PERMISSION',
                                       span=list(identity), coordinate=list(point)))
            if record['kind'] == 'arc' and selected_domain[0] == 'surface':
                if 'parameters' not in record or not isinstance(record['parameters'], dict):
                    raise ValueError('LITERAL_CONTINUOUS_ARC_PARAMETERS_REQUIRED')
                _keys(record['parameters'], ('center_mm', 'radius_mm', 'u_axis', 'v_axis',
                                             'sweep_rad', 'is_closed', 'turn_count'))
                curve = ConcreteCurve({'kind': 'arc', **record['parameters']})
                if curve.record['dimensions'] != 2 or curve.record['is_closed'] != record['is_closed']:
                    raise ValueError('ARC_PARAMETER_FRAME_OR_CLOSURE_MISMATCH')
                if curve.record['turn_count'] > self.anchor_count * CAPACITIES['geometry_span']:
                    raise ValueError('ARC_WINDING_EXCEEDS_FROZEN_GLOBAL_SPAN_CAPACITY')
            elif selected_domain[0] == 'surface' and 'parameters' in record:
                raise ValueError('LINE_POLYLINE_COORDINATES_MUST_REFERENCE_FINITE_SLOTS')
            elif selected_domain[0] == 'carrier':
                if record.get('parameters') != {'source_path_slice': True}:
                    raise ValueError('CARRIER_REQUIRES_SOURCE_PATH_SLICE_NOT_FREE_TEACHER_GEOMETRY')
            if record['is_closed']:
                if ends[0] == ends[1] or ends[0][0] != ends[1][0] or roots[ends[0]] != roots[ends[1]]:
                    raise ValueError('CLOSED_SPAN_REQUIRES_DISTINCT_LOGICAL_SHARED_PHYSICAL_ANCHOR')
            elif ends[0] == ends[1]:
                raise ValueError('POSITIVE_OPEN_SPAN_NEEDS_DISTINCT_LOGICAL_BOUNDARIES')
            span_ends[identity], span_closed[identity] = ends, record['is_closed']
        if internal_references > self.anchor_count * CAPACITIES['geometry_node']:
            raise ValueError('GLOBAL_GEOMETRY_NODE_REFERENCE_OCCURRENCE_CAPACITY_EXCEEDED')
        for identity, record in coordinates.items():
            if 'curve_endpoint' in record:
                reference = slot(record['curve_endpoint']['span'], 'geometry_span')
                if coordinate_domains[identity] != span_domains[reference]:
                    issues.append(dict(reason='UNBOUND_DERIVED_ARC_ENDPOINT_DOMAIN', slot=list(identity)))

        span_owners, coordinate_uses, child_owners = Counter(), set(), Counter()

        def chain(records, first, last, owner):
            if not isinstance(records, list) or not records:
                raise ValueError('EXPLICIT_NONEMPTY_ORDERED_SPAN_CHAIN_REQUIRED')
            selected = [slot(row, 'geometry_span') for row in records]
            if len(set(selected)) != len(selected) or any(value not in spans for value in selected):
                raise ValueError('DUPLICATE_OR_UNDEFINED_SPAN_CHAIN_REF')
            if span_ends[selected[0]][0] != first or span_ends[selected[-1]][1] != last:
                raise ValueError('SPAN_CHAIN_LOGICAL_ENDPOINT_MISMATCH')
            for left, right in zip(selected, selected[1:]):
                a, b = span_ends[left][1], span_ends[right][0]
                if roots[a] != roots[b]:
                    raise ValueError('GEOMETRIC_JOIN_NEEDS_EXPLICIT_SHARED_PHYSICAL_REF')
                if span_domains[left] != span_domains[right]:
                    issues.append(dict(reason='UNBOUND_CROSS_DOMAIN_SPAN_TRANSITION',
                                       owner=list(owner), left=list(left), right=list(right)))
            for value in selected:
                span_owners[value] += 1
                coordinate_uses.update(span_ends[value])
                coordinate_uses.update(slot(row, 'geometry_node') for row in spans[value]['internal_geometry_nodes'])
            return selected

        child_spans, child_ends = {}, {}
        for identity, record in children.items():
            _keys(record, ('slot', 'parent', 'order_index', 'start_boundary', 'end_boundary',
                           'geometry_spans', 'geometry_kind', 'weld_type'))
            parent = slot(record['parent'], 'parent')
            if parent not in parents or type(record['order_index']) is not int or record['order_index'] < 0:
                raise ValueError('EXPLICIT_PARENT_REF_AND_NONNEGATIVE_CHILD_ORDER_REQUIRED')
            if record['geometry_kind'] not in ('line', 'curve'):
                raise ValueError('EXPLICIT_CHILD_GEOMETRY_CATEGORY_REQUIRED')
            if record['weld_type'] not in ('horizontal', 'vertical'):
                raise ValueError('EXPLICIT_H_V_WELD_TYPE_REQUIRED')
            first, last = (slot(record[key], 'boundary') for key in ('start_boundary', 'end_boundary'))
            if first not in coordinates or last not in coordinates:
                raise ValueError('CHILD_BOUNDARY_NOT_DEFINED')
            child_spans[identity] = chain(record['geometry_spans'], first, last, identity)
            if (record['geometry_kind'] == 'line'
                    and any(spans[value]['kind'] == 'arc' and span_domains[value][0] == 'surface'
                            for value in child_spans[identity])):
                raise ValueError('SURFACE_ARC_CANNOT_BE_DECLARED_STRAIGHT_CHILD')
            child_ends[identity] = first, last

        gap_count = shared_count = 0
        for identity, record in parents.items():
            _keys(record, ('slot', 'children', 'start_boundary', 'end_boundary', 'allowed_domains',
                           'direction', 'is_closed', 'phase_boundary', 'turn_count', 'gaps'))
            if type(record['direction']) is not int or record['direction'] not in (-1, 1) or type(record['is_closed']) is not bool:
                raise ValueError('EXPLICIT_PARENT_DIRECTION_AND_CLOSURE_REQUIRED')
            if type(record['turn_count']) is not int or not 0 <= record['turn_count'] <= self.anchor_count * CAPACITIES['geometry_span']:
                raise ValueError('PARENT_TURN_COUNT_EXCEEDS_FROZEN_GLOBAL_CAPACITY')
            if record['is_closed'] != (record['turn_count'] > 0):
                raise ValueError('EXPLICIT_CLOSED_PARENT_TURN_DECLARATION_REQUIRED')
            if not isinstance(record['children'], list) or not record['children']:
                raise ValueError('SELECTED_PARENT_REQUIRES_ORDERED_CHILD_REFS')
            ordered = [slot(row, 'child') for row in record['children']]
            if len(set(ordered)) != len(ordered) or any(value not in children for value in ordered):
                raise ValueError('DUPLICATE_OR_UNDEFINED_ORDERED_CHILD_REF')
            if not isinstance(record['allowed_domains'], list) or not record['allowed_domains']:
                raise ValueError('EXPLICIT_PARENT_ALLOWED_DOMAIN_SET_REQUIRED')
            allowed = [declare_domain(row) for row in record['allowed_domains']]
            if len(set(allowed)) != len(allowed):
                raise ValueError('DUPLICATE_PARENT_DOMAIN_DECLARATION')
            first, last = (slot(record[key], 'boundary') for key in ('start_boundary', 'end_boundary'))
            if first != child_ends[ordered[0]][0] or last != child_ends[ordered[-1]][1]:
                raise ValueError('PARENT_CHILD_ORDERED_ENDPOINT_MISMATCH')
            if slot(record['phase_boundary'], 'boundary') != first:
                raise ValueError('PARENT_PHASE_MUST_REFERENCE_EXPLICIT_LOGICAL_START')
            if record['is_closed'] and (first == last or first[0] != last[0] or roots[first] != roots[last]):
                raise ValueError('CLOSED_PARENT_REQUIRES_DISTINCT_LOGICAL_SHARED_PHYSICAL_ANCHOR')
            if not isinstance(record['gaps'], list) or len(record['gaps']) != len(ordered) - 1:
                raise ValueError('ALL_ORIGINAL_ORDERED_CHILD_ADJACENCIES_REQUIRED')
            for number, child in enumerate(ordered):
                if slot(children[child]['parent'], 'parent') != identity or children[child]['order_index'] != number:
                    raise ValueError('PARENT_CHILD_REFERENCE_OR_ORDER_CONFLICT')
                if any(span_domains[value] not in allowed for value in child_spans[child]):
                    raise ValueError('CHILD_SPAN_OUTSIDE_DECLARED_PARENT_DOMAIN_SET')
                child_owners[child] += 1
            for number, gap in enumerate(record['gaps']):
                _keys(gap, ('relation', 'left_boundary', 'right_boundary', 'geometry_spans'), ('directed_interval_mm',))
                left, right = slot(gap['left_boundary'], 'boundary'), slot(gap['right_boundary'], 'boundary')
                if left != child_ends[ordered[number]][1] or right != child_ends[ordered[number+1]][0]:
                    raise ValueError('EXPLICIT_GAP_BOUNDARY_REF_MISMATCH')
                if gap['relation'] == 'SHARED_BOUNDARY':
                    if left != right or gap['geometry_spans'] or 'directed_interval_mm' in gap:
                        raise ValueError('SHARED_BUSINESS_CUT_MUST_USE_SAME_LOGICAL_BOUNDARY')
                    shared_count += 1
                elif gap['relation'] == 'POSITIVE_GAP':
                    interval = gap.get('directed_interval_mm')
                    if (left == right or not isinstance(interval, list) or len(interval) != 2
                            or not _finite(interval[0]) < _finite(interval[1])):
                        raise ValueError('POSITIVE_GAP_REQUIRES_DISTINCT_REFS_AND_STRICT_PARAMETER_INTERVAL')
                    selected = chain(gap['geometry_spans'], left, right, identity)
                    if any(span_domains[value] not in allowed for value in selected):
                        raise ValueError('GAP_SPAN_OUTSIDE_DECLARED_PARENT_DOMAIN_SET')
                    gap_count += 1
                else:
                    raise ValueError('UNKNOWN_OR_IMPLICIT_BUSINESS_GAP_RELATION')
        if any(child_owners[value] != 1 for value in children) or any(span_owners[value] != 1 for value in spans):
            raise ValueError('ORPHAN_OR_MULTIPLE_INSTANCE_OWNERSHIP')
        for initial in list(coordinate_uses):
            dependency = coordinate_dependencies[initial]
            while dependency is not None:
                coordinate_uses.add(dependency)
                dependency = coordinate_dependencies[dependency]
        if set(coordinates) != coordinate_uses:
            raise ValueError('UNUSED_DEFINING_COORDINATE_SLOT')
        return dict(version=VERSION,
                    status='PASS_COMPLETE_FROZEN_DECODER_GRAPH_STRUCTURE' if not issues else 'UNPROVEN_FROZEN_DECODER_GRAPH_DOMAIN_BINDINGS',
                    source_anchor_count=self.anchor_count, global_capacity=capacity,
                    selected_parent_count=len(parents), selected_child_count=len(children), selected_span_count=len(spans),
                    selected_coordinate_count=len(coordinates), selected_geometry_node_count=sum(value[1] == 'geometry_node' for value in coordinates),
                    internal_node_reference_occurrences=internal_references, physical_coordinate_root_count=len(set(roots.values())),
                    shared_business_cut_count=shared_count, explicit_positive_gap_count=gap_count,
                    unbound_domain_or_transition_issues=issues, structure_and_global_capacity_verified=True,
                    continuous_local_coordinate_permissions_verified=False,
                    actual_positive_gap_metric_and_parent_direction_verified=False,
                    complete_continuous_curves_and_joint_geometry_verified=False,
                    material_and_weld_identity_verified=False, actual_all111_or_all1605_joint_expression_run=False,
                    max_children_per_parent=None, implicit_coordinate_merging_used=False,
                    geometry_corners_added_as_business_children=False, model_inference_calls=0,
                    actual_weld_optimizer_updates=0, full_contract_pass=False)
