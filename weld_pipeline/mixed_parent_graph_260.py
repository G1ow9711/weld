"""Finite mixed source graph ownership and explicit whole-parent structure.

This validates actual typed graph records, without a fabricated coordinate
shadow. Geometric permission and world primitive certificates are additional
gates; syntax or finite capacity never implies a weld prediction is correct.
"""
from collections import Counter

from .decoder_graph_structure_254 import _keys, slot
from .decoder_output_slots_247 import CAPACITIES, validate_slot_capacity
from .concrete_curve_certificate_250 import ConcreteCurve
from .original_curve_carriers_256 import read_fraction
from .raw_triangle_world_charts_258 import exact
from .mixed_world_joint_binding_259 import CarrierOccurrence


VERSION = 'COMPLETE_TYPED_MIXED_SOURCE_PARENT_CHILD_GRAPH_260_v1'
PROTOCOL = dict(version=VERSION,
    domains=['LEGACY_SURFACE', 'RAW_TRIANGLE', 'RAW_CARRIER'],
    authority='UNCHANGED_247_CANONICAL_ANCHOR_ROLE_LOCAL_ORDINAL',
    coordinates='ONE_POSITION_EXPLICIT_PHYSICAL_REF_WITH_OWN_SOURCE_POSITION_OR_ANALYTIC_ARC_ENDPOINT',
    local_permission='EVERY_DECLARED_COORDINATE_AND_LOCATED_ROLE_HAS_OWN_SOURCE_PERMISSION',
    coordinate_equality='EXPLICIT_REFERENCE_AND_EXACT_WORLD_PROOF_NO_NEARNESS_CLUSTER',
    spans=['line', 'arc', 'polyline', 'carrier_slice'],
    arc_body='EXPLICIT_POSITIVE_NORMALIZED_RESTRICTION_OF_ONE_EXISTING_SPAN_CIRCLE_AUTHORITY',
    span_owner='EXACTLY_ONE_CHILD_OR_EXPLICIT_PARENT_GAP',
    child_world_kind=['line', 'arc', 'polyline', 'compound'],
    child_corners='GEOMETRY_NODES_AND_INTRINSIC_CARRIER_CORNERS_NOT_BUSINESS_CHILDREN',
    parent_frames=['line', 'arc', 'carrier', 'piecewise'],
    periodic_limit='EXISTING_SOURCE_ANCHORS_TIMES_GEOMETRY_SPAN_CAPACITY',
    open_complete_turns='DECLARED_FLOOR_COMPLETE_PERIODS_PLUS_EXPLICIT_POSITIVE_REMAINDER',
    closed='EXPLICIT_PHASE_DIRECTION_COMPLETE_TURNS_AND_SHARED_PHYSICAL_ROOT',
    positive_gap='DISTINCT_LOGICAL_BOUNDARIES_EXPLICIT_COMPLETE_SPAN_CHAIN_AND_POSITIVE_EXPORTED_WORLD_METRIC',
    unknowns='RETAIN_IN_FULL_ORIGINAL_DENOMINATORS',
    graph_structure_is_geometry_or_model_acceptance=False, full_contract_pass=False)


def rational(value):
    return read_fraction(value) if isinstance(value, dict) else exact(value)


class MixedGraphValidator:
    def __init__(self, slots, carriers):
        if any(db.execute('PRAGMA query_only').fetchone() != (1,) for db in (slots, carriers)):
            raise ValueError('READ_ONLY_FROZEN_SOURCE_MANIFESTS_REQUIRED')
        self.slots, self.carriers = slots, carriers
        self.anchor_count = slots.execute('SELECT count(*) FROM anchors').fetchone()[0]
        self.span_capacity = self.anchor_count * CAPACITIES['geometry_span']

    def domain(self, record):
        if not isinstance(record, dict) or 'type' not in record:
            raise ValueError('EXPLICIT_TYPED_SOURCE_DOMAIN_REQUIRED')
        kind = record['type']
        if kind == 'LEGACY_SURFACE':
            _keys(record, ('type', 'face_id'))
            face = record['face_id']
            if type(face) is not int or self.slots.execute('SELECT 1 FROM domains WHERE face_id=?', (face,)).fetchone() is None:
                raise ValueError('LEGACY_DOMAIN_NOT_IN_ORIGINAL_SOURCE')
            return kind, face
        if kind == 'RAW_TRIANGLE':
            _keys(record, ('type', 'face_id', 'triangle_ordinal'))
            face, triangle = record['face_id'], record['triangle_ordinal']
            if (type(face) is not int or type(triangle) is not int or self.slots.execute(
                    'SELECT 1 FROM triangles WHERE face_id=? AND triangle_ordinal=?', (face, triangle)).fetchone() is None):
                raise ValueError('RAW_TRIANGLE_NOT_IN_ORIGINAL_SOURCE')
            return kind, face, triangle
        if kind == 'RAW_CARRIER':
            _keys(record, ('type', 'carrier_id'))
            identity = record['carrier_id']
            if not isinstance(identity, str) or self.carriers.execute('SELECT 1 FROM carriers WHERE carrier_id=?', (identity,)).fetchone() is None:
                raise ValueError('CARRIER_NOT_IN_COMPLETE_SOURCE_MANIFEST')
            return kind, identity
        raise ValueError('UNDECLARED_SOURCE_DOMAIN_TYPE')

    def validate(self, graph):
        _keys(graph, ('version', 'coordinates', 'geometry_spans', 'children', 'parents'))
        if graph['version'] != VERSION or any(not isinstance(graph[key], list) for key in graph if key != 'version'):
            raise ValueError('FROZEN_MIXED_GRAPH_VERSION_AND_EXPLICIT_LISTS_REQUIRED')
        coordinates, spans, children, parents = {}, {}, {}, {}
        definitions = []
        for field, table, role in (('coordinates', coordinates, None), ('geometry_spans', spans, 'geometry_span'),
                                   ('children', children, 'child'), ('parents', parents, 'parent')):
            for row in graph[field]:
                if not isinstance(row, dict) or 'slot' not in row:
                    raise ValueError('EXPLICIT_FINITE_SLOT_DEFINITION_REQUIRED')
                identity = slot(row['slot'], role)
                if field == 'coordinates' and identity[1] not in ('boundary', 'geometry_node'):
                    raise ValueError('COORDINATE_ROLE_REQUIRED')
                if identity in table:
                    raise ValueError('DUPLICATE_DEFINING_SLOT')
                table[identity] = row
                definitions.append(row['slot'])
        capacity = validate_slot_capacity(self.slots, definitions)
        coordinate_domains, dependencies = {}, {}
        for identity, row in coordinates.items():
            _keys(row, ('slot', 'domain'), ('position', 'physical_ref', 'local_position', 'curve_endpoint'))
            selected = [key for key in ('position', 'physical_ref', 'curve_endpoint') if key in row]
            if len(selected) != 1:
                raise ValueError('EXACTLY_ONE_COORDINATE_FORM_REQUIRED')
            if 'local_position' in row and selected[0] != 'physical_ref':
                raise ValueError('OWN_SOURCE_POSITION_ONLY_APPLIES_TO_EXPLICIT_PHYSICAL_REF')
            domain = self.domain(row['domain'])
            coordinate_domains[identity] = domain
            if selected[0] == 'position':
                position = row['position']
                if domain[0] == 'LEGACY_SURFACE':
                    _keys(position, ('uv_mm',))
                    values = position['uv_mm']
                    if not isinstance(values, list) or len(values) != 2:
                        raise ValueError('TWO_LEGACY_UV_PARAMETERS_REQUIRED')
                    tuple(map(rational, values))
                elif domain[0] == 'RAW_TRIANGLE':
                    _keys(position, ('barycentric',))
                    values = position['barycentric']
                    if not isinstance(values, list) or len(values) != 2:
                        raise ValueError('TWO_RAW_BARYCENTRIC_PARAMETERS_REQUIRED')
                    a, b = map(rational, values)
                    if a < 0 or b < 0 or a + b > 1:
                        raise ValueError('RAW_BARYCENTRIC_POINT_OUTSIDE_ORIGINAL_TRIANGLE')
                else:
                    _keys(position, ('occurrence',))
                    CarrierOccurrence.from_record(position['occurrence'])
                dependencies[identity] = None
            elif selected[0] == 'physical_ref':
                dependency = slot(row['physical_ref'])
                if dependency not in coordinates:
                    raise ValueError('UNDEFINED_PHYSICAL_COORDINATE_REF')
                if 'local_position' not in row:
                    raise ValueError('EXPLICIT_PHYSICAL_REF_NEEDS_OWN_SOURCE_POSITION')
                position = row['local_position']
                if domain[0] == 'LEGACY_SURFACE':
                    if 'curve_endpoint' in position:
                        _keys(position, ('curve_endpoint',))
                        endpoint = position['curve_endpoint']
                        _keys(endpoint, ('span', 'endpoint_index'))
                        master = slot(endpoint['span'], 'geometry_span')
                        if master not in spans or spans[master].get('kind') != 'arc' or type(endpoint['endpoint_index']) is not int or endpoint['endpoint_index'] not in (0, 1):
                            raise ValueError('OWN_ANALYTIC_SOURCE_ENDPOINT_REF_REQUIRED')
                        values = None
                    else:
                        _keys(position, ('uv_mm',))
                        values = position['uv_mm']
                elif domain[0] == 'RAW_TRIANGLE':
                    _keys(position, ('barycentric',))
                    values = position['barycentric']
                else:
                    _keys(position, ('occurrence',))
                    CarrierOccurrence.from_record(position['occurrence'])
                    values = None
                if values is not None:
                    if not isinstance(values, list) or len(values) != 2:
                        raise ValueError('TWO_OWN_SOURCE_PARAMETERS_REQUIRED')
                    a, b = map(rational, values)
                    if domain[0] == 'RAW_TRIANGLE' and (a < 0 or b < 0 or a + b > 1):
                        raise ValueError('RAW_BARYCENTRIC_POINT_OUTSIDE_ORIGINAL_TRIANGLE')
                dependencies[identity] = dependency
            else:
                endpoint = row['curve_endpoint']
                _keys(endpoint, ('span', 'endpoint_index'))
                span = slot(endpoint['span'], 'geometry_span')
                if span not in spans or spans[span].get('kind') != 'arc' or type(endpoint['endpoint_index']) is not int or endpoint['endpoint_index'] not in (0, 1):
                    raise ValueError('ANALYTIC_ARC_ENDPOINT_REF_REQUIRED')
                if domain[0] != 'LEGACY_SURFACE':
                    raise ValueError('CURRENT_ANALYTIC_ENDPOINT_REQUIRES_DECLARED_LEGACY_SURFACE_FRAME')
                dependencies[identity] = None
        roots = {}
        for initial in coordinates:
            current, chain, seen = initial, [], set()
            while current not in roots:
                if current in seen:
                    raise ValueError('PHYSICAL_REFERENCE_CYCLE')
                seen.add(current)
                chain.append(current)
                dependency = dependencies[current]
                if dependency is None:
                    roots[current] = current
                    break
                current = dependency
            for value in chain:
                roots[value] = roots[current]
        domains, ends, nodes, local_arcs, arc_intervals, arc_authorities = {}, {}, {}, {}, {}, {}
        internal_reference_count = 0
        for identity, row in spans.items():
            _keys(row, ('slot', 'domain', 'kind', 'start_boundary', 'end_boundary', 'internal_geometry_nodes', 'is_closed'), ('parameters', 'parameter_interval', 'authority_span'))
            domain = self.domain(row['domain'])
            if row['kind'] not in ('line', 'arc', 'polyline', 'carrier_slice') or type(row['is_closed']) is not bool:
                raise ValueError('EXPLICIT_SPAN_KIND_AND_CLOSURE_REQUIRED')
            first, last = (slot(row[key], 'boundary') for key in ('start_boundary', 'end_boundary'))
            if first not in coordinates or last not in coordinates:
                raise ValueError('UNDEFINED_SPAN_BOUNDARY')
            if not isinstance(row['internal_geometry_nodes'], list):
                raise ValueError('ORDERED_INTERNAL_NODE_LIST_REQUIRED')
            selected = [slot(ref, 'geometry_node') for ref in row['internal_geometry_nodes']]
            if any(value not in coordinates for value in selected) or (row['kind'] != 'polyline' and selected):
                raise ValueError('INTERNAL_NODES_MUST_BELONG_TO_DECLARED_POLYLINE')
            internal_reference_count += len(selected)
            if row['kind'] == 'arc':
                if domain[0] != 'LEGACY_SURFACE':
                    raise ValueError('WORLD_ARC_DOMAIN_BINDING_REQUIRES_EXPLICIT_ANALYTIC_SOURCE_FRAME')
                _keys(row.get('parameters'), ('center_mm', 'radius_mm', 'u_axis', 'v_axis', 'sweep_rad', 'is_closed', 'turn_count'))
                curve = ConcreteCurve({'kind': 'arc', **row['parameters']})
                if curve.record['dimensions'] != 2 or curve.record['turn_count'] > self.span_capacity:
                    raise ValueError('ANALYTIC_ARC_FRAME_CLOSURE_OR_SOURCE_TURN_CAPACITY_CONFLICT')
                local_arcs[identity] = curve
                interval = row.get('parameter_interval', [0, 1])
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError('EXPLICIT_ARC_PARAMETER_RESTRICTION_REQUIRED')
                first_parameter, last_parameter = map(rational, interval)
                if not 0 <= first_parameter < last_parameter <= 1:
                    raise ValueError('POSITIVE_ARC_RESTRICTION_WITHIN_AUTHORITY_RANGE_REQUIRED')
                master = slot(row.get('authority_span', row['slot']), 'geometry_span')
                if master not in spans or spans[master].get('kind') != 'arc':
                    raise ValueError('EXISTING_GEOMETRY_SPAN_ARC_AUTHORITY_REQUIRED')
                arc_intervals[identity] = first_parameter, last_parameter
                arc_authorities[identity] = master
            elif row['kind'] == 'carrier_slice':
                if domain[0] != 'RAW_CARRIER' or selected:
                    raise ValueError('ORIGINAL_CARRIER_SLICE_WITH_INTRINSIC_CORNERS_REQUIRED')
                _keys(row.get('parameters'), ('direction', 'extra_turns'))
                params = row['parameters']
                if type(params['direction']) is not int or params['direction'] not in (-1, 1) or type(params['extra_turns']) is not int or not 0 <= params['extra_turns'] <= self.span_capacity:
                    raise ValueError('EXPLICIT_CARRIER_DIRECTION_AND_FINITE_COMPLETE_TURNS_REQUIRED')
            elif 'parameters' in row or domain[0] == 'RAW_CARRIER':
                raise ValueError('FREE_SPAN_PARAMETERS_OR_CARRIER_GEOMETRY_FORBIDDEN')
            if row['kind'] != 'arc' and ('parameter_interval' in row or 'authority_span' in row):
                raise ValueError('ONLY_ARCS_CAN_DECLARE_ANALYTIC_RESTRICTIONS')
            if row['is_closed']:
                if row['kind'] == 'line' or first == last or first[0] != last[0] or roots[first] != roots[last]:
                    raise ValueError('CLOSED_SPAN_NEEDS_DISTINCT_LOGICAL_SHARED_PHYSICAL_ANCHOR')
            elif first == last:
                raise ValueError('POSITIVE_OPEN_SPAN_REQUIRES_DISTINCT_LOGICAL_BOUNDARIES')
            domains[identity], ends[identity], nodes[identity] = domain, (first, last), selected
        if internal_reference_count > self.anchor_count * CAPACITIES['geometry_node']:
            raise ValueError('GLOBAL_INTERNAL_NODE_REFERENCE_CAPACITY_EXCEEDED')
        for identity, row in coordinates.items():
            endpoint = row.get('curve_endpoint', row.get('local_position', {}).get('curve_endpoint'))
            if endpoint is not None:
                master = slot(endpoint['span'], 'geometry_span')
                if coordinate_domains[identity] != domains[master]:
                    raise ValueError('ANALYTIC_ENDPOINT_SOURCE_FRAME_CONFLICT')
        for identity, master in arc_authorities.items():
            if arc_authorities[master] != master:
                raise ValueError('ARC_AUTHORITY_MUST_BE_A_DIRECT_SELF_ROOT_NOT_A_REFERENCE_CYCLE')
        span_owners, child_owners, used_coordinates = Counter(), Counter(), set()

        def chain(references, first, last):
            if not isinstance(references, list) or not references:
                raise ValueError('NONEMPTY_COMPLETE_ORDERED_SPAN_CHAIN_REQUIRED')
            selected = [slot(ref, 'geometry_span') for ref in references]
            if len(selected) != len(set(selected)) or any(value not in spans for value in selected):
                raise ValueError('DUPLICATE_OR_UNDEFINED_ORDERED_SPAN')
            if ends[selected[0]][0] != first or ends[selected[-1]][1] != last:
                raise ValueError('LOGICAL_SPAN_CHAIN_ENDPOINT_CONFLICT')
            for left, right in zip(selected, selected[1:]):
                if roots[ends[left][1]] != roots[ends[right][0]]:
                    raise ValueError('SPAN_JOIN_NEEDS_EXPLICIT_SHARED_PHYSICAL_ROOT')
            for value in selected:
                span_owners[value] += 1
                used_coordinates.update(ends[value])
                used_coordinates.update(nodes[value])
            return selected

        child_chains, child_ends = {}, {}
        for identity, row in children.items():
            _keys(row, ('slot', 'parent', 'order_index', 'start_boundary', 'end_boundary', 'geometry_spans', 'geometry_kind', 'world_kind', 'weld_type'))
            parent = slot(row['parent'], 'parent')
            if parent not in parents or type(row['order_index']) is not int or row['order_index'] < 0:
                raise ValueError('ORDERED_DEFINED_PARENT_REF_REQUIRED')
            if row['geometry_kind'] not in ('line', 'curve') or row['world_kind'] not in ('line', 'arc', 'polyline', 'compound') or row['weld_type'] not in ('horizontal', 'vertical'):
                raise ValueError('EXPLICIT_CHILD_WORLD_KIND_AND_H_V_CATEGORY_REQUIRED')
            if (row['geometry_kind'] == 'line') != (row['world_kind'] == 'line'):
                raise ValueError('CHILD_GEOMETRY_CATEGORY_AND_WORLD_KIND_CONFLICT')
            first, last = (slot(row[key], 'boundary') for key in ('start_boundary', 'end_boundary'))
            if first not in coordinates or last not in coordinates:
                raise ValueError('UNDEFINED_CHILD_BOUNDARY')
            child_chains[identity] = chain(row['geometry_spans'], first, last)
            child_ends[identity] = first, last
        parent_chains, shared, positive = {}, 0, 0
        for identity, row in parents.items():
            _keys(row, ('slot', 'children', 'start_boundary', 'end_boundary', 'allowed_domains', 'direction',
                        'is_closed', 'phase_boundary', 'turn_count', 'gaps', 'frame'))
            if type(row['direction']) is not int or row['direction'] not in (-1, 1) or type(row['is_closed']) is not bool:
                raise ValueError('EXPLICIT_PARENT_DIRECTION_AND_CLOSURE_REQUIRED')
            if type(row['turn_count']) is not int or not 0 <= row['turn_count'] <= self.span_capacity or (row['is_closed'] and row['turn_count'] == 0):
                raise ValueError('EXPLICIT_WHOLE_PARENT_TURNS_WITHIN_SOURCE_CAPACITY_REQUIRED')
            if not isinstance(row['children'], list) or not row['children']:
                raise ValueError('NONEMPTY_ORDERED_CHILD_LIST_REQUIRED')
            ordered = [slot(ref, 'child') for ref in row['children']]
            if len(ordered) != len(set(ordered)) or any(value not in children for value in ordered):
                raise ValueError('DUPLICATE_OR_UNDEFINED_CHILD_REF')
            if not isinstance(row['allowed_domains'], list) or not row['allowed_domains']:
                raise ValueError('EXPLICIT_PARENT_ALLOWED_DOMAINS_REQUIRED')
            allowed = [self.domain(domain) for domain in row['allowed_domains']]
            if len(allowed) != len(set(allowed)):
                raise ValueError('DUPLICATE_ALLOWED_DOMAIN')
            first, last = (slot(row[key], 'boundary') for key in ('start_boundary', 'end_boundary'))
            if first != child_ends[ordered[0]][0] or last != child_ends[ordered[-1]][1] or slot(row['phase_boundary'], 'boundary') != first:
                raise ValueError('PARENT_LOGICAL_ENDPOINT_OR_PHASE_CONFLICT')
            if row['is_closed'] and (first == last or first[0] != last[0] or roots[first] != roots[last]):
                raise ValueError('CLOSED_PARENT_NEEDS_DISTINCT_LOGICAL_SHARED_PHYSICAL_ANCHOR')
            frame = row['frame']
            if not isinstance(frame, dict) or frame.get('kind') not in ('line', 'arc', 'carrier', 'piecewise'):
                raise ValueError('EXPLICIT_WHOLE_PARENT_PARAMETRIC_FRAME_REQUIRED')
            if frame['kind'] == 'line':
                _keys(frame, ('kind', 'axis_world'))
                if not isinstance(frame['axis_world'], list) or len(frame['axis_world']) != 3 or not any(map(rational, frame['axis_world'])) or row['is_closed'] or row['turn_count']:
                    raise ValueError('POSITIVE_OPEN_LINE_PARENT_FRAME_REQUIRED')
            elif frame['kind'] == 'arc':
                _keys(frame, ('kind', 'authority_span'))
                master = slot(frame['authority_span'], 'geometry_span')
                if master not in local_arcs:
                    raise ValueError('EXPLICIT_DECLARED_ARC_AUTHORITY_REQUIRED')
            elif frame['kind'] == 'carrier':
                _keys(frame, ('kind', 'domain'))
                if self.domain(frame['domain'])[0] != 'RAW_CARRIER':
                    raise ValueError('EXPLICIT_ORIGINAL_CARRIER_PARENT_FRAME_REQUIRED')
            else:
                _keys(frame, ('kind', 'canonical_span_order'))
                if not isinstance(frame['canonical_span_order'], list):
                    raise ValueError('EXPLICIT_SOURCE_PARAMETRIC_SPAN_ORDER_REQUIRED')
            if not isinstance(row['gaps'], list) or len(row['gaps']) != len(ordered) - 1:
                raise ValueError('ALL_ORDERED_BUSINESS_ADJACENCIES_REQUIRED')
            complete = []
            for index, child in enumerate(ordered):
                record = children[child]
                if slot(record['parent'], 'parent') != identity or record['order_index'] != index:
                    raise ValueError('PARENT_CHILD_OWNER_OR_ORDER_CONFLICT')
                if any(domains[value] not in allowed for value in child_chains[child]):
                    raise ValueError('CHILD_OUTSIDE_DECLARED_PARENT_SOURCE_DOMAINS')
                complete.extend(child_chains[child])
                child_owners[child] += 1
                if index == len(ordered) - 1:
                    continue
                gap = row['gaps'][index]
                _keys(gap, ('relation', 'left_boundary', 'right_boundary', 'geometry_spans'))
                left, right = slot(gap['left_boundary'], 'boundary'), slot(gap['right_boundary'], 'boundary')
                if left != child_ends[child][1] or right != child_ends[ordered[index + 1]][0]:
                    raise ValueError('EXPLICIT_GAP_ENDPOINT_CONFLICT')
                if gap['relation'] == 'SHARED_BOUNDARY':
                    if left != right or gap['geometry_spans']:
                        raise ValueError('SHARED_BUSINESS_CUT_REQUIRES_SAME_LOGICAL_BOUNDARY')
                    shared += 1
                elif gap['relation'] == 'POSITIVE_GAP':
                    if left == right:
                        raise ValueError('POSITIVE_GAP_REQUIRES_DISTINCT_LOGICAL_BOUNDARIES')
                    selected = chain(gap['geometry_spans'], left, right)
                    if any(domains[value] not in allowed for value in selected):
                        raise ValueError('GAP_OUTSIDE_ALLOWED_PARENT_SOURCE_DOMAINS')
                    complete.extend(selected)
                    positive += 1
                else:
                    raise ValueError('EXPLICIT_SHARED_OR_POSITIVE_GAP_RELATION_REQUIRED')
            if frame['kind'] == 'piecewise':
                canonical = [slot(ref, 'geometry_span') for ref in frame['canonical_span_order']]
                if len(canonical) != len(set(canonical)) or (canonical if row['direction'] == 1 else canonical[::-1]) != complete:
                    raise ValueError('WHOLE_SOURCE_SPAN_FRAME_AND_PARENT_DIRECTION_CONFLICT')
            parent_chains[identity] = complete
        if any(span_owners[key] != 1 for key in spans) or any(child_owners[key] != 1 for key in children):
            raise ValueError('ORPHAN_OR_MULTIPLE_INSTANCE_OWNER')
        for value in list(used_coordinates):
            current = dependencies[value]
            while current is not None:
                used_coordinates.add(current)
                current = dependencies[current]
        if set(coordinates) != used_coordinates:
            raise ValueError('UNUSED_DEFINING_COORDINATE')
        return dict(status='PASS_COMPLETE_TYPED_MIXED_GRAPH_STRUCTURE_AND_GLOBAL_CAPACITY', version=VERSION,
                    global_capacity=capacity, source_anchor_count=self.anchor_count,
                    source_global_span_capacity=self.span_capacity,
                    coordinates=coordinates, spans=spans, children=children, parents=parents,
                    coordinate_domains=coordinate_domains, span_domains=domains,
                    physical_roots=roots, span_ends=ends, internal_nodes=nodes, local_arcs=local_arcs,
                    arc_intervals=arc_intervals, arc_authorities=arc_authorities,
                    child_chains=child_chains, parent_chains=parent_chains,
                    shared_business_cut_count=shared, explicit_positive_gap_count=positive,
                    internal_node_reference_occurrences=internal_reference_count,
                    own_source_geometry_or_world_kind_verified=False, joint_GT_geometry_verified=False,
                    actual_weld_optimizer_updates=0, full_contract_pass=False)
