"""Bind real finite-role mixed graphs to immutable original source domains.

Recorded rational positions remain exact. A physical reference includes an
explicit position in its own source domain, so repeated carrier occurrences
cannot be chosen by proximity. Unsupported analytic joins remain unproven.
"""
from fractions import Fraction as F

from .decoder_graph_structure_254 import slot
from .decoder_output_slots_247 import local_region_triangles
from .concrete_curve_certificate_250 import UnprovenCurve, _sqrt_fraction_bounds
from .raw_triangle_world_charts_258 import RawTriangleChart, exact, sub, dot, cross
from .mixed_world_joint_binding_259 import (
    CarrierOccurrence, OccurrenceSlice, FrozenMixedSourceProvider,
    prove_explicit_world_join, certify_streamed_world_line,
    certify_complete_carrier_parent,
)
from .mixed_parent_graph_260 import MixedGraphValidator, rational, VERSION
from .mixed_arc_restrictions_260 import RestrictedSourceArc, matching_points, certify_arc_chain
from .mixed_surface_curve_binding_253 import _local_endpoint_proven
from .mixed_curve_domain_253 import certify_curve_domain


def _affine_parameters(point, context):
    origin = tuple(exact(float(value)) for value in context['origin'])
    columns = [tuple(exact(float(value)) for value in row) for row in context['basis']]
    target = sub(point, origin)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        determinant = columns[0][a] * columns[1][b] - columns[1][a] * columns[0][b]
        if not determinant:
            continue
        u = (target[a] * columns[1][b] - target[b] * columns[1][a]) / determinant
        v = (columns[0][a] * target[b] - columns[0][b] * target[a]) / determinant
        if tuple(origin[k] + u * columns[0][k] + v * columns[1][k] for k in range(3)) != point:
            raise UnprovenCurve('Exact world point is outside the declared legacy affine plane')
        return u, v
    raise UnprovenCurve('Rank-two legacy affine source frame is required')


def _triangle_parameters(chart, point):
    uv = chart.local_mm(point)
    if chart.point_from_local_mm(uv) != point:
        raise UnprovenCurve('Exact world point is outside the declared original raw triangle plane')
    a, b = chart.axes
    first, second = chart.deltas
    det = first[a] * second[b] - first[b] * second[a]
    return ((uv[0] * second[b] - uv[1] * second[a]) / det,
            (first[a] * uv[1] - first[b] * uv[0]) / det)


def _line_interval_in_triangle(first, last, vertices):
    """Exact closed parameter interval inside a rank-two recorded UV triangle."""
    def oriented(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    orientation = oriented(vertices[0], vertices[1], vertices[2])
    if not orientation:
        return None
    sign = 1 if orientation > 0 else -1
    lo, hi = F(0), F(1)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        value = sign * oriented(a, b, first)
        slope = sign * oriented(a, b, last) - value
        if not slope:
            if value < 0:
                return None
        elif slope > 0:
            lo = max(lo, -value / slope)
        else:
            hi = min(hi, -value / slope)
        if lo > hi:
            return None
    return lo, hi


def _legacy_segment_covered(context, first, last):
    """Cover a whole rational segment by original triangles, without sampling."""
    domain = context['domain']
    if first == last:
        if not any(domain.original_triangle_member(i, first) for i in range(len(context['triangles']))):
            raise UnprovenCurve('Degenerate segment has no original finite-domain support')
        return
    intervals = []
    for indices in context['triangles']:
        vertices = [tuple(exact(float(value)) for value in context['uv'][int(index)]) for index in indices]
        interval = _line_interval_in_triangle(first, last, vertices)
        if interval is not None:
            intervals.append(interval)
            continue
        # Preserve rank-one source triangles as exact line supports.
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            delta = (last[0] - first[0], last[1] - first[1])
            axis = 0 if delta[0] else 1
            if ((a[0] - first[0]) * delta[1] != (a[1] - first[1]) * delta[0] or
                    (b[0] - first[0]) * delta[1] != (b[1] - first[1]) * delta[0]):
                continue
            x, y = (a[axis] - first[axis]) / delta[axis], (b[axis] - first[axis]) / delta[axis]
            lo, hi = max(F(0), min(x, y)), min(F(1), max(x, y))
            if lo <= hi:
                intervals.append((lo, hi))
    covered = F(0)
    for lo, hi in sorted(intervals):
        if lo > covered:
            break
        covered = max(covered, hi)
        if covered >= 1:
            return
    raise UnprovenCurve('Whole rational line segment crosses an unsupported source-domain interval')


def _segments_metrics(segments, *, export_required=True):
    count, positive, previous, lower, upper = 0, 0, None, F(0), F(0)
    first_point = last_point = None
    for segment in segments:
        a, b = segment['start_exact'], segment['end_exact']
        if previous is not None and previous != a:
            raise UnprovenCurve('Complete world span chain is discontinuous')
        if first_point is None:
            first_point = a
        previous = last_point = b
        length_squared = dot(sub(b, a), sub(b, a))
        lo, hi = _sqrt_fraction_bounds(length_squared)
        lower, upper = lower + exact(float(lo)), upper + exact(float(hi))
        if length_squared:
            positive += 1
            if export_required and tuple(map(float, a)) == tuple(map(float, b)):
                raise UnprovenCurve('Positive source world edge collapses on numerical export')
        count += 1
    if not count or not positive or lower <= 0:
        raise UnprovenCurve('Positive complete world curve is required')
    return dict(segments=count, positive_segments=positive, physical_lower_mm=lower,
                physical_upper_mm=upper, start_exact=first_point, end_exact=last_point,
                complete_continuity_verified=True, positive_export_verified=export_required)


class FrozenMixedGraphSourceBinder:
    def __init__(self, slots, carriers, raw_xyz_mm):
        self.validator = MixedGraphValidator(slots, carriers)
        self.source = FrozenMixedSourceProvider(slots, carriers, raw_xyz_mm)

    def _context(self, face):
        if face not in self.source.surface_cache:
            self.source.surface_cache[face] = self.source.compiler.face(face)
            while len(self.source.surface_cache) > self.source.maximum_cached_domains:
                self.source.surface_cache.popitem(last=False)
        self.source.surface_cache.move_to_end(face)
        return self.source.surface_cache[face]

    def _chart(self, domain):
        ids = self.source.slots.execute('SELECT raw_a,raw_b,raw_c FROM triangles WHERE face_id=? AND triangle_ordinal=?', domain[1:]).fetchone()
        return RawTriangleChart(self.source.raw[list(ids)], ids, face_id=domain[1], triangle_ordinal=domain[2])

    def _position(self, identity, domain, position):
        anchor = identity[0]
        if domain[0] == 'RAW_TRIANGLE':
            result = self.source.raw_position(anchor, domain[1], domain[2], *map(rational, position['barycentric']))
        elif domain[0] == 'RAW_CARRIER':
            result = self.source.carrier_position(anchor, domain[1], CarrierOccurrence.from_record(position['occurrence']))
        else:
            result = self.source.affine_position(anchor, domain[1], tuple(map(rational, position['uv_mm'])))
        result['domain'] = domain
        result['declared_position'] = position
        result['world_expression'] = ('POINT', result['point_exact'])
        result['world_bounds'] = (result['point_exact'], result['point_exact'])
        return result

    def _arc_position(self, identity, domain, endpoint, local_arcs, world_arcs):
        master = slot(endpoint['span'], 'geometry_span')
        index = endpoint['endpoint_index']
        local, world = local_arcs[master], world_arcs[master]
        local_endpoint = local.endpoint(index)
        context = self._context(domain[1])
        regions = local_region_triangles(self.source.slots, identity[0], domain[1])
        if not regions or not _local_endpoint_proven(context, regions, local_endpoint['world_bounds']):
            raise UnprovenCurve('Analytic endpoint has no own original local permission')
        result = world.endpoint(index)
        result.update(domain=domain, local_bounds=local_endpoint['world_bounds'],
                      own_local_permission_verified=True,root_expression=result['world_expression'])
        return result

    def _locate(self, identity, domain, coordinate):
        point = coordinate['point_exact']
        if point is None:
            if domain[0]!='LEGACY_SURFACE' or coordinate['domain']!=domain or 'local_bounds' not in coordinate:
                raise UnprovenCurve('Symbolic cross-domain point needs an explicit exact source expression')
            regions=local_region_triangles(self.source.slots,identity[0],domain[1])
            if not regions or not _local_endpoint_proven(self._context(domain[1]),regions,coordinate['local_bounds']):
                raise UnprovenCurve('Located role has no own original local permission for its symbolic endpoint')
            return {**coordinate,'own_local_permission_verified':True}
        if domain[0] == 'RAW_TRIANGLE':
            uv = _triangle_parameters(self._chart(domain), point)
            return self._position(identity, domain, {'barycentric': list(uv)})
        if domain[0] == 'LEGACY_SURFACE':
            uv = _affine_parameters(point, self._context(domain[1]))
            return self._position(identity, domain, {'uv_mm': list(uv)})
        if coordinate['domain'] == domain and 'occurrence' in coordinate:
            return self.source.carrier_position(identity[0], domain[1], coordinate['occurrence'])
        raise UnprovenCurve('Cross-domain carrier binding needs an explicit original occurrence')

    @staticmethod
    def _iter_chain(identities, bound_spans):
        for identity in identities:
            span = bound_spans[identity]
            if 'arc' in span:
                raise UnprovenCurve('A positive analytic arc is not a straight line or stored polyline')
            if 'walk' in span:
                yield from span['walk'].segments()
            else:
                yield from span['segments']

    def _chain_metric(self, identities, bound_spans):
        if not any('arc' in bound_spans[key] for key in identities):
            return _segments_metrics(self._iter_chain(identities,bound_spans))
        lower=upper=F(0)
        previous=None;segments=0
        for key in identities:
            span=bound_spans[key]
            if previous is not None and not matching_points(previous,span['start_record']):
                raise UnprovenCurve('Complete mixed analytic world span chain is discontinuous')
            previous=span['end_record']
            metric=span['metric']
            lower+=metric.get('physical_lower_mm',metric.get('physical_length_lower_mm'))
            upper+=metric.get('physical_upper_mm',metric.get('physical_length_upper_mm'))
            segments+=metric.get('segments',metric.get('original_edge_occurrences',span.get('evaluation_point_count',2)-1))
        if lower<=0:
            raise UnprovenCurve('Complete mixed analytic chain has no positive physical length')
        return dict(segments=segments,physical_lower_mm=lower,physical_upper_mm=upper,
                    start_exact=bound_spans[identities[0]]['start_record']['point_exact'],
                    end_exact=bound_spans[identities[-1]]['end_record']['point_exact'],
                    complete_continuity_verified=True,positive_export_verified=True,analytic_arc=True)

    def bind(self, graph):
        structural = self.validator.validate(graph)
        coordinates, roots, joins = {}, structural['physical_roots'], []
        local_arcs,world_arcs={},{}
        for key,prototype in structural['local_arcs'].items():
            domain=structural['span_domains'][key]
            interval=structural['arc_intervals'][key]
            local_arcs[key]=RestrictedSourceArc(prototype,interval)
            world_arcs[key]=RestrictedSourceArc(prototype,interval,context=self._context(domain[1]))
            world_arcs[key].circle_proof()
        for key,master in structural['arc_authorities'].items():
            if world_arcs[key].signature()!=world_arcs[master].signature():
                raise UnprovenCurve('Arc restriction changes its declared authoritative world circle or phase frame')
        for identity in structural['coordinates']:
            chain, current = [], identity
            while current not in coordinates:
                chain.append(current)
                row = structural['coordinates'][current]
                if 'physical_ref' not in row:
                    break
                current = slot(row['physical_ref'])
            for key in reversed(chain):
                row, domain = structural['coordinates'][key], structural['coordinate_domains'][key]
                endpoint=row.get('curve_endpoint',row.get('local_position',{}).get('curve_endpoint'))
                coordinates[key]=(self._arc_position(key,domain,endpoint,local_arcs,world_arcs) if endpoint is not None
                                  else self._position(key,domain,row.get('position',row.get('local_position'))))
                if 'physical_ref' in row:
                    target = coordinates[slot(row['physical_ref'])]
                    if not matching_points(coordinates[key],target):
                        raise UnprovenCurve('Explicit physical reference joins different exact world expressions')
                    joins.append(dict(status='PASS_EXPLICIT_SOURCE_WORLD_POINT_OR_SYMBOLIC_EXPRESSION_JOIN',
                                      both_own_local_permissions_verified=True,source_occurrences_remain_separate=True))
        spans = {}
        for identity, row in structural['spans'].items():
            domain, ends = structural['span_domains'][identity], structural['span_ends'][identity]
            selected = [ends[0], *structural['internal_nodes'][identity], ends[1]]
            points = [coordinates[key]['point_exact'] for key in selected]
            self._locate(identity, domain, coordinates[ends[0]])  # Own defining span permission.
            if row['kind'] == 'arc':
                local,world=local_arcs[identity],world_arcs[identity]
                if not all(matching_points(coordinates[key],world.endpoint(index)) for index,key in enumerate(ends)):
                    raise UnprovenCurve('Declared arc parameters are not coupled to both original symbolic boundaries')
                own=certify_curve_domain(local,self._context(domain[1])['uv'],self._context(domain[1])['triangles'],
                     domain_id=self._context(domain[1])['domain'].domain_id,compiled_domain=self._context(domain[1])['domain'])
                if not own['status'].startswith('PASS'):
                    raise UnprovenCurve('Whole selected arc restriction lacks finite source-domain support: '+own['status'])
                phase=certify_arc_chain([world],authority_ref=structural['arc_authorities'][identity],
                         maximum_turns=self.source.maximum_turns,is_closed=row['is_closed'])
                samples=world.samples()
                metric=world.metric();metric['positive_export_verified']=True
                spans[identity]=dict(arc=world,local_arc=local,metric=metric,source_domain=domain,
                     own_domain_certificate=own,phase_proof=phase,evaluation_point_count=samples['point_count'],
                     start_exact=points[0],end_exact=points[-1])
            elif row['kind'] == 'carrier_slice':
                located = [self._locate(key, domain, coordinates[key]) for key in ends]
                params = row['parameters']
                walk = OccurrenceSlice(self.source.carrier(domain[1]), located[0]['occurrence'], located[1]['occurrence'],
                                       params['direction'], params['extra_turns'], maximum_turns=self.source.maximum_turns)
                metric = walk.metrics(check_export=True)
                spans[identity] = dict(walk=walk, metric=metric, source_domain=domain,
                                       start_exact=points[0], end_exact=points[-1])
            else:
                if domain[0] == 'RAW_TRIANGLE':
                    chart = self._chart(domain)
                    for point in points:
                        _triangle_parameters(chart, point)
                    # A single original triangle is convex, including its closed boundary.
                else:
                    context = self._context(domain[1])
                    local = [_affine_parameters(point, context) for point in points]
                    for first, last in zip(local, local[1:]):
                        _legacy_segment_covered(context, first, last)
                segments = [dict(start_exact=a, end_exact=b) for a, b in zip(points, points[1:])]
                metric = _segments_metrics(iter(segments))
                if row['kind'] == 'line':
                    certify_streamed_world_line(iter(segments))
                spans[identity] = dict(segments=segments, metric=metric, source_domain=domain,
                                       start_exact=points[0], end_exact=points[-1])
            spans[identity].update(start_record=coordinates[ends[0]],end_record=coordinates[ends[1]])
            if row['is_closed'] != matching_points(coordinates[ends[0]],coordinates[ends[1]]):
                raise UnprovenCurve('Declared span closure disagrees with its exact world endpoints')
        children = {}
        for identity, row in structural['children'].items():
            chain = structural['child_chains'][identity]
            metric = self._chain_metric(chain, spans)
            if row['world_kind'] == 'line':
                kind_proof = certify_streamed_world_line(self._iter_chain(chain, spans))
            elif row['world_kind'] == 'arc':
                if any('arc' not in spans[key] for key in chain) or len({structural['arc_authorities'][key] for key in chain})!=1:
                    raise UnprovenCurve('Whole-child arc requires only restrictions of one existing circle authority')
                kind_proof=certify_arc_chain([spans[key]['arc'] for key in chain],authority_ref=structural['arc_authorities'][chain[0]],
                           maximum_turns=self.source.maximum_turns)
            elif row['world_kind']=='compound':
                kind_proof=dict(status='PASS_COMPLETE_ORDERED_EXPLICIT_COMPOUND_WORLD_CURVE',
                                individual_span_kinds=[structural['spans'][key]['kind'] for key in chain],
                                single_world_circle_verified=False)
            else:
                if any('arc' in spans[key] for key in chain):
                    raise UnprovenCurve('A stored polyline cannot implicitly include analytic arcs')
                kind_proof = dict(status='PASS_COMPLETE_RECORDED_WORLD_POLYLINE', analytic_arc_verified=False)
            self._locate(identity, structural['span_domains'][chain[0]], coordinates[slot(row['start_boundary'])])
            children[identity] = dict(metric=metric, whole_world_kind_proof=kind_proof,
                                      own_local_permission_verified=True)
        parents = {}
        for identity, row in structural['parents'].items():
            chain = structural['parent_chains'][identity]
            metric = self._chain_metric(chain, spans)
            self._locate(identity, structural['span_domains'][chain[0]], coordinates[slot(row['start_boundary'])])
            gaps = []
            for gap in row['gaps']:
                if gap['relation'] != 'POSITIVE_GAP':
                    continue
                chosen = [slot(ref) for ref in gap['geometry_spans']]
                gap_metric = self._chain_metric(chosen, spans)
                gaps.append(dict(status='PASS_EXPLICIT_POSITIVE_COMPLETE_WORLD_GAP', metric=gap_metric))
            frame = row['frame']
            if frame['kind'] == 'line':
                proof = certify_streamed_world_line(self._iter_chain(chain, spans))
                axis = tuple(map(rational, frame['axis_world']))
                delta = sub(metric['end_exact'], metric['start_exact'])
                if any(cross(axis, delta)) or row['direction'] * dot(axis, delta) <= 0:
                    raise UnprovenCurve('Whole line parent disagrees with its explicit frame direction')
            elif frame['kind'] == 'carrier':
                if any('walk' not in spans[key] for key in chain):
                    raise UnprovenCurve('A complete carrier parent needs original occurrence slices for every span and gap')
                domain = self.validator.domain(frame['domain'])
                if any(spans[key]['source_domain'] != domain for key in chain):
                    raise UnprovenCurve('Whole parent traversal uses a different original carrier')
                proof = certify_complete_carrier_parent([spans[key]['walk'] for key in chain],
                         anchor_count=self.source.anchor_count, direction=row['direction'],
                         is_closed=row['is_closed'], turn_count=row['turn_count'])
            elif frame['kind']=='arc':
                master=slot(frame['authority_span'])
                if any('arc' not in spans[key] or structural['arc_authorities'][key]!=master for key in chain):
                    raise UnprovenCurve('Whole arc parent mixes authorities or includes a non-arc interval')
                proof=certify_arc_chain([spans[key]['arc'] for key in chain],authority_ref=master,
                       maximum_turns=self.source.maximum_turns,is_closed=row['is_closed'],turn_count=row['turn_count'],direction=row['direction'])
            else:
                if row['turn_count']!=(1 if row['is_closed'] else 0):
                    raise UnprovenCurve('Repeated piecewise periods require an explicit repeated-source frame')
                proof=dict(status='PASS_COMPLETE_EXPLICIT_SINGLE_PIECEWISE_FRAME_TRAVERSAL',
                           direction=row['direction'],canonical_span_count=len(frame['canonical_span_order']),
                           all_original_spans_and_positive_gaps_included=True,is_closed=row['is_closed'],turn_count=row['turn_count'])
            if row['is_closed'] != matching_points(spans[chain[0]]['start_record'],spans[chain[-1]]['end_record']):
                raise UnprovenCurve('Whole parent closure disagrees with its complete world traversal')
            parents[identity] = dict(metric=metric, frame_proof=proof, gap_proofs=gaps,
                                      own_local_permission_verified=True)
        return dict(status='PASS_BOUND_GRAPH_ORIGINAL_SOURCE_GEOMETRY_AND_WHOLE_CHILD_PARENT_TRAVERSAL',
                    version=VERSION, structural=structural, coordinates=coordinates, spans=spans,
                    children=children, parents=parents, explicit_world_joins=joins,
                    all_declared_roles_have_own_local_permission=True,
                    whole_child_world_kind_verified=True, whole_parent_traversal_verified=True,
                    all_general_analytic_and_piecewise_graphs_supported=False,
                    joint_GT_geometry_verified=False, actual_optimizer_updates=0, full_contract_pass=False)
