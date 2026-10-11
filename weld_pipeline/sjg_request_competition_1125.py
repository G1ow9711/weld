"""Competition between alternative outputs of one proven input request.

This is not geometric aliasing or material identification. Both finite domains
survive. Only the two legacy plane-pair constructors share a request, proven by
their frozen prefix, ordered supports and single unbound request interpretation.
Runtime occupied intervals (not entire possible domains) define competition.
"""
from collections import defaultdict
import json
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, digest

BRANCHES = ('global_observed_plane_pair', 'finite_convex_envelope')


def interval_projection(curve, q, origin, direction):
    """Outward interval evaluation, allowing one ULP per frozen float input.

    An uncertified microscopic overlap is not a capacity constraint. This does
    not snap endpoints, merge carriers, or assert that uncertain intervals equal.
    """
    def enclosure(x):
        return (float(np.nextafter(x, -np.inf)), float(np.nextafter(x, np.inf)))
    def add(a, b):
        return (float(np.nextafter(a[0]+b[0], -np.inf)), float(np.nextafter(a[1]+b[1], np.inf)))
    def multiply(a, b):
        values = [x*y for x in a for y in b]
        return (float(np.nextafter(min(values), -np.inf)), float(np.nextafter(max(values), np.inf)))
    total = (0., 0.)
    for start, tangent, offset, axis in zip(curve.ends[0], curve.direction, origin, direction):
        point = add(enclosure(float(start)), multiply(enclosure(float(q)), enclosure(float(tangent))))
        offset_range = enclosure(float(offset))
        relative = add(point, (-offset_range[1], -offset_range[0]))
        total = add(total, multiply(relative, enclosure(float(axis))))
    return total


def freeze_requests(key, graph):
    base = OUT/f'run132/source_{key}'
    current = OUT/f'run185/source_{key}'
    binding = read(base/'input_freeze.json')
    for name in ('profiles.json', 'surfaces.json', 'surface_support.npz'):
        assert digest(base/name) == binding['files'][name]
    original = read(base/'profiles.json')
    faces = read(base/'surfaces.json')
    assert original == graph.profiles[:len(original)]
    assert faces == read(current/'surfaces.json')[:len(faces)]
    grouped = defaultdict(lambda: defaultdict(list))
    for pid, profile in enumerate(original):
        branch = profile.get('proposal_branch')
        if branch not in BRANCHES or profile['representation']['kind'] != 'line':
            continue
        ordered = tuple(profile['face_indices'])
        if len(ordered) != 2:
            continue
        # These constructors instantiate exactly one unresolved interpretation
        # per ordered input pair. A later material/side hypothesis is distinct.
        if any(profile.get(k) is not None for k in ('material_pair', 'weld_type', 'weld_side', 'side_hypothesis')):
            continue
        grouped[ordered][branch].append(pid)
    requests = []
    verified_faces = set()
    with np.load(base/'surface_support.npz') as old_z, np.load(current/'surface_support.npz') as new_z:
        for ordered, branches in grouped.items():
            if set(branches) != set(BRANCHES) or any(len(branches[b]) != 1 for b in BRANCHES):
                continue
            a, b = [faces[i] for i in ordered]
            for fi, f in zip(ordered, (a, b)):
                if fi not in verified_faces:
                    assert np.array_equal(old_z[f['point_ids_key']], new_z[f['point_ids_key']])
                    verified_faces.add(fi)
            direction = np.cross(a['normal'], b['normal'])
            direction /= np.linalg.norm(direction)
            if direction[np.argmax(abs(direction))] < 0:
                direction = -direction
            normals = np.stack([a['normal'], b['normal']])
            origin = normals.T @ np.linalg.solve(normals @ normals.T, [a['offset'], b['offset']])
            members = [branches[b][0] for b in BRANCHES]
            requests.append(dict(members=members, ordered_faces=list(ordered),
                origin=origin.tolist(), direction=direction.tolist(),
                interpretation='single_unbound_interpretation_of_this_generator_request',
                proof='950 passes the same frozen ordered pair to 912 and 941; 941 reuses its support intervals',
                material_identity_proved=False, geometric_alias_proved=False))
    return dict(source=key, requests=requests, frozen_files=binding['files'],
                GT_reads=0, input_profiles_modified=False, branches=list(BRANCHES))


def interval_cliques(intervals, original_ports=()):
    """All positive-length occupied atoms; touching endpoints do not compete.

    Each interval is (lo, hi, node). Input ports and predicted endpoints both
    partition atoms; this permits disjoint short welds within one input span.
    """
    events = defaultdict(lambda: [[], []])
    for port in original_ports:
        events[float(port)]
    for lo, hi, node in intervals:
        if not hi > lo:
            continue
        events[float(lo)][1].append(int(node))
        events[float(hi)][0].append(int(node))
    active = defaultdict(int)
    coordinates = sorted(events)
    result = []
    for lo, hi in zip(coordinates[:-1], coordinates[1:]):
        ending, starting = events[lo]
        for node in ending:
            active[node] -= 1
            assert active[node] >= 0
        for node in starting:
            active[node] += 1
        occupied = sorted(node for node, count in active.items() if count)
        if len(occupied) > 1 and hi > lo:
            result.append(dict(lo=lo, hi=hi, nodes=occupied))
    return result


def competition_resources(nodes, graph, frozen, uncertainty_audit=None):
    lookup = {pid: i for i, request in enumerate(frozen['requests']) for pid in request['members']}
    intervals = defaultdict(list)
    ports = defaultdict(list)
    for index, node in enumerate(nodes):
        pid = graph.profile(node['fragment'])
        if pid not in lookup:
            continue
        group = lookup[pid]
        request = frozen['requests'][group]
        direction = np.asarray(request['direction'])
        origin = np.asarray(request['origin'])
        curve = graph.curve(pid)
        pair = curve.at(np.asarray([0., curve.length]))
        shift = float((pair[0]-origin) @ direction)
        slope = float((pair[1]-pair[0]) @ direction/curve.length)
        assert slope > 0, 'Request direction must be preserved by both constructors'
        resource = (group, int(node['side']))
        profile = graph.profiles[pid]
        for name in ('support_intervals_mm', 'unobserved_gaps_mm'):
            for lo, hi in profile.get(name, []):
                ports[resource].extend((shift+slope*lo, shift+slope*hi))
        ports[resource].extend((shift, shift+slope*curve.length))
        for lo, hi in profile.get('support_intervals_mm', []):
            left, right = max(float(lo), node['lo']), min(float(hi), node['hi'])
            if right > left:
                left_range = interval_projection(curve, left, origin, direction)
                right_range = interval_projection(curve, right, origin, direction)
                # Intersection of every enclosed realization is certified
                # occupied; the fringe remains unresolved, not merged away.
                inner_lo, inner_hi = left_range[1], right_range[0]
                if inner_hi > inner_lo:
                    intervals[resource].append((inner_lo, inner_hi, index))
                if uncertainty_audit is not None:
                    uncertainty_audit.append(dict(node=index, request=group, side=int(node['side']),
                        raw_interval=[shift+slope*left, shift+slope*right],
                        lower_endpoint_enclosure=list(left_range), upper_endpoint_enclosure=list(right_range),
                        certified_nonempty=inner_hi > inner_lo))
    resources = []
    # Multiple ordinal instances can occupy one original fragment. Compete on
    # actual overlapping intervals, never on the fragment key alone.
    local=defaultdict(list)
    for index,node in enumerate(nodes):
        fragment=int(node['fragment']);curve=graph.curve(graph.profile(fragment))
        domain=graph.spans[fragment,:2]
        pieces=[(float(node['lo']),float(node['hi']))]
        if node.get('closed'):pieces=[tuple(domain)]
        elif node.get('wrap'):pieces=[(float(domain[0]),float(node['lo'])),(float(node['hi']),float(domain[1]))]
        for left,right in pieces:
            if not right>left:continue
            if curve.kind=='line':
                lower=interval_projection(curve,left,curve.ends[0],curve.direction)[1]
                upper=interval_projection(curve,right,curve.ends[0],curve.direction)[0]
            else:
                allowance=64*np.finfo(float).eps*max(1.,curve.length)
                lower=left+allowance;upper=right-allowance
            if upper>lower:local[fragment,int(node['side'])].append((lower,upper,index))
    for (fragment,side),occupied in local.items():
        for atom in interval_cliques(occupied):
            resources.append(dict(request='local_fragment_'+str(fragment),side=side,**atom))
    for (group, side), occupied in intervals.items():
        for atom in interval_cliques(occupied, ports[group, side]):
            resources.append(dict(request=group, side=side, **atom))
    return resources


def controls():
    assert len(interval_cliques([(0., 10., 0), (0., 10., 1)])) == 1
    assert interval_cliques([(0., 10., 0), (10., 20., 1)]) == []
    assert interval_cliques([(0., 5., 0), (10., 15., 1)], [0., 20.]) == []
    a = interval_cliques([(0., 10., 0), (5., 15., 1)], [0., 20.])
    assert a == [dict(lo=5., hi=10., nodes=[0, 1])]
    from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
    profiles = [dict(representation=dict(kind='line', endpoints_mm=[[0., 0., 0.], [20., 0., 0.]]),
                     support_intervals_mm=[[0., 20.]], unobserved_gaps_mm=[]) for _ in range(4)]
    graph = JointSegmentGraph.build(profiles)
    frozen = dict(requests=[dict(members=[0, 1], origin=[0., 0., 0.], direction=[1., 0., 0.]),
                            dict(members=[2, 3], origin=[0., 0., 0.], direction=[1., 0., 0.])])
    nodes = [dict(fragment=0, side=0, lo=0., hi=10.), dict(fragment=1, side=0, lo=0., hi=10.)]
    assert competition_resources(nodes, graph, frozen)
    assert competition_resources([nodes[0], dict(nodes[1], side=1)], graph, frozen) == []
    assert competition_resources([nodes[0], dict(nodes[1], fragment=2)], graph, frozen) == []
    assert competition_resources([nodes[0], dict(nodes[1], lo=10., hi=20.)], graph, frozen) == []
    for q in (0., 5., 10., 20.):
        lo, hi = interval_projection(graph.curve(0), q, np.zeros(3), np.array([1., 0., 0.]))
        assert lo <= q <= hi
    return dict(duplicate_interpretations_compete=True, shared_endpoint_not_occupied_twice=True,
                disjoint_short_intervals_not_excluded=True, positive_gap_preserved=True,
                independent_sides_preserved=True, different_request_interpretations_preserved=True,
                exact_axis_projection_enclosed=True)
