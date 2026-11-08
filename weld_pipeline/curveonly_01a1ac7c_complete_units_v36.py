"""Complete-curve units with independently witnessed native continuation.

The published/reference points are never refitted. Internal cuts and corners
remain in the member evidence. This is geometry evaluation, not certification
of weld-side identity, process cuts, direction or required attributes.
"""
import math
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_geometry_units_v26 as legacy

VERSION = 'CURVEONLY_COMPLETE_NATIVE_CONTINUATION_V36'
POLICY = dict(endpoint_identity_mm=.001, native_parameter_identity_mm=.001,
    native_interval_geometry_tolerance_mm=3., fallback_tangent_deflection_deg=5.,
    ambiguous_links_and_duplicates_retained=True, known_material_side_type_changes_block=True,
    confirmed_process_boundary_ends_block=True, all_points_corners_and_original_cuts_retained=True,
    true_gaps_preserved=True, no_truth_driven_prediction_grouping=True,
    no_process_side_direction_or_attribute_acceptance=True)


def link_evidence(a, b, ae, be):
    reason = legacy.compatible(a, b)
    if reason is None:
        return None
    if ae in a.get('_confirmed_process_cut_ends', []) or be in b.get('_confirmed_process_cut_ends', []):
        return None
    d = float(np.linalg.norm(np.asarray(a['points'])[0 if ae == 0 else -1] -
                             np.asarray(b['points'])[0 if be == 0 else -1]))
    if d > .001:
        return None
    na, nb = a.get('_native'), b.get('_native')
    if na is not None and nb is not None and a.get('_carrier') == b.get('_carrier'):
        delta = abs(na['endpoint_s_mm'][ae] - nb['endpoint_s_mm'][be])
        if na['closed'] and nb['closed']:
            length = na['length_mm']
            delta = min(delta, abs(length-delta))
        if delta > .001 or na['inward_sign'][ae] * nb['inward_sign'][be] >= 0:
            return None
        return dict(distance_mm=d, native_parameter_distance_mm=float(delta),
            reason='SAME_FINITE_INPUT_PATH_ORDERED_CONTINUATION',
            process_cut_status='ORIGINAL_BOUNDARY_RETAINED_NOT_CERTIFIED_FITTING_ONLY')
    # Without a legal finite-path witness, retain the earlier conservative rule.
    u, v = legacy.tangent(a, ae), legacy.tangent(b, be)
    if u is None or v is None or -u @ v < math.cos(math.pi/36):
        return None
    return dict(distance_mm=d, reason=reason+'_TANGENT_FALLBACK',
        tangent_deflection_deg=float(math.degrees(math.acos(np.clip(-u @ v, -1, 1)))),
        process_cut_status='ORIGINAL_BOUNDARY_RETAINED_NOT_CERTIFIED_FITTING_ONLY')


def units(curves):
    eligible = [i for i, r in enumerate(curves)
                if np.linalg.norm(np.asarray(r['points'])[0]-np.asarray(r['points'])[-1]) >= .1]
    choices, evidence = {}, {}
    for ai, i in enumerate(eligible):
        for j in eligible[ai+1:]:
            for ae in (0, 1):
                for be in (0, 1):
                    ev = link_evidence(curves[i], curves[j], ae, be)
                    if ev is None:
                        continue
                    a, b = (i, ae), (j, be)
                    choices.setdefault(a, []).append(b)
                    choices.setdefault(b, []).append(a)
                    evidence[tuple(sorted([a, b]))] = dict(ev, left=a, right=b)
    links = {a: b[0] for a, b in choices.items()
             if len(b) == 1 and len(choices.get(b[0], [])) == 1}
    out, visited = [], set()
    for seed in range(len(curves)):
        if seed in visited:
            continue
        component, todo = {seed}, [seed]
        while todo:
            i = todo.pop()
            for end in (0, 1):
                if (i, end) in links:
                    j, _ = links[i, end]
                    if j not in component:
                        component.add(j)
                        todo.append(j)
        free = [(i, end) for i in sorted(component) for end in (0, 1) if (i, end) not in links]
        i, end = free[0] if free else (min(component), 0)
        members, parts, joins = [], [], []
        while i not in {m['curve'] for m in members}:
            r = curves[i]
            p = np.asarray(r['points'], float)
            p = p if end == 0 else p[::-1]
            parts.append(p)
            members.append(dict(curve=i, reversed=bool(end),
                original_id=r.get('id', r.get('parent_id')),
                original_children=r.get('children'), original_members=r.get('members'),
                original_closed=r.get('closed', r.get('is_closed')),
                native_witness=r.get('_native'), source_candidate=r.get('_original_source_candidate'),
                confirmed_process_cut_ends=r.get('_confirmed_process_cut_ends', [])))
            if (i, 1-end) not in links:
                break
            nxt = links[i, 1-end]
            joins.append(evidence[tuple(sorted([(i, 1-end), nxt]))])
            i, end = nxt
        assert {m['curve'] for m in members} == component
        visited.update(component)
        p = np.vstack(parts)
        source_length = sum(float(cad.arclength(q)[-1]) for q in parts)
        bridges = sum(float(np.linalg.norm(a[-1]-b[0])) for a, b in zip(parts[:-1], parts[1:]))
        assert abs(float(cad.arclength(p)[-1])-source_length-bridges) < 1e-7*max(1., source_length)
        first = curves[members[0]['curve']]
        out.append(dict(points=p.tolist(), weld_type=first['weld_type'], weld_side=first.get('weld_side'),
            closed=bool(np.linalg.norm(p[0]-p[-1]) < .1), members=members, connections=joins,
            source_curve_count=len(members), source_length_mm=source_length, numerical_bridge_length_mm=bridges,
            circle=cad.circle_fit(p), _valid_export=all(curves[m['curve']].get('_valid_export', True) for m in members),
            process_boundaries_preserved_in_members=True, geometry_only_not_process_acceptance=True))
    assert sum(len(r['members']) for r in out) == len(curves)
    return out
