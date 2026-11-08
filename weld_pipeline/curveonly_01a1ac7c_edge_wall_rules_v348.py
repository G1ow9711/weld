"""Recover a finite curved base-edge/through-wall fillet structural family.

Contact is only the pre-existing candidate observation. The new permission
requires a single base owner below the free welding side, the same foreign
upright body across the edge above AND below the base, continuous finite-body
ray support, and no confirmed transverse gap. No learned threshold is fitted.
"""
import math
import types
import numpy as np
from . import curveonly_01a1ac7c_preserved_gaps_v342 as previous
from . import curveonly_01a1ac7c_transverse_rules_v338 as finite
from . import curveonly_01a1ac7c_owner_rules_v340 as aliases

prior, provenance = previous.prior, previous.provenance
NUMERIC = finite.NUMERIC
LEGACY_CONTACT_OBSERVATION_MM = 3.  # Original cad contact_observed uses strict <3.


def seed_mask(x):
    """Locate a structural seed; complete permission uses exact finite solids."""
    a = ((x[:,37] > .5) & (x[:,49] > .5) & (x[:,39]+x[:,40] < .01)
         & (x[:,51]+x[:,52] < .01) & (x[:,87]+x[:,88] > .5)
         & (x[:,83]+x[:,84] > .5) & (x[:,85] < .5) & (x[:,89] < .5))
    b = ((x[:,73] > .5) & (x[:,85] > .5) & (x[:,75]+x[:,76] < .01)
         & (x[:,87]+x[:,88] < .01) & (x[:,51]+x[:,52] > .5)
         & (x[:,47]+x[:,48] > .5) & (x[:,49] < .5) & (x[:,53] < .5))
    return a | b


def local_frame(ci, q, packet, geometry):
    c = packet['candidates'][ci]
    points = np.asarray(c['points']); station = finite.cad.arclength(points)
    point, tangent = finite.point_tangent(points, station, q)
    normal = np.cross(tangent, geometry.base)
    normal /= max(np.linalg.norm(normal), 1e-300)
    owner = int(c['material'])
    states = []
    for sign in (-1., 1.):
        free = sign*normal
        if (geometry.inside(owner, point+free-geometry.base)
                and geometry.inside(owner, point+free*8.-geometry.base)
                and not geometry.inside(owner, point+free+geometry.base*3.)
                and not geometry.inside(owner, point+free*8.+geometry.base*3.)):
            states.append((point, tangent, free, owner))
    return states[0] if len(states) == 1 else None


def wall_gap(geometry, material, point, direction):
    if geometry.inside(material, point):
        return 0.
    return geometry.passage(material, point, direction)


def certify(ci, row, packet, geometry, inventory, selected):
    seed = float(row['length_mm']/2.)
    state = local_frame(ci, seed, packet, geometry)
    if state is None:
        return None, dict(candidate=ci, reason='NO_UNIQUE_BASE_FREE_SIDE')
    point, tangent, free, owner = state
    pairs = []
    for mi in range(len(geometry.solids)):
        if mi == owner:
            continue
        gaps = [wall_gap(geometry, mi, point+h*geometry.base, -free) for h in (-1., 3.)]
        if all(g is not None and g < LEGACY_CONTACT_OBSERVATION_MM for g in gaps):
            pairs.append((mi, gaps))
    if len(pairs) != 1:
        return None, dict(candidate=ci, reason='NO_UNIQUE_FINITE_UPRIGHT_PAIR', pairs=pairs)
    wall = pairs[0][0]
    material_faces = [f for f in inventory['faces'] if f['material'] in (owner, wall)]
    if not all(f.get('face_valid', False) for f in material_faces):
        return None, dict(candidate=ci, reason='UNKNOWN_FINITE_FACE_VALIDITY', pair=[owner, wall])
    # Every native endpoint and <=1mm arclength station is checked. A single
    # missed probe neither invents a gap nor recovers a partially supported arc.
    points = np.asarray(packet['candidates'][ci]['points'])
    native = finite.cad.arclength(points)
    samples = np.unique(np.r_[native, np.linspace(0., native[-1], int(np.ceil(native[-1]))+1)])
    maximum, minimum = 0., math.inf
    for q in samples:
        state = local_frame(ci, q, packet, geometry)
        if state is None:
            return None, dict(candidate=ci, reason='BASE_SIDE_NOT_CONTINUOUS', q_mm=float(q))
        point, tangent, free, _ = state
        # The opposite material is the SAME finite upright body at both heights.
        gaps = [wall_gap(geometry, wall, point+h*geometry.base, -free) for h in (-1., 3.)]
        if not all(g is not None and g < LEGACY_CONTACT_OBSERVATION_MM for g in gaps):
            return None, dict(candidate=ci, reason='FINITE_UPRIGHT_SUPPORT_NOT_CONTINUOUS',
                q_mm=float(q), observed_gap_mm=gaps, pair=[owner, wall])
        # The free side contains neither base above nor the upright body.
        if geometry.inside(wall, point+free+geometry.base*3.):
            return None, dict(candidate=ci, reason='NO_LOCAL_FILLET_FREE_SPACE', q_mm=float(q))
        maximum = max(maximum, *gaps); minimum = min(minimum, *gaps)
    def proxy_side(candidate, q, source):
        state = local_frame(candidate, q, source, geometry)
        if state is None:
            return None
        point, tangent, free, owner = state
        return point, tangent, free, wall, [owner]
    proxy = types.SimpleNamespace(**dict(geometry.__dict__, free_side=proxy_side,
        passage=geometry.passage))
    blocks, review = finite.crossings(ci, packet, inventory, proxy)
    if blocks:
        return None, dict(candidate=ci, reason='VERIFIED_TRANSVERSE_PROCESS_GAP', blocks=blocks)
    # Clearance >35 cannot erase another process gap already predicted at this
    # same finite plate pair. This is the V342 preservation condition.
    ranges = []
    for p in selected:
        q = aliases.coincident_parameters(p['points'], points)
        if q is not None:
            s, _ = q; ranges.append((float(s.min()), float(s.max())))
    ranges = finite.union(ranges)
    for event in review:
        if event['measured_two_board_passage_mm'] is not None and any(
                abs(left[1]-event['start_mm']) <= 3. and abs(right[0]-event['end_mm']) <= 3.
                for left, right in zip(ranges[:-1], ranges[1:])):
            return None, dict(candidate=ci, reason='EXISTING_MODEL_STRUCTURAL_GAP_PRESERVED', event=event)
    certificate = dict(candidate=ci, finite_material_pair=[owner, wall],
        full_native_interval_mm=[0., float(native[-1])], sampled_stations=len(samples),
        actual_body_probe_heights_mm=[-1., 3.], free_base_probe_offsets_mm=[1., 8.],
        minimum_wall_ray_gap_mm=minimum, maximum_wall_ray_gap_mm=maximum,
        local_side='BASE_INTERIOR_FREE_SIDE_OPPOSITE_FINITE_UPRIGHT',
        same_upright_body_above_and_below_whole_native_arc=True,
        STEP_nominal_not_actual_acceptance=True,
        contact_observation_alone_does_not_trigger=True, transverse_gaps=[], review=review)
    return certificate, None


def apply(raw, features, packet, inventory, solids):
    proposals, before = previous.apply(raw, features, packet, inventory, solids)
    rows = prior.candidate_rows(features, packet)
    seeds = []
    for ci, (c, row) in enumerate(zip(packet['candidates'], rows)):
        x = row['grid']
        if not (c.get('STEP') and c.get('pose') and c.get('material', -1) >= 0
                and c.get('circle') is not None and not row['closed']):
            continue
        if np.min(x[:,0]) < math.cos(math.radians(3.)) or np.max(x[:,1]) > math.sin(math.radians(3.)):
            continue
        if np.min(x[:,6]) < 1. or np.min(x[:,21]) < 1. or np.max(np.expm1(x[:,22])) >= LEGACY_CONTACT_OBSERVATION_MM:
            continue
        if np.any(seed_mask(x)):
            seeds.append(ci)
    if not seeds:
        return proposals, dict(before, branch='V348', base_rule=before,
            edge_wall_certificates=[], edge_wall_rejected=[], recovered_candidates=[])
    geometry = finite.Geometry(solids, packet, inventory)
    selected, _ = provenance.select(proposals, packet)
    certificates, rejected = [], []
    for ci in seeds:
        cert, failure = certify(ci, rows[ci], packet, geometry, inventory, selected)
        (certificates if cert else rejected).append(cert if cert else failure)
    lengths = [q['length_mm'] for q in rows]
    chosen, footprints = [], {}
    for cert in sorted(certificates, key=lambda q:-lengths[q['candidate']]):
        ci = cert['candidate']
        spans = provenance.primitive_intervals(ci, 0., lengths[ci], packet, lengths)
        if any(all(any(r == s and a >= x-NUMERIC and b <= y+NUMERIC
                       for s, x, y in footprints[old['candidate']]) for r, a, b in spans)
               for old in chosen):
            continue
        chosen.append(cert); footprints[ci] = spans
    kept, removed, alias_receipts = [], [], []
    covered = [s for ci in footprints for s in footprints[ci]]
    for index, proposal in enumerate(proposals):
        ci = proposal['candidate']; r = proposal['interval']
        spans = provenance.primitive_intervals(ci, r['start_mm'], r['end_mm'], packet, lengths)
        overlap = any(root == other and min(b, y)-max(a, x) > provenance.MIN_OVERLAP_MM
                      for root, a, b in spans for other, x, y in covered)
        for cert in chosen if not overlap else []:
            target = cert['candidate']; owner = packet['candidates'][ci].get('material', -1)
            if owner not in cert['finite_material_pair']:
                continue
            alias = aliases.coincident_parameters(proposal['points'], packet['candidates'][target]['points'])
            if alias is not None:
                overlap = True; alias_receipts.append(dict(proposal=index, source=ci, target=target,
                    error_mm=alias[1], finite_material_pair=cert['finite_material_pair'])); break
        if overlap:
            removed.append(index)
        else:
            kept.append(proposal)
    for cert in chosen:
        ci = cert['candidate']; row = rows[ci]
        q = prior.codec.proposals(packet, [row], [prior.previous.intervals_to_codes(row,
            [dict(a=0., b=lengths[ci], confidence=1.)])])[0]
        q['structural_rule'] = 'FINITE_CURVED_BASE_EDGE_THROUGH_UPRIGHT_FILLET_V348'
        kept.append(q)
    audit = dict(before, branch='V348', base_rule=before,
        edge_wall_certificates=chosen, edge_wall_rejected=rejected,
        recovered_candidates=[q['candidate'] for q in chosen],
        chosen_candidates=before.get('chosen_candidates', [])+[q['candidate'] for q in chosen],
        edge_wall_replaced_proposals=removed, edge_wall_aliases=alias_receipts,
        new_rule_proposals=before.get('new_rule_proposals', 0)+len(chosen),
        same_frozen_model_no_new_fit=True)
    return kept, audit
