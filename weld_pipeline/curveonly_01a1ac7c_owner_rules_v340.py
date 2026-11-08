"""Preserve distinct material joints; certify coincident base-edge alternatives."""
import math
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c_transverse_rules_v338 as previous

NUMERIC, PASSAGE_MM = previous.NUMERIC, previous.PASSAGE_MM


def coincident_parameters(query, native):
    """A complete ordered native projection, using the existing .025mm CAD mesh bound."""
    p = np.asarray(native, float); q = np.asarray(query, float)
    s = previous.cad.arclength(p)
    _, nearest = cKDTree(p).query(q, k=min(4, len(p)), workers=1)
    nearest = np.asarray(nearest).reshape(len(q), -1)
    edges = np.clip(np.concatenate((nearest-1, nearest), axis=1), 0, len(p)-2)
    a = p[edges]; vector = p[edges+1]-a
    u = np.clip(((q[:, None]-a)*vector).sum(2)/np.maximum((vector*vector).sum(2), 1e-300), 0., 1.)
    distance = np.linalg.norm(q[:, None]-a-u[:, :, None]*vector, axis=2)
    pick = np.argmin(distance, axis=1)
    best = distance[np.arange(len(q)), pick]
    if best.max() > .025:
        return None
    selected = edges[np.arange(len(q)), pick]
    station = s[selected]+u[np.arange(len(q)), pick]*(s[selected+1]-s[selected])
    if not (np.all(np.diff(station) >= -.025) or np.all(np.diff(station) <= .025)):
        return None
    return station, float(best.max())


def apply(raw, features, packet, inventory, solids):
    rows = previous.prior.candidate_rows(features, packet)
    geometry = previous.Geometry(solids, packet, inventory)
    lengths = [r['length_mm'] for r in rows]
    eligible, audits = [], []
    for ci, (c, row) in enumerate(zip(packet['candidates'], rows)):
        if c.get('material', -1) < 0:
            continue
        grid = row['grid']
        if row['closed'] or not (c.get('STEP') and c.get('pose')):
            continue
        if np.min(grid[:, 0]) < math.cos(math.radians(3.)) or np.max(grid[:, 1]) > math.sin(math.radians(3.)):
            continue
        if np.min(grid[:, 21]) < 1. or np.max(np.expm1(grid[:, 22])) > 3.:
            continue
        states = [geometry.free_side(ci, q, packet) for q in np.linspace(0., lengths[ci], 5)[1:-1]]
        if any(q is None for q in states):
            continue
        blocks, review = previous.crossings(ci, packet, inventory, geometry)
        audits.append(dict(candidate=ci, blocks=blocks, crossing_review=review))
        if blocks:
            eligible.append((ci, blocks))
    footprints = {ci:previous.provenance.primitive_intervals(ci, 0., lengths[ci], packet, lengths)
                  for ci, _ in eligible}
    chosen = []
    for ci, blocks in sorted(eligible, key=lambda q:-lengths[q[0]]):
        if any(all(any(root == oldroot and a >= x-NUMERIC and b <= y+NUMERIC
                       for oldroot, x, y in footprints[old])
                   for root, a, b in footprints[ci]) for old, _ in chosen):
            continue
        chosen.append((ci, blocks))
    covered = [span for ci, _ in chosen for span in footprints[ci]]
    replacements = []
    for ci, blocks in chosen:
        for a, b in previous.complement(lengths[ci], blocks):
            proposal = previous.prior.codec.proposals(packet, [rows[ci]],
                [previous.prior.previous.intervals_to_codes(rows[ci], [dict(a=a, b=b, confidence=1.)])])[0]
            proposal['structural_rule'] = 'SINGLE_OWNER_TRANSVERSE_PASSAGE_V340'
            replacements.append(proposal)
    kept, removed, aliases = [], [], []
    for i, proposal in enumerate(raw):
        r = proposal['interval']; ci = proposal['candidate']
        spans = previous.provenance.primitive_intervals(ci, r['start_mm'], r['end_mm'], packet, lengths)
        overlap = any(root == cr and min(b, y)-max(a, x) > previous.provenance.MIN_OVERLAP_MM
                      for root, a, b in spans for cr, x, y in covered)
        alias = False
        if not overlap:
            owner = packet['candidates'][ci].get('material', -1)
            if owner >= 0 and packet['candidates'][ci].get('STEP'):
                for target, _ in chosen:
                    projection = coincident_parameters(proposal['points'], packet['candidates'][target]['points'])
                    if projection is None:
                        continue
                    station, error = projection
                    state = geometry.free_side(target, float(np.median(station)), packet)
                    if state is not None and owner in [state[3]]+state[4]:
                        alias = True
                        aliases.append(dict(raw_proposal=i, raw_candidate=ci, target_candidate=target,
                            projection_error_mm=error, native_order_verified=True,
                            verified_same_fillet_material_pair=[state[3]]+state[4],
                            target_range_mm=[float(station.min()), float(station.max())]))
                        break
        (removed if overlap or alias else kept).append(i)
    return [raw[i] for i in kept]+replacements, dict(branch='V340',
        candidate_audits=audits, chosen_candidates=[q[0] for q in chosen],
        replaced_raw_proposals=removed, new_rule_proposals=len(replacements),
        coincident_input_alias_certificates=aliases, different_owner_composites_not_merged=True,
        gap_width_from_finite_side_face_pair=True, passage_limit_mm=PASSAGE_MM,
        nominal_STEP_not_actual_geometry_acceptance=True)


prior, provenance = previous.prior, previous.provenance
