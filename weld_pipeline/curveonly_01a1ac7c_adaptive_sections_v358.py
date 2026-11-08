"""Finite local sections with additive representation/numerical uncertainty.

This changes structural observation, not the model, confirmed process thresholds,
labels, or deployment inputs. Unknown/ambiguous finite sections give no recovery
permission. The old candidate contact admission remains explicitly frozen.
"""
import math
import types
import numpy as np
from OCP.gp import gp_Lin, gp_Dir
from . import curveonly_01a1ac7c_edge_wall_rules_v348 as old

finite, aliases = old.finite, old.aliases
prior, provenance, NUMERIC = old.prior, old.provenance, old.NUMERIC
CONTACT = old.LEGACY_CONTACT_OBSERVATION_MM


class Sections:
    def __init__(self, geometry):
        self.geometry = geometry
        self.bounds = [finite.cad.bbox(s) for s in geometry.solids]
        self.horizontal = {}
        for face in geometry.faces:
            if face['status'] != 'CANONICAL_PLANE':
                continue
            n = np.asarray(face['normal'], float)
            dot = float((n @ geometry.R.T) @ geometry.base)
            if abs(abs(dot)-1.) <= NUMERIC:
                self.horizontal.setdefault(face['material'], []).append((n, float(face['offset_mm']), dot))

    def intervals(self, material, point, direction):
        g = self.geometry
        origin = (point-g.t) @ g.R
        direction = np.asarray(direction) @ g.R
        bounds = self.bounds[material]
        center = (bounds[:3]+bounds[3:])/2.
        # Finite material extents determine search reach; no fixed 1e6 ray.
        reach = np.linalg.norm(origin-center)+np.linalg.norm(bounds[3:]-bounds[:3])+NUMERIC
        ray = g.intersectors[material]
        ray.Perform(gp_Lin(finite.cad.gp_Pnt(*map(float, origin)),
            gp_Dir(*map(float, direction))), -float(reach), float(reach))
        if not ray.IsDone():
            return []
        roots = sorted(float(ray.WParameter(i)) for i in range(1, ray.NbPnt()+1))
        unique = []
        for root in roots:
            if not unique or root-unique[-1] > NUMERIC:
                unique.append(root)
        intervals = []
        for a, b in zip(unique[:-1], unique[1:]):
            if b-a > NUMERIC and g.inside(material, point+(a+b)/2.*(direction @ g.R.T)):
                intervals.append((a, b))
        return intervals


def station(candidate, q):
    points = np.asarray(candidate['points'], float)
    length = float(finite.cad.arclength(points)[-1])
    circle = candidate['circle']
    n = np.asarray(circle['normal'], float); n /= np.linalg.norm(n)
    u = np.asarray(circle['start'], float); u -= (u@n)*n; u /= np.linalg.norm(u)
    angle = float(circle['sweep'])*q/max(length, 1e-300)
    radial = math.cos(angle)*u+math.sin(angle)*np.cross(n, u)
    point = np.asarray(circle['center'])+float(circle['radius'])*radial
    tangent = math.copysign(1., circle['sweep'])*np.cross(n, radial)
    return point, tangent


def frame(ci, q, packet, sections):
    g = sections.geometry; c = packet['candidates'][ci]
    point, tangent = station(c, q)
    normal = np.cross(tangent, g.base); normal /= max(np.linalg.norm(normal), 1e-300)
    owner = int(c['material'])
    uncertainty = NUMERIC + float(c['circle'].get('max_fit_error_mm', 0.))
    native = (point-g.t) @ g.R
    heights = [(offset-float(n@native))/dot for n, offset, dot in sections.horizontal.get(owner, [])]
    if not any(abs(h) <= uncertainty for h in heights):
        return None
    choices = []
    for h in sorted(set(h for h in heights if h < -uncertainty), reverse=True):
        for a, b in sections.intervals(owner, point+g.base*h/2., normal):
            if abs(a) <= uncertainty and b > uncertainty:
                sign, width = 1., b-a
            elif abs(b) <= uncertainty and a < -uncertainty:
                sign, width = -1., b-a
            else:
                continue
            # Interior section fractions are sampling coordinates, not welding
            # thresholds. They adapt to thickness, finite width and curvature.
            offset = min(width/2., abs(h)/2., float(c['circle']['radius'])/4.)
            if offset <= uncertainty:
                continue
            free = sign*normal
            vertical = sections.intervals(owner, point+free*offset, g.base)
            valid = [(lo, hi) for lo, hi in vertical if abs(hi) <= uncertainty and lo < -uncertainty]
            if len(valid) != 1:
                continue
            lo, hi = valid[0]
            choices.append(dict(point=point, tangent=tangent, free=free, owner=owner,
                base_bottom_mm=float(lo), base_thickness_mm=float(hi-lo),
                base_interior_probe_offset_mm=float(offset),
                local_finite_width_mm=float(width), numerical_uncertainty_mm=uncertainty))
        if choices:
            break
    if len(choices) != 1:
        return None
    return choices[0]


def upright(state, wall, sections):
    g = sections.geometry; p, free = state['point'], state['free']
    uncertainty = state['numerical_uncertainty_mm']
    intervals = [(a, b) for a, b in sections.intervals(wall, p, -free)
                 if b > uncertainty and a < CONTACT]
    if not intervals:
        return None
    a, b = min(intervals, key=lambda v:max(0., v[0]))
    gap = max(0., a)
    if gap >= CONTACT:
        return None
    inside_point = p-free*(a+b)/2.
    vertical = [(lo, hi) for lo, hi in sections.intervals(wall, inside_point, g.base)
                if lo < -uncertainty and hi > uncertainty]
    if len(vertical) != 1:
        return None
    lo, hi = vertical[0]
    scale = min(state['base_thickness_mm'], b-a)
    below = max(lo, state['base_bottom_mm'])/2.
    above = min(hi, scale)/2.
    if not (below < -uncertainty and above > uncertainty):
        return None
    gaps = [old.wall_gap(g, wall, p+h*g.base, -free) for h in (below, above)]
    if not all(v is not None and v < CONTACT for v in gaps):
        return None
    free_point = p+free*state['base_interior_probe_offset_mm']+g.base*above
    if g.inside(wall, free_point) or g.inside(state['owner'], free_point):
        return None
    return dict(finite_wall=wall, observed_gap_mm=gap, gaps_mm=list(map(float, gaps)),
        wall_section_width_mm=float(b-a), wall_vertical_interval_mm=[float(lo),float(hi)],
        probe_heights_mm=[float(below),float(above)])


def certify(ci, row, packet, geometry, inventory, selected):
    sections = getattr(geometry, '_adaptive_sections_v358', None)
    if sections is None:
        sections = Sections(geometry); geometry._adaptive_sections_v358 = sections
    seed = frame(ci, row['length_mm']/2., packet, sections)
    if seed is None:
        return None, dict(candidate=ci, reason='NO_UNIQUE_LOCAL_FINITE_BASE_SECTION')
    pairs = [q for mi in range(len(geometry.solids)) if mi != seed['owner']
             for q in [upright(seed, mi, sections)] if q is not None]
    if len(pairs) != 1:
        return None, dict(candidate=ci, reason='NO_UNIQUE_FINITE_THROUGH_WALL_SECTION', pairs=pairs)
    wall = pairs[0]['finite_wall']; owner = seed['owner']
    faces = [f for f in inventory['faces'] if f['material'] in (owner,wall)]
    if not all(f.get('face_valid', False) for f in faces):
        return None, dict(candidate=ci, reason='UNKNOWN_FINITE_FACE_VALIDITY')
    points = np.asarray(packet['candidates'][ci]['points']); native = finite.cad.arclength(points)
    samples = np.unique(np.r_[native,np.linspace(0.,native[-1],int(np.ceil(native[-1]))+1)])
    extents = {name:[math.inf,-math.inf] for name in ('base_thickness_mm','base_interior_probe_offset_mm',
        'local_finite_width_mm','wall_section_width_mm','below_probe_mm','above_probe_mm','wall_gap_mm')}
    for q in samples:
        state = frame(ci, q, packet, sections)
        evidence = upright(state, wall, sections) if state is not None else None
        if evidence is None:
            return None, dict(candidate=ci, reason='LOCAL_FINITE_PAIR_SUPPORT_NOT_CONTINUOUS',q_mm=float(q),pair=[owner,wall])
        values = dict(base_thickness_mm=state['base_thickness_mm'],
            base_interior_probe_offset_mm=state['base_interior_probe_offset_mm'],
            local_finite_width_mm=state['local_finite_width_mm'],wall_section_width_mm=evidence['wall_section_width_mm'],
            below_probe_mm=evidence['probe_heights_mm'][0],above_probe_mm=evidence['probe_heights_mm'][1],
            wall_gap_mm=max(evidence['gaps_mm']))
        for name,value in values.items():
            extents[name][0]=min(extents[name][0],value);extents[name][1]=max(extents[name][1],value)
    def proxy_side(candidate,q,source):
        s=frame(candidate,q,source,sections)
        return None if s is None else (s['point'],s['tangent'],s['free'],wall,[s['owner']])
    proxy=types.SimpleNamespace(**dict(geometry.__dict__,free_side=proxy_side,passage=geometry.passage))
    blocks,review=finite.crossings(ci,packet,inventory,proxy)
    if blocks:
        return None,dict(candidate=ci,reason='VERIFIED_TRANSVERSE_PROCESS_GAP',blocks=blocks)
    ranges=[]
    for p in selected:
        q=aliases.coincident_parameters(p['points'],points)
        if q is not None:
            s,_=q;ranges.append((float(s.min()),float(s.max())))
    ranges=finite.union(ranges)
    for event in review:
        if event['measured_two_board_passage_mm'] is not None and any(
                abs(left[1]-event['start_mm'])<=3. and abs(right[0]-event['end_mm'])<=3.
                for left,right in zip(ranges[:-1],ranges[1:])):
            return None,dict(candidate=ci,reason='EXISTING_MODEL_STRUCTURAL_GAP_PRESERVED',event=event)
    return dict(candidate=ci,finite_material_pair=[owner,wall],full_native_interval_mm=[0.,float(native[-1])],
        sampled_stations=len(samples),probe_method='LOCAL_FINITE_MATERIAL_SECTIONS',
        adaptive_geometry_ranges_mm=extents,same_upright_body_above_and_below_whole_native_arc=True,
        contact_observation_alone_does_not_trigger=True,
        remaining_frozen_contact_candidate_admission_mm=CONTACT,
        STEP_nominal_not_actual_acceptance=True,review=review,transverse_gaps=[]),None


def apply(raw,features,packet,inventory,solids):
    # Fixed probe features no longer decide whether this exact kernel is tried.
    fn=types.FunctionType(old.apply.__code__,dict(old.__dict__,certify=certify,
        seed_mask=lambda x:np.ones(len(x),dtype=bool)))
    proposals,audit=fn(raw,features,packet,inventory,solids)
    audit['branch']='V358_ADAPTIVE_SECTIONS'
    audit['fixed_1_8_and_minus1_plus3_seed_probes_removed']=True
    for p in proposals:
        if p.get('structural_rule')=='FINITE_CURVED_BASE_EDGE_THROUGH_UPRIGHT_FILLET_V348':
            p['structural_rule']='FINITE_CURVED_BASE_EDGE_ADAPTIVE_LOCAL_SECTIONS_V358'
    return proposals,audit
