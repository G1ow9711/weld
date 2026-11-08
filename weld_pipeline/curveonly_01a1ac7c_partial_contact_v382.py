"""Observe finite local contact domains on model-anchored native curves.

Carrier roots are partition hypotheses. Actual trimmed face witnesses and
connected finite material sections establish a local joint. Transverse passage
is a separate process decision. Missing/ambiguous observations remain UNKNOWN.
"""
import math
import types
import numpy as np
from . import curveonly_01a1ac7c_step_joint_rules_v373 as step

observation=step.observation
finite,prior,provenance,aliases=step.finite,step.prior,step.provenance,step.aliases
NUMERIC,CONTACT=step.NUMERIC,step.CONTACT


class Context(step.Context):
    def __init__(self,solids,packet,inventory):
        super().__init__(solids,packet,inventory)
        self.face_shapes=[];self.face_boxes=[];self.face_rows=[]
        for mi,solid in enumerate(solids):
            for shape,row in zip(finite.cad.items(solid,finite.cad.TopAbs_FACE),self.faces[mi]):
                self.face_shapes.append(shape);self.face_boxes.append(finite.cad.bbox(shape));self.face_rows.append(row)
        self.face_boxes=np.asarray(self.face_boxes)


def normal(row,native_point,geometry):
    if row['status']=='CANONICAL_PLANE':
        n=np.asarray(row['normal'],float)
    elif row['status']=='CANONICAL_CYLINDER':
        axis=np.asarray(row['axis'],float);axis/=np.linalg.norm(axis)
        delta=native_point-np.asarray(row['axis_point_mm']);n=delta-(delta@axis)*axis
        if np.linalg.norm(n)<=NUMERIC:return None
        n/=np.linalg.norm(n)
    else:return None
    return n@geometry.R.T


def carrier_coherence(state,row,context,sec):
    ci=state['candidate'];candidate=context.packet['candidates'][ci]
    points=np.asarray(candidate['points']);station_s=finite.cad.arclength(points)
    radius=float(candidate['circle']['radius']);eps=state['numerical_uncertainty_mm']
    # Arc length needed to resolve one source-error sagitta. This local span
    # depends on curvature and original representation uncertainty, not labels.
    span=min(float(station_s[-1]),2.*radius*math.acos(1.-min(eps/radius,1.)))
    q=state['query_arclength_mm'];lo=max(0.,q-span/2.);hi=min(float(station_s[-1]),lo+span);lo=max(0.,hi-span)
    world=np.asarray([[np.interp(t,station_s,points[:,k]) for k in range(3)] for t in (lo,q,hi)])
    native=(world-sec.geometry.t)@sec.geometry.R
    if row['status']=='CANONICAL_PLANE':
        values=native@np.asarray(row['normal'])-row['offset_mm']
    elif row['status']=='CANONICAL_CYLINDER':
        axis=np.asarray(row['axis'],float);axis/=np.linalg.norm(axis)
        delta=native-np.asarray(row['axis_point_mm'])
        values=np.linalg.norm(delta-np.outer(delta@axis,axis),axis=1)-row['radius_mm']
    else:return None
    variation=float(np.ptp(values));bound=2.*(eps+float(row.get('conversion_gap_mm') or 0.))
    return dict(compatible=variation<=bound,derived_local_span_mm=span,
        carrier_gap_variation_mm=variation,source_uncertainty_variation_bound_mm=bound)


def facing_faces(state,context,sec):
    g=sec.geometry;p=(state['point']-g.t)@g.R;owner=state['owner']
    lower=np.linalg.norm(np.maximum(np.maximum(context.face_boxes[:,:3]-p,p-context.face_boxes[:,3:]),0.),axis=1)
    found={};unknown=[]
    for fi in np.flatnonzero(lower<CONTACT):
        row=context.face_rows[fi];mi=row['material']
        if mi==owner:continue
        distance,closest=finite.events.native.finite_distance(context.face_shapes[fi],p)
        if distance>=CONTACT:continue
        n=normal(row,closest,g)
        if n is None or not row.get('face_valid',False):
            unknown.append(dict(face=int(fi),material=mi,reason='UNKNOWN_NEARBY_ORIGINAL_FACE',distance_mm=distance));continue
        if abs(n@state['free'])<=abs(n@state['tangent']):continue
        coherence=carrier_coherence(state,row,context,sec)
        if coherence is None or not coherence['compatible']:continue
        found.setdefault(mi,[]).append(dict(face=int(fi),distance_mm=float(distance),
            normal_free_dot=abs(float(n@state['free'])),normal_tangent_dot=abs(float(n@state['tangent'])),carrier_coherence=coherence))
    return found,unknown


def first_interval(sec,material,point,direction,eps):
    intervals=step.connected(sec,material,point,direction)
    eligible=[(a,b) for a,b in intervals if b>eps and a>=-eps and max(0.,a)<CONTACT]
    return min(eligible,key=lambda q:max(0.,q[0])) if eligible else None


def support(state,mate,context,sec):
    g=sec.geometry;point,free=state['point'],state['free'];eps=state['numerical_uncertainty_mm']
    horizontal=first_interval(sec,mate,point,-free,eps)
    if horizontal is None:return None
    a,b=horizontal;offset=min((b-a)/2.,state['base_thickness_mm']/2.,
        float(context.packet['candidates'][state['candidate']]['circle']['radius'])/4.)
    if offset<=eps:return None
    gap=max(0.,a);middle=point-free*(gap+offset)
    vertical=step.connected(sec,mate,middle,g.base)
    vertical=[(lo,hi) for lo,hi in vertical if lo<-eps and hi>eps]
    if len(vertical)!=1:return None
    lo,hi=vertical[0]
    below=max(lo,state['base_bottom_mm'])/2.;above=min(hi,state['base_thickness_mm'],b-a)/2.
    if below>=-eps or above<=eps:return None
    gaps=[]
    for h in (below,above):
        interval=first_interval(sec,mate,point+h*g.base,-free,eps)
        if interval is None:return None
        gaps.append(max(0.,interval[0]))
    free_point=point+free*state['base_interior_probe_offset_mm']+above*g.base
    if g.inside(mate,free_point) or g.inside(state['owner'],free_point):return None
    return dict(mate=mate,minimum_local_wall_depth_mm=float(b-a),gap_mm=float(max(gaps)),
        below_probe_mm=float(below),above_probe_mm=float(above),vertical_material_interval_mm=[float(lo),float(hi)])


def station(ci,q,context,sec):
    state=observation.frame(ci,q,context.packet,sec)
    if state is None:return dict(status='UNKNOWN',reason='NO_UNIQUE_LOCAL_BASE_SECTION',q_mm=float(q))
    state=dict(state,candidate=ci,query_arclength_mm=q)
    faces,unknown=facing_faces(state,context,sec);positive=[]
    for mate,witnesses in faces.items():
        evidence=support(state,mate,context,sec)
        if evidence is not None:positive.append(dict(evidence,finite_facing_faces=witnesses))
    if len(positive)==1 and not unknown:
        return dict(status='POSITIVE',q_mm=float(q),owner=state['owner'],
            mate=positive[0]['mate'],base_thickness_mm=state['base_thickness_mm'],evidence=positive[0])
    if positive or unknown:
        return dict(status='UNKNOWN',reason='AMBIGUOUS_OR_UNKNOWN_LOCAL_WALL',q_mm=float(q),
            positive=positive,unknown_faces=unknown)
    return dict(status='NO_SUPPORTED_LONGITUDINAL_WALL',q_mm=float(q),owner=state['owner'],
        finite_sections_queried=True,near_facing_materials=sorted(faces))


def end_witness(ci,q,mate,context,sec):
    points=np.asarray(context.packet['candidates'][ci]['points']);s=finite.cad.arclength(points)
    point,tangent=finite.point_tangent(points,s,q);native=(point-sec.geometry.t)@sec.geometry.R
    eps=NUMERIC+.025+float(context.packet['candidates'][ci]['circle'].get('max_fit_error_mm',0.))
    result=[]
    for fi,row in enumerate(context.face_rows):
        if row['material']!=mate or not row.get('face_valid',False):continue
        n=normal(row,native,sec.geometry)
        if n is None or abs(n@tangent)<=abs(n@np.cross(tangent,sec.geometry.base)):continue
        box=context.face_boxes[fi];lower=np.linalg.norm(np.maximum(np.maximum(box[:3]-native,native-box[3:]),0.))
        if lower>eps:continue
        distance,_=finite.events.native.finite_distance(context.face_shapes[fi],native)
        if distance<=eps:result.append(dict(face=fi,material=mate,finite_end_distance_mm=float(distance),
            original_face_valid=True,normal_tangent_dot=abs(float(n@tangent))))
    return result


def observe(ci,parents,context,verify=False):
    c=context.packet['candidates'][ci];length=context.lengths[ci]
    own=c.get('material',-1)
    if own not in context.panels:return dict(candidate=ci,reason='NO_QUALIFIED_BASE_PANEL',applicable=False)
    if np.linalg.norm(np.asarray(c['points'][0])-c['points'][-1])<=NUMERIC:
        return dict(candidate=ci,reason='CLOSED_NATIVE_DOMAIN_RETAINS_CURRENT_MODEL',applicable=False)
    sec=context.view(own,1.)
    row=context.inventory['candidates'][ci]
    nodes=sorted(set([0.,length]+[float(q) for q in row['qualified_geometry_nodes_mm']]))
    cells=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        if b-a<=NUMERIC:continue
        observation0=station(ci,(a+b)/2.,context,sec)
        cells.append(dict(start_mm=a,end_mm=b,midpoint=observation0))
    mates=sorted({v['midpoint']['mate'] for v in cells if v['midpoint']['status']=='POSITIVE'})
    if len(mates)!=1 or any(v['midpoint']['status']=='UNKNOWN' for v in cells):
        return dict(candidate=ci,applicable=False,reason='UNKNOWN_OR_MULTIPLE_LONGITUDINAL_JOINTS',cells=cells,mates=mates)
    mate=mates[0];domains=finite.union([(v['start_mm'],v['end_mm']) for v in cells if v['midpoint']['status']=='POSITIVE'])
    no_contact=[v for v in cells if v['midpoint']['status']=='NO_SUPPORTED_LONGITUDINAL_WALL']
    if not domains or not no_contact:
        return dict(candidate=ci,applicable=False,reason='NOT_PARTIALLY_SUPPORTED_NATIVE_CARRIER',cells=cells,mates=mates,domains_mm=domains)
    ends=[]
    for a,b in domains:
        for q in (a,b):
            if q<=NUMERIC or q>=length-NUMERIC:
                ends.append(dict(q_mm=q,native_end=True));continue
            witnesses=end_witness(ci,q,mate,context,sec)
            if not witnesses:return dict(candidate=ci,applicable=False,reason='NO_FINITE_MATERIAL_END_WITNESS',q_mm=q,cells=cells,domains_mm=domains)
            ends.append(dict(q_mm=q,native_end=False,witnesses=witnesses))
    def free_side(candidate,q,packet):
        base=observation.frame(candidate,q,packet,sec)
        if base is None:return None
        base=dict(base,candidate=candidate)
        if support(base,mate,context,sec) is None:return None
        return base['point'],base['tangent'],base['free'],mate,[own]
    proxy=types.SimpleNamespace(**dict(sec.geometry.__dict__,free_side=free_side))
    blocks,review=finite.crossings(ci,context.packet,context.inventory,proxy)
    ranges=finite.union([(p['interval']['start_mm'],p['interval']['end_mm']) for p in parents if p['source_candidate']==ci])
    for event in review:
        if event['measured_two_board_passage_mm'] is not None and any(
                abs(left[1]-event['start_mm'])<=3. and abs(right[0]-event['end_mm'])<=3.
                for left,right in zip(ranges[:-1],ranges[1:])):
            if not any(abs(b['start_mm']-event['start_mm'])<=NUMERIC and abs(b['end_mm']-event['end_mm'])<=NUMERIC for b in blocks):
                blocks.append(dict(event,decision='PRESERVED_MODEL_GAP_AT_VERIFIED_TRANSVERSE_PAIR'))
    allowed=[]
    for a,b in domains:
        cursor=a
        for event in sorted(blocks,key=lambda t:t['start_mm']):
            left,right=max(a,event['start_mm']),min(b,event['end_mm'])
            if right<=left:continue
            if left>cursor+NUMERIC:allowed.append([cursor,left])
            cursor=max(cursor,right)
        if b>cursor+NUMERIC:allowed.append([cursor,b])
    anchored=[]
    for a,b in allowed:
        indices=[i for i,p in enumerate(parents) if p['source_candidate']==ci and
            min(b,p['interval']['end_mm'])-max(a,p['interval']['start_mm'])>provenance.MIN_OVERLAP_MM]
        if indices:anchored.append(dict(start_mm=a,end_mm=b,model_parent_indices=indices))
    if verify:
        native=finite.cad.arclength(np.asarray(c['points']))
        for cell in cells:
            a,b=cell['start_mm'],cell['end_mm'];expected=cell['midpoint']['status']
            samples=np.unique(np.r_[native[(native>a+NUMERIC)&(native<b-NUMERIC)],
                np.linspace(a,b,int(np.ceil(b-a))+2)[1:-1]])
            for q in samples:
                actual=station(ci,float(q),context,sec)
                if actual['status']!=expected or (expected=='POSITIVE' and actual['mate']!=mate):
                    return dict(candidate=ci,applicable=False,reason='UNRESOLVED_INTERIOR_CELL_SUPPORT_CHANGE',q_mm=float(q),
                        expected=cell['midpoint'],actual=actual,cells=cells,domains_mm=domains)
    return dict(candidate=ci,applicable=bool(anchored),finite_material_pair=[own,mate],cells=cells,
        domains_mm=domains,finite_material_ends=ends,transverse_blocks=blocks,transverse_review=review,
        anchored_weld_intervals=anchored,verified_dense=verify,
        no_pointcloud_gap_or_fixed_count_or_label_endpoints=True)
