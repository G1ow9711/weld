"""Resolve model curve hypotheses at a finite parallel-plate thickness step.

A model-selected common-plane edge anchors should-weld intent. The rule only
moves that hypothesis to a source-owned, continuously qualified step corner.
Thickness, normal, sign and search/probe scales come from the actual materials.
No part names, fixed coordinates, radius/count lookup, or new fitted threshold.
"""
import copy
import math
import time
import types
import numpy as np
from . import curveonly_01a1ac7c_adaptive_sections_v365 as observation
from . import curveonly_01a1ac7c_complete_parent_runtime_v368 as baseline
from . import curveonly_01a1ac7c_recovery_resolution_v353 as resolution

finite,prior,provenance,aliases=observation.finite,observation.prior,observation.provenance,observation.aliases
NUMERIC=finite.NUMERIC
CONTACT=observation.CONTACT


class Context:
    def __init__(self,solids,packet,inventory):
        self.geometry=finite.Geometry(solids,packet,inventory)
        self.packet,self.inventory=packet,inventory
        self.lengths=[float(finite.cad.arclength(np.asarray(c['points']))[-1]) for c in packet['candidates']]
        self.faces={};self.panels={};self.views={};index=0
        for mi,solid in enumerate(solids):
            rows=[]
            for face in finite.cad.items(solid,finite.cad.TopAbs_FACE):
                row=inventory['faces'][index];assert row['material']==mi
                props=finite.GProp_GProps();finite.BRepGProp.SurfaceProperties_s(face,props)
                rows.append(dict(row,face_index=index,area_mm2=float(props.Mass())));index+=1
            self.faces[mi]=rows
            ranked=sorted(rows,key=lambda r:-r['area_mm2'])
            if len(ranked)<2 or not all(r['status']=='CANONICAL_PLANE' and r.get('face_valid',False) for r in ranked[:2]):
                continue
            a,b=ranked[:2];n=np.asarray(a['normal']);m=np.asarray(b['normal']);sign=float(np.sign(n@m))
            if np.linalg.norm(n-sign*m)>NUMERIC:continue
            thickness=abs(a['offset_mm']-sign*b['offset_mm'])
            if thickness<=NUMERIC:continue
            axis=n@self.geometry.R.T
            if abs(axis@self.geometry.base)<math.cos(math.radians(3.)):continue
            if len(ranked)>2 and ranked[1]['area_mm2']<=ranked[2]['area_mm2']:continue
            axis*=1. if axis@self.geometry.base>=0. else -1.
            self.panels[mi]=dict(axis=axis,dominant_faces=[a['face_index'],b['face_index']],
                thickness_mm=float(thickness),dominant_areas_mm2=[a['area_mm2'],b['area_mm2']])

    def view(self,owner,sign):
        axis=sign*self.panels[owner]['axis'];key=tuple(axis)
        if key in self.views:return self.views[key]
        g=self.geometry;pairs=[]
        for mi,rows in self.faces.items():
            planes=[r for r in rows if r['status']=='CANONICAL_PLANE' and
                abs((np.asarray(r['normal'])@g.R.T)@axis)<=NUMERIC]
            choices=[]
            for i,a in enumerate(planes):
                for b in planes[:i]:
                    n,m=np.asarray(a['normal']),np.asarray(b['normal']);sgn=float(np.sign(n@m))
                    if np.linalg.norm(n-sgn*m)>NUMERIC:continue
                    thickness=abs(a['offset_mm']-sgn*b['offset_mm'])
                    if thickness>NUMERIC:choices.append((min(a['area_mm2'],b['area_mm2']),a,b,sgn,thickness))
            if choices:
                _,a,b,sgn,thickness=max(choices,key=lambda q:q[0])
                pairs.append(dict(material=mi,faces=[a['face_index'],b['face_index']],
                    normal=np.asarray(a['normal']),limits=sorted((a['offset_mm'],sgn*b['offset_mm'])),thickness_mm=thickness))
        proxy=types.SimpleNamespace(**dict(g.__dict__,base=axis,pairs=pairs,
            inside=g.inside,passage=g.passage))
        result=observation.Sections(proxy);self.views[key]=result;return result


def connected(sec,material,point,direction):
    # Coplanar intersectors may report extra roots within the SAME material.
    # Keep genuine empty intervals; union only touching occupied intervals.
    return finite.union(sec.intervals(material,point,direction))


def at(ci,q,context,sec,mate=None):
    packet=context.packet;g=sec.geometry;owner=packet['candidates'][ci]['material']
    state=observation.frame(ci,q,packet,sec)
    if state is None:return None
    p,free=state['point'],state['free'];T=state['base_thickness_mm'];eps=state['numerical_uncertainty_mm']
    choices=[]
    for mi,panel in context.panels.items():
        if mi==owner or (mate is not None and mi!=mate):continue
        if abs(abs(panel['axis']@g.base)-1.)>NUMERIC:continue
        if not all(f.get('face_valid',False) for f in context.faces[mi]+context.faces[owner]):continue
        intervals=connected(sec,mi,p-g.base*T/2.,-free)
        intervals=[(a,b) for a,b in intervals if b>eps and a>=-eps and max(0.,a)<CONTACT]
        if not intervals:continue
        a,b=min(intervals,key=lambda z:max(0.,z[0]));gap=max(0.,a)
        offset=min((b-a)/2.,T/2.,float(packet['candidates'][ci]['circle']['radius'])/4.)
        if offset<=eps:continue
        vertical=connected(sec,mi,p-free*(gap+offset),g.base)
        material=[(lo,hi) for lo,hi in vertical if lo<-T/2.<hi]
        if len(material)!=1:continue
        lo,hi=material[0]
        if abs(lo+T)>eps or hi<=eps:continue
        native=(p-g.t)@g.R
        planes=[float((off-n@native)/dot) for n,off,dot in sec.horizontal.get(mi,[])]
        if not(any(abs(h-lo)<=eps for h in planes) and any(abs(h-hi)<=eps for h in planes)):continue
        weld_height=min(T,hi)/2.
        probe=p+free*state['base_interior_probe_offset_mm']+g.base*weld_height
        native_probe=(probe-g.t)@g.R
        blockers=[]
        for other,box in enumerate(sec.bounds):
            if np.all(native_probe>=box[:3]-NUMERIC) and np.all(native_probe<=box[3:]+NUMERIC) and g.inside(other,probe):
                blockers.append(other)
        if blockers:continue
        choices.append(dict(state=state,mate=mi,thin_thickness_mm=T,thick_thickness_mm=hi-lo,
            step_height_mm=hi,contact_gap_mm=gap,welding_probe_height_mm=weld_height,
            common_back_plane_height_mm=-T,local_weld_axis=g.base))
    return choices[0] if len(choices)==1 else None


def native_parent(parent,context):
    ci=parent['source_candidate'];r=parent['interval'];p=context.packet['candidates'][ci]['points']
    return finite.cad.slice_curve(p,r['start_mm'],r['end_mm'])


def belongs(parent,pair,context):
    ci=parent['source_candidate'];r=parent['interval']
    spans=provenance.primitive_intervals(ci,r['start_mm'],r['end_mm'],context.packet,context.lengths)
    materials={context.packet['candidates'][root].get('material',-1) for root,a,b in spans}
    return bool(materials) and materials.issubset(set(pair))


def anchors(ci,seed,parents,context):
    owner=context.packet['candidates'][ci]['material'];pair=[owner,seed['mate']]
    points=context.packet['candidates'][ci]['points'];translation=seed['thin_thickness_mm']*seed['local_weld_axis']
    found=[]
    for index,parent in enumerate(parents):
        if parent['weld_type']!='horizontal' or not belongs(parent,pair,context):continue
        projected=aliases.coincident_parameters(native_parent(parent,context)+translation,points)
        if projected is None:continue
        s,error=projected
        found.append(dict(parent_index=index,source_candidate=parent['source_candidate'],
            projected_step_interval_mm=[float(s.min()),float(s.max())],projection_error_mm=error,
            translation_from_verified_common_plane_mm=translation.tolist(),
            confidence=float(parent['confidence'])))
    return found


def certify(ci,sign,row,parents,context):
    owner=context.packet['candidates'][ci]['material'];sec=context.view(owner,sign)
    seed=at(ci,row['length_mm']/2.,context,sec)
    if seed is None:return None,dict(candidate=ci,sign=sign,reason='NO_UNIQUE_FINITE_PLATE_STEP')
    model=anchors(ci,seed,parents,context)
    if not model:return None,dict(candidate=ci,sign=sign,reason='NO_COMPLETE_MODEL_ANCHOR_ON_COMMON_PLANE')
    points=np.asarray(context.packet['candidates'][ci]['points']);native=finite.cad.arclength(points)
    samples=np.unique(np.r_[native,np.linspace(0.,native[-1],int(np.ceil(native[-1]))+1)])
    ranges={name:[math.inf,-math.inf] for name in ('thin_thickness_mm','thick_thickness_mm','step_height_mm','contact_gap_mm','welding_probe_height_mm')}
    for q in samples:
        local=at(ci,q,context,sec,mate=seed['mate'])
        if local is None:return None,dict(candidate=ci,sign=sign,reason='FINITE_STEP_SUPPORT_NOT_CONTINUOUS',q_mm=float(q))
        if abs(local['thin_thickness_mm']-seed['thin_thickness_mm'])>local['state']['numerical_uncertainty_mm']:
            return None,dict(candidate=ci,sign=sign,reason='CHANGING_COMMON_BACK_PLANE')
        for name in ranges:ranges[name]=[min(ranges[name][0],local[name]),max(ranges[name][1],local[name])]
    def free_side(candidate,q,packet):
        local=at(candidate,q,context,sec,mate=seed['mate'])
        if local is None:return None
        s=local['state'];return s['point'],s['tangent'],s['free'],s['owner'],[local['mate']]
    proxy=types.SimpleNamespace(**dict(sec.geometry.__dict__,free_side=free_side))
    blocks,review=finite.crossings(ci,context.packet,context.inventory,proxy)
    if blocks:return None,dict(candidate=ci,sign=sign,reason='VERIFIED_TRANSVERSE_PROCESS_GAP',blocks=blocks)
    model_ranges=finite.union([m['projected_step_interval_mm'] for m in model])
    for event in review:
        if event['measured_two_board_passage_mm'] is not None and any(
                abs(left[1]-event['start_mm'])<=3. and abs(right[0]-event['end_mm'])<=3.
                for left,right in zip(model_ranges[:-1],model_ranges[1:])):
            return None,dict(candidate=ci,sign=sign,reason='EXISTING_MODEL_GAP_AT_VERIFIED_TRANSVERSE_PAIR',event=event)
    return dict(candidate=ci,sign=sign,finite_material_pair=[owner,seed['mate']],
        dominant_panel_faces={str(mi):context.panels[mi]['dominant_faces'] for mi in (owner,seed['mate'])},
        model_common_plane_anchors=model,source_native_interval_mm=[0.,float(native[-1])],
        adaptive_ranges_mm=ranges,sampled_stations=len(samples),
        weld_axis=seed['local_weld_axis'].tolist(),common_plane_shift_mm=(seed['thin_thickness_mm']*seed['local_weld_axis']).tolist(),
        geometry_alone_does_not_create_should_weld_permission=True,
        no_constant_step_height_or_part_switch=True,transverse_review=review),None


def inspect(decoded,features,packet,inventory,solids):
    if not solids or not packet.get('pose',{}).get('passed'):
        return [],dict(not_applicable='NO_QUALIFIED_FINITE_STEP',input_retained=True),None
    context=Context(solids,packet,inventory);rows=prior.candidate_rows(features,packet)
    certs=[];rejected=[]
    for ci,(c,row) in enumerate(zip(packet['candidates'],rows)):
        if not(c.get('STEP') and c.get('pose') and c.get('material',-1) in context.panels and c.get('circle')):continue
        if not resolution.curvature_observation(c)['numerically_resolved_curvature']:continue
        if np.min(row['grid'][:,0])<math.cos(math.radians(3.)) or np.max(row['grid'][:,1])>math.sin(math.radians(3.)):continue
        local=[]
        for sign in (-1.,1.):
            cert,failure=certify(ci,sign,row,decoded['parents'],context)
            (local if cert else rejected).append(cert if cert else failure)
        if len(local)==1:certs+=local
        elif local:rejected.append(dict(candidate=ci,reason='AMBIGUOUS_MATERIAL_STEP_SIDE'))
    chosen=[];used=set()
    for cert in sorted(certs,key=lambda q:-context.lengths[q['candidate']]):
        ids={p['parent_index'] for p in cert['model_common_plane_anchors']}
        if ids&used:continue
        chosen.append(cert);used|=ids
    return chosen,dict(branch='V373_FINITE_PARALLEL_PLATE_STEP',panel_materials=list(context.panels),
        certificates=chosen,rejected=rejected,no_new_fit_or_threshold_calibration=True),context


def correct(decoded,features,packet,inventory,solids):
    start=time.perf_counter();chosen,audit,context=inspect(decoded,features,packet,inventory,solids)
    result=copy.deepcopy(decoded)
    if chosen:
        rows=prior.candidate_rows(features,packet);replacements=[];remove=set()
        for cert in chosen:
            ci=cert['candidate'];row=rows[ci]
            remove.update(p['parent_index'] for p in cert['model_common_plane_anchors'])
            pair=cert['finite_material_pair']
            for i,p in enumerate(decoded['parents']):
                if belongs(p,pair,context) and aliases.coincident_parameters(native_parent(p,context),packet['candidates'][ci]['points']) is not None:
                    remove.add(i)
            confidence=max(p['confidence'] for p in cert['model_common_plane_anchors'])
            replacements+=prior.codec.proposals(packet,[row],[prior.previous.intervals_to_codes(row,
                [dict(a=0.,b=row['length_mm'],confidence=confidence)])])
        selected,source_selection=provenance.select(replacements,packet)
        old=baseline.backbone.publication._fallback
        baseline.backbone.publication._fallback=baseline.backbone.corner.fallback
        try:
            exported=baseline.backbone.exporter.decode(features,packet,baseline.backbone.Proposals(selected,{}))
        finally:baseline.backbone.publication._fallback=old
        parents=[copy.deepcopy(p) for i,p in enumerate(decoded['parents']) if i not in remove]+exported['parents']
        for i,parent in enumerate(parents):parent['parent_id']=f'curve_{i:05d}'
        result['parents']=parents;result['geometry_units']=len(parents)
        result['published_segments']=sum(len(p['segments']) for p in parents)
        audit.update(replaced_upstream_parent_indices=sorted(remove),new_parent_count=len(exported['parents']),
            new_source_selection=source_selection,upstream_parents_not_targeted_retained_exact=True)
    result['finite_step_joint_rule']=audit
    result['finite_step_joint_rule_seconds']=time.perf_counter()-start
    return result
