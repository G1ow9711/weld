"""Resolve competing through-plate T-joint side hypotheses with two models.

Opposite plate faces are distinct potential weld sides. They are never merged.
The rule requires a frozen expert rejection, an endorsed opposite source, an
existing complete-range model anchor and original finite through-wall support.
"""
import copy
import time
import numpy as np
from . import curveonly_01a1ac7c_lower_face_observations_v403 as geometry
from . import curveonly_01a1ac7c_source_expert_v405 as expert

step=geometry.step
finite,aliases=step.finite,step.aliases
NUMERIC,CONTACT=step.NUMERIC,step.CONTACT


def face(ci,context):
    c=context.packet['candidates'][ci];owner=c.get('material',-1)
    if owner not in context.panels or not(c.get('STEP') and c.get('pose')):return None
    low,high,axis=geometry.panel_levels(owner,context);points=np.asarray(c['points'])
    eps=NUMERIC+float(c.get('source_curve_discretization_mm',.025))
    levels=[h for h in (low,high) if np.max(np.abs(points@axis-h))<=eps]
    if len(levels)!=1:return None
    level=levels[0];sign=1. if level==low else -1.
    return dict(owner=owner,axis=axis,inward=sign*axis,thickness=high-low,
        level=level,opposite_level=high if sign>0. else low,uncertainty=eps)


def near_materials(owner,point,context):
    g=context.geometry;native=(point-g.t)@g.R
    if not hasattr(context,'paired_face_boxes'):
        context.paired_face_boxes=np.asarray([finite.cad.bbox(s) for s in g.solids])
    boxes=context.paired_face_boxes
    bounds=np.linalg.norm(np.maximum(np.maximum(boxes[:,:3]-native,native-boxes[:,3:]),0.),axis=1)
    found=[];unknown=[]
    for mi in np.flatnonzero(bounds<=CONTACT+NUMERIC):
        if mi==owner:continue
        d=geometry.body_distance(g.solids[int(mi)],native)
        if d is None:unknown.append(int(mi))
        elif d<=CONTACT+NUMERIC:found.append(dict(material=int(mi),distance_mm=d))
    return found,unknown


def through_wall(parent,state,context):
    ci=parent['source_candidate'];interval=parent['interval'];c=context.packet['candidates'][ci]
    points=np.asarray(c['points']);q=(interval['start_mm']+interval['end_mm'])/2.
    p,tangent=finite.point_tangent(points,finite.cad.arclength(points),q)
    owner,axis,T,eps=state['owner'],state['inward'],state['thickness'],state['uncertainty']
    near,unknown=near_materials(owner,p,context)
    if unknown or len(near)!=1:return None,'UNKNOWN_OR_MULTIPLE_LOCAL_PARTNERS'
    mate=near[0]['material']
    if mate in context.panels:return None,'PARALLEL_PANEL_JOINT_RETAINS_CURRENT_MODEL'
    opposite,unknown2=near_materials(owner,p+axis*T,context)
    if unknown2 or len(opposite)!=1 or opposite[0]['material']!=mate:return None,'OPPOSITE_FACE_HAS_DIFFERENT_OR_UNKNOWN_JOINT'
    sign=1. if axis@context.panels[owner]['axis']>0. else -1.
    sec=context.view(owner,sign);normal=np.cross(tangent,axis)
    if np.linalg.norm(normal)<=NUMERIC:return None,'DEGENERATE_TRANSVERSE_DIRECTION'
    normal/=np.linalg.norm(normal)
    own=geometry.connected(sec,owner,p+axis*T/2.,normal)
    if own is None:return None,'OWNER_INTERSECTOR_NOT_DONE'
    choices=[]
    for a,b in own:
        if abs(a)<=eps and b>eps:choices.append((normal,b-a))
        elif abs(b)<=eps and a < -eps:choices.append((-normal,b-a))
    if len(choices)!=1:return None,'NO_UNIQUE_FINITE_OWNER_SIDE'
    inside,width=choices[0];outward=-inside
    wall=geometry.connected(sec,mate,p+axis*T/2.,outward)
    if wall is None:return None,'MATE_INTERSECTOR_NOT_DONE'
    wall=[(a,b) for a,b in wall if a>=-eps and b>eps and max(a,0.)<=CONTACT]
    if len(wall)!=1:return None,'NO_UNIQUE_FINITE_MATE_SIDE'
    a,b=wall[0];offset=min(T/2.,width/2.,(b-a)/2.)
    if offset<=eps:return None,'UNRESOLVED_INTERIOR_SCALE'
    vertical=geometry.connected(sec,mate,p+outward*(max(a,0.)+offset),axis)
    if vertical is None:return None,'VERTICAL_INTERSECTOR_NOT_DONE'
    crossing=[(lo,hi) for lo,hi in vertical if lo < -eps and hi>T+eps]
    if len(crossing)!=1:return None,'MATE_DOES_NOT_PASS_BOTH_PLATE_FACES'
    return dict(owner=owner,mate=mate,query_native_s_mm=float(q),point_mm=p.tolist(),
        source_face_axis_inward=axis.tolist(),plate_thickness_mm=T,local_probe_offset_mm=float(offset),
        mate_interval_through_slab_mm=list(map(float,crossing[0])),
        source_face_material_distances=near,opposite_face_material_distances=opposite),None


def inspect(decoded,features,packet,inventory,solids,source_expert):
    if not solids or not packet.get('pose',{}).get('passed'):
        return [],dict(not_applicable='NO_QUALIFIED_FINITE_STEP',input_retained=True)
    votes=expert.classify(features,packet,source_expert)
    if not any(votes[p['source_candidate']]['rejected'] for p in decoded['parents']):
        return [],dict(branch='PAIRED_FACE_SOURCE_EXPERT_V407',certificates=[],no_rejected_published_sources=True,
            source_votes=votes)
    context=step.Context(solids,packet,inventory);faces={ci:face(ci,context) for ci in {p['source_candidate'] for p in decoded['parents']}}
    chosen=[];rejected=[]
    for i,parent in enumerate(decoded['parents']):
        ci=parent['source_candidate'];state=faces[ci]
        if state is None or not votes[ci]['rejected']:continue
        r=parent['interval']
        if r['start_mm'] < -NUMERIC or r['end_mm']>context.lengths[ci]+NUMERIC:
            rejected.append(dict(parent_index=i,candidate=ci,reason='UNRESOLVED_PERIODIC_NATIVE_PROJECTION'));continue
        shifted=step.native_parent(parent,context)+state['inward']*state['thickness'];anchors=[]
        for j,other in enumerate(decoded['parents']):
            cj=other['source_candidate'];other_face=faces[cj]
            if not votes[cj]['endorsed'] or other_face is None or other_face['owner']!=state['owner']:continue
            if abs(other_face['level']-state['opposite_level'])>state['uncertainty']:continue
            if parent['weld_type']!=other['weld_type']:continue
            if parent.get('weld_side') is not None and other.get('weld_side') is not None and parent['weld_side']!=other['weld_side']:continue
            projected=aliases.coincident_parameters(shifted,packet['candidates'][cj]['points'])
            if projected is None:continue
            s,error=projected;lo,hi=float(s.min()),float(s.max());other_range=other['interval']
            if lo>=other_range['start_mm']-3. and hi<=other_range['end_mm']+3.:
                anchors.append(dict(parent_index=j,candidate=cj,projected_interval_mm=[lo,hi],
                    model_interval=other_range,ordered_native_projection_error_mm=error))
        if not anchors:
            rejected.append(dict(parent_index=i,candidate=ci,reason='NO_ENDORSED_OPPOSITE_FACE_MODEL_ANCHOR'));continue
        physical,reason=through_wall(parent,state,context)
        if physical is None:
            rejected.append(dict(parent_index=i,candidate=ci,reason=reason));continue
        chosen.append(dict(parent_index=i,candidate=ci,geometry=physical,opposite_face_anchors=anchors,
            rejected_source_vote=votes[ci],endorsed_opposite_votes=[votes[a['candidate']] for a in anchors],
            decision='FROZEN_LEARNED_SIDE_REJECTION_WITH_OPPOSITE_MODEL_AND_FINITE_THROUGH_WALL',
            physical_contact_is_not_treated_as_should_weld=True))
    return chosen,dict(branch='PAIRED_FACE_SOURCE_EXPERT_V407',certificates=chosen,rejected=rejected,source_votes=votes,
        source_count_classes_not_used_to_limit_output_weld_count=True,
        faces_not_merged_or_geometrically_deduplicated=True,no_hard_upper_or_lower_face_ban=True,
        no_new_fit_or_VALcalibration=True)


def correct(decoded,features,packet,inventory,solids,source_expert):
    start=time.perf_counter();chosen,audit=inspect(decoded,features,packet,inventory,solids,source_expert)
    result=copy.deepcopy(decoded)
    if chosen:
        remove={v['parent_index'] for v in chosen};parents=[p for i,p in enumerate(result['parents']) if i not in remove]
        for i,p in enumerate(parents):p['parent_id']=f'curve_{i:05d}'
        result.update(parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments']) for p in parents))
        audit.update(replaced_upstream_parent_indices=sorted(remove),removed_parent_indices=sorted(remove),
            new_parent_count=0,upstream_parents_not_targeted_retained_exact=True)
    result['paired_face_source_rule']=audit;result['paired_face_source_rule_seconds']=time.perf_counter()-start
    return result
