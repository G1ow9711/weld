"""Observe finite two-panel interfaces with a common outer material plane.

This observes a joint subtype, not whether it should be welded. The original
source path, finite occupied intervals and individual panel levels are retained.
"""
import math
import numpy as np
from . import curveonly_01a1ac7c_lower_face_observations_v403 as original

step,finite=original.step,original.step.finite
NUMERIC=step.NUMERIC


def protected_sources(decoded):
    return {int(c['candidate']) for c in decoded.get('finite_step_joint_rule',{}).get('certificates',[])
        if c.get('model_common_plane_anchors')}


def targets(decoded,packet):
    votes={int(v['candidate']):v for v in decoded.get('paired_face_source_rule',{}).get('source_votes',[])}
    protected=protected_sources(decoded);found=[]
    for index,parent in enumerate(decoded['parents']):
        ci=parent['source_candidate'];c=packet['candidates'][ci];vote=votes.get(ci)
        if vote is None or vote['endorsed'] or ci in protected:continue
        if parent['weld_type']!='horizontal' or not(c.get('STEP') and c.get('pose')) or c.get('material',-1)<0:continue
        found.append(dict(parent_index=index,candidate=ci,source_vote=vote))
    return found


def uncertainty(ci,context):
    c=context.packet['candidates'][ci];owner=c['material'];panel=context.panels[owner]
    conversion=max(float(context.inventory['faces'][i].get('conversion_gap_mm',0.)) for i in panel['dominant_faces'])
    return NUMERIC+float(c.get('source_curve_discretization_mm',.025))+float((c.get('circle') or {}).get('max_fit_error_mm',0.))+conversion


def configuration(owner,mate,eps,context):
    if owner not in context.panels or mate not in context.panels or owner==mate:return None
    lo,hi,axis=original.panel_levels(owner,context);ml,mh,other=original.panel_levels(mate,context)
    if np.linalg.norm(axis-other)>NUMERIC:return None
    shared=[dict(owner_level_mm=a,mate_level_mm=b,delta_mm=abs(a-b)) for a in (lo,hi) for b in (ml,mh) if abs(a-b)<=eps]
    if not shared or min(hi,mh)-max(lo,ml)<=2.*eps:return None
    return dict(owner=owner,mate=mate,owner_limits_mm=[lo,hi],mate_limits_mm=[ml,mh],axis=axis,
        shared_outer_planes=shared,overlap_middle_mm=(max(lo,ml)+min(hi,mh))/2.,uncertainty_mm=eps)


def choices(intervals,eps,sign=None):
    found=[]
    for a,b in intervals:
        if (sign is None or sign>0.) and abs(a)<=eps and b>eps:found.append((1.,b-a))
        if (sign is None or sign<0.) and abs(b)<=eps and a < -eps:found.append((-1.,b-a))
    return found


def station(ci,q,config,context):
    c=context.packet['candidates'][ci];points=np.asarray(c['points']);s=finite.cad.arclength(points)
    point,tangent=finite.point_tangent(points,s,q);axis=config['axis'];eps=config['uncertainty_mm']
    height=float(point@axis);lo,hi=config['owner_limits_mm']
    if min(abs(height-lo),abs(height-hi))>eps:return dict(passed=False,reason='NATIVE_SOURCE_NOT_ON_OWNER_OUTER_FACE')
    normal=np.cross(tangent,axis);norm=float(np.linalg.norm(normal))
    if norm<=NUMERIC:return dict(passed=False,reason='UNRESOLVED_TRANSVERSE_DIRECTION')
    normal/=norm;middle=config['overlap_middle_mm'];p=point+axis*(middle-height)
    owner,mate=config['owner'],config['mate'];sec=context.view(owner,1.)
    own=original.connected(sec,owner,p,normal)
    if own is None:return dict(passed=False,reason='OWNER_FINITE_QUERY_NOT_DONE')
    interior=choices(own,eps)
    if len(interior)!=1:return dict(passed=False,reason='NO_UNIQUE_OWNER_EDGE_SIDE')
    sign,width=interior[0];other=original.connected(sec,mate,p,normal)
    if other is None:return dict(passed=False,reason='MATE_FINITE_QUERY_NOT_DONE')
    opposite=choices(other,eps,-sign)
    if len(opposite)!=1:return dict(passed=False,reason='NO_OPPOSITE_FINITE_PANEL_EDGE_SIDE')
    _,mate_width=opposite[0];ml,mh=config['mate_limits_mm']
    offset=min(width/2.,mate_width/2.,(hi-lo)/2.,(mh-ml)/2.)
    if offset<=eps:return dict(passed=False,reason='UNRESOLVED_INTERIOR_SCALE')
    vertical=[]
    for material,probe,limits in [(owner,p+sign*normal*offset,[lo,hi]),(mate,p-sign*normal*offset,[ml,mh])]:
        values=original.connected(sec,material,probe,axis)
        if values is None:return dict(passed=False,reason='VERTICAL_FINITE_QUERY_NOT_DONE')
        occupied=[(a,b) for a,b in values if a<0.<b]
        if len(occupied)!=1:return dict(passed=False,reason='NO_UNIQUE_PANEL_MATERIAL_INTERVAL')
        a,b=occupied[0]
        if max(abs(a+middle-limits[0]),abs(b+middle-limits[1]))>eps:
            return dict(passed=False,reason='LOCAL_MATERIAL_DIFFERS_FROM_DOMINANT_PANEL_LIMITS')
        vertical.append(dict(material=material,interval_relative_middle_mm=[float(a),float(b)]))
    return dict(passed=True,q_mm=float(q),source_height_mm=height,finite_overlap_height_mm=middle,
        owner_side=float(sign),interior_probe_offset_mm=float(offset),owner_width_mm=float(width),mate_width_mm=float(mate_width),
        vertical_finite_intervals=vertical)


def qualify(parent,context,verify=True):
    ci=parent['source_candidate'];c=context.packet['candidates'][ci];owner=c.get('material',-1)
    if owner not in context.panels:return dict(passed=False,reason='NO_DOMINANT_OWNER_PANEL')
    r=parent.get('interval');length=context.lengths[ci]
    if not r:return dict(passed=False,reason='NO_ORIGINAL_NATIVE_INTERVAL')
    a,b=float(r['start_mm']),float(r['end_mm'])
    if a<0. or b>length+1e-8 or b<=a:return dict(passed=False,reason='PRESERVE_UNRESOLVED_NATIVE_PERIODIC_RANGE')
    eps=uncertainty(ci,context);found=[]
    for mate in context.panels:
        config=configuration(owner,mate,eps,context)
        if config is None:continue
        middle=station(ci,(a+b)/2.,config,context)
        if middle['passed']:found.append((config,middle))
    if len(found)!=1:return dict(passed=False,reason='NO_UNIQUE_FINITE_COPLANAR_INTERFACE',midpoint_matches=len(found))
    config,middle=found[0];source_s=finite.cad.arclength(np.asarray(c['points']))
    grid=np.unique(np.r_[np.linspace(a,b,max(2,int(math.ceil(b-a))+1)),source_s[(source_s>a)&(source_s<b)],a,b])
    if verify:grid=np.unique(np.r_[grid,(grid[:-1]+grid[1:])/2.])
    else:grid=np.array([a,(a+b)/2.,b])
    checked=[];failures=[]
    for q in grid:
        state=station(ci,float(q),config,context)
        if state['passed']:checked.append(state)
        else:failures.append(dict(q_mm=float(q),evidence=state))
    return dict(candidate=ci,passed=not bool(failures),reason='FINITE_COPLANAR_PANEL_INTERFACE' if not failures else 'UNRESOLVED_FINITE_INTERFACE_RANGE',
        finite_material_pair=[owner,config['mate']],owner_limits_mm=config['owner_limits_mm'],mate_limits_mm=config['mate_limits_mm'],
        common_outer_planes=config['shared_outer_planes'],panel_axis=config['axis'].tolist(),source_interval_mm=[a,b],
        dominant_panel_faces={str(m):context.panels[m]['dominant_faces'] for m in (owner,config['mate'])},
        original_source_uncertainty_mm=eps,checked_stations=len(grid),verified_dense=bool(verify),
        failures=failures[:12],failure_count=len(failures),observations=checked,
        joint_subtype_only_not_should_weld_or_geometric_acceptance=True)


def observe(decoded,packet,inventory,solids,verify=True):
    selected=targets(decoded,packet)
    if not selected:return dict(targets=[],qualifications=[],reason='NO_UNENDORSED_UNPROTECTED_SOURCE',no_input_removed=True)
    # The preceding finite-step module already supplied an exact input panel
    # list. An absent owner can be skipped without loading any new geometry.
    panels=set(decoded.get('finite_step_joint_rule',{}).get('panel_materials',[]))
    possible=[q for q in selected if packet['candidates'][q['candidate']].get('material') in panels]
    if not possible:return dict(targets=selected,qualifications=[],reason='NO_PRECEDING_DOMINANT_PANEL_OWNER',no_input_removed=True)
    if not solids or not packet.get('pose',{}).get('passed'):
        return dict(targets=selected,qualifications=[],reason='NO_QUALIFIED_FINITE_STEP',no_input_removed=True)
    context=step.Context(solids,packet,inventory);out=[]
    for target in possible:
        q=qualify(decoded['parents'][target['parent_index']],context,verify)
        out.append(dict(target,geometry=q))
    return dict(targets=selected,qualifications=out,verified_dense=bool(verify),
        no_new_should_weld_or_process_gap_decision=True)
