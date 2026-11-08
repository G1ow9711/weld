"""Additional connected-curve geometry view, never a process-identity acceptance.

Every original curve and cut remains traceable. Only uniquely adjacent, tangent
compatible arcs on the same carrier/circle can connect. Duplicate/overlapping
geometry is not erased. Prediction grouping never reads reference geometry.
"""
import math
import numpy as np
from . import curveonly_01a1ac7c as cad

VERSION='CURVEONLY_CONNECTED_GEOMETRY_DIAGNOSTIC_V3'
POLICY=dict(endpoint_identity_mm=.001,circle_center_radius_identity_mm=.001,
    maximum_tangent_deflection_deg=5.,known_different_material_owners_block_join=True,
    known_side_changes_block_join=True,ambiguous_endpoint_links_not_joined=True,
    true_positive_gaps_above_identity_tolerance_preserved=True,
    original_records_and_all_internal_cuts_retained=True,
    geometry_only_not_process_or_joint_identity_acceptance=True,
    no_new_fitted_prediction_geometry=True,no_truth_driven_prediction_grouping=True)


def tangent(r,end):
    p=np.asarray(r['points'],float);fit=r.get('circle')or cad.circle_fit(p)
    if fit:
        v=np.cross(np.array(fit['normal']),p[0 if end==0 else -1]-np.array(fit['center']))
        v*=np.sign(fit['sweep'])*(1 if end==0 else -1)
    else:
        q=p if end==0 else p[::-1];q=q[np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-10]]
        if len(q)<2:return None
        if len(q)==2:v=q[1]-q[0]
        else:
            s=cad.arclength(q[:3]);coeff=np.polyfit(s,q[:3]-q[0],2);v=coeff[1]
    length=np.linalg.norm(v)
    return v/length if length>1e-12 else None


def compatible(a,b):
    if a['weld_type']!=b['weld_type']or a.get('weld_side')!=b.get('weld_side'):return None
    am,bm=a.get('_material'),b.get('_material')
    if am is not None and bm is not None and am>=0 and bm>=0 and am!=bm:return None
    ac,bc=a.get('_carrier'),b.get('_carrier')
    if ac is not None and bc is not None and ac==bc:return 'SAME_INPUT_CARRIER'
    x,y=a.get('circle'),b.get('circle')
    if x is None:x=cad.circle_fit(np.asarray(a['points']))
    if y is None:y=cad.circle_fit(np.asarray(b['points']))
    if x is None or y is None:return None
    if np.linalg.norm(np.array(x['center'])-y['center'])>.001 or abs(x['radius']-y['radius'])>.001:return None
    if abs(np.dot(x['normal'],y['normal']))<1-1e-8:return None
    return 'SAME_ANALYTIC_CIRCLE_GEOMETRY_ONLY'


def units(curves):
    ends=[];eligible=[]
    for i,r in enumerate(curves):
        p=np.asarray(r['points']);ends.extend([p[0],p[-1]])
        if np.linalg.norm(p[0]-p[-1])>=.1:eligible.append(i)
    choices={};evidence={}
    for ai,i in enumerate(eligible):
        for j in eligible[ai+1:]:
            reason=compatible(curves[i],curves[j])
            if reason is None:continue
            for a in (0,1):
                for b in (0,1):
                    distance=float(np.linalg.norm(ends[2*i+a]-ends[2*j+b]))
                    if distance>.001:continue
                    u=tangent(curves[i],a);v=tangent(curves[j],b)
                    if u is None or v is None:continue
                    cosine=float(-u@v)
                    if cosine<math.cos(math.pi/36):continue
                    left,right=(i,a),(j,b);choices.setdefault(left,[]).append(right);choices.setdefault(right,[]).append(left)
                    evidence[tuple(sorted([left,right]))]=dict(left=left,right=right,distance_mm=distance,
                        tangent_deflection_deg=float(math.degrees(math.acos(np.clip(cosine,-1,1)))),reason=reason,
                        process_cut_reason='UNKNOWN_RETAINED_NOT_CANCELED',joint_side_identity_not_accepted=True)
    links={k:v[0]for k,v in choices.items()if len(v)==1 and len(choices.get(v[0],[]))==1}
    visited=set();out=[]
    for seed in range(len(curves)):
        if seed in visited:continue
        component={seed};todo=[seed]
        while todo:
            i=todo.pop()
            for end in (0,1):
                if(i,end)in links:
                    j,_=links[i,end]
                    if j not in component:component.add(j);todo.append(j)
        free=[(i,e)for i in sorted(component)for e in (0,1)if(i,e)not in links]
        start=free[0]if free else(min(component),0);i,end=start;members=[];parts=[];joins=[]
        while i not in {m['curve']for m in members}:
            p=np.asarray(curves[i]['points']);p=p if end==0 else p[::-1]
            parts.append(p);members.append(dict(curve=i,reversed=bool(end),original_id=curves[i].get('id',curves[i].get('parent_id')),
                original_children=curves[i].get('children'),original_closed=curves[i].get('closed',curves[i].get('is_closed'))))
            exit_end=1-end
            if(i,exit_end)not in links:break
            nxt=links[i,exit_end];joins.append(evidence[tuple(sorted([(i,exit_end),nxt]))]);i,end=nxt
        assert {m['curve']for m in members}==component
        visited.update(component);points=np.vstack(parts);source_length=sum(float(cad.arclength(p)[-1])for p in parts)
        bridge=sum(float(np.linalg.norm(a[-1]-b[0]))for a,b in zip(parts[:-1],parts[1:]))
        length=float(cad.arclength(points)[-1]);assert abs(length-source_length-bridge)<1e-7*max(1.,length)
        first=curves[members[0]['curve']]
        out.append(dict(points=points.tolist(),weld_type=first['weld_type'],weld_side=first.get('weld_side'),
            closed=bool(np.linalg.norm(points[0]-points[-1])<.1),members=members,connections=joins,
            source_curve_count=len(members),source_length_mm=source_length,numerical_bridge_length_mm=bridge,
            circle=cad.circle_fit(points),_valid_export=all(curves[m['curve']].get('_valid_export',True)for m in members),
            process_boundaries_preserved_in_members=True,geometry_diagnostic_not_formal_acceptance=True))
    assert sum(len(r['members'])for r in out)==len(curves)
    return out
