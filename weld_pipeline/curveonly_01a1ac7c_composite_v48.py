"""Append finite curve paths while preserving all original input candidates.

Local finite support permits geometric continuation proposals. Exact circular
material contours without a joint witness are admitted only as closed loops,
with that UNKNOWN retained. No source continuation declares a weld/process cut.
"""
import time,math
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_geometry_units_v26 as geometry
from . import curveonly_01a1ac7c_contact_relations_v47 as relations
from .curveonly_01a1ac7c_set_v18 import reverse_features

ENDPOINT_MM=relations.TOLERANCE_MM
TANGENT_DEGREES=5.
EXACT_CONTOUR_ENDPOINT_MM=1e-6


def same_circle(a,b):
    x,y=a.get('circle'),b.get('circle')
    if x is None or y is None:return False
    return bool(np.linalg.norm(np.asarray(x['center'])-y['center'])<=.001 and abs(x['radius']-y['radius'])<=.001
        and abs(np.asarray(x['normal'])@y['normal'])>=1-1e-8)


def append(features,packet,context):
    start=time.perf_counter();original=packet['candidates'];n=len(original)
    summary=dict(original_candidates=n,new_candidates=0,local_finite_face_queries=0,
        original_features_and_geometry_preserved=True,new_point_neighborhood_queries=0,
        endpoint_proposal_radius_mm=ENDPOINT_MM,maximum_tangent_deflection_deg=TANGENT_DEGREES,
        should_weld_or_process_continuity_accepted=False)
    if context is None or n<2:
        return features,dict(packet,finite_continuation=dict(summary,seconds=time.perf_counter()-start))
    points=[];owners=[]
    for ci,c in enumerate(original):
        p=np.asarray(c['points']);mi=c.get('material',-1)
        if mi<0 or not c.get('STEP') or not c.get('pose') or np.linalg.norm(p[0]-p[-1])<.1:continue
        for end in (0,1):points.append(p[0 if end==0 else -1]);owners.append((ci,end))
    choices={};evidence={};queried=0;circle_only=0
    if points:
        tree=cKDTree(points)
        for left,right in sorted(tree.query_pairs(ENDPOINT_MM)):
            ci,ae=owners[left];cj,be=owners[right]
            if ci==cj:continue
            a,b=original[ci],original[cj]
            if a['material']==b['material']:continue
            u,v=geometry.tangent(a,ae),geometry.tangent(b,be)
            if u is None or v is None or -u@v<math.cos(math.radians(TANGENT_DEGREES)):continue
            gap=float(np.linalg.norm(np.asarray(points[left])-points[right]));queried+=1
            relation=relations.relation(a,b,ae,be,context)
            if relation is None:
                if gap>EXACT_CONTOUR_ENDPOINT_MM or not same_circle(a,b):continue
                relation=dict(kind='FINITE_CIRCULAR_MATERIAL_CONTOUR_NO_JOINT_CERTIFICATE',
                    source_materials=[a['material'],b['material']],nominal_endpoint_clearance_mm=gap,
                    should_weld_and_process_continuity=False);circle_only+=1
            aa,bb=(ci,ae),(cj,be)
            choices.setdefault(aa,[]).append(bb);choices.setdefault(bb,[]).append(aa)
            evidence[tuple(sorted([aa,bb]))]=dict(left=aa,right=bb,
                tangent_deflection_deg=float(math.degrees(math.acos(np.clip(-u@v,-1,1)))),
                endpoint_clearance_mm=gap,relation=relation)
    links={a:v[0] for a,v in choices.items() if len(v)==1 and len(choices.get(v[0],[]))==1}
    visited=set();new=[];blocks=[]
    for seed in range(n):
        if seed in visited or not any((seed,e) in links for e in (0,1)):continue
        component={seed};todo=[seed]
        while todo:
            ci=todo.pop()
            for end in (0,1):
                if (ci,end) not in links:continue
                cj,_=links[ci,end]
                if cj not in component:component.add(cj);todo.append(cj)
        visited.update(component)
        if len(component)<2:continue
        free=[(ci,e) for ci in sorted(component) for e in (0,1) if (ci,e) not in links]
        ci,end=free[0] if free else(min(component),0);cycle=not bool(free)
        members=[];parts=[];station_parts=[];observations=[];joins=[]
        while ci not in {m['curve'] for m in members}:
            c=original[ci];a,b=packet['offsets'][ci:ci+2];x=np.asarray(features[a:b],np.float32).copy()
            p=np.asarray(c['points'],float);stations=np.asarray(c['stations'],float)
            if end:p=p[::-1];stations=stations[::-1];x=reverse_features(x)
            if station_parts and np.linalg.norm(station_parts[-1][-1]-stations[0])<1e-9:
                stations=stations[1:];x=x[1:]
            parts.append(p);station_parts.append(stations);observations.append(x)
            members.append(dict(curve=ci,material=c['material'],reversed=bool(end),original_native_members=c.get('original_native_members')))
            if (ci,1-end) not in links:break
            nxt=links[ci,1-end];joins.append(evidence[tuple(sorted([(ci,1-end),nxt]))]);ci,end=nxt
        assert {m['curve'] for m in members}==component
        p=np.vstack(parts);stations=np.vstack(station_parts);x=np.concatenate(observations)
        closing=float(np.linalg.norm(p[-1]-p[0])) if cycle else 0.
        if cycle and closing>0.:
            p=np.vstack([p,p[0]])
            if np.linalg.norm(stations[-1]-stations[0])>0.:
                stations=np.vstack([stations,stations[0]]);x=np.vstack([x,x[0]])
        fit=cad.circle_fit(p)
        weak=any(j['relation']['kind']=='FINITE_CIRCULAR_MATERIAL_CONTOUR_NO_JOINT_CERTIFICATE' for j in joins)
        if weak and (not cycle or fit is None):continue
        s=cad.arclength(stations);length=float(cad.arclength(p)[-1]);first=original[members[0]['curve']]
        scale=float(np.expm1(x[0,0])/max(x[0,33],1e-12))
        tangent=np.gradient(stations,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        turn=abs(fit['sweep']) if fit else float(np.arccos(np.clip((tangent[1:]*tangent[:-1]).sum(1),-1,1)).sum())
        x[:,0]=np.log1p(length);x[:,1]=np.log1p(fit['radius']) if fit else 0.;x[:,2]=turn
        x[:,3]=fit is not None;x[:,4]=np.linalg.norm(p[0]-p[-1])<.1
        x[:,9]=s/max(s[-1],1e-12);x[:,10]=np.log1p(np.minimum(s,s[-1]-s));x[:,33]=length/max(scale,1.)
        cursor=0.;path=[];clearances=[]
        for i,(m,part) in enumerate(zip(members,parts)):
            if i:
                gap=float(np.linalg.norm(parts[i-1][-1]-part[0]))
                clearances.append(dict(start_mm=cursor,end_mm=cursor+gap,length_mm=gap,
                    source='MEASURED_NOMINAL_MATERIAL_EDGE_CLEARANCE',weld_gap_or_process_cut_status='UNKNOWN_NOT_AUTOMATIC'))
                cursor+=gap
            L=float(cad.arclength(part)[-1]);path.append(dict(start_mm=cursor,end_mm=cursor+L,**m));cursor+=L
        if cycle and closing>0.:
            clearances.append(dict(start_mm=cursor,end_mm=cursor+closing,length_mm=closing,
                source='MEASURED_NOMINAL_CLOSING_CLEARANCE',weld_gap_or_process_cut_status='UNKNOWN_NOT_AUTOMATIC'));cursor+=closing
        assert abs(cursor-length)<1e-7*max(1.,length)
        candidate=dict(points=p.tolist(),stations=stations.tolist(),station_s_mm=s.tolist(),circle=fit,
            material=-2,STEP=True,pose=True,material_extent=first.get('material_extent',[0,0,0]),
            other_distance=[float(x[0,12]),float(x[len(x)//2,13]),float(x[-1,14])],
            composite_carrier_members=members,source_material_path=path,source_connections=joins,
            nominal_clearance_intervals=clearances,source_curve_count=sum(original[m['curve']].get('source_curve_count',1) for m in members),
            source_length_mm=sum(float(cad.arclength(q)[-1]) for q in parts),
            nominal_clearance_length_mm=sum(g['length_mm'] for g in clearances),
            joint_identity_accepted=False,process_continuity_accepted=False,
            source_kind='FINITE_MATERIAL_CONTOUR_CANDIDATE' if weak else'LOCALLY_SUPPORTED_FINITE_COMPOSITE_CANDIDATE')
        assert x.shape==(len(stations),34) and np.isfinite(x).all()
        new.append(candidate);blocks.append(x)
    offsets=list(packet['offsets'])
    for x in blocks:offsets.append(offsets[-1]+len(x))
    summary.update(new_candidates=len(new),local_finite_face_queries=context.calls,near_endpoint_pairs_tested=queried,
        contour_only_endpoint_links=circle_only,ambiguous_endpoint_count=sum(len(v)>1 for v in choices.values()),
        seconds=time.perf_counter()-start)
    out=np.concatenate([features,*blocks]) if blocks else features
    assert np.array_equal(out[:len(features)],features)
    return out,dict(packet,candidates=original+new,offsets=offsets,finite_continuation=summary)
