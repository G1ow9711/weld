"""Finite material-relative height and nearest-other-material observations.

Material height extrema are whole-solid extents in the frozen base direction,
not local board heights, thicknesses, physical joint identities or weld rules.
"""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from .curveonly_01a1ac7c_finite_shadow_v73 import triangles,owner_at

NAMES=('owner_available_fraction','owner_below_signed_log_mm','owner_above_signed_log_mm',
       'other_available_fraction','other_below_signed_log_mm','other_above_signed_log_mm',
       'other_distance_log_median_mm','other_distance_log_max_mm')
FRACTIONS=np.linspace(0.,1.,5)
MESH_DEFLECTION_MM=.025


def signed_log(value):
    return float(np.sign(value)*np.log1p(abs(value)))


def build(packet,solids):
    start=time.perf_counter();out=np.zeros((len(packet['candidates']),len(NAMES)),np.float32)
    pose=packet.get('pose',{});transform=np.asarray(pose.get('transform_STEP_to_XYZ',[]),float)
    qualified=bool(solids and pose.get('passed') and transform.shape==(4,4)
        and np.linalg.norm(transform[:3,:3].T@transform[:3,:3]-np.eye(3))<1e-6)
    if not qualified:
        return dict(values=out.tolist(),names=list(NAMES),available=False,
            reason='NO_QUALIFIED_SOLID_POSE',new_geometry_queries=0,seconds=time.perf_counter()-start,
            physical_joint_or_weld_decision_accepted=False)
    base=np.asarray(packet['base_axis'],float);base/=np.linalg.norm(base)
    native_axis=transform[:3,:3].T@base
    heights=[];boxes=[]
    for solid in solids:
        # Make this observation independent of registration's earlier mesh state.
        cad.BRepTools.Clean_s(solid)
        vertices=triangles(solid,MESH_DEFLECTION_MM)
        projected=vertices@native_axis
        heights.append((float(projected.min()),float(projected.max())))
        boxes.append(cad.bbox(solid))
    boxes=np.asarray(boxes);observations=[];queries=0
    for ci,candidate in enumerate(packet['candidates']):
        probes=[]
        if not(candidate.get('STEP') and candidate.get('pose')):
            observations.append(dict(candidate=ci,probes=[],reason='NO_QUALIFIED_STEP_CARRIER'));continue
        points=np.asarray(candidate['points']);arc=cad.arclength(points);length=float(arc[-1])
        for fraction in FRACTIONS:
            s=float(fraction*length);world=np.array([np.interp(s,arc,points[:,j]) for j in range(3)])
            point=(world-transform[:3,3])@transform[:3,:3];height=float(point@native_axis)
            owner=owner_at(ci,s,packet);record=dict(fraction=float(fraction),native_s_mm=s,owner_material=owner)
            if not 0<=owner<len(solids):
                probes.append(dict(record,available=False));continue
            record.update(available=True,owner_below_mm=height-heights[owner][0],owner_above_mm=heights[owner][1]-height)
            lower=np.linalg.norm(np.maximum(np.maximum(boxes[:,:3]-point,point-boxes[:,3:]),0.),axis=1)
            best=np.inf;other=None
            for mi in np.argsort(lower,kind='stable'):
                mi=int(mi)
                if mi==owner:continue
                if lower[mi]>best+1e-9:break
                distance=cad.finite_distance(solids[mi],point);queries+=1
                if not np.isfinite(distance):raise ValueError('Nonfinite finite-material distance')
                if distance<best-1e-9 or (abs(distance-best)<=1e-9 and (other is None or mi<other)):
                    best=float(distance);other=mi
            if other is not None:
                record.update(other_material=other,other_distance_mm=best,
                    other_below_mm=height-heights[other][0],other_above_mm=heights[other][1]-height)
            probes.append(record)
        own=[q for q in probes if q['available']];other=[q for q in own if 'other_material' in q]
        out[ci,0]=len(own)/len(FRACTIONS);out[ci,3]=len(other)/len(FRACTIONS)
        if own:
            out[ci,1]=signed_log(float(np.median([q['owner_below_mm'] for q in own])))
            out[ci,2]=signed_log(float(np.median([q['owner_above_mm'] for q in own])))
        if other:
            out[ci,4]=signed_log(float(np.median([q['other_below_mm'] for q in other])))
            out[ci,5]=signed_log(float(np.median([q['other_above_mm'] for q in other])))
            d=[q['other_distance_mm'] for q in other];out[ci,6]=np.log1p(np.median(d));out[ci,7]=np.log1p(max(d))
        observations.append(dict(candidate=ci,probes=probes))
    assert np.isfinite(out).all()
    return dict(values=out.tolist(),names=list(NAMES),available=True,material_height_extrema_mm=heights,
        observations=observations,mesh_deflection_mm=MESH_DEFLECTION_MM,uniform_arclength_fractions=FRACTIONS.tolist(),
        mesh_initialization='CLEAN_PER_INVOCATION_TRIANGULATIONS_BREP_GEOMETRY_UNCHANGED',
        nearest_other_identity='DISCRETE_FINITE_GEOMETRY_WITNESS_NOT_CONFIRMED_JOINT',
        extrema_are_not_local_board_height_or_thickness=True,physical_joint_or_weld_decision_accepted=False,
        new_geometry_queries=queries,seconds=time.perf_counter()-start)
