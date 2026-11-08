"""Publish fitted circular curves with explicit consistent ARC30 parameters.

No reference geometry participates. The existing circle-fit qualification is
unchanged; conversion is accepted only within0.1mm of the published polyline.
General B-splines retain their own representation instead of an average radius.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad

MAX_CONVERSION_ERROR_MM=.1
TURN_LIMIT=math.pi/6

def distance_upper(query,p):
    p=np.asarray(p,float);q=np.asarray(query,float);tree=cKDTree(p)
    _,idx=tree.query(q,k=min(4,len(p)),workers=1);idx=np.asarray(idx).reshape(len(q),-1)
    segment=np.clip(np.concatenate([idx-1,idx],axis=1),0,len(p)-2)
    a=p[segment];v=p[segment+1]-a;delta=q[:,None,:]-a
    t=np.clip((delta*v).sum(2)/np.maximum((v*v).sum(2),1e-24),0,1)
    return np.sqrt(((delta-t[:,:,None]*v)**2).sum(2)).min(1)

def _fallback(parent):
    result=dict(parent);segments=[]
    for original in parent['segments']:
        stack=[np.asarray(original['points'],float)]
        while stack:
            p=stack.pop();fit=cad.circle_fit(p)
            if fit is not None and abs(fit['sweep'])>TURN_LIMIT:
                length=cad.arclength(p)[-1]
                if length<=1e-8:raise ValueError('A zero-length curve cannot have a certified circular turn')
                stack.append(cad.slice_curve(p,length/2,length));stack.append(cad.slice_curve(p,0,length/2));continue
            segment=dict(original,segment_index=len(segments),points=p.tolist(),length_mm=float(cad.arclength(p)[-1]),
                status='NOMINAL_ARC_TURN_UNCERTIFIED'if fit is None else 'NOMINAL_POLYLINE_ARC_FIT_TURN_CHECKED',
                reason='CIRCULAR_CONVERSION_ERROR_BOUND_EXCEEDED')
            segments.append(segment)
    result['segments']=segments;result['arc_publication_status']='CONVERSION_UNCERTIFIED_ORIGINAL_GEOMETRY_RETAINED'
    return result

def circular_parent(parent):
    fit=parent.get('circle')
    if fit is None:return dict(parent),dict(projected=False,general_curve=True)
    center=np.asarray(fit['center']);radius=float(fit['radius']);normal=np.asarray(fit['normal']);u=np.asarray(fit['start']);v=np.cross(normal,u);sweep=float(fit['sweep'])
    if radius<=0 or not np.isfinite([radius,sweep]).all():raise ValueError('Invalid predicted circle')
    count=max(1,int(math.ceil(abs(sweep)/TURN_LIMIT)))
    angles=np.linspace(0.,sweep,count+1)
    while abs(np.diff(angles)).max()>TURN_LIMIT:
        count+=1;angles=np.linspace(0.,sweep,count+1)
    segments=[];allpoints=[]
    for index,(start,end)in enumerate(zip(angles[:-1],angles[1:])):
        delta=float(end-start);n=max(3,int(math.ceil(abs(delta)*radius))+1,int(math.ceil(abs(delta)/.02))+1)
        grid=np.linspace(start,end,n);points=center+radius*(np.cos(grid)[:,None]*u+np.sin(grid)[:,None]*v)
        start_direction=np.cos(start)*u+np.sin(start)*v
        parameters=dict(schema_version='contract_3d_circular_arc_chord_v1',center_mm=center.tolist(),radius_mm=radius,
            plane_normal=normal.tolist(),start_direction=start_direction.tolist(),sweep_rad=delta,is_closed=False,units='mm',angle_units='rad',
            parameterization='C + R*(cos(sign(sweep)*s/R)*U + sin(sign(sweep)*s/R)*(N cross U))')
        segments.append(dict(segment_index=index,points=points.tolist(),length_mm=radius*abs(delta),geometry_type='arc',
            arc_parameters=parameters,status='NOMINAL_CIRCULAR_PARAMETERS_AND_TURN_CERTIFIED',reason='ARC_TURN_AT_MOST_30_DEGREES'))
        allpoints.extend(points if not allpoints else points[1:])
    new=np.array(allpoints);old=np.asarray(parent['points'],float)
    a=float(distance_upper(old,new).max());b=float(distance_upper(new,old).max());error=max(a,b)
    if error>MAX_CONVERSION_ERROR_MM:
        return _fallback(parent),dict(projected=False,general_curve=False,conversion_error_mm=error,uncertified=True)
    result=dict(parent,points=new.tolist(),segments=segments,arc_publication_status='PARAMETRIC_CIRCLE_WITH_LEGAL_TURN',
        source_polyline_conversion_error_upper_mm=error,source_circle_fit=fit)
    result['circle']=dict(fit,max_fit_error_mm=0.)
    return result,dict(projected=True,general_curve=False,conversion_error_mm=error,segments=len(segments),maximum_turn_rad=float(abs(np.diff(angles)).max()))

def publish(decoded):
    parents=[];records=[]
    for parent in decoded['parents']:
        updated,record=circular_parent(parent);parents.append(updated);records.append(record)
    return dict(decoded,parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments'])for p in parents),
        circular_publication=dict(records=records,conversion_limit_mm=MAX_CONVERSION_ERROR_MM,arc_turn_limit_rad=TURN_LIMIT,
            labels_or_reference_fitted=False,general_curves_not_forced_to_average_circle=True))
