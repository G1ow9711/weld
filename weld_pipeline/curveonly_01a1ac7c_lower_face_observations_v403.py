"""Observe original material sections at published plate-edge hypotheses.

This kernel does not decide should-weld status or change an output. Published
arc points are evaluated analytically; explicit polyline points stay on their
ordered segments. Plate levels and probe offsets come from finite materials.
"""
import math
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from . import curveonly_01a1ac7c_step_joint_rules_v373 as step

finite=step.finite
NUMERIC=step.NUMERIC


def panel_levels(owner,context):
    panel=context.panels[owner];axis=panel['axis'];g=context.geometry
    rows=[context.inventory['faces'][i] for i in panel['dominant_faces']]
    values=[float(r['offset_mm']/(np.asarray(r['normal'])@g.R.T@axis)+g.t@axis) for r in rows]
    return min(values),max(values),axis


def published_station(parent):
    segment=max(parent['segments'],key=lambda s:s['length_mm'])
    if segment['geometry_type']=='arc':
        a=segment['arc_parameters'];center=np.asarray(a['center_mm']);u=np.asarray(a['start_direction'])
        n=np.asarray(a['plane_normal']);v=np.cross(n,u);theta=a['sweep_rad']/2.
        point=center+a['radius_mm']*(math.cos(theta)*u+math.sin(theta)*v)
        tangent=np.sign(a['sweep_rad'])*(-math.sin(theta)*u+math.cos(theta)*v)
        return point,tangent,dict(kind='ANALYTIC_PUBLISHED_ARC_MIDPOINT',segment_index=segment['segment_index'])
    if segment['geometry_type'] in ('bspline_polyline','polyline','line'):
        points=np.asarray(segment['points']);s=finite.cad.arclength(points)
        point,tangent=finite.point_tangent(points,s,float(s[-1])/2.)
        return point,tangent,dict(kind='DECLARED_PUBLISHED_POLYLINE_MIDPOINT',segment_index=segment['segment_index'])
    return None,None,dict(reason='UNSUPPORTED_PUBLISHED_CURVE_REPRESENTATION')


def body_distance(shape,native_point):
    vertex=BRepBuilderAPI_MakeVertex(finite.cad.gp_Pnt(*map(float,native_point))).Vertex()
    query=BRepExtrema_DistShapeShape(vertex,shape);query.Perform()
    if not query.IsDone() or query.NbSolution()<1:return None
    return float(query.Value())


def nearest_other(owner,point,context):
    g=context.geometry;native=(point-g.t)@g.R
    boxes=np.asarray([finite.cad.bbox(s) for s in g.solids])
    lower=np.linalg.norm(np.maximum(np.maximum(boxes[:,:3]-native,native-boxes[:,3:]),0.),axis=1)
    best=math.inf;mate=None;unresolved=[];queried=[]
    for mi in np.argsort(lower):
        if mi==owner:continue
        if lower[mi]>best+NUMERIC:continue
        distance=body_distance(g.solids[mi],native)
        if distance is None:unresolved.append(int(mi));continue
        queried.append(dict(material=int(mi),distance_mm=distance,original_body_bbox_lower_bound_mm=float(lower[mi])))
        if distance<best:best=distance;mate=int(mi)
    return dict(material=mate,distance_mm=best if mate is not None else None,queried=queried,
        unresolved_materials=unresolved,certified_nearest=not unresolved)


def connected(sec,material,point,direction):
    result=step.connected(sec,material,point,direction)
    return result if sec.geometry.intersectors[material].IsDone() else None


def at(parent,context):
    ci=parent['source_candidate'];c=context.packet['candidates'][ci];owner=c.get('material',-1)
    if owner not in context.panels:
        return dict(applicable=False,reason='NO_DOMINANT_PARALLEL_OWNER_PANEL',candidate=ci,owner=owner)
    lower,upper,axis=panel_levels(owner,context);thickness=upper-lower
    point,tangent,source=published_station(parent)
    if point is None:return dict(applicable=False,reason=source['reason'],candidate=ci,owner=owner)
    uncertainty=NUMERIC+float(c.get('source_curve_discretization_mm',.025))+max(
        float(parent.get('source_polyline_conversion_error_upper_mm',0.)),
        float(parent.get('source_conversion_continuous_error_upper_mm',0.)))
    height=float(point@axis);record=dict(candidate=ci,owner=owner,published_witness=source,point_mm=point.tolist(),
        panel_axis=axis.tolist(),owner_lower_level_mm=lower,owner_upper_level_mm=upper,
        owner_thickness_mm=thickness,point_above_lower_mm=height-lower,
        original_source_and_conversion_uncertainty_mm=uncertainty)
    if abs(height-lower)>uncertainty:
        return dict(record,applicable=False,reason='PUBLISHED_WITNESS_NOT_ON_OWNER_LOWER_FACE')
    nearest=nearest_other(owner,point,context);record['nearest_other_material']=nearest
    mate=nearest['material']
    if mate is None or not nearest['certified_nearest']:
        return dict(record,applicable=False,reason='UNKNOWN_OTHER_MATERIAL_QUERY')
    sec=context.view(owner,1.);normal=np.cross(tangent,axis);normal/=max(np.linalg.norm(normal),1e-300)
    own=connected(sec,owner,point+axis*thickness/2.,normal)
    if own is None:return dict(record,applicable=False,reason='OWNER_INTERSECTOR_NOT_DONE')
    choices=[]
    for a,b in own:
        if abs(a)<=uncertainty and b>uncertainty:choices.append((normal,b-a))
        elif abs(b)<=uncertainty and a < -uncertainty:choices.append((-normal,b-a))
    if len(choices)!=1:
        return dict(record,applicable=False,reason='NO_UNIQUE_OWNER_INTERIOR_SIDE',owner_sections_mm=own)
    interior,width=choices[0];outward=-interior
    lateral=connected(sec,mate,point+axis*thickness/2.,outward)
    if lateral is None:return dict(record,applicable=False,reason='MATE_INTERSECTOR_NOT_DONE')
    eligible=[(a,b) for a,b in lateral if b>uncertainty and a>=-uncertainty]
    record.update(mate=mate,mate_is_parallel_panel=mate in context.panels,mate_lateral_sections_mm=lateral,
        owner_interior=interior.tolist())
    if not eligible:return dict(record,applicable=False,reason='NO_LOCAL_MATE_INTERIOR_SECTION')
    a,b=min(eligible,key=lambda v:max(v[0],0.));offset=min(thickness/2.,width/2.,(b-a)/2.)
    if offset<=uncertainty:return dict(record,applicable=False,reason='UNRESOLVED_INTERIOR_PROBE')
    mate_point=point+outward*(max(a,0.)+offset);vertical=connected(sec,mate,mate_point,axis)
    if vertical is None:return dict(record,applicable=False,reason='VERTICAL_INTERSECTOR_NOT_DONE')
    ranges=[(lo,hi) for lo,hi in vertical if lo<=thickness/2.+uncertainty and hi>=thickness/2.-uncertainty]
    record.update(adaptive_probe_offset_mm=offset,mate_vertical_sections_mm=vertical)
    if len(ranges)!=1:return dict(record,applicable=False,reason='NO_UNIQUE_MATE_VERTICAL_SECTION')
    lo,hi=ranges[0];quadrants={}
    for label,origin in [('lower',point),('upper',point+axis*thickness)]:
        quadrants[label]=[]
        for side in (-1.,1.):
            for sign in (-1.,1.):
                q=origin+side*interior*offset+sign*axis*offset
                occupied=[mi for mi in (owner,mate) if sec.geometry.inside(mi,q)]
                quadrants[label].append(dict(owner_side=side,up_side=sign,occupied_materials=occupied))
    return dict(record,applicable=True,observation_only=True,mate_bottom_relative_mm=float(lo),
        mate_top_relative_mm=float(hi),material_quadrants=quadrants,
        not_a_should_weld_or_global_bottom_ban=True)
