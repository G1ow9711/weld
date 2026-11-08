"""Observe native upright-owned curves over finite supporting base material.

This is an input-only observation kernel. It publishes no welds or process cuts.
Native compound paths may change wall material while retaining their provenance.
"""
import numpy as np
from . import curveonly_01a1ac7c_step_joint_rules_v373 as step

finite=step.finite
NUMERIC=step.NUMERIC


class UnresolvedSection(RuntimeError):pass


def section(sec,material,point,direction):
    result=step.connected(sec,material,point,direction)
    if not sec.geometry.intersectors[material].IsDone():
        raise UnresolvedSection('FINITE_INTERSECTOR_NOT_DONE')
    return result


def primitive(ci,q,context):
    path=context.packet['candidates'][ci].get('source_material_path')
    if not path:return ci,q
    for row in path:
        if row['start_mm']-NUMERIC<=q<=row['end_mm']+NUMERIC:
            local=np.clip(q-row['start_mm'],0.,row['end_mm']-row['start_mm'])
            if row['reversed']:local=context.lengths[row['curve']]-local
            return primitive(row['curve'],local,context)
    return None,None


def station(ci,q,context):
    root,local_q=primitive(ci,q,context)
    if root is None:return dict(status='UNKNOWN',reason='NATIVE_MATERIAL_CLEARANCE_HAS_NO_OWNER',q_mm=float(q))
    source=context.packet['candidates'][root];wall=source.get('material',-1);circle=source.get('circle')
    if wall<0 or wall in context.panels or circle is None:
        return dict(status='UNKNOWN',reason='NOT_A_RESOLVED_UPRIGHT_OWNED_CURVE',primitive=root,wall=wall,q_mm=float(q))
    points=np.asarray(context.packet['candidates'][ci]['points'])
    point,tangent=finite.point_tangent(points,finite.cad.arclength(points),q)
    eps=NUMERIC+float(circle.get('max_fit_error_mm',0.))+float(source.get('source_curve_discretization_mm',.025))
    positive=[];review=[]
    for base in context.panels:
        sec=context.view(base,1.);axis=sec.geometry.base
        normal=np.cross(tangent,axis);norm=np.linalg.norm(normal)
        if norm<=NUMERIC:continue
        normal/=norm
        try:
            base_intervals=section(sec,base,point,axis)
            below=[(lo,hi) for lo,hi in base_intervals if abs(hi)<=eps and lo<-eps]
            if len(below)!=1:
                native=(point-sec.geometry.t)@sec.geometry.R
                distance,_=finite.events.native.finite_distance(sec.geometry.solids[base],native)
                review.append(dict(base=base,reason='NO_UNIQUE_BASE_TOP_SECTION_OBSERVED',
                    base_intervals_mm=base_intervals,original_finite_body_distance_mm=float(distance),
                    not_a_process_gap_certificate=True));continue
            bottom,top=below[0];thickness=top-bottom
            edge_vertical=section(sec,wall,point,axis)
            above=[(lo,hi) for lo,hi in edge_vertical if lo<=eps and hi>eps]
            if len(above)!=1:
                review.append(dict(base=base,reason='NO_UNIQUE_WALL_ABOVE_SOURCE',wall_intervals_mm=edge_vertical));continue
            wall_bottom,wall_top=above[0]
            height=min(wall_top/2.,thickness/2.,float(circle['radius'])/4.)
            if height<=eps:continue
            wall_intervals=section(sec,wall,point+height*axis,normal)
            sides=[]
            for a,b in wall_intervals:
                if abs(a)<=eps and b>eps:sides.append((normal,b-a))
                elif abs(b)<=eps and a<-eps:sides.append((-normal,b-a))
            if len(sides)!=1:
                review.append(dict(base=base,reason='NO_UNIQUE_WALL_MATERIAL_SIDE',wall_intervals_mm=wall_intervals));continue
            inward,depth=sides[0];offset=min(depth/2.,thickness/2.,float(circle['radius'])/4.)
            if offset<=eps:continue
            inside_vertical=section(sec,wall,point+inward*offset,axis)
            local=[(lo,hi) for lo,hi in inside_vertical if lo<=eps and hi>height]
            if len(local)!=1:
                review.append(dict(base=base,reason='WALL_INTERIOR_NOT_CONTINUOUS',wall_intervals_mm=inside_vertical));continue
            free=-inward;free_point=point+free*offset
            free_base=section(sec,base,free_point,axis)
            supported=[(lo,hi) for lo,hi in free_base if abs(hi)<=eps and abs(lo-bottom)<=eps]
            if len(supported)!=1 or sec.geometry.inside(wall,free_point+height*axis) or sec.geometry.inside(base,free_point+height*axis):
                review.append(dict(base=base,reason='FREE_FILLET_QUADRANT_NOT_SUPPORTED',free_base_intervals_mm=free_base));continue
            positive.append(dict(base=base,wall=wall,source_primitive=root,base_thickness_mm=float(thickness),
                wall_depth_mm=float(depth),wall_vertical_interval_mm=list(map(float,local[0])),
                adaptive_probe_height_mm=float(height),adaptive_interior_offset_mm=float(offset),
                numerical_source_uncertainty_mm=float(eps),free=free.tolist()))
        except UnresolvedSection as exc:review.append(dict(base=base,reason=str(exc),unknown=True))
    if len(positive)==1:
        return dict(status='POSITIVE_FINITE_BASE_FILLET',q_mm=float(q),evidence=positive[0],other_base_review=review)
    return dict(status='UNKNOWN' if positive or any(v.get('unknown') for v in review) else 'NO_POSITIVE_JOINT_OBSERVED',
        q_mm=float(q),source_primitive=root,wall=wall,positive=positive,review=review,
        no_absence_or_should_weld_or_process_cut_inferred=True)


def observe(ci,context):
    row=context.inventory['candidates'][ci];c=context.packet['candidates'][ci]
    nodes=sorted(set([0.,context.lengths[ci]]+[float(q) for q in row['qualified_geometry_nodes_mm']]+
        [float(m[key]) for m in c.get('source_material_path',[]) for key in ('start_mm','end_mm')]))
    cells=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        if b-a<=NUMERIC:continue
        cells.append(dict(start_mm=a,end_mm=b,midpoint=station(ci,(a+b)/2.,context)))
    positive=finite.union([(v['start_mm'],v['end_mm']) for v in cells if v['midpoint']['status']=='POSITIVE_FINITE_BASE_FILLET'])
    return dict(candidate=ci,cells=cells,positive_midpoint_hypothesis_domains_mm=positive,
        not_dense_verified=True,not_final_model_output=True,no_geometry_alone_should_weld_permission=True)
