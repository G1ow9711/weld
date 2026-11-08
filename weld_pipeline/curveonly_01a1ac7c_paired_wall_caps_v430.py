"""Finite end-face certificates for opposite wall boundary endpoints."""
import types
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Line
from OCP.TopAbs import TopAbs_FACE,TopAbs_WIRE,TopAbs_EDGE,TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from . import curveonly_01a1ac7c_paired_wall_observations_v428 as previous
from . import curveonly_01a1ac7c_lower_face_observations_v403 as safe

observation,step,finite=previous.observation,previous.step,previous.finite
NUMERIC=previous.NUMERIC


def convex_caps(wall,context):
    if not hasattr(context,'paired_convex_cap_cache'):context.paired_convex_cap_cache={}
    if wall in context.paired_convex_cap_cache:return context.paired_convex_cap_cache[wall]
    result=[];g=context.geometry
    faces=list(finite.cad.items(g.solids[wall],TopAbs_FACE))
    for face,row in zip(faces,context.faces[wall]):
        if row['status']!='CANONICAL_PLANE' or not row.get('face_valid'):continue
        if len(list(finite.cad.items(face,TopAbs_WIRE)))!=1:continue
        edges=list(finite.cad.items(face,TopAbs_EDGE))
        if len(edges)<3 or any(BRepAdaptor_Curve(TopoDS.Edge_s(e)).GetType()!=GeomAbs_Line for e in edges):continue
        vertices=[]
        for vertex in finite.cad.items(face,TopAbs_VERTEX):
            p=finite.cad.vec(BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertex)))
            if not any(np.linalg.norm(p-q)<=NUMERIC for q in vertices):vertices.append(p)
        if len(vertices)<3:continue
        points=np.asarray(vertices);_,_,basis=np.linalg.svd(points-points.mean(0),full_matrices=False)
        try:hull=ConvexHull((points-points.mean(0))@basis[:2].T)
        except QhullError:continue
        area=float(row['area_mm2'])
        if area<=NUMERIC or abs(hull.volume-area)>NUMERIC*max(1.,area):continue
        result.append(dict(face=face,face_index=row['face_index'],normal=np.asarray(row['normal'])@g.R.T,
            area_mm2=area,convex_hull_area_mm2=float(hull.volume),vertices=len(vertices)))
    context.paired_convex_cap_cache[wall]=result
    return result


def endpoint_state(root,local,context):
    L=context.lengths[root];eps=previous.source_uncertainty(root,context)
    if abs(local)<=NUMERIC:end=0;exact=0.;probe=min(L/2.,eps)
    elif abs(local-L)<=NUMERIC:end=1;exact=L;probe=L-min(L/2.,eps)
    else:return None
    state=observation.station(root,probe,context)
    if state['status']!='POSITIVE_FINITE_BASE_FILLET':return None
    points=np.asarray(context.packet['candidates'][root]['points']);s=finite.cad.arclength(points)
    point,tangent=finite.point_tangent(points,s,exact)
    return dict(point=point,tangent_inward=tangent if end==0 else -tangent,
        state=state,endpoint=end,local=exact,uncertainty=eps)


def counterpart(outer,q,inner,peer_interval,context):
    initial=previous.counterpart(outer,q,inner,peer_interval,context)
    if initial['passed']:return initial
    root,local=observation.primitive(outer,q,context)
    if root is None:return initial
    start=endpoint_state(root,local,context)
    if start is None:return initial
    evidence=start['state']['evidence'];base,wall=evidence['base'],evidence['wall'];g=context.geometry
    _,top,axis=safe.panel_levels(base,context);matches=[]
    po=start['point'];native_o=(po-g.t)@g.R
    if abs(po@axis-top)>start['uncertainty']:return initial
    base_distance_o=safe.body_distance(g.solids[base],native_o)
    if base_distance_o is None or base_distance_o>start['uncertainty']:return initial
    for leaf in previous.leaves(inner,context):
        if leaf['material']!=wall:continue
        for local_i in (0.,context.lengths[leaf['root']]):
            end=endpoint_state(leaf['root'],local_i,context)
            if end is None:continue
            state_i=end['state']['evidence']
            if state_i['base']!=base or state_i['wall']!=wall:continue
            if np.asarray(evidence['free'])@np.asarray(state_i['free'])>=0.:continue
            pi=end['point'];native_i=(pi-g.t)@g.R;budget=start['uncertainty']+end['uncertainty']
            mapped=leaf['start_mm']+(context.lengths[leaf['root']]-local_i if leaf['reversed'] else local_i)
            if mapped<peer_interval['start_mm']-budget or mapped>peer_interval['end_mm']+budget:continue
            if abs(pi@axis-top)>end['uncertainty']:continue
            base_distance_i=safe.body_distance(g.solids[base],native_i)
            if base_distance_i is None or base_distance_i>end['uncertainty']:continue
            for cap in convex_caps(wall,context):
                n=cap['normal'];a=float(n@start['tangent_inward']);b=float(n@end['tangent_inward'])
                if abs(a)<=NUMERIC or abs(b)<=NUMERIC or a*b<=0.:continue
                do=safe.body_distance(cap['face'],native_o);di=safe.body_distance(cap['face'],native_i)
                if do is None or di is None or do>start['uncertainty'] or di>end['uncertainty']:continue
                matches.append(dict(passed=True,outer_candidate=outer,outer_native_s_mm=float(q),
                    outer_primitive=root,wall=wall,base=base,inner_primitive=leaf['root'],inner_station_mm=local_i,
                    peer_native_s_mm=float(mapped),cross_wall_depth_mm=float(np.linalg.norm(pi-po)),
                    cross_wall_exit_to_peer_error_mm=float(max(do,di)),source_uncertainty_budget_mm=budget,
                    opposite_free_dot=float(np.asarray(evidence['free'])@np.asarray(state_i['free'])),
                    method='ACTUAL_CONVEX_FINITE_END_FACE',finite_end_face_index=cap['face_index'],
                    finite_face_area_mm2=cap['area_mm2'],convex_hull_area_mm2=cap['convex_hull_area_mm2'],
                    finite_face_vertex_count=cap['vertices'],one_sided_curve_inward_dots=[a,b],
                    original_finite_base_distances_mm=[base_distance_o,base_distance_i],
                    outer_point_mm=po.tolist(),inner_point_mm=pi.tolist(),
                    previous_boundary_query_reason=initial.get('reason')))
    if len(matches)==1:return matches[0]
    return dict(initial,finite_end_face_alternative_matches=matches,
        finite_end_face_alternative_reason='NO_UNIQUE_COMPLETE_FINITE_CAP_CERTIFICATE')


def observe(decoded,features,packet,inventory,solids,cached_observations=None,verify=False):
    run=types.FunctionType(previous.observe.__code__,dict(previous.__dict__,counterpart=counterpart))
    result=run(decoded,features,packet,inventory,solids,cached_observations,verify)
    result['finite_end_face_boundary_proof_version']=430
    return result
