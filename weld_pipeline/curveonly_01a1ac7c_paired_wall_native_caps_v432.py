"""Finite caps with certified straight edges independent of STEP curve type."""
import types
import numpy as np
from OCP.GeomAbs import GeomAbs_Line,GeomAbs_BSplineCurve,GeomAbs_BezierCurve
from OCP.GeomConvert import GeomConvert
from OCP.TopoDS import TopoDS
from OCP.TopAbs import TopAbs_FACE,TopAbs_WIRE,TopAbs_EDGE,TopAbs_VERTEX
from . import curveonly_01a1ac7c_paired_wall_caps_v430 as previous
from . import curveonly_01a1ac7c_sampler_v5 as sampler

finite,NUMERIC,safe=previous.finite,previous.NUMERIC,previous.safe


def linear_edge(edge):
    adaptor=previous.BRepAdaptor_Curve(TopoDS.Edge_s(edge));kind=adaptor.GetType()
    if kind==GeomAbs_Line:return dict(type=str(kind),error_upper_mm=0.,positive_weights=True)
    if kind not in (GeomAbs_BSplineCurve,GeomAbs_BezierCurve):return None
    spline=adaptor.BSpline() if kind==GeomAbs_BSplineCurve else GeomConvert.CurveToBSplineCurve_s(adaptor.Bezier())
    a,b=float(adaptor.FirstParameter()),float(adaptor.LastParameter())
    if not np.isfinite([a,b]).all() or b<=a:return None
    controls=[];starts=[];ends=[]
    for arc in sampler._arcs(spline,a,b):
        p=np.asarray([finite.cad.vec(arc.Pole(i)) for i in range(1,arc.NbPoles()+1)])
        w=np.asarray([arc.Weight(i) for i in range(1,arc.NbPoles()+1)])
        if not np.isfinite(p).all() or not np.isfinite(w).all() or np.min(w)<=0.:return None
        control=np.c_[p*w[:,None],w]
        if not np.isfinite(control).all():return None
        controls.append(control);starts.append(p[0]);ends.append(p[-1])
    if not controls:return None
    if np.linalg.norm(starts[0]-finite.cad.vec(adaptor.Value(a)))>NUMERIC:return None
    if np.linalg.norm(ends[-1]-finite.cad.vec(adaptor.Value(b)))>NUMERIC:return None
    if any(np.linalg.norm(x-y)>NUMERIC for x,y in zip(ends[:-1],starts[1:])):return None
    error,monotone=sampler._bound(np.vstack(controls))
    if error>NUMERIC or not monotone:return None
    return dict(type=str(kind),error_upper_mm=float(error),positive_weights=True,
        Bezier_spans=len(controls),control_hull_and_monotone_projection=True)


def convex_caps(wall,context):
    if not hasattr(context,'paired_native_convex_cap_cache'):context.paired_native_convex_cap_cache={}
    if wall in context.paired_native_convex_cap_cache:return context.paired_native_convex_cap_cache[wall]
    result=[];g=context.geometry
    for face,row in zip(finite.cad.items(g.solids[wall],TopAbs_FACE),context.faces[wall]):
        if row['status']!='CANONICAL_PLANE' or not row.get('face_valid'):continue
        if len(list(finite.cad.items(face,TopAbs_WIRE)))!=1:continue
        edges=[linear_edge(e) for e in finite.cad.items(face,TopAbs_EDGE)]
        if len(edges)<3 or any(e is None for e in edges):continue
        vertices=[]
        for vertex in finite.cad.items(face,TopAbs_VERTEX):
            p=finite.cad.vec(previous.BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertex)))
            if not any(np.linalg.norm(p-q)<=NUMERIC for q in vertices):vertices.append(p)
        if len(vertices)<3:continue
        points=np.asarray(vertices);_,_,basis=np.linalg.svd(points-points.mean(0),full_matrices=False)
        try:hull=previous.ConvexHull((points-points.mean(0))@basis[:2].T)
        except previous.QhullError:continue
        area=float(row['area_mm2'])
        if area<=NUMERIC or abs(hull.volume-area)>NUMERIC*max(1.,area):continue
        result.append(dict(face=face,face_index=row['face_index'],normal=np.asarray(row['normal'])@g.R.T,
            area_mm2=area,convex_hull_area_mm2=float(hull.volume),vertices=len(vertices),
            native_edge_linearity_certificates=edges))
    context.paired_native_convex_cap_cache[wall]=result
    return result


def counterpart(outer,q,inner,peer_interval,context):
    run=types.FunctionType(previous.counterpart.__code__,dict(previous.__dict__,convex_caps=convex_caps))
    return run(outer,q,inner,peer_interval,context)


def observe(decoded,features,packet,inventory,solids,cached_observations=None,verify=False):
    source=previous.previous
    run=types.FunctionType(source.observe.__code__,dict(source.__dict__,counterpart=counterpart))
    result=run(decoded,features,packet,inventory,solids,cached_observations,verify)
    result['finite_end_face_boundary_proof_version']=432
    result['native_Bspline_edges_certified_by_positive_control_hulls']=True
    return result
