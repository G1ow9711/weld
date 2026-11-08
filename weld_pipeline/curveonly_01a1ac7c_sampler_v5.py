"""Bounded positive-weight rational Bezier sampling of native trimmed curves.

Each accepted chord contains its Bezier control polygon within TOLERANCE_MM.
Positive weights and monotone chord projection give a bidirectional chord
bound in floating-point geometry, assuming the kernel's Bezier conversion.
No label, weld file, fitted model or workpiece identifier is read.
"""
from __future__ import annotations
import numpy as np
from . import curveonly_01a1ac7c as cad
from OCP.GeomAbs import GeomAbs_BSplineCurve,GeomAbs_BezierCurve
from OCP.GeomConvert import GeomConvert_BSplineCurveToBezierCurve

TOLERANCE_MM=.025
AUDIT=[]

def reset_audit():AUDIT.clear()

def audit():
    return dict(method='positive_weight_Bezier_control_polygon_subdivision',
        requested_chord_bound_mm=TOLERANCE_MM,curves=list(AUDIT),
        finite_precision_and_native_conversion_assumed=True,
        labels_or_learned_parameters=False)

def _split(control):
    value=control.copy();left=[value[0].copy()];right=[value[-1].copy()]
    while len(value)>1:
        value=.5*(value[:-1]+value[1:]);left.append(value[0].copy());right.append(value[-1].copy())
    return np.array(left),np.array(right[::-1])

def _bound(control):
    p=control[:,:3]/control[:,3,None];delta=p[-1]-p[0];den=float(delta@delta)
    if den<=1e-24:
        error=float(np.linalg.norm(p-p[0],axis=1).max())
        return error,error<=1e-10
    t=(p-p[0])@delta/den;nearest=p[0]+np.clip(t,0,1)[:,None]*delta
    error=float(np.linalg.norm(p-nearest,axis=1).max())
    monotone=bool(np.all(np.diff(t)>=-1e-10)and t.min()>=-1e-10 and t.max()<=1+1e-10)
    return error,monotone

def sample_bezier(poles,weights,tolerance=TOLERANCE_MM):
    p=np.asarray(poles,float);w=np.asarray(weights,float)
    if p.ndim!=2 or p.shape[1]!=3 or w.shape!=(len(p),)or not np.isfinite(p).all()or not np.isfinite(w).all()or np.min(w)<=0:
        raise ValueError('Native rational Bezier requires finite positive weights')
    w=w/w.max();control=np.c_[p*w[:,None],w];stack=[(control,0)];points=[p[0]];bounds=[];depth=0
    while stack:
        current,d=stack.pop();error,monotone=_bound(current);depth=max(depth,d)
        if error<=tolerance and monotone and d>=1:
            points.append(current[-1,:3]/current[-1,3]);bounds.append(error);continue
        if d>=40:raise ValueError('Native curve subdivision did not satisfy its declared chord bound')
        left,right=_split(current);stack.append((right,d+1));stack.append((left,d+1))
    return np.asarray(points),dict(chords=len(bounds),maximum_control_polygon_distance_mm=max(bounds,default=0.),maximum_depth=depth,positive_weights=True)

def densify(p,step=1.):
    p=np.asarray(p,float);s=cad.arclength(p);keep=np.r_[True,np.diff(s)>1e-10];p=p[keep];s=s[keep]
    if len(p)<2:return p
    grid=np.unique(np.r_[s,np.linspace(0,s[-1],max(2,int(np.ceil(s[-1]/step))+1))])
    return np.column_stack([np.interp(grid,s,p[:,i])for i in range(3)])

def _straight(p):
    direction=p[np.argmax(np.linalg.norm(p-p[0],axis=1))]-p[0];length=np.linalg.norm(direction)
    return bool(length<=1e-10 or np.linalg.norm(np.cross(p-p[0],direction/length),axis=1).max()<1e-5)

def _arcs(spline,a,b):
    lo,hi=float(spline.FirstParameter()),float(spline.LastParameter())
    eps=32*np.finfo(float).eps*max(1.,abs(lo),abs(hi),abs(a),abs(b))
    if abs(a-lo)<=eps:a=lo
    if abs(b-hi)<=eps:b=hi
    if a==lo and b==hi:
        conversion=GeomConvert_BSplineCurveToBezierCurve(spline)
        return [conversion.Arc(i)for i in range(1,conversion.NbArcs()+1)]
    intervals=[(a,b)]
    if spline.IsPeriodic():
        period=float(spline.Period());length=b-a
        if length>period+eps:raise ValueError('A native edge repeats a periodic curve more than once')
        start=lo+((a-lo)%period)
        if abs(start-hi)<=eps or abs(start-lo)<=eps:start=lo
        end=start+length
        if end<=hi+eps:intervals=[(start,min(end,hi))]
        else:intervals=[(start,hi),(lo,lo+(end-hi))]
    arcs=[]
    for start,end in intervals:
        if end-start<=eps:continue
        conversion=GeomConvert_BSplineCurveToBezierCurve(spline,start,end,min(1e-12,(end-start)*1e-4))
        arcs.extend(conversion.Arc(i)for i in range(1,conversion.NbArcs()+1))
    return arcs

def curve_points(edge):
    ad=cad.BRepAdaptor_Curve(edge);kind=ad.GetType()
    if kind not in (GeomAbs_BSplineCurve,GeomAbs_BezierCurve):return cad.curve_points(edge)
    a,b=float(ad.FirstParameter()),float(ad.LastParameter())
    if not np.isfinite([a,b]).all()or b<=a:raise ValueError('Invalid native curve trim parameters')
    if kind==GeomAbs_BSplineCurve:
        spline=ad.BSpline();poles=np.array([cad.vec(spline.Pole(i))for i in range(1,spline.NbPoles()+1)])
        if _straight(poles):return None
        arcs=_arcs(spline,a,b)
        degree=spline.Degree();knots=spline.NbKnots()
    else:
        # Trimming an analytic Bezier by its B-spline representation retains
        # the exact parameter interval before the same rational subdivision.
        from OCP.GeomConvert import GeomConvert
        spline=GeomConvert.CurveToBSplineCurve_s(ad.Bezier())
        arcs=_arcs(spline,a,b)
        degree=spline.Degree();knots=spline.NbKnots()
    result=[];records=[]
    for arc in arcs:
        p=np.array([cad.vec(arc.Pole(i))for i in range(1,arc.NbPoles()+1)])
        w=np.array([arc.Weight(i)for i in range(1,arc.NbPoles()+1)])
        sampled,record=sample_bezier(p,w)
        if result and np.linalg.norm(result[-1]-sampled[0])>1e-5:raise ValueError('Discontinuous native spline must not be bridged')
        result.extend(sampled if not result else sampled[1:]);records.append(record)
    p=np.array(result)
    if len(p)<2:raise ValueError('Empty native Bezier conversion')
    start_error=np.linalg.norm(p[0]-cad.vec(ad.Value(a)));end_error=np.linalg.norm(p[-1]-cad.vec(ad.Value(b)))
    if max(start_error,end_error)>1e-5:raise ValueError('Native Bezier placement or trim disagrees with its original edge')
    # A genuinely nonlinear control polygon must not be reclassified as a
    # straight edge merely because its curvature is below the chord tolerance.
    p=densify(p)
    AUDIT.append(dict(degree=degree,native_knots=knots,Bezier_spans=len(arcs),sampled_points=len(p),
        chords=sum(r['chords']for r in records),maximum_control_polygon_distance_mm=max(r['maximum_control_polygon_distance_mm']for r in records),
        native_endpoint_check_mm=float(max(start_error,end_error)),minimum_positive_weight_preserved=True))
    return p
