"""Faithful native B-spline sampling across all trimmed knot spans.

The old whole-edge C1 sampler can return only two points for a degree1 C0
polyline and thereby erase an entire curved profile. No label determines knots.
"""
import numpy as np
from OCP.GeomAbs import GeomAbs_BSplineCurve
from . import curveonly_01a1ac7c as cad

def densify(points,step=1.):
    p=np.array(points,float);s=cad.arclength(p);keep=np.r_[True,np.diff(s)>1e-10];p=p[keep];s=s[keep]
    if len(p)<2:return p
    grid=np.unique(np.r_[s,np.linspace(0,s[-1],max(2,int(np.ceil(s[-1]/step))+1))])
    return np.column_stack([np.interp(grid,s,p[:,i])for i in range(3)])

def curve_points(edge):
    ad=cad.BRepAdaptor_Curve(edge)
    if ad.GetType()!=GeomAbs_BSplineCurve:return cad.curve_points(edge)
    spline=ad.BSpline();a,b=ad.FirstParameter(),ad.LastParameter();degree=spline.Degree()
    poles=np.array([cad.vec(spline.Pole(i))for i in range(1,spline.NbPoles()+1)])
    delta=poles[np.argmax(np.linalg.norm(poles-poles[0],axis=1))]-poles[0];norm=np.linalg.norm(delta)
    if norm<1e-8:return None
    if np.linalg.norm(np.cross(poles-poles[0],delta/norm),axis=1).max()<1e-5:return None
    knots=sorted(set([a,b]+[spline.Knot(i)for i in range(1,spline.NbKnots()+1)if a<spline.Knot(i)<b]));points=[]
    if degree==1:
        points=[cad.vec(ad.Value(float(t)))for t in knots]
    else:
        for lo,hi in zip(knots[:-1],knots[1:]):
            if hi-lo<1e-12:continue
            # Each open knot span is smooth; mandatory quarter parameters also
            # retain inflection/loop evidence before the deflection refinement.
            quarters=np.linspace(lo,hi,5)
            for start,end in zip(quarters[:-1],quarters[1:]):
                sample=cad.GCPnts_QuasiUniformDeflection(ad,.025,float(start),float(end))
                if sample.IsDone()and sample.NbPoints()>=2:points.extend(cad.vec(sample.Value(i))for i in range(1,sample.NbPoints()+1))
                else:points.extend([cad.vec(ad.Value(float(start))),cad.vec(ad.Value(float(end)))])
    p=np.array(points)
    if len(p)<3:return None
    delta=p[-1]-p[0];norm=np.linalg.norm(delta)
    if norm>1e-8 and np.linalg.norm(np.cross(p-p[0],delta/norm),axis=1).max()<1e-5:return None
    return densify(p)
