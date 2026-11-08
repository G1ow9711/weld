"""Input-only linear referencing of feature stations on their native polyline."""
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad

PROJECTION_NUMERIC_MM = 1e-7


def parameters(candidate):
    points=np.asarray(candidate['points'],float);query=np.asarray(candidate['stations'],float)
    old=np.asarray(candidate['station_s_mm'],float)
    signature=hashlib.sha256(points.tobytes()+query.tobytes()+old.tobytes()).hexdigest()
    cached=candidate.get('_native_station_reference_v154')
    if cached is not None and cached.get('input_sha256')==signature:
        return np.asarray(cached['native_s_mm']),cached['receipt']
    arc=cad.arclength(points);length=float(arc[-1])
    assert len(query)==len(old) and length>0 and old[-1]>0 and np.all(np.diff(old)>0)
    a=points[:-1];v=np.diff(points,axis=0);den=(v*v).sum(1)
    valid=den>0;ids=np.flatnonzero(valid);a=a[valid];v=v[valid];den=den[valid]
    center=a+v/2;radius=np.sqrt(den)/2;tree=cKDTree(center)
    _,initial=tree.query(query,workers=1)
    t=np.clip(((query-a[initial])*v[initial]).sum(1)/den[initial],0,1)
    first=np.linalg.norm(query-a[initial]-t[:,None]*v[initial],axis=1)
    # Any segment closer than the initial exact distance has its midpoint
    # within max(segment half-length)+that distance. No nearest-k heuristic.
    neighbors=tree.query_ball_point(query,float(radius.max())+first+1e-9,workers=1)
    count=np.array([len(q)for q in neighbors]);assert np.all(count>0)
    qi=np.repeat(np.arange(len(query)),count);si=np.concatenate(neighbors).astype(np.int64)
    tt=np.clip(((query[qi]-a[si])*v[si]).sum(1)/den[si],0,1)
    distance=np.linalg.norm(query[qi]-a[si]-tt[:,None]*v[si],axis=1)
    start=np.r_[0,np.cumsum(count)[:-1]];best=np.minimum.reduceat(distance,start)
    close=distance<=best[qi]+1e-9
    qi=qi[close];si=si[close];tt=tt[close]
    native=arc[ids[si]]+tt*np.sqrt(den[si])
    expected=old/old[-1]*length
    # Geometric ties at closed endpoints/self intersections use the existing
    # ordered station path, not workpiece IDs or any reference welds.
    order=np.lexsort((np.abs(native-expected[qi]),qi));ordered_q=qi[order]
    first=np.r_[True,np.diff(ordered_q)!=0];chosen=order[first]
    assert np.array_equal(qi[chosen],np.arange(len(query)))
    result=native[chosen]
    end_error=np.linalg.norm(query[[0,-1]]-points[[0,-1]],axis=1)
    assert np.all(end_error<=PROJECTION_NUMERIC_MM)
    result[0]=0.;result[-1]=length
    unadjusted=result.copy()
    step=4*float(np.spacing(length))
    for i in range(1,len(result)):
        if result[i]<=result[i-1]:result[i]=result[i-1]+step
    result[-1]=length
    for i in range(len(result)-2,-1,-1):
        if result[i]>=result[i+1]:result[i]=result[i+1]-step
    result[0]=0.
    adjustment=float(np.abs(result-unadjusted).max())
    assert adjustment<=PROJECTION_NUMERIC_MM,('Genuine nonmonotone station path',adjustment)
    assert np.all(np.diff(result)>0),('Nonmonotone native station reference',np.diff(result).min())
    assert float(best.max())<=PROJECTION_NUMERIC_MM,('Station not on native input curve',float(best.max()))
    uniform=np.linspace(0,length,len(query))
    receipt=dict(method='EXACT_POINT_SEGMENT_PROJECTION_WITH_BOUNDING_SPHERE_ENUMERATION',
        stations=len(query),native_length_mm=length,station_chord_length_mm=float(old[-1]),
        max_station_to_native_mm=float(best.max()),max_native_vs_chord_parameter_mm=float(np.abs(result-old).max()),
        max_native_vs_uniform_parameter_mm=float(np.abs(result-uniform).max()),
        numeric_order_adjustment_max_mm=adjustment,
        strictly_ordered=True,no_GT_pose_or_geometry_change=True)
    candidate['_native_station_reference_v154']=dict(native_s_mm=result.tolist(),receipt=receipt,input_sha256=signature)
    return result,receipt
