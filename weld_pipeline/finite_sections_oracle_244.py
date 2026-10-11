"""Independent finite-domain oracle using oriented triangle-edge halfplanes.

This module imports no production section or interior-construction routine.
It checks sampled interpolation only, never filled-material or weld identity.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

VERSION='ORIENTED_EDGE_HALFPLANE_FINITE_SECTION_ORACLE_244_v1'


class UnresolvedOracle(ValueError):
    pass


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def edge_halfplane_sections(vertices_uv,point_uv,direction_uv):
    vertices=np.asarray(vertices_uv,dtype=np.float64)
    point=np.asarray(point_uv,dtype=np.float64)
    direction=np.asarray(direction_uv,dtype=np.float64)
    if (vertices.ndim!=3 or vertices.shape[1:]!=(3,2) or point.shape!=(2,) or direction.shape!=(2,)
            or not all(np.isfinite(x).all() for x in (vertices,point,direction))):
        raise ValueError('Finite N triangular UV vertices and UV line required')
    if np.linalg.norm(direction)==0:
        raise UnresolvedOracle('Projected direction is a point')
    winding=cross2(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
    if (winding==0).any():
        raise UnresolvedOracle('Degenerate original triangle retained as unresolved')
    orientation=np.sign(winding)
    lower=np.full(len(vertices),-np.inf);upper=np.full(len(vertices),np.inf)
    possible=np.ones(len(vertices),dtype=bool)
    for start,end in ((0,1),(1,2),(2,0)):
        a,b=vertices[:,start],vertices[:,end]
        reverse=(b[:,0]<a[:,0])|((b[:,0]==a[:,0])&(b[:,1]<a[:,1]))
        anchor=np.where(reverse[:,None],b,a)
        other=np.where(reverse[:,None],a,b)
        edge=other-anchor
        inside_orientation=orientation*np.where(reverse,-1.,1.)
        intercept=cross2(edge,point-anchor)*inside_orientation
        slope=cross2(edge,direction)*inside_orientation
        forward=slope>0;reverse=slope<0
        lower[forward]=np.maximum(lower[forward],-intercept[forward]/slope[forward])
        upper[reverse]=np.minimum(upper[reverse],-intercept[reverse]/slope[reverse])
        possible &= (slope!=0)|(intercept>=0)
    ids=np.flatnonzero(possible&(lower<=upper))
    intervals=np.column_stack((lower[ids],upper[ids]))
    if not np.isfinite(intervals).all():
        raise UnresolvedOracle('Infinite section of a finite triangle')
    return ids,intervals


def event_union(intervals):
    """Closed union by start/end events, preserving each positive gap."""
    rows=np.asarray(intervals,dtype=np.float64).reshape(-1,2)
    if not np.isfinite(rows).all() or (rows[:,0]>rows[:,1]).any():
        raise ValueError('Finite ordered interval bounds required')
    events=defaultdict(lambda:[0,0])
    for start,end in rows:
        events[float(start)][0]+=1;events[float(end)][1]+=1
    output=[];active=0;begin=None
    for position,(starts,ends) in sorted(events.items()):
        if active==0:
            begin=position
        active+=starts
        assert active>=ends
        active-=ends
        if active==0:
            output.append((begin,position))
    assert active==0
    return np.asarray(output,dtype=np.float64).reshape(-1,2)


def common_support_union(support_intervals):
    """Closed intersection using membership at all endpoint events."""
    unions=[event_union(rows) for rows in support_intervals]
    if not unions or any(len(rows)==0 for rows in unions):
        return np.empty((0,2))
    events=defaultdict(lambda:defaultdict(lambda:[0,0]))
    for number,rows in enumerate(unions):
        for start,end in rows:
            events[float(start)][number][0]+=1
            events[float(end)][number][1]+=1
    active=np.zeros(len(unions),dtype=np.int64);pieces=[];positions=sorted(events)
    for index,position in enumerate(positions):
        for number,(starts,ends) in events[position].items():
            active[number]+=starts
        if (active>0).all():
            pieces.append((position,position))
        for number,(starts,ends) in events[position].items():
            active[number]-=ends
        assert (active>=0).all()
        if index+1<len(positions) and (active>0).all():
            pieces.append((position,positions[index+1]))
    assert (active==0).all()
    return event_union(np.asarray(pieces,dtype=np.float64).reshape(-1,2))


def projected_box_by_corners(low,high,normal,offset):
    corners=np.array([[low[i] if not (bits>>i)&1 else high[i] for i in range(3)]
                      for bits in range(8)],dtype=np.float64)
    normal=np.asarray(normal,dtype=np.float64)
    projected=corners-(corners@normal-float(offset))[:,None]*normal
    return projected.min(axis=0),projected.max(axis=0)


def reference_midplane(a,da,b,db,parallel_cosine):
    """Numerical unit-plane bisector, with physical interpretation unknown."""
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    product=float(a@b)
    if abs(product)<parallel_cosine:
        raise UnresolvedOracle('Outside frozen parallel family')
    oriented=b if product>=0 else -b
    second_offset=db if product>=0 else -db
    if np.array_equal(a,oriented) and float(da)==float(second_offset):
        raise UnresolvedOracle('Coincident original planes')
    combined=a+oriented;denominator=float(np.linalg.norm(combined))
    return combined/denominator,(float(da)+float(second_offset))/denominator
