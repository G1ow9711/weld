"""Shared line/3D-circle carrier and point-query interface in local millimetres."""
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc

class CurveCarrier:
    def __init__(self,representation):
        self.kind=representation['kind']
        if self.kind=='line':
            self.ends=np.asarray(representation['endpoints_mm'],np.float64)
            if self.ends.shape!=(2,3) or not np.isfinite(self.ends).all():raise ValueError('Invalid line')
            self.length=float(np.linalg.norm(self.ends[1]-self.ends[0]))
            if self.length<=0:raise ValueError('Zero-length line')
            self.direction=(self.ends[1]-self.ends[0])/self.length
        elif self.kind=='arc':
            self.arc=CircularArc.from_parameters(representation['arc_parameters']);self.length=self.arc.length_mm
        else:raise ValueError('Unsupported geometric carrier type')

    def at(self,q):
        q=np.asarray(q,np.float64)
        if not np.isfinite(q).all() or np.any(q<0) or np.any(q>self.length):raise ValueError('Arc-length query outside carrier')
        if self.kind=='line':return self.ends[0]+q[...,None]*self.direction
        return self.arc.point_at(q)

    def tangent(self,q):
        points=self.at(q)
        if self.kind=='line':return np.broadcast_to(self.direction,points.shape).copy()
        radial=(points-self.arc.center_mm)/self.arc.radius_mm
        return np.sign(self.arc.sweep_rad)*np.cross(self.arc.plane_normal,radial)

    def project(self,xyz):
        p=np.asarray(xyz,np.float64)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite Nx3 XYZ required')
        if self.kind=='line':q=np.clip((p-self.ends[0])@self.direction,0,self.length)
        else:
            a=self.arc;relative=p-a.center_mm;second=np.cross(a.plane_normal,a.start_direction)
            angle=np.sign(a.sweep_rad)*np.arctan2(relative@second,relative@a.start_direction)
            angle=np.mod(angle,2*np.pi);span=abs(a.sweep_rad)
            q=np.minimum(angle,span)*a.radius_mm
            # Outside the open angular interval, nearest point is an endpoint.
            outside=angle>span
            first,last=self.at(np.array([0.,self.length]))
            nearer_start=np.linalg.norm(p-first,axis=1)<=np.linalg.norm(p-last,axis=1)
            q=np.where(outside,np.where(nearer_start,0.,self.length),q)
            q=np.clip(q,0,self.length)
        return q,np.linalg.norm(p-self.at(q),axis=1)

    def segment(self,lo,hi):
        if not 0<=lo<hi<=self.length:raise ValueError('Invalid interval')
        if self.kind=='line':return dict(kind='line',endpoints_mm=self.at(np.array([lo,hi])).tolist())
        return dict(kind='arc',arc_parameters=self.arc.subarc(lo,hi).to_parameters())

    def grid(self,step_mm=1.):
        if not np.isfinite(step_mm) or step_mm<=0:raise ValueError('Positive finite step required')
        return np.linspace(0,self.length,max(1,int(np.ceil(self.length/step_mm)))+1)

def curve_point_queries(representation,q,xyz,point_features,tree,support_normal=None):
    """Actual neural input for both types; no label, source ID or STEP arguments.

    Relative positions use the local curve frame. The learned embedding remains
    that of the supplied PTv3 backbone; full rotation invariance is not claimed.
    """
    curve=CurveCarrier(representation);q=np.asarray(q,np.float64);positions=curve.at(q);tangent=curve.tangent(q)
    if curve.kind=='arc':
        normal=np.broadcast_to(curve.arc.plane_normal,tangent.shape).copy()
    else:
        if support_normal is None:raise ValueError('Line queries need an observed support-plane normal')
        normal=np.broadcast_to(np.asarray(support_normal,float),tangent.shape).copy()
        normal-=np.einsum('ij,ij->i',normal,tangent)[:,None]*tangent
        norm=np.linalg.norm(normal,axis=1)
        if np.any(norm<1e-8):raise ValueError('Support normal parallel to line')
        normal/=norm[:,None]
    side=np.cross(normal,tangent);frame=np.stack([tangent,side,normal],axis=-1)
    if len(xyz)<16:raise ValueError('At least 16 observed points required')
    distance,ids=tree.query(positions,k=16,workers=2)
    relative=np.einsum('nki,nij->nkj',xyz[ids]-positions[:,None],frame)
    observed_normal=np.abs(np.einsum('nki,nij->nkj',point_features['normal'][ids],frame))
    patch=np.concatenate([point_features['embedding'][ids],observed_normal,
        point_features['boundary'][ids,None],np.clip(relative/50,-20,20)],axis=-1).astype(np.float32)
    curvature=0. if curve.kind=='line' else 1./curve.arc.radius_mm
    query=np.c_[q/curve.length,q/1000,(curve.length-q)/1000,distance[:,0]/50,
                 np.full(len(q),float(curve.kind=='arc')),np.full(len(q),curvature*100),
                 np.full(len(q),float(curve.kind=='arc' and curve.arc.is_closed))].astype(np.float32)
    assert patch.shape==(len(q),16,23) and query.shape==(len(q),7)
    return dict(patch=patch,query=query,positions_mm=positions,tangent=tangent,
                original_point_ids=ids.astype(np.int64),frame=frame)
