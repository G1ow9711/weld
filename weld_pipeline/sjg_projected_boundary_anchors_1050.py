"""Anchor proposals from frozen predicted boundary points on input faces."""
import numpy as np
from weld_pipeline.sjg_geometry_query_bank_1025 import basis

class ProjectedBoundaryAnchors:
    def __init__(self,xyz,boundary,faces,support,radius_mm=1.):
        self.xyz=xyz;self.boundary=boundary;self.faces=faces;self.support=support;self.radius=float(radius_mm)
        self.cache={};self.point_candidates=0;self.near_curve_points=0;self.anchor_count=0
    def face_ids(self,index):
        if index not in self.cache:
            ids=self.support[self.faces[index]['point_ids_key']]
            self.cache[index]=ids[self.boundary[ids]>=.5]
        return self.cache[index]
    def family(self,family,face_indices):
        groups=[self.face_ids(int(i)) for i in face_indices]
        ids=np.unique(np.concatenate(groups)) if groups else np.empty(0,np.int64)
        self.point_candidates+=len(ids)
        if not len(ids):return []
        d=np.asarray(family['descriptor']);relative=np.asarray(self.xyz[ids],np.float64)-d[:3]
        if family['kind']=='line':
            q=relative@d[3:];distance=np.sqrt(np.maximum(0,np.square(relative).sum(1)-q*q))
        else:
            normal=d[3:6];u,v=basis(normal);height=relative@normal;radial=np.sqrt(np.maximum(0,np.square(relative).sum(1)-height*height))
            distance=np.hypot(height,radial-d[6]);q=(np.arctan2(relative@v,relative@u)%(2*np.pi))*d[6]
        q=q[distance<=self.radius];self.near_curve_points+=len(q);anchors=[]
        for lo,hi in family['domains_mm']:
            points=np.unique(q[(q>=lo)&(q<=hi)])
            if not len(points):continue
            gaps=np.diff(points);spacing=gaps[gaps>1e-5]
            separation=max(1e-4,2*float(np.median(spacing))) if len(spacing) else 1e-4
            starts=np.r_[0,np.flatnonzero(gaps>separation)+1];ends=np.r_[starts[1:]-1,len(points)-1]
            anchors.extend(points[starts].tolist());anchors.extend(points[ends].tolist())
        self.anchor_count+=len(anchors);return anchors
    def metadata(self):
        return dict(boundary_score_threshold=.5,curve_tube_radius_mm=self.radius,original_point_features_only=True,
            grouping='endpoints of projected boundary-evidence runs; separation=2*median positive input spacing',
            candidate_point_visits=self.point_candidates,near_curve_point_visits=self.near_curve_points,
            proposed_anchors=self.anchor_count,GT_reads=0,projected_runs_not_declared_physical_welds_or_gaps=True)

def controls():
    x=np.r_[np.arange(40),np.arange(61,101)].astype(float);xyz=np.c_[x,np.zeros_like(x),np.zeros_like(x)]
    xyz=np.r_[xyz,[[50,5,0]]];support={'p':np.arange(len(xyz))};face=[{'point_ids_key':'p'}]
    f=dict(kind='line',descriptor=[0,0,0,1,0,0],domains_mm=[[0,100]])
    provider=ProjectedBoundaryAnchors(xyz,np.ones(len(xyz)),face,support)
    assert provider.family(f,[0])==[0.,61.,39.,100.]
    return dict(projected_boundary_runs_keep_observed_break=True,off_curve_boundary_evidence_excluded=True,GT_reads=0)
