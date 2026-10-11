"""Dominant planes from ideal sampled XYZ; pure geometry, no material labels.

Peeling is discovery only: support is recovered from all original XYZ. Curved
and unresolved points remain explicit, and are not claimed to be absent.
"""
import numpy as np
from scipy.spatial import Delaunay,cKDTree,QhullError

def extract_planes(xyz,progress=None,maximum_planes=512):
    rng=np.random.default_rng(20261010);remaining=np.arange(len(xyz));hypotheses=[]
    for k in range(maximum_planes):
        if len(remaining)<32:break
        sample=xyz[rng.choice(remaining,min(8192,len(remaining)),replace=False)]
        triples=sample[rng.integers(len(sample),size=(512,3))]
        normals=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0]);length=np.linalg.norm(normals,axis=1);good=length>1e-6
        if not good.any():break
        normals=normals[good]/length[good,None];offsets=(triples[good,0]*normals).sum(1)
        scores=(abs(sample@normals.T-offsets)<.01).sum(0);best=int(scores.argmax());n=normals[best];offset=offsets[best]
        ids=remaining[abs(xyz[remaining]@n-offset)<.01]
        if len(ids)<32:break
        points=xyz[ids];center=points.mean(0);e,v=np.linalg.eigh((points-center).T@(points-center)/len(points))
        if e[1]<=1e-8:break
        n=v[:,0]
        if n[np.argmax(abs(n))]<0:n=-n
        offset=float(center@n);mask=abs(xyz[remaining]@n-offset)<.01
        if mask.sum()<32:break
        remaining=remaining[~mask]
        hypotheses.append(dict(normal=n,offset=offset,center=center,basis=v[:,1:],discovery_support=int(mask.sum())))
        if progress:progress('GLOBAL_DOMINANT_PLANE',len(hypotheses),'SUPPORT',int(mask.sum()),'REMAINING',len(remaining))
    faces=[];failures=[]
    for i,p in enumerate(hypotheses):
        from weld_pipeline.sjg_union_sampling_mesh_995 import local_support
        ids=np.flatnonzero(abs(xyz@p['normal']-p['offset'])<.01)
        face=local_support(xyz,dict(p,point_ids=ids))
        if face is None:failures.append(i);continue
        faces.append(face)
        if progress:progress('GLOBAL_FINITE_SUPPORT',i+1,len(hypotheses),'POINTS',len(face['point_ids']),'TRIANGLES',len(face['triangles']))
    return faces,dict(original_points=len(xyz),plane_hypotheses=len(hypotheses),planes=len(faces),remaining_unassigned_points=len(remaining),unassigned_point_ids=remaining.tolist(),failed_mesh_hypotheses=failures,plane_budget=maximum_planes,plane_budget_reached=len(hypotheses)>=maximum_planes,ideal_sample_tolerance_mm=.01,noisy_scan_applicability_unvalidated=True)
