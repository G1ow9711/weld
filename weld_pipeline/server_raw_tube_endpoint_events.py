"""Observed XYZ projections inside a finite line tube; not weld/contact decisions.

Sphere queries cover the entire finite tube; exact axial/radial checks follow.
No labels, synthetic grid event positions, proximity merging, or GT snaps.
"""
import numpy as np
from scipy.spatial import cKDTree
class RawTubeEndpointEvents:
 def __init__(self,xyz,radius_mm=3.):
  self.xyz=np.asarray(xyz);self.radius=float(radius_mm)
  if not np.isfinite(self.radius) or self.radius<=0:raise ValueError('Positive finite tube radius required')
  self.tree=cKDTree(self.xyz)
 def events(self,curve):
  if curve.kind!='line':raise ValueError('Line-only proposal branch')
  # Center spacing <=2r gives a conservative ball cover at sqrt(2)*r.
  count=max(2,int(np.ceil(curve.length/(2*self.radius)))+1);centers=curve.at(np.linspace(0,curve.length,count));ids=[]
  for begin in range(0,count,1024):
   balls=self.tree.query_ball_point(centers[begin:begin+1024],np.sqrt(2)*self.radius+1e-9,workers=2)
   ids.extend(i for group in balls for i in group)
  if not ids:return np.empty(0,np.float64),np.empty(0,np.int64)
  ids=np.unique(ids);delta=self.xyz[ids]-curve.ends[0];q=delta@curve.direction;radial=np.linalg.norm(delta-q[:,None]*curve.direction,axis=1);keep=(q>=0)&(q<=curve.length)&(radial<=self.radius);q=q[keep];ids=ids[keep];values,first=np.unique(q,return_index=True)
  return values,ids[first]
