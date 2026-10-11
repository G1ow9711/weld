"""Input finite observed-face hull events on analytic carriers; not weld labels.

Convex hulls retain finite extent but do not certify holes/material/contact.
No GT or STEP is read. Event coordinates are analytic intersections, not snaps.
"""
import numpy as np
from .sjg_finite_face_domains_1235 import face_halfspaces
class FaceEvents:
 def __init__(self,xyz,faces,support):
  self.faces=faces;self.domains={};self.normals=[];self.offsets=[]
  for i,f in enumerate(faces):
   normal=np.asarray(f.get('normal',[0,0,0]),float);norm=np.linalg.norm(normal);self.normals.append(normal/norm if norm else normal);self.offsets.append(float(f.get('offset',0))/norm if norm else 0)
   key=f.get('point_ids_key')
   if not norm or key not in support:continue
   try:A,b,_=face_halfspaces(np.asarray(xyz[support[key]]),normal);self.domains[i]=(A,b)
   except (ValueError,RuntimeError):continue
  self.normals=np.asarray(self.normals);self.offsets=np.asarray(self.offsets)
 def events(self,curve):
  candidates=[]
  if curve.kind=='line':
   slopes=self.normals@curve.direction;valid=np.flatnonzero(abs(slopes)>1e-8);q=(self.offsets[valid]-self.normals[valid]@curve.ends[0])/slopes[valid];candidates=[(int(i),float(t)) for i,t in zip(valid,q) if 0<=t<=curve.length]
  else:
   a=curve.arc;sign=np.sign(a.sweep_rad);u=a.start_direction;v=sign*np.cross(a.plane_normal,u);aa=a.radius_mm*(self.normals@u);bb=a.radius_mm*(self.normals@v);cc=self.offsets-self.normals@a.center_mm;rad=np.hypot(aa,bb)
   for i in np.flatnonzero((rad>1e-8)&(abs(cc)<=rad)):
    phase=np.arctan2(bb[i],aa[i]);delta=np.arccos(np.clip(cc[i]/rad[i],-1,1))
    for root in (phase-delta,phase+delta):
     for k in (-1,0,1,2):
      angle=root+2*np.pi*k
      if 0<=angle<=abs(a.sweep_rad):candidates.append((int(i),float(angle*a.radius_mm)))
  q=[0.,float(curve.length)];sources=[-1,-1]
  for face,t in candidates:
   if face not in self.domains:continue
   A,b=self.domains[face];point=curve.at(np.array([t]))[0]
   if np.max(A@point+b)<=1e-6:q.append(t);sources.append(face)
  return np.asarray(q),np.asarray(sources)
