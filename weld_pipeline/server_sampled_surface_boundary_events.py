"""Input sampled-surface boundary proposals; no material/contact certification."""
import numpy as np
from scipy.spatial import Delaunay,cKDTree,QhullError
from .server_face_boundary_endpoint_events import FaceBoundaryEvents

def cross2(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
class SampledSurfaceBoundaryEvents(FaceBoundaryEvents):
 def __init__(self,xyz,faces,support):
  super().__init__(xyz,faces,support);self.boundaries={};self.boundary_statistics=[]
  for face in self.domains:
   n=self.normals[face];key=faces[face].get('point_ids_key');points=np.asarray(xyz[support[key]],np.float64);origin=points.mean(0);axis=np.eye(3)[np.argmin(abs(n))];u=axis-n*(axis@n);u/=np.linalg.norm(u);basis=np.column_stack((u,np.cross(n,u)));uv=np.unique((points-origin)@basis,axis=0)
   if len(uv)<3:continue
   nearest=cKDTree(uv).query(uv,k=2,workers=2)[0][:,1];positive=nearest[nearest>1e-6]
   if not len(positive):continue
   spacing=float(np.median(positive));cutoff=6*spacing
   try:tri=Delaunay(uv).simplices
   except QhullError:continue
   vertices=uv[tri];edge_lengths=np.linalg.norm(vertices-np.roll(vertices,1,axis=1),axis=2);tri=tri[edge_lengths.max(1)<=cutoff]
   if not len(tri):continue
   edges=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1);unique,count=np.unique(edges,axis=0,return_counts=True);boundary=uv[unique[count==1]];self.boundaries[face]=(origin,basis,boundary);self.boundary_statistics.append(dict(face=face,points=len(uv),median_nearest_mm=spacing,max_triangle_edge_mm=cutoff,boundary_edges=len(boundary),kept_triangles=len(tri)))
 def events(self,curve):
  q,source=super().events(curve);events=q.tolist();sources=source.tolist()
  for face,(origin,basis,edges) in self.boundaries.items():
   normal=self.normals[face];offset=self.offsets[face];a=edges[:,0];e=edges[:,1]-a
   if curve.kind=='line':
    if abs(float(normal@curve.direction))>1e-8 or abs(float(normal@curve.ends[0])-offset)>3:continue
    o=(curve.ends[0]-origin)@basis;d=curve.direction@basis;den=cross2(d,e);valid=abs(den)>1e-12;t=np.zeros(len(e));s=np.zeros(len(e));t[valid]=cross2(a[valid]-o,e[valid])/den[valid];s[valid]=cross2(a[valid]-o,d)/den[valid];take=valid&(t>=0)&(t<=curve.length)&(s>=0)&(s<=1);values=t[take]
   else:
    arc=curve.arc
    if 1-abs(float(normal@arc.plane_normal))>1e-8 or abs(float(normal@arc.center_mm)-offset)>3:continue
    center=(arc.center_mm-origin)@basis;v=a-center;aa=(e*e).sum(1);bb=2*(v*e).sum(1);cc=(v*v).sum(1)-arc.radius_mm**2;discriminant=bb*bb-4*aa*cc;valid=(aa>0)&(discriminant>=0);roots=[]
    for sign in (-1,1):
     ids=np.flatnonzero(valid);s=(-bb[ids]+sign*np.sqrt(discriminant[ids]))/(2*aa[ids]);keep=(s>=0)&(s<=1);roots.append(a[ids[keep]]+s[keep,None]*e[ids[keep]])
    uv=np.concatenate(roots)
    if not len(uv):continue
    points=origin+uv@basis.T;values,_=curve.project(points);good=(values>=0)&(values<=curve.length);projected=(curve.at(values)-origin)@basis;good&=np.linalg.norm(projected-uv,axis=1)<=1e-6;values=values[good]
   events.extend(values.tolist());sources.extend([face]*len(values))
  return np.asarray(events),np.asarray(sources)
