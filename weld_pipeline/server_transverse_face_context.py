"""Finite observed boundary ray descriptors, not plate identity or process truth."""
import numpy as np
from .server_sampled_surface_boundary_events import cross2

def transverse_face_descriptor(point,tangent,normal,offset,boundary):
 # 6 global values + two sign-invariant ray-side tuples(has,count,near,far).
 out=np.zeros(14,np.float64);n=np.asarray(normal,np.float64);norm=np.linalg.norm(n)
 if norm<=1e-12:return out
 n=n/norm;t=np.asarray(tangent,np.float64);t=t/np.linalg.norm(t);distance=abs(n@point-offset/norm);out[:3]=[1.,distance/1000.,abs(n@t)]
 if boundary is None:return out
 origin,basis,edges=boundary;direction=np.cross(n,t);size=np.linalg.norm(direction)
 if size<=1e-12 or distance>3.:return out
 direction/=size;o=(point-origin)@basis;d=direction@basis;a=edges[:,0];e=edges[:,1]-a;den=cross2(d,e);regular=abs(den)>1e-12;roots=[]
 if regular.any():
  u=cross2(a[regular]-o,e[regular])/den[regular];v=cross2(a[regular]-o,d)/den[regular];roots.extend(u[(v>=-1e-10)&(v<=1+1e-10)].tolist())
 collinear=(~regular)&(abs(cross2(a-o,d))<=1e-9)
 if collinear.any():roots.extend(((edges[collinear]-o)@d).reshape(-1).tolist())
 out[3]=1;out[5]=float(collinear.any());q=np.sort(np.asarray(roots));q=q[np.r_[True,np.diff(q)>1e-7]] if len(q) else q;out[4]=float(np.any(abs(q)<=1e-7));sides=[]
 for side in [-1,1]:
  values=side*q[side*q>1e-7];sides.append(np.array([1.,np.log1p(len(values)),values.min()/1000.,values.max()/1000.]) if len(values) else np.zeros(4))
 # Sign changes in normals/tangents may exchange ray sides, never field meaning.
 sides.sort(key=lambda v:tuple(np.round(v,9)));out[6:]=np.concatenate(sides);assert np.isfinite(out).all();return out

def transverse_endpoint_context(point,tangent,face_ids,generator):
 descriptors=[transverse_face_descriptor(np.asarray(point),tangent,generator.normals[i],generator.offsets[i],generator.boundaries.get(i)) for i in face_ids[:2]]
 if not descriptors:return np.zeros(43,np.float32)
 a=np.asarray(descriptors);return np.r_[len(a),a.mean(0),a.min(0),a.max(0)].astype(np.float32)
