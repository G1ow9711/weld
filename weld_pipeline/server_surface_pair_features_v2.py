"""XYZ-only rigid-invariant, point-order-symmetric surface pair features."""
import numpy as np
from scipy.spatial import cKDTree

def local_geometry(xyz,query,k=32):
 tree=cKDTree(xyz);_,indices=tree.query(query,k=min(k,len(xyz)),workers=2)
 points=np.asarray(xyz[indices],np.float64);centered=points-points.mean(1,keepdims=True)
 cov=np.einsum('bki,bkj->bij',centered,centered)/points.shape[1]
 values,vectors=np.linalg.eigh(cov);values=np.maximum(values,0)
 return vectors[:,:,0],values/np.maximum(values.sum(1,keepdims=True),1e-20)

def pair_features(xyz,normals,spectrum,pairs):
 a,b=pairs.T;delta=xyz[b]-xyz[a];distance=np.linalg.norm(delta,axis=1);resolution=64*np.finfo(np.float64).eps*max(1.,float(np.abs(xyz).max()));unit=np.divide(delta,distance[:,None],out=np.zeros_like(delta),where=distance[:,None]>resolution)
 ca=np.abs(np.einsum('bi,bi->b',unit,normals[a]));cb=np.abs(np.einsum('bi,bi->b',unit,normals[b]));dot=np.abs(np.einsum('bi,bi->b',normals[a],normals[b]))
 return np.column_stack((distance/1000,np.minimum(ca,cb),np.maximum(ca,cb),dot,np.minimum(spectrum[a],spectrum[b]),np.maximum(spectrum[a],spectrum[b]))).astype(np.float32)
