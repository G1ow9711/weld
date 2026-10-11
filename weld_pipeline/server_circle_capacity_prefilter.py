"""Conservative infinite-circle lower-bound filter for offline finite-arc scoring."""
import numpy as np


def possible_circle_indices(centers,normals,radii,points,tolerance_mm=3.):
    # Probe actual reference samples only. A finite arc is a subset of its circle.
    points=np.asarray(points,float);probes=points[np.unique([0,len(points)//2,len(points)-1])];accepted=[]
    for start in range(0,len(radii),16384):
        stop=min(start+16384,len(radii));delta=probes[None]-centers[start:stop,None];height=np.einsum('bpi,bi->bp',delta,normals[start:stop]);planar=delta-height[:,:,None]*normals[start:stop,None];radial=np.linalg.norm(planar,axis=2)
        distance=np.hypot(height,radial-radii[start:stop,None])
        scale=np.maximum(1.,np.maximum(np.max(np.abs(delta),axis=(1,2)),radii[start:stop]))
        guard=1e-7+128*np.finfo(float).eps*scale
        keep=np.any(~np.isfinite(distance),axis=1)|(distance.max(1)<=tolerance_mm+guard)
        accepted.extend((np.flatnonzero(keep)+start).tolist())
    return accepted
