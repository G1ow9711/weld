"""Offline-only exact STEP query adapter; not imported by XYZ inference.

The installed Gmsh backend can return placeholder [0,0] parameters from
getClosestPoint for these BSpline faces. Recompute parameters from its actual
3D closest point and verify the inverse map before finite-trim queries.
"""
import numpy as np
import gmsh

def closest_with_verified_parameters(face,point):
 q,unused=gmsh.model.getClosestPoint(2,int(face),np.asarray(point,float).tolist());q=np.asarray(q);uv=np.asarray(gmsh.model.getParametrization(2,int(face),q.tolist()));reconstructed=np.asarray(gmsh.model.getValue(2,int(face),uv.tolist()));error=float(np.linalg.norm(reconstructed-q))
 if not np.isfinite(uv).all() or error>1e-4:raise ValueError('STEP inverse parameter map not verified: '+str(error))
 return q,uv,error
