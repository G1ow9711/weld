"""All-input spatial index and exact, bounded-memory local geometric moments."""
import math
import numpy as np
from scipy.spatial import cKDTree

RADII_MM=(16.,64.,256.)
FEATURES_PER_RADIUS=13

def local_point_features(source, world_positions_mm, block_size=16384):
    """Positions are supplied by caller; inference supplies only its predictions.

    Every raw row remains indexed. No nearest point is returned as a weld answer.
    Radius membership is detached; learned residuals use relative moments.
    """
    positions=np.asarray(world_positions_mm,np.float64)
    if positions.ndim!=2 or positions.shape[1]!=3: raise ValueError("Finite Nx3 observation positions required")
    if not hasattr(source,"_295_raw_spatial_index"):
        source._295_raw_spatial_index=cKDTree(np.asarray(source.xyz_mm),copy_data=False)
    tree=source._295_raw_spatial_index
    result=np.zeros((len(positions),len(RADII_MM)*FEATURES_PER_RADIUS),np.float32)
    for i,center in enumerate(positions):
        if not np.isfinite(center).all(): continue
        for ri,radius in enumerate(RADII_MM):
            members=tree.query_ball_point(center,radius,workers=1,return_sorted=False)
            count=len(members)
            if not count: continue
            total=np.zeros(3); products=np.zeros(6); dmin=np.inf; dmax=0.
            for first in range(0,count,block_size):
                relative=(np.asarray(source.xyz_mm[np.asarray(members[first:first+block_size],np.int64)])-center)/radius
                total+=relative.sum(axis=0)
                x,y,z=relative.T
                products+=np.asarray([x@x,x@y,x@z,y@y,y@z,z@z])
                distance=np.linalg.norm(relative,axis=1)
                dmin=min(dmin,float(distance.min())); dmax=max(dmax,float(distance.max()))
            mean=total/count
            covariance=products/count-np.asarray([mean[0]**2,mean[0]*mean[1],mean[0]*mean[2],mean[1]**2,mean[1]*mean[2],mean[2]**2])
            vector=[1.,math.log1p(count)/math.log1p(len(source.xyz_mm)),*mean,*covariance,dmin,dmax]
            assert len(vector)==FEATURES_PER_RADIUS
            result[i,ri*13:(ri+1)*13]=vector
    return result
