"""Local triangle interpolation for reference-frame audits, not inference."""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def reference_surface_distances(reference, query, max_edge_mm=20., neighbors=64, voxel_mm=2.):
    reference = np.asarray(reference, float); query = np.asarray(query, float)
    if voxel_mm > 0:
        # Keep actual vertices; uniform cells prevent a densely sampled column
        # from filling the entire neighborhood before the next column appears.
        _, keep = np.unique(np.floor(reference/voxel_mm).astype(np.int64),axis=0,return_index=True)
        reference = reference[np.sort(keep)]
    tree = cKDTree(reference); k = min(neighbors, len(reference))
    if k < 3:
        return tree.query(query)[0]
    # Include close and farther vertices so narrow rows of densely sampled
    # points do not hide adjacent rows in the same surface grid.
    ranks = sorted(set([*range(1, min(17,k)), *[r for r in [20,24,32,40,48,63] if r<k]]))
    pairs = np.asarray(list(itertools.combinations(ranks,2)))
    result = []
    for start in range(0,len(query),256):
        q = query[start:start+256]; distance, indices = tree.query(q,k=k,workers=2)
        p = reference[indices]; a = p[:,0,None,:]
        u = p[:,pairs[:,0],:]-a; v = p[:,pairs[:,1],:]-a; w = q[:,None,:]-a
        uu = np.sum(u*u,axis=2); vv = np.sum(v*v,axis=2); uv = np.sum(u*v,axis=2)
        uw = np.sum(u*w,axis=2); vw = np.sum(v*w,axis=2)
        denom = uu*vv-uv*uv
        safe = np.where(denom>1e-12,denom,1.)
        s = (vv*uw-uv*vw)/safe; t = (uu*vw-uv*uw)/safe
        valid = (denom>1e-12)&(s>=0)&(t>=0)&(s+t<=1)&(uu<=max_edge_mm**2)&(vv<=max_edge_mm**2)
        valid &= np.sum((u-v)**2,axis=2)<=max_edge_mm**2
        projection = w-s[:,:,None]*u-t[:,:,None]*v
        errors = np.where(valid,np.linalg.norm(projection,axis=2),np.inf)
        result.append(np.minimum(distance[:,0],errors.min(axis=1)))
    return np.concatenate(result) if result else np.empty(0)
