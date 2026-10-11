"""Finite plane support with rank-two, density-normalized normal seeds."""
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from weld_pipeline.sjg_observed_cylinders_919 import canonical_normals,dominant_axes,basis_for,plane_levels
from weld_pipeline.sjg_local_runtime_904 import check_resources


def extract_planes(xyz,progress=None):
    data=canonical_normals(xyz,progress);axes=dominant_axes(data,maximum=64)
    origin=xyz.mean(0);faces=[];stats=[]
    for ai,axis in enumerate(axes):
        col=dict(axis=axis,origin=origin)
        levels,eligible,height=plane_levels(data,col)
        if progress:progress('RANK2_PLANE_AXIS',ai+1,len(axes),'LEVELS',len(levels))
        for level in levels:
            # Only points with a stable, aligned planar normal seed an offset.
            seed=data['xyz'][eligible[abs(height-level)<=.2]]
            if len(seed)<32:continue
            center=seed.mean(0);_,s,v=np.linalg.svd(seed-center,full_matrices=False)
            if s[1]<.5:continue
            normal=v[-1]
            if normal@axis<0:normal=-normal
            if normal@axis<np.cos(np.deg2rad(2.)):continue
            offset=float(center@normal)
            if any(abs(normal@f['normal'])>1-1e-8 and abs(offset-np.sign(normal@f['normal'])*f['offset'])<.2 for f in faces):continue
            # Boundary normals may be mixed; recover all observed XYZ near
            # the seeded plane instead of eroding the boundary by normal mask.
            ids=np.flatnonzero(abs(xyz@normal-offset)<=.2);basis=basis_for(normal)
            uv=(xyz[ids]-center)@basis
            _,unique=np.unique(np.round(uv,5),axis=0,return_index=True);ids=ids[unique];uv=uv[unique]
            if len(ids)<32:continue
            try:triangles=Delaunay(uv).simplices
            except QhullError:continue
            distances=cKDTree(uv).query(uv,k=min(24,len(uv)),workers=2)[0];radius=distances[:,-1]
            q=uv[triangles];keep=np.ones(len(triangles),bool)
            for i,j in ((0,1),(1,2),(2,0)):
                keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=np.maximum(2.,1.5*np.minimum(radius[triangles[:,i]],radius[triangles[:,j]]))
            triangles=triangles[keep]
            if not len(triangles):continue
            faces.append(dict(normal=normal,center=center,offset=offset,basis=basis,point_ids=ids,triangles=triangles,
                proposal_branch='raw_geometry',normal_seed='canonical_1mm_rank2_32NN',planar_seed_points=len(seed),
                material_id=None,material_status='UNKNOWN_NOT_INFERRED',unresolved_subsampling_holes=True,
                max_plane_residual_mm=float(abs(xyz[ids]@normal-offset).max())))
            check_resources()
            if progress:progress('RANK2_FINITE_PLANE',len(faces),'SEEDS',len(seed),'SUPPORT',len(ids))
        stats.append(dict(axis=axis.tolist(),candidate_levels=len(levels)))
    return faces,dict(original_points=len(xyz),canonical_points=len(data['xyz']),
        rank2_points=int(data['rank2'].sum()),planar_points=int(data['planar'].sum()),
        axis_count=len(axes),axis_budget=64,axis_budget_reached=len(axes)==64,axes=stats)
