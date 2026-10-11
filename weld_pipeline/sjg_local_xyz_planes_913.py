"""Geometry-only planar support using the prior 794 rules on native local XYZ.

No Torch, STEP, targets, learned instance IDs or original source identifiers.
"""
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from weld_pipeline.sjg_local_runtime_904 import check_resources

def normal_cloud(xyz):
    tree=cKDTree(xyz);normals=np.empty_like(xyz);planar=np.empty(len(xyz),bool)
    for begin in range(0,len(xyz),16384):
        end=min(begin+16384,len(xyz));near=tree.query(xyz[begin:end],k=min(12,len(xyz)),workers=2)[1]
        cloud=xyz[near];center=cloud.mean(1);delta=cloud-center[:,None]
        covariance=np.einsum('nki,nkj->nij',delta,delta)/cloud.shape[1]
        eigen,vectors=np.linalg.eigh(covariance);normals[begin:end]=vectors[:,:,0]
        planar[begin:end]=(eigen[:,0]/eigen[:,1].clip(1e-8)<.02)&(eigen[:,1]>1e-6)
    return normals,planar

def extract_planes(xyz,maximum_planes=512,progress=None):
    normals,planar=normal_cloud(xyz)
    signs=np.sign(normals[np.arange(len(normals)),np.argmax(np.abs(normals),axis=1)]);normals*=signs[:,None]
    keys,inverse,counts=np.unique(np.round(normals[planar]/.02).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    usable=normals[planar];families=[]
    for k in np.argsort(-counts):
        if counts[k]<32:continue
        cluster=usable[inverse==k]
        _,fine,fine_counts=np.unique(np.round(cluster/1e-5).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
        mode=int(fine_counts.argmax())
        if fine_counts[mode]<16:continue
        normal=np.median(cluster[fine==mode],axis=0);normal/=np.linalg.norm(normal)
        if not any(abs(normal@n)>1-1e-8 for n in families):families.append(normal)
    observed=list(families)
    for i,a in enumerate(observed[:8]):
        for b in observed[i+1:8]:
            normal=np.cross(a,b)
            if np.linalg.norm(normal)<.15:continue
            normal/=np.linalg.norm(normal)
            if normal[np.argmax(np.abs(normal))]<0:normal=-normal
            if not any(abs(normal@n)>1-1e-8 for n in families):families.append(normal)
    del normals,usable,inverse
    candidates=[]
    for ni,normal in enumerate(families):
        projection=xyz@normal;order=np.argsort(projection);values=projection[order]
        bins,counts=np.unique(np.round(values/.2).astype(np.int64),return_counts=True)
        lookup={int(b):int(n) for b,n in zip(bins,counts)};seeds=[]
        for b,count in zip(bins,counts):
            b=int(b);left=lookup.get(b-1,0);right=lookup.get(b+1,0)
            if count<32 or count<left or count<right or count==left:continue
            lo=np.searchsorted(values,b*.2-.3);hi=np.searchsorted(values,b*.2+.3,side='right');center=float(np.median(values[lo:hi]))
            if any(abs(center-old)<.3 for old in seeds):continue
            ids=order[np.searchsorted(values,center-.2):np.searchsorted(values,center+.2,side='right')].copy()
            if len(ids)<32:continue
            candidates.append(dict(normal=normal,ids=ids));seeds.append(center)
        if progress and ni%20==0:progress('XYZ_PLANE_FAMILIES',ni+1,len(families),'CANDIDATES',len(candidates))
    remaining=np.ones(len(xyz),bool);faces=[]
    for iteration in range(maximum_planes):
        if not candidates:break
        counts=np.array([len(c['ids']) for c in candidates]);index=int(counts.argmax())
        if counts[index]<32:break
        candidate=candidates.pop(index);support=xyz[candidate['ids']];center=support.mean(0)
        _,singular,v=np.linalg.svd(support-center,full_matrices=False)
        if singular[1]<.5:continue
        normal=v[-1]
        if normal[np.argmax(np.abs(normal))]<0:normal=-normal
        offset=float(center@normal)
        if any(abs(normal@p['normal'])>1-1e-10 and abs(offset-np.sign(normal@p['normal'])*p['offset'])<.1 for p in faces):continue
        ids=np.flatnonzero(np.abs(xyz@normal-offset)<.2);basis=v[:2].T;uv=(xyz[ids]-center)@basis
        _,unique=np.unique(np.round(uv,5),axis=0,return_index=True);ids=ids[unique];uv=uv[unique]
        try:triangles=Delaunay(uv).simplices
        except QhullError:continue
        distance=cKDTree(uv).query(uv,k=min(24,len(uv)),workers=2)[0];nearest=distance[:,1];radius=distance[:,-1]
        triangle_uv=uv[triangles];keep=np.ones(len(triangles),bool)
        for i,j in ((0,1),(1,2),(2,0)):
            edge=np.linalg.norm(triangle_uv[:,i]-triangle_uv[:,j],axis=1)
            keep &= edge<=np.maximum(2.,1.5*np.minimum(radius[triangles[:,i]],radius[triangles[:,j]]))
        triangles=triangles[keep]
        if not len(triangles):continue
        full_ids=np.flatnonzero(np.abs(xyz@normal-offset)<.2);core_count=int(remaining[full_ids].sum());remaining[full_ids]=False
        nonzero=nearest[nearest>1e-6]
        faces.append(dict(normal=normal,center=center,offset=offset,basis=basis,point_ids=ids,triangles=triangles,
            sample_spacing_mm=float(np.median(nonzero)) if len(nonzero) else 0.,adaptive_radius_p50_p95_mm=np.percentile(radius,[50,95]),
            support_rule='edge <= 1.5 * min(endpoint 24-neighbor radii), floor 2mm',unresolved_subsampling_holes=True,
            core_points=core_count,max_plane_residual_mm=float(np.abs(xyz[ids]@normal-offset).max()),
            material_id=None,material_status='UNKNOWN_NOT_INFERRED',overlapping_plane_hypothesis=True,proposal_branch='raw_geometry'))
        if progress:progress('XYZ_FINITE_PLANE',len(faces),'SUPPORT',len(ids),'UNASSIGNED',int(remaining.sum()))
        check_resources()
    return faces,dict(planar_seed_points=int(planar.sum()),remaining_unassigned_points=int(remaining.sum()),
        orientation_families=len(families),observed_orientation_families=len(observed),candidate_planes_remaining=len(candidates),
        maximum_candidate_iterations=maximum_planes,candidate_iteration_cap_reached=bool(iteration+1==maximum_planes and candidates))
