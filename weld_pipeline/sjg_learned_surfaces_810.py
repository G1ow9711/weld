"""PTv3 affinity proposals with finite, raw-point-supported planar fitting."""
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def components(xyz,embedding,normal,progress=None):
    tree=cKDTree(xyz);rows=[];cols=[];accepted=0
    for start in range(0,len(xyz),8192):
        ids=np.arange(start,min(start+8192,len(xyz)),dtype=np.int64)
        distance,near=tree.query(xyz[ids],k=min(17,len(xyz)),workers=2)
        a=np.repeat(ids,near.shape[1]-1);b=near[:,1:].ravel().astype(np.int64)
        difference=np.linalg.norm(embedding[a]-embedding[b],axis=1)
        cosine=np.abs((normal[a]*normal[b]).sum(1))
        # Fixed from the registered TRAIN-only audit; never relaxed on VALID.
        keep=(difference<=.4)&(cosine>=np.cos(np.deg2rad(20)))&(a!=b)
        rows.append(a[keep]);cols.append(b[keep]);accepted+=int(keep.sum())
        if progress and start%65536==0:progress('AFFINITY_POINTS',min(start+8192,len(xyz)),len(xyz))
    a=np.concatenate(rows);b=np.concatenate(cols)
    graph=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(xyz),len(xyz))).tocsr()
    count,labels=connected_components(graph,directed=False)
    return labels,dict(components=int(count),directed_links=accepted,embedding_threshold=.4,normal_angle_deg=20.)


def refine_plane(points,ids,normal,tolerance=.2):
    offset=float(np.median(points@normal));support=ids[np.abs(points@normal-offset)<=tolerance]
    if len(support)<16:return None
    # ids index the full point cloud; the local membership array is kept here.
    mask=np.abs(points@normal-offset)<=tolerance
    for _ in range(3):
        p=points[mask]
        if len(p)<16:return None
        center=p.mean(0);_,singular,v=np.linalg.svd(p-center,full_matrices=False)
        if singular[1]<.2:return None
        normal=v[-1];offset=float(center@normal)
        updated=np.abs(points@normal-offset)<=tolerance
        if np.array_equal(mask,updated):break
        mask=updated
    if normal[np.argmax(abs(normal))]<0:normal=-normal;offset=-offset
    return dict(normal=normal,center=center,offset=offset,point_ids=ids[mask],basis=v[:2].T)


def split_component(xyz,ids,pred_normal,boundary,rng):
    remaining=ids.copy();planes=[];attempts=0
    while len(remaining)>=16 and attempts<64:
        attempts+=1;points=xyz[remaining];center=points.mean(0)
        _,singular,v=np.linalg.svd(points-center,full_matrices=False)
        if singular[1]<.2:break
        if np.quantile(np.abs((points-center)@v[-1]),.95)<=.2:
            candidates=v[-1:]
        else:
            triples=points[rng.integers(len(points),size=(128,3))]
            normals=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0])
            lengths=np.linalg.norm(normals,axis=1);good=lengths>1e-6
            normals=normals[good]/lengths[good,None]
            # RANSAC offsets retain the sampled plane rather than forcing every
            # small surface through the component's median projected position.
            offsets=(triples[good,0]*normals).sum(1)
            pick=rng.choice(len(points),min(2048,len(points)),replace=False)
            sample=points[pick];weights=.25+.75*(1-boundary[remaining[pick]])
            score=(np.abs(sample@normals.T-offsets)<.2).T@weights if len(normals) else np.empty(0)
            if not len(score) or score.max()<8:break
            best=int(score.argmax());initial=np.abs(points@normals[best]-offsets[best])<=.2
            if initial.sum()<16:break
            local=points[initial];c=local.mean(0);_,s,vv=np.linalg.svd(local-c,full_matrices=False)
            if s[1]<.2:break
            selected=remaining[np.abs((points-c)@vv[-1])<=.2]
            plane=refine_plane(xyz[selected],selected,vv[-1])
            if plane is None:break
            planes.append(plane);remaining=remaining[~np.isin(remaining,plane['point_ids'])]
            continue
        plane=refine_plane(points,remaining,candidates[0])
        if plane is None:break
        planes.append(plane);remaining=remaining[~np.isin(remaining,plane['point_ids'])]
    return planes,remaining,attempts


def merge_planes(xyz,planes):
    merged=[]
    for plane in sorted(planes,key=lambda p:-len(p['point_ids'])):
        found=False
        for previous in merged:
            if abs(plane['normal']@previous['normal'])<np.cos(np.deg2rad(1.)):continue
            if abs((plane['center']-previous['center'])@previous['normal'])>.2:continue
            if abs((plane['center']-previous['center'])@plane['normal'])>.2:continue
            ids=np.unique(np.r_[previous['point_ids'],plane['point_ids']]);points=xyz[ids];center=points.mean(0)
            _,_,v=np.linalg.svd(points-center,full_matrices=False);normal=v[-1]
            if np.abs((points-center)@normal).max()>.2:continue
            if normal[np.argmax(abs(normal))]<0:normal=-normal
            previous.update(point_ids=ids,normal=normal,center=center,offset=float(center@normal),basis=v[:2].T)
            previous['learned_component_ids']=sorted(set(previous['learned_component_ids']+plane['learned_component_ids']))
            found=True;break
        if not found:merged.append(plane)
    return merged


def finite_support(xyz,plane):
    ids=plane['point_ids'];uv=(xyz[ids]-plane['center'])@plane['basis']
    _,unique=np.unique(np.round(uv,5),axis=0,return_index=True);ids=ids[unique];uv=uv[unique]
    if len(ids)<16:return None
    try:triangles=Delaunay(uv).simplices
    except QhullError:return None
    distance=cKDTree(uv).query(uv,k=min(24,len(uv)),workers=2)[0]
    radius=distance[:,-1];keep=np.ones(len(triangles),bool);q=uv[triangles]
    for i,j in ((0,1),(1,2),(2,0)):
        keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=np.maximum(2.,1.5*np.minimum(radius[triangles[:,i]],radius[triangles[:,j]]))
    triangles=triangles[keep]
    if not len(triangles):return None
    nonzero=distance[:,1][distance[:,1]>1e-6]
    return dict(**{k:v for k,v in plane.items() if k!='point_ids'},point_ids=ids,triangles=triangles,
        sample_spacing_mm=float(np.median(nonzero)) if len(nonzero) else 0.,
        adaptive_radius_p50_p95_mm=np.percentile(radius,[50,95]),
        support_rule='same geometric plane; local 24-neighbor adaptive triangle edges, floor 2mm',
        unresolved_subsampling_holes=True,material_id=None,material_status='UNKNOWN_NOT_INFERRED',
        max_plane_residual_mm=float(np.abs(xyz[ids]@plane['normal']-plane['offset']).max()))


def extract(xyz,pred,progress=None):
    labels,stats=components(xyz,pred['embedding'],pred['normal'],progress)
    order=np.argsort(labels,kind='stable');splits=np.r_[0,np.flatnonzero(np.diff(labels[order]))+1,len(labels)]
    planes=[];unresolved=0;attempts=0;exhausted=0;rng=np.random.default_rng(810)
    for k,(lo,hi) in enumerate(zip(splits[:-1],splits[1:])):
        ids=order[lo:hi].astype(np.int64)
        if len(ids)<16:unresolved+=len(ids);continue
        subset,remaining,trials=split_component(xyz,ids,pred['normal'],pred['boundary'],rng)
        exhausted+=int(trials==64 and len(remaining)>=16)
        for plane in subset:plane['learned_component_ids']=[int(labels[ids[0]])]
        planes.extend(subset);unresolved+=len(remaining);attempts+=trials
        if progress and k%100==0:progress('FITTED_COMPONENTS',k+1,len(splits)-1)
    before=len(planes);planes=merge_planes(xyz,planes);faces=[]
    for plane in planes:
        face=finite_support(xyz,plane)
        if face is not None:faces.append(face)
    stats.update(raw_fitted_planes=before,merged_planes=len(planes),finite_surfaces=len(faces),
        unresolved_points_before_meshing=unresolved,plane_fit_attempts=attempts,
        components_exhausting_fit_budget=exhausted,
        per_component_fit_budget=64,unsupported_curved_surfaces=True)
    return faces,labels,stats
