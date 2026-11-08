"""Hull rescue with identical-pose removal and inherited certified early exit."""
from .curveonly_01a1ac7c_registration_v20 import *
from . import curveonly_01a1ac7c_registration_v20 as base


def rescue(solids,xyz,samples,normals):
    start=time.perf_counter();raw,rn,tree=raw_normals(xyz);na=previous.axes(normals);ra=previous.axes(rn)
    vertices=material_vertices(solids);source_frames=rectangles(vertices,na[:,0]);target_frames=rectangles(xyz,ra[:,0]);proposals=[]
    witness=samples[np.linspace(0,len(samples)-1,32,dtype=int)];placements=[]
    for si,s in enumerate(source_frames):
        for ti,t in enumerate(target_frames):
            for flip in (False,True):
                for quarter in range(4):
                    local=Rotation.from_rotvec(np.array([0.,0.,quarter*math.pi/2])).as_matrix()
                    if flip:local=local@np.diag([1.,-1.,-1.])
                    R=t['frame']@local@s['frame'].T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=t['center']-R@s['center']
                    where=witness@R.T+T[:3,3]
                    if any(np.linalg.norm(where-old,axis=1).max()<1e-5 for old in placements):continue
                    placements.append(where);moved=samples[::16]@R.T+T[:3,3]
                    d=tree.query(moved,workers=1,distance_upper_bound=300.)[0];rank=float(np.quantile(np.minimum(d,300.),.8))
                    proposals.append((rank,T,dict(native_rectangle=si,raw_rectangle=ti,flip=flip,quarter=quarter)))
    ordered=sorted(proposals,key=lambda p:p[0]);trials=[];boxes=np.array([cad.bbox(s)for s in solids]);early=False
    for i,(rank,T,origin)in enumerate(ordered[:3]):
        q,dist=cad.pose_quality(solids,boxes,xyz,T);trials.append((q,T,dist,'HULL',rank,origin))
        # Same1mm certificate and10mm rank separation as old registration_v2.
        if i==0 and q<=1. and(len(ordered)==1 or ordered[1][0]-rank>=10.):early=True;break
        fitted=refine(T,raw,samples,normals,na);q,dist=cad.pose_quality(solids,boxes,xyz,fitted)
        trials.append((q,fitted,dist,'HULL_ICP',rank,origin))
        if i==0 and q<=1. and(len(ordered)==1 or ordered[1][0]-rank>=10.):early=True;break
    q,T,dist,stage,rank,origin=min(trials,key=lambda x:(x[0],x[4]))
    return T,dict(sample_p95_mm=q,passed=q<=3.,sample_distances_mm=dist,transform_STEP_to_XYZ=T.tolist(),
        input_only=True,globally_unique_pose_proved=False,method=stage,pose_threshold_unchanged=True,
        candidate_count=len(proposals),raw_proposal_count=8*len(source_frames)*len(target_frames),maximum_candidates=72,
        maximum_refined=3,ICP_iterations_limit=16,verified_early_exit=early,
        early_exit='Existing1mm finite certificate plus10mm distinct input mesh rank advantage'if early else None,
        native_hull_frames=len(source_frames),raw_hull_frames=len(target_frames),candidate_rank=rank,origin=origin,
        distinct_candidate_ranks=[v[0]for v in ordered],
        tested=[dict(p95_mm=v[0],method=v[3],rank=v[4],origin=v[5],transform_STEP_to_XYZ=v[1].tolist())for v in trials],seconds=time.perf_counter()-start)


def choose_pose(solids,xyz,provided=None):
    function=types.FunctionType(base.choose_pose.__code__,dict(base.__dict__,rescue=rescue))
    return function(solids,xyz,provided)
