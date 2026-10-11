"""Point-only circular fillet proposals from sparse arc columns and tangent walls."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.point_frame import dominant_xy_frame
from weld_pipeline.circle_candidates import cylinder_columns


def supported_wall_positions(values,minimum_support):
    bins=np.round(values/.25).astype(np.int64);keys,counts=np.unique(bins,return_counts=True)
    positions=[float(np.median(values[bins==k])) for k in keys[counts>=minimum_support]]
    groups=[]
    for p in positions:
        if groups and p-groups[-1][-1]<.4:groups[-1].append(p)
        else:groups.append([p])
    return np.array([np.mean(g) for g in groups])


def propose_tangent_circles(xyz,tolerance_mm=.3,wall_min_support=None):
    frame=dominant_xy_frame(xyz);anchors=cylinder_columns(xyz);local=anchors@frame.T
    required=max(8,int(np.ceil(len(local)*.001))) if wall_min_support is None else int(wall_min_support)
    if required<3:raise ValueError('At least three wall columns are required')
    planes=[supported_wall_positions(local[:,i],required) for i in range(2)]
    stats={'columns':len(local),'wall_positions':[len(p) for p in planes],'wall_support_required':required}
    if any(len(p)==0 for p in planes):return [],stats
    plane_trees=[cKDTree(p[:,None]) for p in planes]
    distances=[tree.query(local[:,i,None])[0] for i,tree in enumerate(plane_trees)]
    seeds=local[(distances[0]>tolerance_mm)&(distances[1]>tolerance_mm)]
    stats['off_wall_seeds']=len(seeds);tree=cKDTree(local)
    maximum_radius=float(np.linalg.norm(np.ptp(local,axis=0))/2);candidates=[]
    for p in seeds:
        near=[plane_trees[i].query([[p[i]]],k=min(6,len(planes[i])))[1].reshape(-1) for i in range(2)]
        for ix in near[0]:
            for iy in near[1]:
                corner=np.array([planes[0][ix],planes[1][iy]]);delta=p-corner
                signs=np.sign(delta);d=np.abs(delta);base=d.sum();root=np.sqrt(2*d.prod())
                for radius in [base+root,base-root]:
                    if not 2.<=radius<=maximum_radius:continue
                    center=corner+signs*radius
                    if np.any((p-center)*signs>tolerance_mm):continue
                    tangent=np.array([[corner[0],center[1]],[center[0],corner[1]]])
                    errors,ids=tree.query(tangent)
                    if errors.max()>tolerance_mm:continue
                    if np.min(np.linalg.norm(tangent-p,axis=1))<2.:continue
                    candidates.append({'center_local':center,'radius_mm':float(radius),
                        'tangent_error_mm':float(errors.max()),'seed':p,'tangent_points':local[ids]})
    selected=[]
    for c in sorted(candidates,key=lambda c:c['tangent_error_mm']):
        if any(np.linalg.norm(c['center_local']-r['center_local'])+abs(c['radius_mm']-r['radius_mm'])<1. for r in selected):continue
        selected.append(c)
    result=[]
    for c in selected:
        result.append({'center_xy_mm':(c['center_local']@frame).tolist(),'radius_mm':c['radius_mm'],
            'tangent_error_mm':c['tangent_error_mm'],'arc_seed_xy_mm':(c['seed']@frame).tolist(),
            'tangent_points_xy_mm':(c['tangent_points']@frame).tolist()})
    stats['raw_hypotheses']=len(candidates);stats['cylinder_proposals']=len(result)
    return result,stats
