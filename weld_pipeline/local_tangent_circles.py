"""Sparse fillet hypotheses using local tangent walls, without reference geometry."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.point_frame import dominant_xy_frame
from weld_pipeline.circle_candidates import cylinder_columns
from weld_pipeline.tangent_circle_candidates import supported_wall_positions


def propose_local_tangent_circles(xyz,tolerance_mm=.3):
    xyz=np.asarray(xyz,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite XYZ points in millimeters')
    if not np.isfinite(tolerance_mm) or tolerance_mm<=0:
        raise ValueError('Tolerance must be positive and finite')
    anchors=cylinder_columns(xyz) if len(xyz) else np.empty((0,2))
    if len(anchors)<4:
        return [],{'columns':len(anchors),'off_local_wall_seeds':0,'local_neighbors':0,
                   'raw_hypotheses':0,'cylinder_proposals':0}
    frame=dominant_xy_frame(xyz);p=anchors@frame.T
    tree=cKDTree(p);d,ids=tree.query(p,k=min(12,len(p)),workers=2)
    delta=p[ids]-p[:,None]
    on_wall=np.zeros(len(p),bool)
    for axis in range(2):on_wall|=((np.abs(delta[:,:,axis])<=tolerance_mm)&(d<=25.)).sum(1)>=3
    seeds=p[~on_wall];maximum_radius=float(np.linalg.norm(np.ptp(p,axis=0))/2);candidates=[]
    _,neighborhoods=tree.query(seeds,k=min(128,len(p)),workers=2)
    for seed,neighbors in zip(seeds,neighborhoods):
        local=p[neighbors];planes=[supported_wall_positions(local[:,i],3) for i in range(2)]
        for x in planes[0]:
            for y in planes[1]:
                corner=np.array([x,y]);offset=seed-corner;extent=np.abs(offset);signs=np.sign(offset)
                if extent.min()<=tolerance_mm:continue
                radius=extent.sum()+np.sqrt(2*extent.prod())
                if not 2.<=radius<=maximum_radius:continue
                center=corner+signs*radius
                tangent=np.array([[x,center[1]],[center[0],y]])
                errors,tangent_ids=tree.query(tangent)
                if errors.max()>tolerance_mm or np.linalg.norm(tangent-seed,axis=1).min()<2.:continue
                candidates.append({'center_local':center,'radius_mm':float(radius),
                    'tangent_error_mm':float(errors.max()),'seed':seed,'tangent_points':p[tangent_ids]})
    selected=[]
    for c in sorted(candidates,key=lambda c:c['tangent_error_mm']):
        if any(np.linalg.norm(c['center_local']-r['center_local'])+abs(c['radius_mm']-r['radius_mm'])<1. for r in selected):continue
        selected.append(c)
    result=[{'center_xy_mm':(c['center_local']@frame).tolist(),'radius_mm':c['radius_mm'],
             'tangent_error_mm':c['tangent_error_mm'],'arc_seed_xy_mm':(c['seed']@frame).tolist(),
             'tangent_points_xy_mm':(c['tangent_points']@frame).tolist()} for c in selected]
    return result,{'columns':len(p),'off_local_wall_seeds':len(seeds),'local_neighbors':min(128,len(p)),
                   'raw_hypotheses':len(candidates),'cylinder_proposals':len(result)}
