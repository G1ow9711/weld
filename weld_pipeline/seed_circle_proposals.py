"""General partial-circle proposals; no perpendicular-tangent assumption."""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.circle_candidates import cylinder_columns,fit_xy_circle
from weld_pipeline.point_frame import dominant_xy_frame


def propose_seed_circles(xyz,tolerance_mm=.06,max_trials=6000):
    xyz=np.asarray(xyz,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('Finite XYZ required')
    if len(xyz)==0:return [],{'columns':0,'seeds':0,'proposals':0}
    anchors=cylinder_columns(xyz)
    if len(anchors)<6:return [],{'columns':len(anchors),'seeds':0,'proposals':0}
    frame=dominant_xy_frame(xyz);local=anchors@frame.T;tree=cKDTree(local)
    distance,ids=tree.query(local,k=min(12,len(local)),workers=2);delta=local[ids]-local[:,None]
    on_wall=np.zeros(len(local),bool)
    for axis in range(2):on_wall|=((abs(delta[:,:,axis])<=.3)&(distance<=25)).sum(1)>=3
    seeds=anchors[~on_wall];n=len(seeds);rng=np.random.default_rng(391)
    if n<3:return [],{'columns':len(anchors),'seeds':n,'proposals':0}
    count=n*(n-1)*(n-2)//6
    triples=list(itertools.combinations(range(n),3)) if count<=max_trials else [rng.choice(n,3,replace=False) for _ in range(max_trials)]
    maximum=float(np.linalg.norm(np.ptp(xyz[:,:2],axis=0)));hypotheses=[]
    for ids in triples:
        fit=fit_xy_circle(seeds[list(ids)])
        if fit is None:continue
        center,radius,_=fit
        if not 2<=radius<=maximum:continue
        error=abs(np.linalg.norm(seeds-center,axis=1)-radius);support=np.flatnonzero(error<=tolerance_mm)
        if len(support)<6:continue
        fit=fit_xy_circle(seeds[support])
        if fit is None:continue
        center,radius,residual=fit
        if residual.max()>tolerance_mm:continue
        angles=np.sort(np.arctan2(seeds[support,1]-center[1],seeds[support,0]-center[0]))
        span=2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
        if span<np.deg2rad(10):continue
        hypotheses.append({'center_xy_mm':center.tolist(),'radius_mm':radius,'seed_support':len(support),
                           'radial_max_error_mm':float(residual.max()),'angle_span_degrees':float(np.degrees(span))})
    result=[]
    for candidate in sorted(hypotheses,key=lambda c:(-c['seed_support'],c['radial_max_error_mm'])):
        if any(np.linalg.norm(np.array(candidate['center_xy_mm'])-r['center_xy_mm'])+abs(candidate['radius_mm']-r['radius_mm'])<.5 for r in result):continue
        result.append(candidate)
    return result,{'columns':len(anchors),'seeds':n,'trials':len(triples),'raw_hypotheses':len(hypotheses),'proposals':len(result)}
