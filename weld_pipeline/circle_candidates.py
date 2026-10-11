"""Point-only vertical-cylinder proposals for horizontal circular welds."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def cylinder_columns(xyz):
    xy=np.round(xyz[:,:2]/.01).astype(np.int64)
    xy-=xy.min(0);key=xy[:,0]*(int(xy[:,1].max())+1)+xy[:,1]
    z=np.round(xyz[:,2]/.01).astype(np.int64);order=np.lexsort((z,key))
    keys=key[order];heights=z[order];starts=np.r_[0,np.flatnonzero(np.diff(keys)!=0)+1]
    ends=np.r_[starts[1:]-1,len(order)-1]
    distinct=np.r_[True,(np.diff(heights)!=0)|(np.diff(keys)!=0)]
    counts=np.add.reduceat(distinct.astype(int),starts)
    keep=(counts>=3)&((heights[ends]-heights[starts])*.01>=5.)
    return xyz[order[starts[keep]],:2]


def fit_xy_circle(xy):
    mean=xy.mean(0);q=xy-mean
    coef,_,rank,_=np.linalg.lstsq(np.column_stack([2*q,np.ones(len(q))]),(q*q).sum(1),rcond=None)
    if rank<3:return None
    r2=coef[2]+coef[:2]@coef[:2]
    if r2<=0:return None
    center=mean+coef[:2];radius=float(np.sqrt(r2))
    residual=np.abs(np.linalg.norm(xy-center,axis=1)-radius)
    return center,radius,residual


def propose_cylinders(xyz,max_samples=25000):
    anchors=cylinder_columns(xyz)
    if len(anchors)<16:return [],{'columns':len(anchors),'local_hypotheses':0}
    tree=cKDTree(anchors);rng=np.random.default_rng(271)
    selected=rng.choice(len(anchors),min(max_samples,len(anchors)),replace=False)
    _,neighbors=tree.query(anchors[selected],k=16,workers=2)
    hypotheses=[];support=[];max_radius=float(np.linalg.norm(np.ptp(xyz[:,:2],axis=0)))
    for ids in neighbors:
        points=anchors[ids];q=points-points.mean(0)
        singular=np.linalg.svd(q,compute_uv=False)
        if singular[0]<1. or singular[1]<.01*singular[0]:continue
        fitted=fit_xy_circle(points)
        if fitted is None:continue
        center,radius,residual=fitted
        if not 2.<=radius<=max_radius or residual.max()>.08:continue
        angle=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
        span=2*np.pi-np.diff(np.r_[np.sort(angle),np.sort(angle)[0]+2*np.pi]).max()
        if span<np.deg2rad(5):continue
        hypotheses.append([*center,radius]);support.append(ids)
    if not hypotheses:return [],{'columns':len(anchors),'local_hypotheses':0}
    hypotheses=np.asarray(hypotheses)
    labels=DBSCAN(eps=1.,min_samples=3).fit_predict(hypotheses)
    result=[]
    for label in sorted(set(labels)-{-1}):
        indices=np.unique(np.concatenate([support[i] for i in np.flatnonzero(labels==label)]))
        fitted=fit_xy_circle(anchors[indices])
        if fitted is None:continue
        center,radius,residual=fitted
        if np.quantile(residual,.95)>.10:continue
        distances=np.abs(np.linalg.norm(anchors-center,axis=1)-radius)
        points=anchors[distances<.12]
        if len(points)<12:continue
        theta=np.sort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
        angular_span=2*np.pi-np.diff(np.r_[theta,theta[0]+2*np.pi]).max()
        if angular_span<np.deg2rad(10):continue
        result.append({'center_xy_mm':center.tolist(),'radius_mm':radius,'xy_support':len(points),
                       'fit_max_residual_mm':float(residual.max()),'occupied_angle_degrees':float(np.degrees(angular_span))})
    return result,{'columns':len(anchors),'sampled_columns':len(selected),'local_hypotheses':len(hypotheses),
                   'cylinder_proposals':len(result)}
