"""Estimate the dominant plate frame, including densely sampled CAD exports."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


def dominant_xy_frame(xyz):
    if len(xyz)<4 or not np.isfinite(xyz).all():
        raise ValueError('At least four finite points are required')
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    major=z[np.argsort(counts)[-2:]]
    q=xyz[~np.isin(np.round(xyz[:,2],2),major),:2]
    if len(q)<20:q=xyz[:,:2]
    q=np.unique(np.round(q,3),axis=0)
    if len(q)<4:raise ValueError('Insufficient distinct XY points for a plate frame')
    if len(q)>50000:q=q[np.random.default_rng(17).choice(len(q),50000,replace=False)]
    distances,ids=cKDTree(q).query(q,k=min(4,len(q)),workers=2)
    delta=q[ids[:,1:]]-q[:,None]
    length=distances[:,1:]
    good=(length>2)&(length<50)
    # The old 2 mm lower bound rejects every neighbour of dense sub-mm grids.
    # Keep its result when supported; otherwise use resolved sub-mm pairs.
    if good.sum()<16:good=(length>.05)&(length<50)
    if not good.any():raise ValueError('No resolved local XY directions')
    angle=np.arctan2(delta[:,:,1][good],delta[:,:,0][good])%(np.pi/2)
    bins=1800
    hist=np.bincount(np.minimum((angle/(np.pi/2)*bins).astype(int),bins-1),minlength=bins)
    peak=np.argmax(gaussian_filter1d(hist.astype(float),1,mode='wrap'))
    center=(peak+.5)/bins*(np.pi/2)
    residual=(angle-center+np.pi/4)%(np.pi/2)-np.pi/4
    selected=angle[np.abs(residual)<np.deg2rad(.2)]
    if not len(selected):raise ValueError('Plate direction peak has no support')
    theta=np.angle(np.mean(np.exp(4j*selected)))/4
    return np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]],float)
