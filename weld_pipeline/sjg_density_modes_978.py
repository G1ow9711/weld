"""Density-mode instance seeds; all input points retain a predicted assignment."""
import numpy as np
from scipy.spatial.distance import cdist
from weld_pipeline.sjg_overlap_instances_973 import local_partition


def density_partition(values,radius=.4,seeds=128,iterations=12):
    x=np.asarray(values,np.float32)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite embeddings required')
    anchors=np.linspace(0,len(x)-1,min(seeds,len(x)),dtype=np.int64)
    centers=x[anchors].astype(np.float64);counts=np.zeros(len(centers),np.int64)
    for _ in range(iterations):
        distance=cdist(centers,x,'sqeuclidean');inside=distance<=radius**2;counts=inside.sum(1)
        updated=inside@x/np.maximum(counts[:,None],1)
        updated[counts==0]=centers[counts==0]
        change=float(np.linalg.norm(updated-centers,axis=1).max());centers=updated
        if change<1e-4:break
    # Select modes by density, suppressing overlapping basins rather than
    # placing one seed at every farthest tail point and splitting the bulk.
    selected=[]
    for i in np.argsort(-counts,kind='stable'):
        if counts[i]<8:continue
        if selected and np.min(np.linalg.norm(centers[selected]-centers[i],axis=1))<=radius:continue
        selected.append(int(i))
    if not selected:return local_partition(x,radius)
    labels=cdist(x,centers[selected],'sqeuclidean').argmin(1)
    _,labels=np.unique(labels,return_inverse=True)
    return labels.astype(np.int32)


def controls():
    rng=np.random.default_rng(978)
    x=np.r_[rng.normal(0,.02,(256,16)),rng.normal(0,.02,(256,16))+np.r_[1.5,np.zeros(15)]]
    labels=density_partition(x)
    assert len(np.unique(labels))==2 and np.all(labels[:256]==labels[0]) and np.all(labels[256:]==labels[256]) and labels[0]!=labels[256]
    # A single diffuse density basin must not become dozens of tail seeds.
    diffuse=rng.normal(0,.08,(1024,16));labels=density_partition(diffuse)
    assert len(np.unique(labels))==1
    return dict(two_dense_groups_separated=True,diffuse_single_mode_not_fragmented=True,all_points_assigned=True)
