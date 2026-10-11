"""Local embedding partitions and conservative overlap stitching, without labels.

Embedding distances are compared only within one PTv3 context. Cross-context
identities are linked by shared original point IDs, never by embedding values.
The resulting groups are hypotheses; ambiguous contacts need multiple owners.
"""
import numpy as np
from scipy.spatial.distance import cdist


def local_partition(values, radius=.4):
    x=np.asarray(values,np.float32)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('A finite, nonempty embedding matrix is required')
    # Deterministic farthest-point covering; no assumed instance count or cap.
    seed=int(np.argmin(np.sum((x-x.mean(0))**2,axis=1)))
    seeds=[seed];best=np.sum((x-x[seed])**2,axis=1);labels=np.zeros(len(x),np.int32)
    while float(best.max())>radius**2:
        seed=int(np.argmax(best));distance=np.sum((x-x[seed])**2,axis=1)
        take=distance<best;labels[take]=len(seeds);best[take]=distance[take];seeds.append(seed)
    # Refit centers without combining distinct local instances.
    centers=x[seeds].copy()
    for _ in range(3):
        counts=np.bincount(labels,minlength=len(centers));sums=np.zeros_like(centers)
        np.add.at(sums,labels,x);nonempty=counts>0;centers[nonempty]=sums[nonempty]/counts[nonempty,None]
        updated=np.empty_like(labels)
        for start in range(0,len(x),1024):
            updated[start:start+1024]=cdist(x[start:start+1024],centers,'sqeuclidean').argmin(1)
        if np.array_equal(updated,labels):break
        labels=updated
    _,labels=np.unique(labels,return_inverse=True)
    return labels.astype(np.int32)


def stitch(node_blocks,pairs,votes,min_votes=16,min_agreement=.9):
    """Merge mutually consistent overlap votes, vetoing within-block conflicts."""
    blocks=np.asarray(node_blocks,np.int32);n=len(blocks)
    pairs=np.asarray(pairs,np.int64).reshape(-1,2);votes=np.asarray(votes,np.int64)
    parent=np.arange(n,dtype=np.int32);size=np.ones(n,np.int64)
    blocksets=[{int(b)} for b in blocks]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    if len(pairs)==0:return parent,dict(accepted=0,conflicts=0,eligible=0,raw_pairs=0)
    assert len(pairs)==len(votes) and np.all(votes>0) and pairs.min()>=0 and pairs.max()<n
    assert np.all(blocks[pairs[:,0]]!=blocks[pairs[:,1]])
    # Denominators are all votes between a local cluster and another block,
    # not its total point count outside that block's visible overlap.
    denominator={}
    for (a,b),v in zip(pairs,votes):
        for node,other in ((a,b),(b,a)):
            key=(int(node),int(blocks[other]));denominator[key]=denominator.get(key,0)+int(v)
    agreement=np.array([min(v/denominator[(int(a),int(blocks[b]))],
                            v/denominator[(int(b),int(blocks[a]))]) for (a,b),v in zip(pairs,votes)])
    eligible=np.flatnonzero((votes>=min_votes)&(agreement>=min_agreement))
    order=sorted(eligible,key=lambda i:(-agreement[i],-int(votes[i]),int(pairs[i,0]),int(pairs[i,1])))
    accepted=conflicts=0
    for i in order:
        a,b=map(root,map(int,pairs[i]))
        if a==b:continue
        if blocksets[a]&blocksets[b]:conflicts+=1;continue
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b];blocksets[a].update(blocksets[b]);blocksets[b].clear();accepted+=1
    final=np.array([root(i) for i in range(n)],np.int32)
    _,final=np.unique(final,return_inverse=True)
    return final.astype(np.int32),dict(accepted=accepted,conflicts=conflicts,eligible=len(eligible),raw_pairs=len(pairs))


def controls():
    rng=np.random.default_rng(973)
    x=np.r_[rng.normal(0,.015,(64,16)),rng.normal(0,.015,(64,16))+np.r_[1.5,np.zeros(15)]]
    a=local_partition(x);assert len(np.unique(a))==2 and np.all(a[:64]==a[0]) and np.all(a[64:]==a[64]) and a[0]!=a[64]
    q=np.linalg.qr(rng.normal(size=(16,16)))[0];b=local_partition(x@q+3)
    assert np.array_equal(a[:,None]==a,b[:,None]==b)
    merged,s=stitch([0,0,1,1,2],[[0,2],[1,3],[2,4],[3,4]],[40,40,40,16])
    assert merged[0]==merged[2] and merged[1]==merged[3] and merged[0]!=merged[1]
    # Strong pairwise bridges must still not combine two IDs from one block.
    merged,s=stitch([0,0,1,2],[[0,2],[1,3],[2,3]],[40,40,40])
    assert merged[0]!=merged[1] and s['conflicts']==1
    separate,_=stitch([0,1],[[0,1]],[1]);assert separate[0]!=separate[1]
    return dict(separated_local_groups=True,embedding_isometry_control=True,
                overlap_bridge_conflict_veto=True,one_point_bridge_rejected=True)
