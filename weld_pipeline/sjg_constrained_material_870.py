"""Input-only signed grouping with boundary overlap and explicit unknown points.

    Experimental structure hypotheses, not approved material identity. Predicted
    boundary points cannot bridge components. Every observed negative relation
    forbids its endpoints sharing a returned hypothesis, even at boundaries.
"""
import numpy as np


def constrained_memberships(neighbors,distance,boundary,threshold=.5):
    n=len(neighbors);interior=np.asarray(boundary)<.5
    valid=(neighbors>=0)&np.isfinite(distance)
    a=np.broadcast_to(np.arange(n)[:,None],neighbors.shape)[valid]
    b=neighbors[valid].astype(np.int64);d=distance[valid]
    neg=d>threshold;na=a[neg];nb=b[neg]
    positive=~neg;pa=a[positive];pb=b[positive];pd=d[positive]
    parent=np.arange(n,dtype=np.int64);size=np.ones(n,np.int64);mutex={}

    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=int(parent[x])
        return x

    for u,v in zip(na,nb):
        u=int(u);v=int(v)
        if interior[u] and interior[v]:
            mutex.setdefault(u,set()).add(v);mutex.setdefault(v,set()).add(u)
    keep=interior[pa]&interior[pb];ia=pa[keep];ib=pb[keep];values=pd[keep]
    rejected=0;merged=0
    # Stable original-edge order breaks equal-distance ties, independent of GT.
    for j in np.argsort(values,kind='stable'):
        u=root(int(ia[j]));v=root(int(ib[j]))
        if u==v:continue
        if v in mutex.get(u,()):rejected+=1;continue
        if size[u]<size[v] or (size[u]==size[v] and u>v):u,v=v,u
        parent[v]=u;size[u]+=size[v];merged+=1
        old=mutex.pop(v,set())
        for other in old:
            assert parent[other]==other and other!=u
            mutex[other].discard(v);mutex[other].add(u)
        if old:mutex.setdefault(u,set()).update(old)
    ids=np.flatnonzero(interior);roots=np.array([root(int(i)) for i in ids],np.int64)
    _,labels=np.unique(roots,return_inverse=True)
    component=np.full(n,-1,np.int32);component[ids]=labels.astype(np.int32)
    memberships={}
    # Boundaries may attach to several existing interior hypotheses, never join them.
    for u,v,val in zip(pa,pb,pd):
        u=int(u);v=int(v)
        if interior[u]==interior[v]:continue
        if interior[u]:u,v=v,u
        label=int(component[v]);score=float(threshold-val)
        choices=memberships.setdefault(u,{})
        choices[label]=max(choices.get(label,-float('inf')),score)
    removed=0
    # All negatives win, including contradictory directed observations.
    for u,v in zip(na,nb):
        u=int(u);v=int(v)
        if interior[u] and interior[v]:
            assert component[u]!=component[v];continue
        if interior[v]:u,v=v,u
        if interior[u]:
            choices=memberships.get(v,{})
            if int(component[u]) in choices:del choices[int(component[u])];removed+=1
        else:
            left=memberships.get(u,{});right=memberships.get(v,{})
            for label in sorted(left.keys()&right.keys()):
                if left[label]<right[label] or (left[label]==right[label] and u>v):del left[label]
                else:del right[label]
                removed+=1
    counts=interior.astype(np.int64)
    for point,choices in memberships.items():counts[point]=len(choices)
    offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)];membership=np.empty(int(offsets[-1]),np.int32)
    membership[offsets[ids]]=component[ids]
    for point,choices in memberships.items():membership[offsets[point]:offsets[point+1]]=sorted(choices)
    # A point remains in the denominator even when its membership list is empty.
    return dict(offsets=offsets,membership=membership,interior_component=component,
        boundary_uncertain=(~interior).astype(np.uint8)),dict(original_points=n,interior_points=int(interior.sum()),
        boundary_points=int((~interior).sum()),hypotheses=int(len(np.unique(labels))),
        unknown_membership_points=int((counts==0).sum()),multi_membership_points=int((counts>1).sum()),
        observed_negative_edges=int(neg.sum()),positive_merges=merged,positive_merges_rejected=rejected,
        conflicting_boundary_memberships_removed=removed,negative_constraints_preserved=True)


def verify_constraints(neighbors,distance,output,threshold=.5):
    off=output['offsets'];mem=output['membership'];n=len(neighbors)
    assert len(off)==n+1 and off[0]==0 and off[-1]==len(mem) and np.all(np.diff(off)>=0)
    valid=(neighbors>=0)&np.isfinite(distance)&(distance>threshold)
    a=np.broadcast_to(np.arange(n)[:,None],neighbors.shape)[valid];b=neighbors[valid]
    for u,v in zip(a,b):
        assert not np.intersect1d(mem[off[u]:off[u+1]],mem[off[v]:off[v+1]],assume_unique=True).size
    return int(len(a))


def controls():
    # Positive chain cannot erase an explicit negative across its endpoints.
    near=np.array([[1,2],[0,2],[0,1]],np.int32);d=np.array([[.1,.9],[.1,.2],[.9,.2]],np.float32)
    o,s=constrained_memberships(near,d,np.zeros(3));verify_constraints(near,d,o)
    assert s['hypotheses']==2 and s['positive_merges_rejected']>0
    # Two disconnected interiors share a boundary point without being merged.
    near=np.array([[2,1],[2,0],[0,1]],np.int32);d=np.array([[.1,.9],[.1,.9],[.1,.1]],np.float32)
    o,s=constrained_memberships(near,d,np.array([0,0,1.]));verify_constraints(near,d,o)
    assert s['hypotheses']==2 and s['multi_membership_points']==1 and np.diff(o['offsets']).tolist()==[1,1,2]
    # Contradictory boundary observation is resolved by the negative edge.
    d[0,0]=.9;o,s=constrained_memberships(near,d,np.array([0,0,1.]));verify_constraints(near,d,o)
    assert s['conflicting_boundary_memberships_removed']==1 and np.diff(o['offsets']).tolist()==[1,1,1]
    o,s=constrained_memberships(near,np.full((3,2),np.nan),np.ones(3));verify_constraints(near,np.full((3,2),np.nan),o)
    assert s['unknown_membership_points']==3 and len(o['membership'])==0
    return dict(negative_chain_constraint=True,contact_multiple_membership=True,
        negative_overrides_positive=True,unknown_retains_all_points=True,optimizer_updates=0)
