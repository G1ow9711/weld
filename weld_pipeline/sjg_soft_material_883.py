"""Soft signed agglomeration of input-derived patches, with overlap attachments."""
import heapq
import numpy as np


def merge_signed_graph(node_count,left,right,cost,mass):
    """Positive summed edge gain favors a merge; negative edges are not absolute vetoes."""
    left=np.asarray(left,np.int64);right=np.asarray(right,np.int64);cost=np.asarray(cost,np.float64)
    parent=np.arange(node_count,dtype=np.int64);size=np.asarray(mass,np.int64).copy();adj=[{} for _ in range(node_count)]
    for u,v,w in zip(left,right,cost):
        u=int(u);v=int(v);w=float(w)
        if u==v:continue
        adj[u][v]=adj[u].get(v,0.)+w;adj[v][u]=adj[v].get(u,0.)+w
    heap=[(-w,u,v) for u in range(node_count) for v,w in adj[u].items() if u<v and w>0];heapq.heapify(heap)
    merges=0;gain=0.
    while heap:
        minus,u,v=heapq.heappop(heap);weight=-minus
        if parent[u]!=u or parent[v]!=v or adj[u].get(v)!=weight:continue
        if size[u]<size[v] or (size[u]==size[v] and u>v):u,v=v,u
        parent[v]=u;size[u]+=size[v];merges+=1;gain+=weight
        adj[u].pop(v,None);adj[v].pop(u,None)
        for other,w in list(adj[v].items()):
            assert parent[other]==other and other!=u
            value=adj[u].get(other,0.)+w;adj[u][other]=value;adj[other].pop(v,None);adj[other][u]=value
            if value>0:heapq.heappush(heap,(-value,min(u,other),max(u,other)))
        adj[v].clear()
    for i in range(node_count):
        j=i
        while parent[j]!=j:j=int(parent[j])
        parent[i]=j
    _,labels=np.unique(parent,return_inverse=True)
    merged_gain=float(cost[(left!=right)&(labels[left]==labels[right])].sum())
    assert np.isclose(gain,merged_gain,rtol=1e-9,atol=1e-7)
    # No remaining cross-component aggregate gain may be positive.
    for u in range(node_count):
        if parent[u]==u:assert all(w<=0 for w in adj[u].values())
    return labels.astype(np.int32),dict(merges=merges,total_positive_merge_gain=gain,final_groups=len(np.unique(labels)),merge_gain_accounting_verified=True)


def decode_soft(neighbors,flat,logit,seed_offsets,seed_membership):
    n=len(neighbors);seed_counts=np.diff(seed_offsets);single=seed_counts==1
    initial=np.full(n,-1,np.int32);initial[single]=seed_membership[seed_offsets[:-1][single]]
    atoms,remap=np.unique(initial[single],return_inverse=True);initial[single]=remap.astype(np.int32);m=len(atoms)
    a=flat//neighbors.shape[1];b=neighbors.ravel()[flat].astype(np.int64)
    assert np.all(b>=0) and np.isfinite(logit).all();weight=np.clip(-np.asarray(logit,np.float64),-5.,5.)
    both=(initial[a]>=0)&(initial[b]>=0)&(initial[a]!=initial[b]);u=initial[a[both]];v=initial[b[both]]
    lo=np.minimum(u,v).astype(np.int64);hi=np.maximum(u,v).astype(np.int64);codes=lo*m+hi
    keys,inverse=np.unique(codes,return_inverse=True);cost=np.bincount(inverse,weights=weight[both],minlength=len(keys))
    mass=np.bincount(initial[single],minlength=m)
    mapping,stats=merge_signed_graph(m,keys//max(1,m),keys%max(1,m),cost,mass)
    component=np.full(n,-1,np.int32);component[single]=mapping[initial[single]];groups=stats['final_groups']
    # Ambiguous/unassigned old points can attach to several new groups but never
    # act as bridges during the patch merge. No GT or desired object count enters.
    cross=(component[a]<0)&(component[b]>=0);reverse=(component[b]<0)&(component[a]>=0)
    point=np.r_[a[cross],b[reverse]];group=np.r_[component[b[cross]],component[a[reverse]]]
    votes=np.r_[weight[cross],weight[reverse]];codes=point*max(1,groups)+group
    keys,inv=np.unique(codes,return_inverse=True);sum_vote=np.bincount(inv,weights=votes,minlength=len(keys));keys=keys[sum_vote>0]
    attach_point=keys//max(1,groups);attach_group=(keys%max(1,groups)).astype(np.int32)
    rows=np.r_[np.flatnonzero(single),attach_point];cols=np.r_[component[single],attach_group]
    order=np.lexsort((cols,rows));rows=rows[order];cols=cols[order].astype(np.int32)
    counts=np.bincount(rows,minlength=n);offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    assert len(offsets)==n+1 and offsets[-1]==len(cols)
    stats.update(original_points=n,seed_atoms=m,old_single_membership_points=int(single.sum()),
        unknown_membership_points=int((counts==0).sum()),multi_membership_points=int((counts>1).sum()),
        hypothesis_count=groups,soft_negative_edges_not_hard_constraints=True,no_small_group_removal=True)
    return dict(offsets=offsets,membership=cols,seed_single=single.astype(np.uint8),single_component=component),stats


def controls():
    labels,s=merge_signed_graph(3,[0,1,0],[1,2,2],[3.,1.,-5.],[1,1,1])
    assert labels[0]==labels[1] and labels[0]!=labels[2] and s['merges']==1
    labels,s=merge_signed_graph(3,[0,1,0],[1,2,2],[3.,4.,-1.],[1,1,1])
    assert len(np.unique(labels))==1 and s['merges']==2
    near=np.array([[1,2],[0,2],[0,1]],np.int32);flat=np.arange(6)
    # 0 and 1 have repulsive evidence; old ambiguous point 2 may join both.
    logit=np.array([2.,-2.,2.,-2.,-2.,-2.]);off=np.array([0,1,2,4]);mem=np.array([0,1,0,1])
    out,s=decode_soft(near,flat,logit,off,mem);assert np.diff(out['offsets']).tolist()==[1,1,2] and s['hypothesis_count']==2
    out,s=decode_soft(near,flat,logit,np.zeros(4,np.int64),np.empty(0,np.int32));assert s['unknown_membership_points']==3
    return dict(negative_evidence_can_stop_merge=True,combined_positive_evidence_can_overrule_one_negative=True,
        contact_overlap_preserved=True,no_seed_means_unknown_not_dropped=True)
