"""Pure-input ordinal interval inference and complete vectorized link features."""
import numpy as np
from weld_pipeline.sjg_joint_forest_train_1109 import attributes
from weld_pipeline.sjg_ordered_interval_targets_1128 import decode_count_category
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry


def predict_instances(case, models, include_instances=()):
    obj=models['object'];reg=models['endpoint'];K=int(models['K'])
    columns=int(case['x'].shape[1])
    assert obj.n_features_in_==columns and reg.n_features_in_==columns+1
    classes=[decode_count_category(v) for v in obj.classes_]
    cats=np.asarray([v[0] for v in classes]);counts=np.asarray([v[1] for v in classes])
    states={};invalid={};admitted=0;max_probability=0.;saturated=set()
    required={tuple(map(int,v)) for v in include_instances}
    for first in range(0,len(case['x']),32768):
        x=np.asarray(case['x'][first:first+32768],dtype=np.float32)
        assert np.isfinite(x).all(), 'Object predictor requires finite float32 input'
        probs=obj.predict_proba(x)
        required_local={(r-first,k) for r,k in required if first<=r<first+len(x)}
        for k in range(1,K+1):
            cols=np.flatnonzero(counts>=k)
            if not len(cols):continue
            presence=probs[:,cols].sum(1);max_probability=max(max_probability,float(presence.max(initial=0)))
            selected=set(np.flatnonzero(presence>=.5).tolist());admitted+=len(selected)
            if k==K:saturated.update(first+r for r in selected)
            selected.update(r for r,slot in required_local if slot==k)
            if not selected:continue
            take=np.asarray(sorted(selected),np.int64)
            ep_x=np.empty((len(take),columns+1),np.float32);ep_x[:,:columns]=x[take];ep_x[:,columns]=k/K
            assert np.isfinite(ep_x).all(), 'Endpoint predictor requires finite float32 input'
            ratios=reg.predict(ep_x)
            category_scores=np.zeros((len(take),16),np.float64)
            for c in cols:category_scores[:,cats[c]-1]+=probs[take,c]
            chosen=category_scores.argmax(1)+1
            for local,q,value in zip(take,ratios,chosen):
                row=int(first+local);fragment=int(case['canonical'][row//2]);left,right=case['graph'].spans[fragment,:2]
                node=dict(object_row=row,k=k,fragment=fragment,side=row%2,probability=float(presence[local]),**attributes(value))
                try:
                    q=np.asarray(q,np.float64)
                    if not np.isfinite(q).all() or np.any(q<0) or np.any(q>1) or not q[1]>q[0]:
                        raise ValueError('Invalid ordinal endpoints: no sorting or clipping repair')
                    node.update(lo=float(left+q[0]*(right-left)),hi=float(left+q[1]*(right-left)))
                    node_geometry(node,case['graph']);states[row,k]=node
                except ValueError as error:
                    invalid[row,k]=dict(node=node,reason=str(error),counted_false_children=1)
    return dict(nodes=[states[k] for k in sorted(states)],invalid=[invalid[k] for k in sorted(invalid)],
        admitted_instances=admitted,objects_scored=len(case['x']),maximum_probability=max_probability,
        saturated_object_rows=sorted(saturated),K=K,GT_endpoint_overrides=0)


def link_blocks(nodes,graph,gap_resolver=None):
    """Every legal directed pair; fast local predicates, exact gap fallback."""
    n=len(nodes)
    if not n:return
    geometry=[node_geometry(node,graph) for node in nodes]
    ends=np.asarray([g[0] for g in geometry]);tangents=np.asarray([g[1] for g in geometry])
    fragments=np.asarray([n['fragment'] for n in nodes],np.int64)
    profiles=np.asarray([graph.profile(f) for f in fragments],np.int64)
    boxes=np.asarray([graph.bounds(f) for f in fragments])
    sides=np.asarray([n['side'] for n in nodes]);forward=np.asarray([n['forward'] for n in nodes])
    weld=np.asarray([n['weld_type'] for n in nodes]);spans=graph.spans[fragments]
    faces=np.full((n,2),-1,np.int64);roles=faces.copy();arc=np.zeros(n,bool);closed=np.zeros(n,bool);has_gaps=np.zeros(n,bool)
    for i,p in enumerate(profiles):
        f=graph.profiles[p].get('face_indices',[])[:2];faces[i,:len(f)]=f
        ordered=f[::-1] if sides[i] else f;roles[i,:len(ordered)]=ordered
        curve=graph.curve(p);arc[i]=curve.kind=='arc';closed[i]=arc[i] and curve.arc.is_closed
        has_gaps[i]=bool(graph.profiles[p].get('unobserved_gaps_mm'))
    gap_cache={}
    for i,node in enumerate(nodes):
        delta=np.maximum(0.,np.maximum(boxes[i,0]-boxes[:,1],boxes[:,0]-boxes[i,1]))
        broad=np.linalg.norm(delta,axis=1);distance=np.linalg.norm(ends[i,1]-ends[:,0],axis=1)
        same_profile=profiles==profiles[i]
        adjacent=fragments==fragments[i]+(1 if forward[i] else -1)
        if closed[i]:
            first,last=graph.offsets[profiles[i]],graph.offsets[profiles[i]+1]-1
            adjacent |= ((fragments[i]==first)&(fragments==last))|((fragments[i]==last)&(fragments==first))
        same=same_profile&(forward==forward[i])&(sides==sides[i])&(weld==weld[i])&adjacent
        shared=(broad<=6.)&(distance<=6.)
        gap=(distance>0)&(same_profile|(broad<=6.))
        common=np.zeros(n,bool)
        for face in faces[i]:
            if face>=0:common|=np.any(faces==face,axis=1)
        pending=np.flatnonzero((distance>0)&~gap&common&(has_gaps|has_gaps[i]))
        if gap_resolver is not None:
            gap[pending]=gap_resolver.resolve(fragments[i],fragments[pending])
        else:
            for j in pending:
                key=tuple(sorted((int(fragments[i]),int(fragments[j]))))
                if key not in gap_cache:gap_cache[key]=graph.connection(*key,'POSITIVE_GAP')
                gap[j]=gap_cache[key]
        legal=np.c_[same,shared,gap];legal[i]=False;take=np.flatnonzero(legal.any(1))
        if not len(take):continue
        features=np.zeros((len(take),32),np.float32)
        features[:,0]=broad[take];features[:,1]=distance[take]
        features[:,2:5]=tangents[i,1];features[:,5:8]=tangents[take,0]
        features[:,8]=tangents[take,0]@tangents[i,1]
        features[:,9]=not arc[i];features[:,10]=arc[i];features[:,11]=~arc[take];features[:,12]=arc[take]
        features[:,13]=same_profile[take]
        for col,(a,b) in enumerate(((0,0),(0,1),(1,0),(1,1)),14):
            features[:,col]=(roles[i,a]>=0)&(roles[take,b]>=0)&(roles[i,a]==roles[take,b])
        features[:,18]=sides[i]==0;features[:,19]=sides[i]==1
        features[:,20]=sides[take]==0;features[:,21]=sides[take]==1
        features[:,22]=spans[i,2];features[:,23]=spans[take,2]
        features[:,24]=spans[i,3];features[:,25]=spans[take,3];features[:,28]=1.
        features[:,29:32]=legal[take]
        yield np.c_[np.full(len(take),i,np.int32),take.astype(np.int32)],features,legal[take]
