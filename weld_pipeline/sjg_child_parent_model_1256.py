"""Child validity, duplicate suppression, then explicit whole-parent assembly."""
import argparse
import gc
import json
import pickle
import time
import traceback
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_child_parent_supervision_1255 import (
    SCHEMA,TRAIN,child_inputs,state_geometry,link_features)
from weld_pipeline.sjg_additive_parent_precision_1254 import endpoint_distances,candidate_pair_possible,pair_features
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations


def solve_oriented(count,edges,weights,limit=10.):
    """Exactly one orientation per child, path/cycle degree constraints, no gap cycles."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    begin=time.perf_counter();edges=np.asarray(edges,int).reshape(-1,3);weights=np.asarray(weights,float)
    assert len(edges)==len(weights) and np.all(weights>0) and np.isfinite(weights).all()
    if not count:return dict(states=[],links=[],objective=0.,gap_cycle_cuts=0,seconds=0.,optimal=True)
    width=2*count+len(edges);objective=np.r_[np.zeros(2*count),-weights]
    rows=[];cols=[];values=[];low=[];high=[]
    def constraint(entries,lo,hi):
        ri=len(low);low.append(lo);high.append(hi)
        for col,v in entries:rows.append(ri);cols.append(col);values.append(v)
    for i in range(count):constraint([(2*i,1),(2*i+1,1)],1,1)
    incoming=[[] for _ in range(2*count)];outgoing=[[] for _ in range(2*count)]
    for e,(a,b,kind) in enumerate(edges):
        assert a//2!=b//2 and kind in (1,2)
        col=2*count+e;incoming[b].append(col);outgoing[a].append(col)
        constraint([(col,1),(int(a),-1)],-np.inf,0);constraint([(col,1),(int(b),-1)],-np.inf,0)
    for i in range(2*count):
        for attached in (incoming[i],outgoing[i]):
            if attached:constraint([(col,1) for col in attached]+[(i,-1)],-np.inf,0)
    cuts=0;seen_cuts=set()
    while True:
        remaining=limit-(time.perf_counter()-begin)
        if remaining<=0:raise TimeoutError('Complete parent assembly exceeded cumulative budget')
        matrix=coo_matrix((values,(rows,cols)),shape=(len(low),width)).tocsc()
        matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
        result=milp(objective,integrality=np.ones(width,np.int32),bounds=Bounds(0,1),
            constraints=LinearConstraint(matrix,np.asarray(low),np.asarray(high)),
            options={'time_limit':remaining,'presolve':True})
        if result.status!=0 or result.x is None:raise RuntimeError('No complete optimal parent assembly: '+str(result.message))
        states=np.flatnonzero(result.x[:2*count]>.5).tolist();chosen=np.flatnonzero(result.x[2*count:]>.5).tolist()
        assert sorted(i//2 for i in states)==list(range(count))
        successor={int(edges[e,0]):(int(edges[e,1]),e) for e in chosen};visited=set();bad=[]
        for start in states:
            if start in visited:continue
            order=[];position={};node=start
            while node not in visited and node not in position and node in successor:
                position[node]=len(order);order.append(node);node=successor[node][0]
            if node in position:
                cycle=[successor[v][1] for v in order[position[node]:]]
                if any(edges[e,2]==2 for e in cycle):bad.append(tuple(sorted(cycle)))
            visited.update(order)
        if not bad:break
        for cycle in bad:
            assert cycle not in seen_cuts;seen_cuts.add(cycle)
            constraint([(2*count+e,1) for e in cycle],-np.inf,len(cycle)-1);cuts+=1
    return dict(states=states,links=edges[chosen].tolist(),objective=float(weights[chosen].sum()),
        gap_cycle_cuts=cuts,seconds=time.perf_counter()-begin,optimal=True)


def controls(dest):
    # Both traversal directions of a 3-child ordered path, no positive omitted.
    edges=np.array([[0,2,1],[2,4,2],[5,3,2],[3,1,1]])
    r=solve_oriented(3,edges,np.ones(4))
    assert len(r['states'])==3 and len(r['links'])==2 and r['objective']==2
    shared=np.array([[0,2,1],[2,4,1],[4,0,1]])
    closed=solve_oriented(3,shared,np.ones(3));assert len(closed['links'])==3
    gap=shared.copy();gap[2,2]=2
    opened=solve_oriented(3,gap,np.ones(3));assert len(opened['links'])==2 and opened['gap_cycle_cuts']>=1
    lone=solve_oriented(1,[],[]);assert len(lone['states'])==1 and not lone['links']
    # Exhaustive feasible-link score oracle for the above small graphs.
    for ee,solution in ((edges,r),(shared,closed),(gap,opened)):
        best=0
        for bits in range(1<<len(ee)):
            subset=[tuple(map(int,ee[i])) for i in range(len(ee)) if bits>>i&1]
            state_sets=[{v for edge in subset for v in edge[:2] if v//2==i} for i in range(3)]
            if any(len(v)>1 for v in state_sets):continue
            if len({a for a,b,k in subset})!=len(subset) or len({b for a,b,k in subset})!=len(subset):continue
            successor={a:(b,k) for a,b,k in subset};valid=True
            for start in successor:
                node=start;path=[]
                while node in successor and node not in path:
                    path.append(node);node=successor[node][0]
                if node in path and any(successor[v][1]==2 for v in path[path.index(node):]):valid=False
            if valid:best=max(best,len(subset))
        assert best==solution['objective']
    write(dest/'assembly_controls.json',dict(status='PASS',three_child_path=True,
        legitimate_shared_closed_cycle=True,gap_cycle_excluded=True,isolated_positive_retained=True,
        one_orientation_per_child=True,brute_force_objective_equal=True))


def train(dest,input_dest):
    begin=time.perf_counter();assert read(input_dest/'prefit_audit.json')['status']=='PASS'
    controls(dest);r=read(input_dest/'supervision_receipt.json');models={};traces={};bindings={}
    train_keys=tuple(c['source'] for c in r['cases'])
    assert len(train_keys)==len(set(train_keys)) and '64' not in train_keys
    assert r['G']==sum(c['G'] for c in r['cases'])
    for name in ('validity','duplicate','link'):
        xs=[];ys=[];ws=[]
        for case in r['cases']:
            key=case['source'];folder=input_dest/f'source_{key}';receipt=read(folder/'receipt.json')
            assert digest(folder/'training_targets.npz')==case['labels_sha256']
            assert digest(folder/'features.npy')==receipt['feature_sha256']
            bindings[key]=dict(labels=case['labels_sha256'],features=receipt['feature_sha256'])
            with np.load(folder/'training_targets.npz') as z:
                if name=='validity':x=np.load(folder/'features.npy');y=z['validity'];base=np.ones(len(y))
                elif name=='duplicate':x=z['duplicate_x'];y=z['duplicate_y'];base=np.ones(len(y))
                else:x=z['link_x'];y=z['link_y'];base=z['link_weight']
                mask=y>=0;x=x[mask];y=y[mask];base=base[mask]
            if not len(y):continue
            classes=np.unique(y);w=np.zeros(len(y))
            for c in classes:w[y==c]=base[y==c]/(len(train_keys)*len(classes)*base[y==c].sum())
            xs.append(x);ys.append(y);ws.append(w)
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
        observed=set(y.tolist())
        assert (0 in observed and observed-{0} and observed<={0,1,2}) if name=='link' else observed=={0,1}
        model=ExtraTreesClassifier(n_estimators=16,max_depth=16,min_samples_leaf=2,
            max_features=.75,bootstrap=False,random_state=1031,n_jobs=2,warm_start=True)
        trace=[]
        for batch in range(1,5):
            check_resources();model.set_params(n_estimators=16*batch);model.fit(x,y,sample_weight=w)
            trace.append(dict(batch=batch,trees=len(model.estimators_),seconds=time.perf_counter()-begin))
            print('FOREST',name,'BATCH',batch,'/4 TREES',len(model.estimators_),'/64',flush=True)
        model.set_params(n_jobs=1,warm_start=False);models[name]=model;traces[name]=trace
        assert np.array_equal(model.predict_proba(x),model.predict_proba(x[::-1])[::-1])
    payload=dict(schema=SCHEMA,models=models,input_run=input_dest.name,threshold=.5)
    atomic_binary_write(dest/'models.pkl',lambda f:pickle.dump(payload,f,protocol=5))
    with (dest/'models.pkl').open('rb') as f:loaded=pickle.load(f)
    assert np.array_equal(models['link'].predict_proba(x),loaded['models']['link'].predict_proba(x))
    write(dest/'training_receipt.json',dict(status='CHILD_VALIDITY_DUPLICATE_AND_PARENT_LINK_MODELS_TRAINED',
        TRAIN=list(train_keys),G=r['G'],traces=traces,bindings=bindings,model_sha256=digest(dest/'models.pkl'),
        input_run=input_dest.name,DEV_reads=0,STEP_reads=0,full_contract_pass=False,
        learned_link_classes=models['link'].classes_.tolist(),
        unobserved_link_classes_not_learned=sorted({0,1,2}-set(models['link'].classes_.tolist())),
        whole_parent_acceptance_still_required=True,seconds=time.perf_counter()-begin))


def assemble(context,model,dest):
    state=state_geometry(context);count=len(context.parents);ids=[]
    for a in range(2*count):
        if state['closed'][a]:continue
        for b in range(2*count):
            if a//2!=b//2 and not state['closed'][b]:ids.append((a,b))
    ids=np.asarray(ids,int).reshape(-1,2)
    features=link_features(state,ids);probability=np.zeros((len(ids),3))
    if len(ids):probability[:,model.classes_.astype(int)]=model.predict_proba(features)
    edges=[];weights=[]
    for i,(a,b) in enumerate(ids):
        distance=float(np.linalg.norm(state['ends'][a,1]-state['ends'][b,0]))
        for kind in (1,2):
            legal=distance<=6.00001 if kind==1 else distance>1e-9
            if not legal:continue
            score=float(np.log(max(probability[i,kind],1e-6))-np.log(max(probability[i,0],1e-6)))
            # Nonpositive edges cannot improve a graph with optional links.
            if score>0:edges.append((int(a),int(b),kind));weights.append(score)
    result=solve_oriented(count,edges,weights)
    successor={a:(b,kind) for a,b,kind in result['links']};predecessor={b:a for a,b,kind in result['links']}
    visited=set();paths=[]
    for start in result['states']:
        if start in predecessor:continue
        chain=[];kinds=[];node=start
        while node not in visited:
            visited.add(node);chain.append(node)
            if node not in successor:break
            node,kind=successor[node];kinds.append(kind)
        paths.append((chain,kinds,False))
    for start in result['states']:
        if start in visited:continue
        chain=[];kinds=[];node=start
        while node not in visited:
            visited.add(node);chain.append(node);node,kind=successor[node];kinds.append(kind)
        assert node==start and all(k==1 for k in kinds)
        paths.append((chain,kinds[:-1],True))
    assert visited==set(result['states'])
    parents=[]
    for chain,kinds,cycle in paths:
        segments=[state['segments'][i] for i in chain]
        closed=bool(cycle or (len(chain)==1 and state['closed'][chain[0]]))
        parents.append(dict(predicted_parent_query=len(parents),child_count=len(chain),is_closed=closed,
            gaps=['SHARED' if k==1 else 'POSITIVE_GAP' for k in kinds],segments=segments))
    assert sum(p['child_count'] for p in parents)==count
    write(dest/'assembly.json',dict(result,all_directed_state_pairs_scored=len(ids),positive_link_states=len(edges),
        kept_children=count,output_parents=len(parents),no_positive_child_dropped=True,
        nonpositive_links_only_omitted_by_objective_dominance=True,GT_reads=0))
    return parents


def infer(dest,key):
    begin=time.perf_counter();r=read(dest/'training_receipt.json');assert digest(dest/'models.pkl')==r['model_sha256']
    with (dest/'models.pkl').open('rb') as f:payload=pickle.load(f)
    assert payload['schema']==SCHEMA;input_dest=OUT/payload['input_run']
    receipt=read(input_dest/f'source_{key}/receipt.json')
    assert digest(input_dest/f'source_{key}/features.npy')==receipt['feature_sha256']
    baseline,additions,context,origins,base,binding=child_inputs(key,input_dest)
    assert binding['baseline_sha256']==receipt['baseline_sha256'] and binding['new_sha256']==receipt['new_sha256']
    x=np.load(input_dest/f'source_{key}/features.npy');p=payload['models']['validity'].predict_proba(x)[:,1]
    distance=endpoint_distances(context.parents);selected=list(range(base));accepted=[];decisions=[]
    for local in sorted(range(len(p)),key=lambda i:(-p[i],-float(x[i,523]),i)):
        index=base+local;duplicate=None
        if p[local]>.5:
            close=[i for i in selected if candidate_pair_possible(context,distance,index,i)]
            if close:
                features=np.stack([pair_features(context,index,i) for i in close])
                q=payload['models']['duplicate'].predict_proba(features)[:,1];j=int(q.argmax())
                if q[j]>.5:duplicate=dict(selected_child_index=close[j],probability=float(q[j]))
            if duplicate is None:selected.append(index);accepted.append(local)
        decisions.append(dict(candidate_index=local,validity_probability=float(p[local]),duplicate_of=duplicate,selected=local in accepted))
    folder=dest/f'test_{key}';folder.mkdir()
    remaining=ParentRelations([context.parents[base+i] for i in accepted],[context.metadata[base+i] for i in accepted])
    new_parents=assemble(remaining,payload['models']['link'],folder)
    result=dict(baseline);result['parents']=baseline['parents']+new_parents
    result['invalid_predictions']=baseline['invalid_predictions']+additions['invalid_predictions']
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(v['counted_false_children'] for v in result['invalid_predictions'])
    result.update(runtime_model=SCHEMA,GT_reads=0,STEP_reads=0)
    with (folder/'final_decoded.json').open('w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(',',':'),
            default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())
    write(folder/'decisions.json',dict(decisions=decisions,accepted_child_indices=accepted))
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_CHILD_PARENT_OUTPUT_FROZEN',source=key,
        decoded_sha256=digest(folder/'final_decoded.json'),model_sha256=r['model_sha256'],**binding,
        base_parents=len(baseline['parents']),new_children=len(accepted),new_parents=len(new_parents),
        GT_reads=0,STEP_reads=0,material_role_rules_not_fully_implemented=True,full_contract_pass=False,
        seconds=time.perf_counter()-begin))
    print('CHILD_PARENT_OUTPUT',key,'NEW_CHILDREN',len(accepted),'NEW_PARENTS',len(new_parents),'SECONDS',round(time.perf_counter()-begin,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['train','infer','controls'],required=True)
    ap.add_argument('--source');ap.add_argument('--input-run',default='run411');a=ap.parse_args();dest=OUT/a.run
    if a.stage=='train':
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(schema=SCHEMA,source_sha256=digest(__file__),
            input_module_sha256=digest(ROOT/'weld_pipeline/sjg_child_parent_supervision_1255.py'),
            TRAIN=list(TRAIN),DEV=['64'],original_G=213,trees_each=64,threshold=.5,
            exactly_one_orientation_per_retained_child=True,all_retained_children_must_be_output=True,
            frozen_best379=True,external_consultation_paused=True,full_contract_pass=False))
    try:
        if a.stage=='train':train(dest,OUT/a.input_run)
        elif a.stage=='controls':controls(dest)
        else:infer(dest,a.source)
    except BaseException:write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
