"""Separate native child geometry labels from complete-parent link labels."""
import argparse
from collections import defaultdict
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, check_resources
from weld_pipeline.sjg_additive_parent_precision_1254 import (
    TRAIN, KEYS, source, endpoint_distances, candidate_pair_possible, pair_features)
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.contract_curve_geometry import CircularArc

SCHEMA='CHILD_VALIDITY_AND_DIRECTED_PARENT_GROUPING_V1'


def direction_free(x):
    assert x.shape[1]==528
    return np.c_[x[:,:524],x[:,524]+x[:,526],x[:,525]+x[:,527]].astype(np.float32)


def singleton(segment):
    closed=bool(segment.get('arc_parameters',{}).get('is_closed',False)) if segment['kind']=='arc' else False
    return dict(child_count=1,is_closed=closed,gaps=[],segments=[segment])


def input_route(key, input_dest=None):
    path = input_dest/'input_routes.json' if input_dest is not None else None
    if path is not None and path.exists() and (input_dest/'all_inputs_frozen.json').exists():
        assert digest(path)==read(input_dest/'all_inputs_frozen.json')['routes_sha256']
    return read(path)['routes'][key] if path is not None and path.exists() else {}


def child_inputs(key, input_dest=None):
    route=input_route(key,input_dest)
    baseline, additions, bm, binding=source(key,route)
    raw=OUT/route.get('feature_run','run409')/f'source_{key}'
    assert digest(raw/'metadata.json')==read(raw/'receipt.json')['metadata_sha256']
    new_metadata=read(raw/'metadata.json')[len(baseline['parents']):]
    parents=[];metadata=[];origin=[]
    for pi,(parent,m) in enumerate(zip(baseline['parents'],bm)):
        for ci,segment in enumerate(parent['segments']):
            parents.append(singleton(segment));metadata.append(dict(m,
                length_mm=segment['length_mm'],line_fraction=float(segment['kind']=='line')))
            origin.append(dict(base_parent=pi,base_child=ci))
    base_children=len(parents)
    for i,(parent,m) in enumerate(zip(additions['parents'],new_metadata)):
        assert parent['child_count']==1
        parents.append(parent);metadata.append(m);origin.append(dict(addition=i))
    return baseline,additions,ParentRelations(parents,metadata),origin,base_children,binding


def state_geometry(context):
    segments=[];ends=[];tangents=[];lengths=[];line=[];horizontal=[];closed=[];faces=[];pairs=[];known=[]
    for parent,m in zip(context.parents,context.metadata):
        original=parent['segments'][0]
        for segment in (original,dict(original,**reverse_segment(original))):
            p=np.asarray(segment['points'],float);ep=p[[0,-1]]
            if segment['kind']=='line':
                tangent=(ep[1]-ep[0])/np.linalg.norm(ep[1]-ep[0]);tt=np.stack((tangent,tangent))
            else:
                arc=CircularArc.from_parameters(segment['arc_parameters'])
                tt=np.sign(arc.sweep_rad)*np.cross(arc.plane_normal,(ep-arc.center_mm)/arc.radius_mm)
            segments.append(segment);ends.append(ep);tangents.append(tt);lengths.append(m['length_mm'])
            line.append(segment['kind']=='line');horizontal.append(segment['weld_type']=='horizontal')
            closed.append(parent['is_closed']);known.append(m['side_known'])
            faces.append({f for pair in m['face_pairs'] for f in pair if f>=0})
            pairs.append({tuple(p) for p in m['face_pairs'] if len(p)==2 and min(p)>=0})
    return dict(segments=segments,ends=np.asarray(ends).reshape(-1,2,3),
        tangent=np.asarray(tangents).reshape(-1,2,3),length=np.asarray(lengths),
        line=np.asarray(line),horizontal=np.asarray(horizontal),closed=np.asarray(closed),
        faces=faces,face_pairs=pairs,known=np.asarray(known))


def link_features(state, edges):
    edges=np.asarray(edges,np.int64).reshape(-1,2);a,b=edges.T
    ea,eb=state['ends'][a],state['ends'][b]
    ta,tb=state['tangent'][a,1],state['tangent'][b,0]
    delta=eb[:,0]-ea[:,1];distance=np.linalg.norm(delta,axis=1)
    unit=np.zeros_like(delta); nonzero=distance>1e-9
    unit[nonzero]=delta[nonzero]/distance[nonzero,None]
    la,lb=state['length'][a],state['length'][b]
    raw=np.c_[np.log1p(distance/10),np.log1p(np.linalg.norm(ea[:,0]-eb[:,0],axis=1)/10),
        np.log1p(np.linalg.norm(ea[:,1]-eb[:,1],axis=1)/10),np.log1p(np.linalg.norm(ea[:,0]-eb[:,1],axis=1)/10),
        np.log1p(la/10),np.log1p(lb/10),abs(np.log(la/lb)),np.sum(ta*tb,axis=1),
        np.sum(unit*ta,axis=1),np.sum(unit*tb,axis=1),distance>1e-9,
        state['line'][a],state['line'][b],state['closed'][a],state['closed'][b],
        state['horizontal'][a],state['horizontal'][b],
        [len(state['faces'][i]&state['faces'][j])/max(1,len(state['faces'][i]|state['faces'][j])) for i,j in edges],
        [len(state['face_pairs'][i]&state['face_pairs'][j])/max(1,len(state['face_pairs'][i]|state['face_pairs'][j])) for i,j in edges],
        state['known'][a],state['known'][b],state['horizontal'][a]==state['horizontal'][b],
        np.log1p(distance/np.minimum(la,lb)),np.log1p(distance/np.maximum(la,lb))]
    assert raw.shape==(len(edges),24) and np.isfinite(raw).all()
    return raw.astype(np.float32)


def feature_controls(dest):
    # Label-free rigid-transform control, including a real TRAIN candidate curve.
    _,_,context,_,_,_=child_inputs('21');s=state_geometry(context)
    edges=np.array([[0,2],[1,3],[2,0],[3,1]])
    reference=link_features(s,edges)
    from weld_pipeline.sjg_relative_geometry_1135 import transformations
    maximum=0.
    for _,R,shift in transformations():
        moved=dict(s,ends=s['ends']@R.T+shift,tangent=s['tangent']@R.T)
        error=float(abs(link_features(moved,edges)-reference).max());maximum=max(maximum,error)
        assert error<1e-4
    x=np.load(OUT/'run409/source_21/features.npy');reverse=x.copy();reverse[:,524:528]=x[:,[526,527,524,525]]
    assert np.array_equal(direction_free(x),direction_free(reverse))
    write(dest/'feature_controls.json',dict(link_rigid_transform_max_abs=maximum,
        child_validity_traversal_invariant_exact=True,GT_reads=0))


def prepare(dest, keys=KEYS):
    feature_controls(dest)
    for key in keys:
        folder=dest/f'source_{key}';folder.mkdir()
        route=input_route(key,dest);raw=OUT/route.get('feature_run','run409')/f'source_{key}'
        r=read(raw/'receipt.json')
        assert digest(raw/'features.npy')==r['feature_sha256']
        assert digest(raw/'metadata.json')==r['metadata_sha256']
        x=direction_free(np.load(raw/'features.npy'))
        np.save(folder/'features.npy',x)
        _,_,context,origins,base,binding=child_inputs(key,dest)
        write(folder/'origin.json',dict(base_children=base,origins=origins))
        write(folder/'receipt.json',dict(source=key,schema=SCHEMA,**binding,columns=526,
            feature_sha256=digest(folder/'features.npy'),original_input_receipt_sha256=digest(raw/'receipt.json'),
            base_children=base,new_children=len(x),GT_reads=0,STEP_reads=0))
        print('CHILD_INPUT_READY',key,'BASE',base,'NEW',len(x),flush=True)
    write(dest/'all_inputs_frozen.json',dict(GT_reads=0,STEP_reads=0,
        receipts={k:digest(dest/f'source_{k}/receipt.json') for k in keys},
        routes_sha256=digest(dest/'input_routes.json') if (dest/'input_routes.json').exists() else None))


def targets(dest, train=TRAIN, original_g=213):
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
    from weld_pipeline.sjg_parent_compatibility_1161 import compile_compatibility
    from weld_pipeline.contract_geometry_d3_409 import child_error
    from weld_pipeline.world_set_evaluation_285 import reference_segment
    assert read(dest/'all_inputs_frozen.json')['GT_reads']==0
    cases=[]
    for key in train:
        check_resources();begin=time.perf_counter();_,_,context,origins,base,_=child_inputs(key,dest)
        original,binding=target_for(key);flat=[];owners=[];by_owner={}
        for pi,parent in enumerate(original['parents']):
            for ci,child in enumerate(parent['children']):
                by_owner[pi,ci]=len(flat);owners.append([pi,ci])
                flat.append(dict(original_child_count=1,is_closed=bool(child.get('original_is_closed',False)),gaps=[],children=[child]))
        target=dict(original,parents=flat)
        decoded=dict(parents=context.parents,invalid_predictions=[],total_prediction_children=len(context.parents))
        compatible=compile_compatibility(decoded,target);edges=[set(v) for v in compatible['compatible_targets']]
        refs=[reference_segment(p['children'][0]) for p in native_targets(target)['parents']]
        truths=[p['children'][0] for p in target['parents']]
        states=state_geometry(context);state_owners=[[] for _ in states['segments']];reverse_index=defaultdict(list)
        for i,options in enumerate(edges):
            for orientation in (0,1):
                si=2*i+orientation;segment=states['segments'][si]
                for owner in options:
                    closed=flat[owner]['is_closed']
                    if child_error(segment,truths[owner],refs[owner],closed,False) is not None:
                        state_owners[si].append(owner);reverse_index[owner].append(si)
        assert all(not e or state_owners[2*i] or state_owners[2*i+1] for i,e in enumerate(edges))
        route=input_route(key,dest)
        known_path=OUT/route.get('known_background_file',f'run406/source_{key}/compatibility.json')
        known={v['row']:v['complete_background_supervision_allowed'] for v in read(known_path)['rows']}
        intervals=read(OUT/route.get('proposal_run','run408')/f'proposal_{key}/selected_intervals.json')['intervals']
        y=np.array([1 if edges[base+i] else (0 if known.get(v['row'],False) else -1) for i,v in enumerate(intervals)],np.int8)
        distances=endpoint_distances(context.parents);eligible=[i for i,v in enumerate(edges) if v]
        pairs=set()
        for a in eligible:
            negative=[]
            for b in eligible:
                if a==b:continue
                if candidate_pair_possible(context,distances,a,b):pairs.add(tuple(sorted((a,b))))
                if len(edges[a]|edges[b])>1:negative.append((float(distances[a,b]),b))
            for _,b in sorted(negative)[:4]:pairs.add(tuple(sorted((a,b))))
        pair_ids=sorted(pairs);duplicate_x=np.stack([pair_features(context,a,b) for a,b in pair_ids]) if pair_ids else np.empty((0,32),np.float32)
        duplicate_y=np.array([len(edges[a]|edges[b])==1 for a,b in pair_ids],np.int8)
        # Positive directed links retain the parent identity, child order, gaps,
        # and a common whole-parent orientation. Reversed arcs use true geometry.
        positive=defaultdict(set);missing=[];target_links=0
        for pi,parent in enumerate(original['parents']):
            count=len(parent['children']);steps=list(range(count-1))
            if parent['is_closed'] and count>1:steps.append(count-1)
            for ci in steps:
                cj=(ci+1)%count;kind=1 if (ci==count-1 or parent['gaps'][ci]=='SHARED') else 2
                assert ci==count-1 or parent['gaps'][ci] in ('SHARED','POSITIVE_GAP')
                target_links+=1;a=by_owner[pi,ci];b=by_owner[pi,cj]
                if not reverse_index[a] or not reverse_index[b]:missing.append([pi,ci,cj]);continue
                for sa in reverse_index[a]:
                    for sb in reverse_index[b]:
                        if sa//2==sb//2:continue
                        positive[sa,sb].add(kind);positive[sb^1,sa^1].add(kind)
        # All explainable positive links, plus four nearest known negatives per state.
        known_children=set(eligible)|{base+i for i,v in enumerate(y) if v==0}
        state_ids=np.array([2*i+o for i in sorted(known_children) for o in (0,1) if not context.parents[i]['is_closed']],int)
        link_ids=set(positive)
        for sa in state_ids:
            ds=np.linalg.norm(states['ends'][state_ids,0]-states['ends'][sa,1],axis=1)
            count=0
            for pos in np.argsort(ds,kind='stable'):
                sb=int(state_ids[pos])
                if sa//2==sb//2 or (int(sa),sb) in positive:continue
                link_ids.add((int(sa),sb));count+=1
                if count==4:break
        ids=[];labels=[];weights=[]
        for pair in sorted(link_ids):
            allowed=sorted(positive.get(pair,{0}))
            for label in allowed:ids.append(pair);labels.append(label);weights.append(1/len(allowed))
        lx=link_features(states,ids);ly=np.asarray(labels,np.int8)
        if np.any(ly==1):assert np.expm1(lx[ly==1,0]).max()*10<=6.0001
        folder=dest/f'source_{key}'
        np.savez(folder/'training_targets.npz',validity=y,duplicate_x=duplicate_x,duplicate_y=duplicate_y,
            link_x=lx,link_y=ly,link_weight=np.asarray(weights,float))
        write(folder/'child_compatibility.json',dict(compatible,original_parent_child_owners=owners,
            forward_state_owners=state_owners,original_target_sha256=binding['target_sha256'],
            only_child_training_scope_not_parent_acceptance=True))
        write(folder/'training_pair_indices.json',dict(duplicate_pairs=pair_ids,link_state_pairs=ids))
        types=defaultdict(int)
        for i,v in enumerate(y):
            if v==1:types[context.parents[base+i]['segments'][0]['weld_type']]+=1
        r=dict(source=key,G=original['original_children'],positive=int((y==1).sum()),negative=int((y==0).sum()),unknown=int((y<0).sum()),
            positive_types=dict(types),covered_original_children=compatible['maximum_matched_children'],
            duplicate_pairs=len(pair_ids),link_training_rows=len(ly),link_classes={str(c):int((ly==c).sum()) for c in np.unique(ly)},
            original_parent_links=target_links,unrepresented_parent_links=missing,
            labels_sha256=digest(folder/'training_targets.npz'),compatibility_sha256=digest(folder/'child_compatibility.json'),
            seconds=time.perf_counter()-begin)
        cases.append(r);print('CHILD_PARENT_TARGETS',r,flush=True)
        del context,states,distances,duplicate_x,lx;gc.collect()
    assert sum(c['G'] for c in cases)==original_g
    assert sum(c['positive_types'].get('vertical',0) for c in cases)>0
    write(dest/'supervision_receipt.json',dict(cases=cases,G=original_g,TRAIN=list(train),DEV_reads=0,
        original_parent_scope_preserved=True,child_labels_not_full_parent_acceptance=True,full_contract_pass=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['prepare','targets'],required=True);a=ap.parse_args()
    dest=OUT/a.run
    if a.stage=='prepare':
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(schema=SCHEMA,source_sha256=digest(__file__),TRAIN=list(TRAIN),DEV=['64'],
            original_G=213,STEP_reads=0,external_consultation_paused=True,full_contract_pass=False))
    try:prepare(dest) if a.stage=='prepare' else targets(dest)
    except BaseException:write(dest/f'{a.stage}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
