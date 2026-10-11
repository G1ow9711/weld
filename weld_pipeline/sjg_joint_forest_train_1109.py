"""One Pro286 CPU fit: object, endpoints, then conditional complete links."""
import argparse
import json
import pickle
import time
import traceback
from pathlib import Path


def category(weld_type,forward,closed=False,wrap=False):
    return 1+int(weld_type)+2*int(forward)+4*int(closed)+8*int(wrap)


def attributes(value):
    v=int(value)-1
    if v<0:raise ValueError('NO_OUTPUT has no geometric category')
    return dict(weld_type=v&1,forward=bool(v&2),closed=bool(v&4),wrap=bool(v&8))


def load_case(key,supervision=False,build_features=True):
    import numpy as np
    from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest
    from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
    from weld_pipeline.sjg_joint_object_features_1106 import side_rows
    if key not in ('21','4','10','25','36','42','BM79A_native_local','64'):raise ValueError('Unregistered conditional experiment source')
    if key=='64' and supervision:raise ValueError('DEV64 must never enter TRAIN supervision')
    feature_run='run293' if key=='64' else 'run308' if key in ('4','25','36','42') else 'run287'
    alias_run='run293' if key=='64' else 'run288'
    geometry_run='run293' if key=='64' else 'run285'
    fdir=OUT/feature_run/f'source_{key}';fr=read(fdir/'receipt.json')
    assert digest(fdir/'object_features.npy')==fr['features_sha256']
    nr=read(OUT/alias_run/'receipt.json')
    assert nr['status']==('PURE_DEV_INPUT_READY_UNSCORED' if key=='64' else 'ALIAS_NORMALIZATION_AND_WITNESS_REINDEX_PASSED')
    nf=next(r for r in read(OUT/alias_run/'all_alias_maps_frozen_before_labels.json')['cases'] if r['source']==key)
    assert digest(OUT/alias_run/f'source_{key}_alias_map.npz')==nf['map_sha256']
    with np.load(OUT/alias_run/f'source_{key}_alias_map.npz') as z:canonical=z['canonical_fragments'];rep=z['fragment_representative']
    gf=next(r for r in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if r['source']==key)
    assert digest(OUT/f'run185/source_{key}/profiles.json')==gf['source_files']['profiles.json']
    assert digest(OUT/geometry_run/gf['geometry_file'])==gf['geometry_sha256']
    profiles=read(OUT/f'run185/source_{key}/profiles.json')
    with np.load(OUT/geometry_run/gf['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
    n=len(canonical);x=None
    if build_features:
        base=np.load(fdir/'object_features.npy',mmap_mode='r')
        x=np.empty((n*2,128),np.float32);x[0::2]=base[canonical];x[1::2]=side_rows(base,canonical,np.ones(n,np.int8))
        base._mmap.close()
    result=dict(key=key,graph=graph,canonical=canonical,x=x,feature_receipt_sha256=digest(fdir/'receipt.json'),
                alias_map_sha256=nf['map_sha256'],geometry_sha256=gf['geometry_sha256'])
    if not supervision:return result
    target_file=OUT/f'run264/source_{key}_targets.json'
    target_bound=next(r for r in read(OUT/'run264/receipt.json')['cases'] if r['source']==key)
    assert digest(target_file)==target_bound['compiled_sha256']
    target=read(target_file);witness=read(OUT/f'run285/source_{key}_joint_witnesses.json')['witnesses']
    y=np.zeros(n*2,np.int16);q=np.zeros((n*2,2),np.float64);teacher={};true_edges={}
    index={int(f):i for i,f in enumerate(canonical)}
    for parent in witness:
        chain=[];chain_child=[];original=target['parents'][parent['parent']]
        for child in parent['children']:
            alt=child['alt'];source_curve=graph.curve(alt['profile']);truth=original['children'][child['child']]
            for fragment in child['fragments']:
                canonical_fragment=int(rep[fragment]);slot=index[canonical_fragment]*2+int(child['side_hypothesis'])
                assert y[slot]==0,'Duplicate canonical output responsibility'
                curve=graph.curve(graph.profile(canonical_fragment));lo,hi=graph.spans[canonical_fragment,:2]
                projected,_=curve.project(source_curve.at(np.asarray([alt['lo'],alt['hi']])))
                ca,cb=sorted(projected.tolist())
                if alt['closed']:left,right=float(lo),float(hi)
                elif alt['wrap']:
                    if hi<=ca:left,right=float(lo),min(float(hi),ca)
                    elif lo>=cb:left,right=max(float(lo),cb),float(hi)
                    else:raise ValueError('Wrapped teacher cannot occupy this fragment once')
                else:left,right=max(float(lo),ca),min(float(hi),cb)
                if not right>left:raise ValueError('Canonical positive has no finite interval; do not drop a witness')
                closed=bool(alt['closed'] and len(child['fragments'])==1)
                value=category(truth['weld_type'],alt['forward'],closed,False)
                y[slot]=value;q[slot]=[(left-lo)/(hi-lo),(right-lo)/(hi-lo)]
                teacher[slot]=dict(fragment=canonical_fragment,side=int(child['side_hypothesis']),
                    probability=1.,lo=left,hi=right,**attributes(value),object_row=slot)
                chain.append(slot);chain_child.append(child['child'])
        for i,(a,b) in enumerate(zip(chain[:-1],chain[1:])):
            kind=1 if chain_child[i]==chain_child[i+1] else {'SHARED':2,'POSITIVE_GAP':3}[parent['gaps'][chain_child[i]]]
            true_edges[a,b]=kind
        if parent['closed'] and len(chain)>1:true_edges[chain[-1],chain[0]]=1 if len(original['children'])==1 else 2
    result.update(y=y,q=q,teacher=teacher,true_edges=true_edges,G=target['G'],parents=len(target['parents']),
                  compiled_target_sha256=target_bound['compiled_sha256'])
    return result


def predict_nodes(case,classifier,regressor):
    import numpy as np
    from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry
    nodes=[];invalid=[];max_probability=0.;count=0
    positive_columns=np.flatnonzero(classifier.classes_!=0)
    for first in range(0,len(case['x']),32768):
        x=case['x'][first:first+32768];proba=classifier.predict_proba(x)
        probability=proba[:,positive_columns].sum(1);max_probability=max(max_probability,float(probability.max(initial=0)))
        selected=np.flatnonzero(probability>=.5);count+=len(selected)
        if not len(selected):continue
        geometry=regressor.predict(x[selected])
        chosen=classifier.classes_[positive_columns[proba[selected][:,positive_columns].argmax(1)]]
        for row,ratios,value in zip(selected,geometry,chosen):
            absolute=int(first+row);fragment=int(case['canonical'][absolute//2]);side=absolute%2
            left,right=case['graph'].spans[fragment,:2];ratios=np.sort(np.asarray(ratios,np.float64))
            node=dict(fragment=fragment,side=side,probability=float(probability[row]),object_row=absolute,**attributes(value))
            try:
                if not np.isfinite(ratios).all() or np.any(ratios<0) or np.any(ratios>1):raise ValueError('Regressed endpoint outside domain')
                node.update(lo=float(left+ratios[0]*(right-left)),hi=float(left+ratios[1]*(right-left)))
                node_geometry(node,case['graph']);nodes.append(node)
            except ValueError as exc:
                invalid.append(dict(node=node,reason=str(exc),counted_false_children=1))
    return dict(nodes=nodes,invalid=invalid,positive_count=int(count),maximum_probability=max_probability,
                object_rows_scored=len(case['x']),fixed_presence_threshold=.5)


def compact_connections(nodes,graph):
    import numpy as np
    from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry,pair_features
    geometry=[node_geometry(n,graph) for n in nodes];n=len(nodes)
    boxes=np.asarray([graph.bounds(s['fragment']) for s in nodes]).reshape(-1,2,3)
    profiles=np.asarray([graph.profile(s['fragment']) for s in nodes])
    faces=np.full((n,2),-1,np.int64);has_gaps=np.zeros(n,bool)
    for i,pid in enumerate(profiles):
        p=graph.profiles[int(pid)];f=p.get('face_indices',[])[:2];faces[i,:len(f)]=f
        has_gaps[i]=bool(p.get('unobserved_gaps_mm'))
    links=[];masks=[]
    for i,a in enumerate(nodes):
        delta=np.maximum(0.,np.maximum(boxes[i,0]-boxes[:,1],boxes[:,0]-boxes[i,1]))
        near=np.einsum('ij,ij->i',delta,delta)<=36.
        common=np.zeros(n,bool)
        for face in faces[i]:
            if face>=0:common|=np.any(faces==face,axis=1)
        candidate=near|(profiles==profiles[i])|(common&(has_gaps|has_gaps[i]));candidate[i]=False
        for j in np.flatnonzero(candidate):
            _,mask=pair_features(a,nodes[j],graph,(geometry[i],geometry[j]))
            if mask.any():links.append((i,int(j)));masks.append(mask)
    return np.asarray(links,np.int32).reshape(-1,2),np.asarray(masks,bool).reshape(-1,3),geometry


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run289');args=ap.parse_args()
    from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
    dest=OUT/args.run;dest.mkdir(exist_ok=True);start=time.perf_counter()
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,utc=utc(),**extra))
        print('CPU_FOREST_PHASE',name,'SECONDS',round(time.perf_counter()-start,3),extra,flush=True)
    try:
        phase('IMPORTING_CPU_LIBRARIES')
        import faulthandler
        faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
        import sys
        import numpy as np
        from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
        from weld_pipeline.sjg_joint_link_runtime_1107 import pair_features
        from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
        assert 'torch' not in sys.modules
        phase('LOADING_COMPLETE_OBJECT_TABLES');cases=[]
        for key in ('21','BM79A_native_local','10'):
            case=load_case(key,supervision=True);cases.append(case);check_resources()
            print('COMPLETE_OBJECT_TABLE',key,len(case['x']),'POSITIVE',len(case['teacher']),flush=True)
        assert sum(c['G'] for c in cases)==213
        x=np.concatenate([c['x'] for c in cases]);y=np.concatenate([c['y'] for c in cases]);q=np.concatenate([c['q'] for c in cases])
        settings=dict(n_estimators=64,max_depth=16,min_samples_leaf=1,max_features='sqrt',bootstrap=False,n_jobs=1,random_state=1031,verbose=2)
        phase('FIT_1_OF_3_OBJECT',rows=len(x));object_model=ExtraTreesClassifier(**settings).fit(x,y)
        check_resources();positive=y!=0
        phase('FIT_2_OF_3_ENDPOINT',rows=int(positive.sum()));endpoint_model=ExtraTreesRegressor(**settings).fit(x[positive],q[positive])
        del x,y,q,positive
        import gc;gc.collect();check_resources();tables=[];bindings=[]
        for case in cases:
            key=case['key'];phase('FROZEN_OBJECT_SCREEN_AND_EXACT_LINK_COUNT',source=key)
            predicted=predict_nodes(case,object_model,endpoint_model)
            states={n['object_row']:n for n in predicted['nodes']};states.update(case['teacher'])
            nodes=[states[k] for k in sorted(states)]
            links,masks,geometry=compact_connections(nodes,case['graph']);m=len(links)
            write(dest/f'source_{key}_conditional_count.json',dict(source=key,complete_objects=len(case['x']),
                teacher_positive_objects=len(case['teacher']),predicted_positive_objects=predicted['positive_count'],
                predicted_invalid_objects=predicted['invalid'],conditional_objects=len(nodes),legal_directed_edges=m,
                compact_count_completed_before_feature_allocation=True,feature_bytes=m*32*4,
                maximum_probability=predicted['maximum_probability'],same_TRAIN_screen_not_independent_validation=True))
            check_resources()
            # Keep the original 5 GiB reserve before allocating any link table.
            import shutil
            if shutil.disk_usage(dest).free-m*(128+8+3+2)<5368709120:raise RuntimeError('Conditional link table exceeds original disk guard')
            folder=dest/f'links_{key}';folder.mkdir(exist_ok=False)
            np.save(folder/'links.npy',links);np.save(folder/'legal.npy',masks)
            features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(m,32))
            labels=np.zeros(m,np.int8);known=np.zeros(m,bool);seen_true=set()
            for first in range(0,m,32768):
                for e in range(first,min(m,first+32768)):
                    a,b=map(int,links[e]);features[e],_=pair_features(nodes[a],nodes[b],case['graph'],(geometry[a],geometry[b]))
                    u,v=nodes[a]['object_row'],nodes[b]['object_row']
                    if u in case['teacher'] and v in case['teacher']:
                        known[e]=True;labels[e]=case['true_edges'].get((u,v),0)
                    if (u,v) in case['true_edges']:
                        if not masks[e,int(labels[e])-1]:raise ValueError('Complete positive connection lost its registered legal class')
                        seen_true.add((u,v))
                features.flush();check_resources()
            if seen_true!=set(case['true_edges']):raise ValueError('Conditional graph lost a complete positive parent connection')
            np.save(folder/'labels.npy',labels);np.save(folder/'supervision_known.npy',known)
            tables.append((features,labels,known));bindings.append(dict(source=key,G=case['G'],
                feature_receipt_sha256=case['feature_receipt_sha256'],alias_map_sha256=case['alias_map_sha256'],
                geometry_sha256=case['geometry_sha256'],compiled_target_sha256=case['compiled_target_sha256'],
                nodes=len(nodes),edges=m,supervised_edges=int(known.sum()),masked_edges=int((~known).sum())))
            print('CONDITIONAL_LINK_TABLE',key,m,'KNOWN',int(known.sum()),flush=True)
        phase('FIT_3_OF_3_CONNECTION')
        lx=np.concatenate([np.asarray(f[k]) for f,y,k in tables]);ly=np.concatenate([y[k] for f,y,k in tables])
        if not len(ly):raise ValueError('No known conditional connection supervision')
        check_resources();connection_model=ExtraTreesClassifier(**settings).fit(lx,ly)
        phase('ATOMIC_MODEL_SAVE');check_resources()
        payload=dict(object=object_model,endpoint=endpoint_model,connection=connection_model,
                     settings=settings,bindings=bindings,fixed_presence_threshold=.5,
                     nodes_fixed_selected=True,classification='NO_OUTPUT=0; positive=1+weld_type+2*forward+4*closed+8*wrap')
        atomic_binary_write(dest/'model.pkl',lambda stream:pickle.dump(payload,stream,protocol=5))
        assert 'torch' not in sys.modules
        write(dest/'worker_receipt.json',dict(status='THREE_CPU_FORESTS_FIT_AND_SAVED',model_sha256=digest(dest/'model.pkl'),
            G=213,cases=bindings,forest_fit_calls=3,settings=settings,seconds=time.perf_counter()-start,
            Torch_imported=False,threshold_search=False,negative_subsampling=False,GT_only_TRAIN=True,
            extraction_correct_rate=None,extraction_precision_mm=None,utc=utc()))
        phase('COMPLETE')
        faulthandler.cancel_dump_traceback_later()
    except BaseException:
        write(dest/'worker_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,utc=utc()))
        raise


if __name__=='__main__':main()
