"""Pro297: refit links once on frozen upstream predicted geometry."""
import argparse
import pickle
import sys
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_joint_forest_train_1109 import load_case,predict_nodes,compact_connections,attributes
from weld_pipeline.sjg_joint_link_runtime_1107 import pair_features,node_geometry


def union_states(case,screen,payload):
    """GT contributes row membership only; all state fields come from predictors."""
    states={n['object_row']:n for n in screen['nodes']}
    invalid={e['node']['object_row']:e for e in screen['invalid']}
    absent=sorted(set(case['teacher'])-set(states)-set(invalid))
    if absent:
        x=case['x'][absent];obj=payload['object'];cols=np.flatnonzero(obj.classes_!=0)
        probabilities=obj.predict_proba(x);ratios=payload['endpoint'].predict(x)
        categories=obj.classes_[cols[probabilities[:,cols].argmax(1)]]
        for row,ratio,value,p in zip(absent,ratios,categories,probabilities[:,cols].sum(1)):
            fragment=int(case['canonical'][row//2]);left,right=case['graph'].spans[fragment,:2]
            ratio=np.sort(np.asarray(ratio,np.float64))
            node=dict(object_row=row,fragment=fragment,side=row%2,probability=float(p),**attributes(value))
            if not np.isfinite(ratio).all() or (ratio<0).any() or (ratio>1).any():
                invalid[row]=dict(node=node,reason='Predicted endpoint outside finite domain');continue
            node.update(lo=float(left+ratio[0]*(right-left)),hi=float(left+ratio[1]*(right-left)))
            try:node_geometry(node,case['graph']);states[row]=node
            except ValueError as exc:invalid[row]=dict(node=node,reason=str(exc))
    return [states[k] for k in sorted(states)],list(invalid.values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run298');args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter()
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,utc=utc(),**extra))
        print('PREDICTED_LINK_PHASE',name,'SECONDS',round(time.perf_counter()-start,3),extra,flush=True)
    try:
        phase('IMPORTING_CPU_LIBRARIES')
        import faulthandler
        faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
        from sklearn.ensemble import ExtraTreesClassifier
        from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
        original=read(OUT/'run294/receipt.json');path=OUT/'run294/model.pkl'
        assert digest(path)==original['model_sha256']
        with path.open('rb') as f:payload=pickle.load(f)
        frozen={k:forest_state_sha256(payload[k]) for k in ('object','endpoint')}
        settings=payload['connection'].get_params(deep=False)
        assert settings['n_estimators']==64 and settings['max_depth']==16 and not settings['warm_start']
        cases=[];tables=[];bindings=[];frozen_tables=[]
        for key in ('21','BM79A_native_local','10'):
            phase('PREDICT_GEOMETRY_AND_ENUMERATE_FULL_UNION',source=key)
            case=load_case(key,supervision=True)
            prior=next(c for c in original['cases'] if c['source']==key)
            for field in ('feature_receipt_sha256','alias_map_sha256','geometry_sha256','compiled_target_sha256'):
                assert prior[field]==case[field]
            screen=predict_nodes(case,payload['object'],payload['endpoint'])
            nodes,invalid=union_states(case,screen,payload)
            links,legal,geometry=compact_connections(nodes,case['graph']);m=len(links)
            folder=dest/f'links_{key}';folder.mkdir(exist_ok=False)
            import shutil
            if shutil.disk_usage(dest).free-m*(128+8+3+2)<5368709120:raise RuntimeError('Original disk reserve')
            np.save(folder/'links.npy',links);np.save(folder/'legal.npy',legal)
            write(folder/'predicted_nodes.json',dict(nodes=nodes,invalid=invalid,GT_endpoint_overrides=0))
            x=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(m,32))
            for first in range(0,m,32768):
                for e in range(first,min(first+32768,m)):
                    a,b=map(int,links[e]);x[e],_=pair_features(nodes[a],nodes[b],case['graph'],(geometry[a],geometry[b]))
                x.flush();check_resources()
            frozen_tables.append(dict(source=key,nodes=len(nodes),edges=m,feature_sha256=digest(folder/'features.npy'),
                legal_sha256=digest(folder/'legal.npy'),links_sha256=digest(folder/'links.npy'),
                predicted_nodes_sha256=digest(folder/'predicted_nodes.json')))
            cases.append(case);tables.append((folder,x,nodes,links,legal))
            print('PREDICTED_GEOMETRY_FROZEN',key,len(nodes),m,flush=True)
        write(dest/'all_predicted_geometry_frozen_before_edge_labels.json',dict(cases=frozen_tables,
            teacher_use='union row membership and later supervision only',GT_endpoint_overrides=0,utc=utc()))
        fitx=[];fity=[]
        for case,(folder,x,nodes,links,legal),bound in zip(cases,tables,frozen_tables):
            phase('LABEL_FROZEN_PREDICTED_GEOMETRY',source=case['key'])
            y=np.zeros(len(links),np.int8);known=np.zeros(len(links),bool);seen=set();misses=[]
            truth=case['true_edges'];teacher=case['teacher']
            for e,(a,b) in enumerate(links):
                u,v=nodes[int(a)]['object_row'],nodes[int(b)]['object_row'];kind=truth.get((u,v),0)
                if u in teacher and v in teacher:
                    if kind and not legal[e,kind-1]:continue
                    known[e]=True;y[e]=kind
                if kind and legal[e,kind-1]:seen.add((u,v))
            for (u,v),kind in truth.items():
                if (u,v) not in seen:
                    misses.append(dict(status='PREDICTED_GEOMETRY_EDGE_MISS',object_rows=[u,v],true_kind=kind,
                        original_teacher_nodes=[teacher[u],teacher[v]],
                        reason='Original true class not legal under frozen predicted geometry; not inserted and not labeled NO_LINK'))
            np.save(folder/'labels.npy',y);np.save(folder/'supervision_known.npy',known)
            write(folder/'geometry_edge_misses.json',dict(misses=misses,original_G=case['G'],denominator_unchanged=True,
                true_edges_by_type={str(k):sum(v==k for v in truth.values()) for k in (1,2,3)},
                legal_true_edges_by_type={str(k):sum(truth[e]==k for e in seen) for k in (1,2,3)}))
            assert digest(folder/'features.npy')==bound['feature_sha256']
            prior=next(c for c in original['cases'] if c['source']==case['key'])
            bindings.append(dict(prior,nodes=len(nodes),edges=len(links),supervised_edges=int(known.sum()),
                masked_edges=int((~known).sum()),predicted_geometry_true_edge_misses=len(misses)))
            fitx.append(np.asarray(x[known]));fity.append(y[known]);check_resources()
        x=np.concatenate(fitx);y=np.concatenate(fity)
        phase('FIT_ONE_NEW_CONNECTION_FOREST',rows=len(y),trees=64)
        payload['connection']=ExtraTreesClassifier(**settings).fit(x,y)
        payload.update(base_model_run='run294',frozen_component_sha256=frozen,bindings=bindings,
            connection_training_geometry='FROZEN_UPSTREAM_PREDICTED_ENDPOINTS',connection_settings=settings)
        assert {k:forest_state_sha256(payload[k]) for k in frozen}==frozen
        check_resources();phase('ATOMIC_MODEL_SAVE_AND_NUMERIC_RELOAD_VERIFY')
        atomic_binary_write(dest/'model.pkl',lambda f:pickle.dump(payload,f,protocol=5))
        with (dest/'model.pkl').open('rb') as f:restored=pickle.load(f)
        assert {k:forest_state_sha256(restored[k]) for k in frozen}==frozen
        assert forest_state_sha256(restored['connection'])==forest_state_sha256(payload['connection'])
        assert digest(path)==original['model_sha256'] and 'torch' not in sys.modules
        write(dest/'worker_receipt.json',dict(status='ONE_CONNECTION_FOREST_FIT_UPSTREAM_FROZEN',
            model_sha256=digest(dest/'model.pkl'),base_model_run='run294',base_model_sha256=original['model_sha256'],
            frozen_component_sha256=frozen,frozen_component_hash_format='EXACT_FOREST_PREDICTIVE_STATE_v1',
            frozen_components_verified_after_save_reload=True,forest_fit_calls=1,object_fit_calls=0,endpoint_fit_calls=0,
            connection_fit_calls=1,connection_settings=settings,connection_training_rows=len(y),
            connection_training_geometry='FROZEN_UPSTREAM_PREDICTED_ENDPOINTS',GT_endpoint_overrides=0,
            cases=bindings,G=213,fixed_presence_threshold=.5,all_positive_nodes_fixed_selected=True,
            negative_subsampling=False,threshold_search=False,Torch_imported=False,
            seconds=time.perf_counter()-start,extraction_correct_rate=None,extraction_precision_mm=None,utc=utc()))
        phase('COMPLETE');faulthandler.cancel_dump_traceback_later()
    except BaseException:
        write(dest/'worker_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,utc=utc()));raise


if __name__=='__main__':main()
