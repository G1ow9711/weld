"""One 152-channel three-TRAIN refit, all three forests once, streamed tables."""
import argparse
import gc
import pickle
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_relative_forest_runtime_1139 import load_relative_case,relative_link_blocks,predict_nodes
from weld_pipeline.sjg_predicted_link_train_1117 import union_states
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

KEYS=('21','BM79A_native_local','10')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter();fit_calls=0
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,**extra))
        print('RELATIVE_FOREST_PHASE',name,round(time.perf_counter()-start,3),extra,flush=True)
    try:
        phase('IMPORT_AND_STREAM_ORIGINAL_TABLES')
        from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
        assert read(OUT/'run329/receipt.json')['status']=='RELATIVE_SUPERVISION_READY'
        receipts=[read(OUT/f'run329/source_{k}/receipt.json') for k in KEYS]
        rows=sum(r['rows'] for r in receipts);assert rows==336344
        x=np.lib.format.open_memmap(dest/'object_train.npy',mode='w+',dtype=np.float32,shape=(rows,152))
        ys=[];ex=[];eq=[];bindings=[];offset=0
        for key,r in zip(KEYS,receipts):
            folder=OUT/f'run329/source_{key}';assert digest(folder/'features.npy')==r['features_sha256']
            source=np.load(folder/'features.npy',mmap_mode='r')
            with np.load(folder/'original_supervision.npz') as z:y=z['labels'];q=z['ratios']
            assert len(source)==len(y)==r['rows']
            for first in range(0,len(y),8192):x[offset+first:offset+min(first+8192,len(y))]=source[first:first+8192]
            positive=y!=0;ex.append(np.asarray(source[positive]));eq.append(q[positive]);ys.append(y)
            bindings.append(dict(source=key,rows=len(y),positive_rows=int(positive.sum()),G={'21':1,'BM79A_native_local':4,'10':208}[key],
                features_sha256=r['features_sha256'],supervision_sha256=digest(folder/'original_supervision.npz')))
            source._mmap.close();offset+=len(y);check_resources()
        x.flush();y=np.concatenate(ys);ys.clear();gc.collect()
        settings=dict(n_estimators=64,max_depth=32,min_samples_leaf=1,max_features='sqrt',bootstrap=False,
                      n_jobs=1,random_state=1031,warm_start=False,verbose=2)
        phase('FIT_1_OF_3_OBJECT',rows=rows,columns=152)
        obj=ExtraTreesClassifier(**settings).fit(x,y);fit_calls+=1
        x._mmap.close();del x,y;gc.collect();check_resources()
        x=np.concatenate(ex);q=np.concatenate(eq);ex.clear();eq.clear();assert x.shape==(782,152)
        phase('FIT_2_OF_3_ENDPOINT',rows=len(x))
        endpoint=ExtraTreesRegressor(**settings).fit(x,q);fit_calls+=1;del x,q;gc.collect();check_resources()
        models=dict(object=obj,endpoint=endpoint,model_family='RELATIVE_OBSERVED_FACE_152_THREE_FORESTS',
            feature_run='run329',training_sources=list(KEYS),feature_columns=152,link_columns=32)
        frozen=[]
        for key in KEYS:
            phase('FREEZE_NEW_PREDICTED_CONNECTION_GEOMETRY',source=key)
            case=load_relative_case(key,supervision=True)
            screen=predict_nodes(case,obj,endpoint);nodes,invalid=union_states(case,screen,models)
            folder=dest/f'links_{key}';folder.mkdir();write(folder/'predicted_nodes.json',dict(nodes=nodes,invalid=invalid))
            m=0
            with (folder/'features.f32').open('wb') as fx,(folder/'links.i32').open('wb') as fl,(folder/'legal.u8').open('wb') as fm:
                for links,features,legal in relative_link_blocks(nodes,case):
                    fx.write(features.astype('<f4').tobytes());fl.write(links.astype('<i4').tobytes());fm.write(legal.astype('u1').tobytes());m+=len(links)
                    if m%10000<len(links):check_resources()
            frozen.append(dict(source=key,nodes=len(nodes),edges=m,invalid=len(invalid),screen_positive=screen['positive_count'],
                features_sha256=digest(folder/'features.f32'),nodes_sha256=digest(folder/'predicted_nodes.json'),
                links_sha256=digest(folder/'links.i32'),legal_sha256=digest(folder/'legal.u8')))
            case['x']._mmap.close();del case,nodes,screen;gc.collect();check_resources()
        write(dest/'all_predicted_geometry_frozen_before_edge_labels.json',dict(cases=frozen,GT_endpoint_overrides=0))
        fitx=[];fity=[]
        for entry in frozen:
            key=entry['source'];phase('LABEL_FROZEN_CONNECTION_TABLE',source=key)
            case=load_relative_case(key,supervision=True);folder=dest/f'links_{key}';m=entry['edges']
            nodes=read(folder/'predicted_nodes.json')['nodes'];truth=case['true_edges'];teacher=case['teacher']
            labels=np.zeros(m,np.int8);known=np.zeros(m,bool);seen=set()
            if m:
                links=np.memmap(folder/'links.i32',dtype='<i4',mode='r',shape=(m,2))
                legal=np.memmap(folder/'legal.u8',dtype='u1',mode='r',shape=(m,3))
                features=np.memmap(folder/'features.f32',dtype='<f4',mode='r',shape=(m,32))
                for e,(a,b) in enumerate(links):
                    u,v=nodes[int(a)]['object_row'],nodes[int(b)]['object_row'];kind=truth.get((u,v),0)
                    if u in teacher and v in teacher:
                        if kind and not legal[e,kind-1]:continue
                        known[e]=True;labels[e]=kind
                    if kind and legal[e,kind-1]:seen.add((u,v))
                fitx.append(np.asarray(features[known]));fity.append(labels[known])
                links._mmap.close();legal._mmap.close();features._mmap.close()
            misses=[dict(first=u,second=v,kind=k) for (u,v),k in truth.items() if (u,v) not in seen]
            np.savez(folder/'supervision.npz',labels=labels,known=known)
            write(folder/'predicted_geometry_edge_misses.json',dict(misses=misses,GT_endpoint_overrides=0))
            entry.update(supervised_edges=int(known.sum()),masked_edges=int((~known).sum()),true_edge_misses=len(misses))
            case['x']._mmap.close();del case;gc.collect();check_resources()
        x=np.concatenate(fitx);y=np.concatenate(fity);fitx.clear();fity.clear();gc.collect()
        phase('FIT_3_OF_3_CONNECTION',rows=len(y))
        models['connection']=ExtraTreesClassifier(**dict(settings,max_depth=16)).fit(x,y);fit_calls+=1
        link_rows=len(y);del x,y;gc.collect();check_resources()
        models.update(frozen_connection_tables=frozen,original_responsibility_preserved=True)
        states={k:forest_state_sha256(models[k]) for k in ('object','endpoint','connection')}
        phase('ATOMIC_SAVE_AND_RELOAD_VERIFY')
        atomic_binary_write(dest/'model.pkl',lambda f:pickle.dump(models,f,protocol=5))
        with (dest/'model.pkl').open('rb') as f:loaded=pickle.load(f)
        assert states=={k:forest_state_sha256(loaded[k]) for k in states}
        write(dest/'worker_receipt.json',dict(status='THREE_RELATIVE_FORESTS_FIT_AND_VERIFIED',forest_fit_calls=fit_calls,
            model_sha256=digest(dest/'model.pkl'),component_state_sha256=states,cases=bindings,G=213,
            object_rows=rows,endpoint_rows=782,connection_rows=link_rows,connection_tables=frozen,
            feature_columns=152,link_columns=32,settings=settings,connection_max_depth=16,
            features_run='run329',seconds=time.perf_counter()-start,full_contract_pass=False,
            extraction_correct_rate=None,extraction_precision_mm=None))
        phase('COMPLETE')
    except BaseException:
        write(dest/'worker_failure.json',dict(error=traceback.format_exc(),forest_fit_calls=fit_calls,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
