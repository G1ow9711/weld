"""Pro336 single seven-case fit, all original compatible weights and interval instances."""
import argparse,gc,pickle,time,traceback
from contextlib import ExitStack
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS,feature_path
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case,predict_instances,relative_link_blocks
from weld_pipeline.sjg_interval_forest_train_1130 import instance_edge_truth
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run
    started=time.perf_counter();fits=0
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-started,fit_calls=fits,**extra))
        print('RELATIVE_ORDINAL_PHASE',name,round(time.perf_counter()-started,3),extra,flush=True)
    try:
        phase('IMPORT_AND_STREAM_ALL_SEVEN_TABLES')
        from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
        gate=read(OUT/'run342/receipt.json');assert gate['status']=='SEVEN_RELATIVE_K12_SUPERVISION_READY'
        n=0;e=0
        with ExitStack() as stack:
            streams={name:stack.enter_context((dest/(name+'.bin')).open('xb')) for name in ('object_x','object_y','object_w','endpoint_x','endpoint_y','endpoint_w')}
            for key in KEYS:
                bound=next(v for v in gate['cases'] if v['source']==key)
                f=feature_path(key)/'features.npy';lf=OUT/f'run318/source_{key}_count_targets.npz';ef=OUT/f'run318/source_{key}_interval_instances.json'
                assert digest(f)==bound['features_sha256'] and digest(lf)==bound['labels_sha256'] and digest(ef)==bound['instances_sha256']
                x=np.load(f,mmap_mode='r')
                with np.load(lf) as z:y=z['labels'];w=z['weights']
                known=np.flatnonzero(y>=0);instances=[v for v in read(ef)['instances'] if v['count_is_known']]
                for first in range(0,len(known),4096):
                    ix=known[first:first+4096]
                    streams['object_x'].write(np.asarray(x[ix],dtype='<f4').tobytes())
                    streams['object_y'].write(y[ix].astype('<i2').tobytes());streams['object_w'].write(w[ix].astype('<f8').tobytes())
                for first in range(0,len(instances),4096):
                    part=instances[first:first+4096];ex=np.empty((len(part),153),np.float32)
                    ex[:,:152]=x[[v['row'] for v in part]];ex[:,152]=np.array([v['k'] for v in part])/12
                    streams['endpoint_x'].write(ex.astype('<f4',copy=False).tobytes())
                    streams['endpoint_y'].write(np.array([v['ratio'] for v in part],dtype='<f8').tobytes())
                    streams['endpoint_w'].write(np.array([v['weight'] for v in part],dtype='<f8').tobytes())
                n+=len(known);e+=len(instances);x._mmap.close();del x,y,w,known,instances;gc.collect();check_resources()
        assert (n,e)==(1134182,13533)
        settings=dict(n_estimators=64,max_depth=32,min_samples_leaf=1,max_features='sqrt',bootstrap=False,n_jobs=1,random_state=1031,warm_start=False,verbose=2)
        def mapped(name,dtype,shape):
            path=dest/(name+'.bin');assert path.stat().st_size==np.prod(shape)*np.dtype(dtype).itemsize
            return np.memmap(path,dtype=dtype,mode='r',shape=shape)
        phase('FIT_1_OF_3_COMPATIBLE_COUNT',rows=n)
        x=mapped('object_x','<f4',(n,152));y=mapped('object_y','<i2',(n,));w=mapped('object_w','<f8',(n,))
        obj=ExtraTreesClassifier(**settings).fit(x,y,sample_weight=w);fits+=1
        for a in (x,y,w):a._mmap.close()
        del x,y,w;gc.collect();check_resources()
        phase('FIT_2_OF_3_ORDINAL_ENDPOINT',rows=e)
        x=mapped('endpoint_x','<f4',(e,153));y=mapped('endpoint_y','<f8',(e,2));w=mapped('endpoint_w','<f8',(e,))
        endpoint=ExtraTreesRegressor(**settings).fit(x,y,sample_weight=w);fits+=1
        for a in (x,y,w):a._mmap.close()
        del x,y,w;gc.collect();check_resources()
        models=dict(object=obj,endpoint=endpoint,K=12,feature_columns=152,feature_run='run329',feature_runs={key:feature_path(key).parent.name for key in (*KEYS,'64')},
            model_family='RELATIVE_COMPATIBLE_K12_FORESTS',training_sources=list(KEYS),supervision_run='run318',compatible_run='run317',input_request_run='run312')
        bindings=[]
        # Freeze every new predicted graph before reading edge association targets.
        for key in KEYS:
            phase('FREEZE_PREDICTED_INSTANCE_CONNECTIONS',source=key)
            case=load_case(key);instances=[v for v in read(OUT/f'run318/source_{key}_interval_instances.json')['instances'] if v['count_is_known']]
            screen=predict_instances(case,models,[(v['row'],v['k']) for v in instances]);nodes=screen['nodes']
            folder=dest/f'links_{key}';folder.mkdir();write(folder/'predicted_nodes.json',screen);m=0
            with (folder/'features.f32').open('xb') as fx,(folder/'links.i32').open('xb') as fl,(folder/'legal.u8').open('xb') as fm:
                for links,features,legal in relative_link_blocks(nodes,case):
                    fx.write(features.astype('<f4',copy=False).tobytes());fl.write(links.astype('<i4').tobytes());fm.write(legal.astype('u1').tobytes());m+=len(links)
                    if m%100000<len(links):check_resources()
            bindings.append(dict(source=key,nodes=len(nodes),edges=m,invalid_instances=len(screen['invalid']),admitted_instances=screen['admitted_instances'],
                features_sha256=digest(folder/'features.f32'),nodes_sha256=digest(folder/'predicted_nodes.json'),links_sha256=digest(folder/'links.i32'),legal_sha256=digest(folder/'legal.u8')))
            case['x']._mmap.close();del case,nodes,screen,instances;gc.collect();check_resources()
        write(dest/'all_predicted_geometry_frozen_before_edge_labels.json',dict(cases=bindings,GT_endpoint_overrides=0))
        total=0
        with (dest/'link_train.f32').open('xb') as fx,(dest/'link_labels.i8').open('xb') as fy:
            for bound in bindings:
                key=bound['source'];phase('LABEL_COMPLETE_INSTANCE_CONNECTIONS',source=key)
                folder=dest/f'links_{key}';nodes=read(folder/'predicted_nodes.json')['nodes'];m=bound['edges']
                instances=[v for v in read(OUT/f'run318/source_{key}_interval_instances.json')['instances'] if v['count_is_known']]
                owners,truth,compiled=instance_edge_truth(key,instances)
                with np.load(OUT/f'run318/source_{key}_count_targets.npz') as z:count_labels=z['labels']
                ids=[(v['object_row'],v['k']) for v in nodes];seen=set();supervised=0
                if m:
                    links=np.memmap(folder/'links.i32',dtype='<i4',mode='r',shape=(m,2));legal=np.memmap(folder/'legal.u8',dtype='u1',mode='r',shape=(m,3))
                    features=np.memmap(folder/'features.f32',dtype='<f4',mode='r',shape=(m,32))
                    for first in range(0,m,8192):
                        part=links[first:first+8192];labels=np.zeros(len(part),np.int8);known=np.zeros(len(part),bool)
                        for j,(a,b) in enumerate(part):
                            u,v=ids[int(a)],ids[int(b)];kind=truth.get((u,v))
                            if kind is not None:
                                if legal[first+j,kind-1]:labels[j]=kind;known[j]=True;seen.add((u,v))
                                continue
                            ou,ov=owners.get(u),owners.get(v)
                            if count_labels[u[0]]==0 or count_labels[v[0]]==0:known[j]=True
                            elif ou is not None and ov is not None:
                                if ou[0]!=ov[0]:known[j]=True
                                elif ou[1]!=ov[1]:
                                    parent=compiled['parents'][ou[0]]
                                    adjacent=ov[1]==ou[1]+1 or (parent['closed'] and ou[1]==parent['child_count']-1 and ov[1]==0)
                                    if not adjacent:known[j]=True
                        fx.write(np.asarray(features[first:first+len(part)][known]).tobytes());fy.write(labels[known].tobytes());supervised+=int(known.sum())
                    for a in (links,legal,features):a._mmap.close()
                misses=[dict(first=list(u),second=list(v),kind=k) for (u,v),k in truth.items() if (u,v) not in seen]
                write(folder/'predicted_geometry_edge_misses.json',dict(misses=misses,GT_endpoint_overrides=0))
                bound.update(supervised_edges=supervised,undetermined_edges=m-supervised,predicted_geometry_true_edge_misses=len(misses));total+=supervised
                del nodes,instances,count_labels,compiled,owners,truth;gc.collect();check_resources()
        assert total>0;phase('FIT_3_OF_3_INSTANCE_CONNECTION',rows=total)
        x=np.memmap(dest/'link_train.f32',dtype='<f4',mode='r',shape=(total,32));y=np.memmap(dest/'link_labels.i8',dtype='i1',mode='r',shape=(total,))
        assert np.any(y==0);models['connection']=ExtraTreesClassifier(**dict(settings,max_depth=16)).fit(x,y);fits+=1
        x._mmap.close();y._mmap.close();del x,y;gc.collect();check_resources()
        states={k:forest_state_sha256(models[k]) for k in ('object','endpoint','connection')}
        phase('ATOMIC_SAVE_AND_VERIFY');atomic_binary_write(dest/'model.pkl',lambda stream:pickle.dump(models,stream,protocol=5))
        with (dest/'model.pkl').open('rb') as f:restored=pickle.load(f)
        assert states=={k:forest_state_sha256(restored[k]) for k in states}
        write(dest/'worker_receipt.json',dict(status='THREE_RELATIVE_ORDINAL_FORESTS_FIT_AND_VERIFIED',forest_fit_calls=fits,model_sha256=digest(dest/'model.pkl'),
            component_state_sha256=states,cases=bindings,G=363,N=n,E=e,K=12,connection_rows=total,settings=settings,
            seconds=time.perf_counter()-started,full_contract_pass=False,extraction_correct_rate=None,extraction_precision_mm=None))
        phase('COMPLETE')
    except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),forest_fit_calls=fits,seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
