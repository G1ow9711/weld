"""One seven-TRAIN count/ordinal/instance-link forest experiment."""
import argparse
import gc
import pickle
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_forest_train_1109 import load_case
from weld_pipeline.sjg_interval_forest_runtime_1129 import predict_instances,link_blocks
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

KEYS=('21','4','10','25','36','42','BM79A_native_local')


def instance_edge_truth(key,instances):
    mapping={(e['row'],tuple(e['owner'])):(e['row'],e['k']) for e in instances if e['count_is_known']}
    owners={(e['row'],e['k']):tuple(e['owner']) for e in instances if e['count_is_known']}
    compiled=read(OUT/f'run264/source_{key}_targets.json')
    truth={}
    for path in read(OUT/f'run317/source_{key}_complete_path_proofs.json'):
        if not path['passed']:continue
        chain=path['instance_chain'];parent=compiled['parents'][path['parent']]
        pairs=list(zip(chain[:-1],chain[1:]))
        if parent['closed'] and len(chain)>1:pairs.append((chain[-1],chain[0]))
        for a,b in pairs:
            u=mapping.get((a['object_row'],tuple(a['owner'])));v=mapping.get((b['object_row'],tuple(b['owner'])))
            if u is None or v is None:continue
            if a['owner'][1]==b['owner'][1]:kind=1
            elif b is chain[0]:kind=2
            else:kind={'SHARED':2,'POSITIVE_GAP':3}[parent['gaps'][a['owner'][1]]]
            if (u,v) in truth and truth[u,v]!=kind:raise ValueError('Instance edge class conflict')
            truth[u,v]=kind
    return owners,truth,compiled


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter();fit_calls=0
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,utc=utc(),**extra))
        print('INTERVAL_FOREST_PHASE',name,'SECONDS',round(time.perf_counter()-start,3),extra,flush=True)
    try:
        phase('IMPORTING_CPU_FORESTS')
        from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
        supervision=read(OUT/'run318/receipt.json');assert supervision['status']=='ORDERED_INTERVAL_SUPERVISION_READY'
        K=int(supervision['K']);cases=[];object_x=[];object_y=[];object_w=[];endpoint_x=[];endpoint_y=[];endpoint_w=[]
        for key in KEYS:
            phase('LOAD_COMPLETE_TRAIN_TABLE',source=key)
            case=load_case(key,False)
            with np.load(OUT/f'run318/source_{key}_count_targets.npz') as z:y=z['labels'];w=z['weights']
            instances=read(OUT/f'run318/source_{key}_interval_instances.json')['instances']
            instances=[e for e in instances if e['count_is_known']]
            case.update(instances=instances,count_labels=y);cases.append(case)
            known=y>=0;object_x.append(np.asarray(case['x'][known]));object_y.append(y[known]);object_w.append(w[known])
            ex=np.empty((len(instances),129),np.float32)
            ex[:,:128]=case['x'][[e['row'] for e in instances]];ex[:,128]=np.asarray([e['k'] for e in instances])/K
            endpoint_x.append(ex);endpoint_y.extend(e['ratio'] for e in instances);endpoint_w.extend(e['weight'] for e in instances)
            check_resources()
        x=np.concatenate(object_x);y=np.concatenate(object_y);w=np.concatenate(object_w)
        object_x.clear();object_y.clear();object_w.clear();gc.collect()
        settings=dict(n_estimators=64,max_depth=32,min_samples_leaf=1,max_features='sqrt',bootstrap=False,
                      n_jobs=1,random_state=1031,warm_start=False,verbose=2)
        phase('FIT_1_OF_3_COUNT',rows=len(y),K=K)
        obj=ExtraTreesClassifier(**settings).fit(x,y,sample_weight=w);fit_calls+=1
        object_rows=len(y);del x,y,w;gc.collect();check_resources()
        ex=np.concatenate(endpoint_x);ey=np.asarray(endpoint_y);ew=np.asarray(endpoint_w)
        endpoint_x.clear();endpoint_y.clear();endpoint_w.clear()
        phase('FIT_2_OF_3_ORDINAL_ENDPOINT',rows=len(ey),columns=129)
        endpoint=ExtraTreesRegressor(**settings).fit(ex,ey,sample_weight=ew);fit_calls+=1
        E=len(ey);del ex,ey,ew;gc.collect();check_resources()
        models=dict(object=obj,endpoint=endpoint,K=K,model_family='ORDERED_LOCAL_INTERVAL_FORESTS',training_sources=list(KEYS))
        bindings=[];tables=[]
        for case in cases:
            key=case['key'];phase('FREEZE_PREDICTED_INSTANCE_GEOMETRY',source=key)
            required=[(e['row'],e['k']) for e in case['instances']]
            prediction=predict_instances(case,models,required);nodes=prediction['nodes']
            folder=dest/f'links_{key}';folder.mkdir(exist_ok=False)
            write(folder/'predicted_nodes.json',prediction)
            m=0
            with (folder/'features.f32').open('wb') as fx,(folder/'links.i32').open('wb') as fl,(folder/'legal.u8').open('wb') as fm:
                for links,features,legal in link_blocks(nodes,case['graph']):
                    fx.write(features.tobytes());fl.write(links.astype(np.int32).tobytes());fm.write(legal.astype(np.uint8).tobytes());m+=len(links)
                    if m%100000<len(links):check_resources()
            check_resources()
            if shutil.disk_usage(dest).free<5368709120:raise RuntimeError('Original 5 GiB reserve')
            bound=dict(source=key,nodes=len(nodes),edges=m,admitted_instances=prediction['admitted_instances'],
                invalid_instances=len(prediction['invalid']),saturated_objects=len(prediction['saturated_object_rows']),
                predicted_nodes_sha256=digest(folder/'predicted_nodes.json'),
                features_sha256=digest(folder/'features.f32'),links_sha256=digest(folder/'links.i32'),legal_sha256=digest(folder/'legal.u8'))
            bindings.append(bound);tables.append((folder,nodes,m));print('ALL_INSTANCE_LINKS_FROZEN',key,len(nodes),m,flush=True)
        write(dest/'all_predicted_geometry_frozen_before_edge_labels.json',dict(cases=bindings,GT_endpoint_overrides=0))
        fitx=[];fity=[]
        for case,(folder,nodes,m),binding in zip(cases,tables,bindings):
            key=case['key'];phase('LABEL_FULL_INSTANCE_CONNECTIONS',source=key,edges=m)
            owners,truth,compiled=instance_edge_truth(key,case['instances'])
            ids=[(n['object_row'],n['k']) for n in nodes]
            labels=np.zeros(m,np.int8);known=np.zeros(m,bool);seen=set()
            if m:
                links=np.memmap(folder/'links.i32',dtype=np.int32,mode='r',shape=(m,2))
                legal=np.memmap(folder/'legal.u8',dtype=np.uint8,mode='r',shape=(m,3))
                for i,(a,b) in enumerate(links):
                    u,v=ids[int(a)],ids[int(b)];kind=truth.get((u,v))
                    if kind is not None:
                        if legal[i,kind-1]:labels[i]=kind;known[i]=True;seen.add((u,v))
                        continue
                    ou,ov=owners.get(u),owners.get(v)
                    if case['count_labels'][u[0]]==0 or case['count_labels'][v[0]]==0:
                        known[i]=True
                    elif ou is not None and ov is not None:
                        if ou[0]!=ov[0]:known[i]=True
                        elif ou[1]!=ov[1]:
                            parent=compiled['parents'][ou[0]]
                            adjacent=ov[1]==ou[1]+1 or (parent['closed'] and ou[1]==parent['child_count']-1 and ov[1]==0)
                            if not adjacent:known[i]=True
                features=np.memmap(folder/'features.f32',dtype=np.float32,mode='r',shape=(m,32))
                fitx.append(np.asarray(features[known]));fity.append(labels[known])
            misses=[dict(first=list(u),second=list(v),kind=kind) for (u,v),kind in truth.items() if (u,v) not in seen]
            np.savez(folder/'edge_supervision.npz',labels=labels,known=known)
            write(folder/'predicted_geometry_edge_misses.json',dict(misses=misses,GT_endpoint_overrides=0,denominator_unchanged=True))
            binding.update(supervised_edges=int(known.sum()),undetermined_edges=int((~known).sum()),predicted_geometry_true_edge_misses=len(misses))
            check_resources()
        x=np.concatenate(fitx);y=np.concatenate(fity);fitx.clear();fity.clear();gc.collect()
        if not len(y) or not np.any(y==0):raise ValueError('No complete supervised link training table')
        phase('FIT_3_OF_3_INSTANCE_CONNECTION',rows=len(y))
        models['connection']=ExtraTreesClassifier(**dict(settings,max_depth=16)).fit(x,y);fit_calls+=1
        link_rows=len(y);del x,y;gc.collect();check_resources()
        models.update(bindings=bindings,supervision_run='run318',compatible_run='run317',input_request_run='run312')
        states={k:forest_state_sha256(models[k]) for k in ('object','endpoint','connection')}
        phase('ATOMIC_SAVE_AND_RELOAD_VERIFY')
        atomic_binary_write(dest/'model.pkl',lambda f:pickle.dump(models,f,protocol=5))
        with (dest/'model.pkl').open('rb') as f:restored=pickle.load(f)
        assert states=={k:forest_state_sha256(restored[k]) for k in states}
        write(dest/'worker_receipt.json',dict(status='THREE_INTERVAL_FORESTS_FIT_AND_VERIFIED',forest_fit_calls=fit_calls,
            model_sha256=digest(dest/'model.pkl'),component_state_sha256=states,K=K,E=E,
            object_training_rows=object_rows,connection_training_rows=link_rows,cases=bindings,G=363,
            forest_settings=settings,connection_max_depth=16,GT_endpoint_overrides=0,
            fixed_presence_threshold=.5,all_positive_nodes_fixed_selected=False,negative_subsampling=False,
            seconds=time.perf_counter()-start,full_contract_pass=False,extraction_correct_rate=None,extraction_precision_mm=None))
        phase('COMPLETE')
    except BaseException:
        write(dest/'worker_failure.json',dict(error=traceback.format_exc(),forest_fit_calls=fit_calls,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
