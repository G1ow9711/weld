"""Three-TRAIN whole-carrier proposal pilot. No selector replacement.

Presence, count, per-slot category, and interval endpoints are separate learned
heads. Entire saved proposal pools are frozen before any evaluation labels.
"""
import argparse
import gc
import pickle
import time
import traceback
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from weld_pipeline.sjg_local_runtime_904 import OUT,ROOT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_whole_joint_features_1228 import SCHEMA
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN


def fit(dest,input_run):
    begin=time.perf_counter();source=OUT/input_run;gate=read(source/'joint_supervision_receipt.json')
    assert gate['original_G']==213 and gate['TRAIN']==list(TRAIN)
    assert all(not r['category_conflicts'] and not r['exact_feature_count_conflicts'] for r in gate['cases'])
    K=max(r['maximum_known_count'] for r in gate['cases']);xs=[];ys=[];ws=[];exs=[];cats=[];ratios=[];ews=[]
    for key in TRAIN:
        folder=source/f'source_{key}';r=read(folder/'receipt.json');assert r['schema']==SCHEMA
        assert digest(folder/'joint_features.npy')==r['feature_sha256']
        x=np.load(folder/'joint_features.npy',mmap_mode='r')
        with np.load(folder/'joint_count_targets.npz') as z:y=z['labels']
        known=np.flatnonzero(y>=0);w=np.zeros(len(known));positive=y[known]>0
        for mask in (positive,~positive):
            if mask.any():w[mask]=1/(len(TRAIN)*2*int(mask.sum()))
        xs.append(np.asarray(x[known]));ys.append(y[known]);ws.append(w)
        events=[e for e in read(folder/'joint_interval_instances.json')['instances'] if e['count_is_known']]
        ex=np.column_stack((x[[e['row'] for e in events]],np.array([e['k']/K for e in events]))).astype(np.float32)
        exs.append(ex);cats.extend(e['category'] for e in events);ratios.extend(e['ratio'] for e in events)
        ew=np.array([e['weight'] for e in events]);ew/=ew.sum()*len(TRAIN);ews.append(ew);x._mmap.close()
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);ex=np.concatenate(exs)
    q=np.array(ratios);cat=np.array(cats);ew=np.concatenate(ews);del xs,ys,ws,exs,ratios,cats,ews;gc.collect()
    # Known floating-point endpoint normalization, never geometric snapping.
    assert q.min()>=-1e-12 and q.max()<=1+1e-12;q=np.clip(q,0,1)
    signatures={};conflicts=[]
    for i,row in enumerate(ex):
        key=row.tobytes()
        if key in signatures:
            j=signatures[key]
            if cat[i]!=cat[j] or np.max(np.abs(q[i]-q[j]))>1e-10:conflicts.append([int(j),i])
        else:signatures[key]=i
    write(dest/'training_input_audit.json',dict(rows=len(x),positive_rows=int((y>0).sum()),endpoint_rows=len(ex),K=K,
        identical_input_endpoint_or_category_conflicts=conflicts,TRAIN=list(TRAIN),DEV_reads=0))
    if conflicts:raise ValueError('Whole-joint endpoint supervision has identical-input conflicts; no model fitted')
    settings=dict(max_depth=24,min_samples_leaf=1,max_features=.75,bootstrap=False,n_jobs=2,random_state=1031,warm_start=True)
    models={};trace=[]
    tasks=[('presence',ExtraTreesClassifier,x,(y>0).astype(int),w,96),
           ('count',ExtraTreesClassifier,x[y>0],y[y>0],w[y>0],64),
           ('category',ExtraTreesClassifier,ex,cat,ew,64),
           ('endpoint',ExtraTreesRegressor,ex,q,ew,96)]
    for name,cls,features,target,weights,trees in tasks:
        model=cls(n_estimators=16,**settings)
        for count in range(16,trees+1,16):
            check_resources();model.set_params(n_estimators=count);model.fit(features,target,sample_weight=weights)
            trace.append(dict(head=name,trees=count,seconds=time.perf_counter()-begin))
            write(dest/'training_progress.json',dict(status='TRAINING_NOT_FINAL',trace=trace))
            print('TRAIN_HEAD',name,'TREES',count,'/',trees,'SECONDS',round(time.perf_counter()-begin,2),flush=True)
        model.set_params(n_jobs=1,warm_start=False);models[name]=model
    payload=dict(schema=SCHEMA,K=K,models=models,TRAIN=list(TRAIN),input_run=input_run)
    atomic_binary_write(dest/'whole_joint_model.pkl',lambda stream:pickle.dump(payload,stream,protocol=5))
    with (dest/'whole_joint_model.pkl').open('rb') as stream:loaded=pickle.load(stream)
    for name in models:
        data=ex[:256] if name in ('category','endpoint') else x[:256]
        assert np.array_equal(models[name].predict(data),loaded['models'][name].predict(data))
    write(dest/'training_receipt.json',dict(status='WHOLE_JOINT_PROPOSAL_MODEL_TRAINED',schema=SCHEMA,K=K,TRAIN=list(TRAIN),G=213,
        input_run=input_run,trace=trace,trees={n:len(m.estimators_) for n,m in models.items()},
        model_sha256=digest(dest/'whole_joint_model.pkl'),reload_equal=True,DEV_reads=0,STEP_reads=0,
        best_model_replaced=False,full_contract_pass=False,seconds=time.perf_counter()-begin))
    print('TRAIN_COMPLETE',round(time.perf_counter()-begin,2),flush=True)


def infer(dest,input_run,key):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_joint_graph_capacity_1105 import ContinuousGraphBank
    from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices
    from weld_pipeline.sjg_joint_forest_train_1109 import attributes
    begin=time.perf_counter();r=read(dest/'training_receipt.json');assert digest(dest/'whole_joint_model.pkl')==r['model_sha256']
    with (dest/'whole_joint_model.pkl').open('rb') as stream:payload=pickle.load(stream)
    x=np.load(OUT/input_run/f'source_{key}/joint_features.npy',mmap_mode='r');models=payload['models']
    scores=[]
    for first in range(0,len(x),8192):scores.append(models['presence'].predict_proba(x[first:first+8192])[:,1])
    scores=np.concatenate(scores);selected=np.flatnonzero(scores>=.5);counts=models['count'].predict(x[selected]) if len(selected) else []
    rows=[];ks=[]
    for row,count in zip(selected,counts):
        for k in range(1,int(count)+1):rows.append(int(row));ks.append(k)
    print('WHOLE_JOINT_PREDICT',key,'ADMITTED_CARRIERS',len(selected),'SLOTS',len(rows),flush=True)
    choices=[];case=load_case(key);bank=ContinuousGraphBank(case['graph'])
    for first in range(0,len(rows),4096):
        rr=rows[first:first+4096];kk=ks[first:first+4096]
        features=np.column_stack((x[rr],np.array(kk)/payload['K'])).astype(np.float32)
        cats=models['category'].predict(features);q=models['endpoint'].predict(features)
        for row,k,cat,ratio in zip(rr,kk,cats,q):
            attr=attributes(cat)
            choices.append(dict(query=len(choices),probability=float(scores[row]),child_count=1,closed=attr['closed'],
                children=[dict(profile=row//2,intervals=[0,0],fractions=ratio.tolist(),side='UNKNOWN',connection='UNKNOWN',**attr)]))
    decoded=decode_choices(choices,bank)
    decoded.update(runtime_model=SCHEMA,scope='RAW_FRONTEND_PROPOSAL_POOL_NOT_FINAL_SELECTED_OUTPUT',GT_reads=0,STEP_reads=0)
    folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False);write(folder/'decoded.json',decoded)
    np.savez(folder/'presence_scores.npz',scores=scores,selected_rows=selected)
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION',source=key,
        original_profile_rows=len(x),admitted_profile_rows=len(selected),proposed_slots=len(rows),
        valid_parent_candidates=len(decoded['parents']),invalid_children=sum(p['counted_false_children'] for p in decoded['invalid_predictions']),
        decoded_sha256=digest(folder/'decoded.json'),model_sha256=r['model_sha256'],GT_reads=0,STEP_reads=0,
        selector_run379_not_replaced=True,seconds=time.perf_counter()-begin))
    x._mmap.close();case['x']._mmap.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--input-run',required=True);ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=True)
    try:
        if args.stage=='train':
            if (dest/'whole_joint_model.pkl').exists():raise ValueError('Refuse to overwrite existing model')
            fit(dest,args.input_run)
        else:infer(dest,args.input_run,args.source)
    except BaseException:write(dest/f'{args.stage}_{args.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
