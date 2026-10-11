"""Train and freeze inward endpoint-role scores on the original three TRAINs.

This component does not by itself assemble or select final weld intervals.
"""
import argparse
import gc
import hashlib
import pickle
import time
import traceback
from collections import defaultdict
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_endpoint_role_features_1245 import SCHEMA,COLS
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


def inputs(key,input_run='run396'):
    folder=OUT/f'{input_run}/source_{key}';r=read(folder/'receipt.json')
    assert r['GT_reads']==0 and r['schema']==SCHEMA
    assert digest(folder/'role_map.npz')==r['map_sha256']
    with np.load(folder/'role_map.npz') as z:m={k:z[k] for k in z.files}
    if (folder/'feature_overlay.json').exists():
        from weld_pipeline.sjg_endpoint_occupancy_overlay_1272 import load_overlay
        return load_overlay(folder,r),m,r
    assert digest(folder/'features.npy')==r['feature_sha256']
    return np.load(folder/'features.npy',mmap_mode='r'),m,r


def train(dest):
    # This target import is confined to the training path, never inference.
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    start=time.perf_counter();plans=[];total=0
    for key in TRAIN:
        x,m,r=inputs(key);x._mmap.close();folder=OUT/f'run380/source_{key}'
        sr=read(folder/'joint_supervision_receipt.json')
        assert digest(folder/'joint_count_targets.npz')==sr['labels_sha256']
        assert digest(folder/'joint_interval_instances.json')==sr['instances_sha256']
        with np.load(folder/'joint_count_targets.npz') as z:counts=z['labels']
        selected=[i for i,row in enumerate(m['rows']) if counts[row]>=0]
        size=sum(int(2*(m['offsets'][i+1]-m['offsets'][i])) for i in selected)
        plans.append((key,m,selected,size));total+=size
    xx=np.empty((total,COLS),np.float32);yy=np.zeros(total,np.uint8);ww=np.zeros(total,float)
    audits=[];cursor=0
    for key,m,selected,size in plans:
        x,_,r=inputs(key);target,binding=target_for(key);case=load_case(key);g=case['graph'];case['x']._mmap.close()
        events=defaultdict(list);owners=set();covered=set();missing=[];begin=cursor
        for event in read(OUT/f'run380/source_{key}/joint_interval_instances.json')['instances']:
            owners.add(tuple(event['owner']))
            if event['count_is_known']:events[event['row']].append(event)
        for i in selected:
            row=int(m['rows'][i]);a,b=map(int,m['offsets'][i:i+2]);n=b-a;q=m['q_mm'][a:b]
            xx[cursor:cursor+2*n]=x[2*a:2*b]
            curve=g.curve(row//2);points=curve.at(q)
            for event in events[row]:
                pi,ci=event['owner'];truth=target['parents'][pi]['children'][ci]
                endpoints=np.asarray(truth['original_points_mm'])[[0,-1]]
                native=curve.at(np.array(event['ratio'])*event['length_mm'])
                if np.linalg.norm(native-endpoints[::-1],axis=1).sum()<np.linalg.norm(native-endpoints,axis=1).sum():
                    endpoints=endpoints[::-1]
                found=[]
                for role in (0,1):
                    mask=np.linalg.norm(points-endpoints[role],axis=1)<=3.
                    found.append(bool(mask.any()));lo=cursor+role*n
                    yy[lo:lo+n][mask]=1
                    if mask.any():ww[lo:lo+n][mask]+=event['weight']/int(mask.sum())
                if all(found):covered.add(tuple(event['owner']))
            cursor+=2*n
        y=yy[begin:cursor];w=ww[begin:cursor];pos=y>0;neg=~pos
        if pos.any():w[pos]/=(len(TRAIN)*2*w[pos].sum())
        if neg.any():w[neg]=1/(len(TRAIN)*2*int(neg.sum()))
        audit=dict(source=key,known_carriers=len(selected),directional_anchor_rows=cursor-begin,
            positive_roles=int(pos.sum()),original_children=len(owners),both_endpoints_supervised=len(covered),
            missing_owners=[list(o) for o in sorted(owners-covered)],target_sha256=binding['target_sha256'],
            input_features_sha256=r['feature_sha256'],unknown_carriers_excluded_not_negative=True)
        audits.append(audit);x._mmap.close();del case,g;gc.collect()
        print('ENDPOINT_SUPERVISION',audit,flush=True)
    assert cursor==total and not any(a['missing_owners'] for a in audits)
    write(dest/'supervision_receipt.json',dict(cases=audits,G=sum(a['original_children'] for a in audits),
        rows=total,columns=COLS,DEV_reads=0,STEP_reads=0,original_labels_retained=True))
    seen={};conflicts=[]
    for i in range(total):
        h=hashlib.sha256(xx[i].tobytes()).digest()
        previous=seen.get(h)
        if previous is None:seen[h]=(i,int(yy[i]))
        elif previous[1]!=yy[i]:conflicts.append([previous[0],i])
    del seen;gc.collect()
    write(dest/'input_collision_audit.json',dict(rows=total,opposite_role_label_pairs=conflicts,
        original_labels_unchanged=True,no_rows_deleted=True))
    check_resources();model=ExtraTreesClassifier(n_estimators=16,max_depth=18,min_samples_leaf=4,
        max_features=.75,random_state=1031,n_jobs=2,warm_start=True,bootstrap=False)
    trace=[]
    for trees in range(16,65,16):
        check_resources();model.set_params(n_estimators=trees);model.fit(xx,yy,sample_weight=ww)
        trace.append(dict(trees=trees,seconds=time.perf_counter()-start))
        write(dest/'training_progress.json',dict(trace=trace,total_trees=64))
        print('ENDPOINT_TRAIN TREES',trees,'/64 SECONDS',round(time.perf_counter()-start,2),flush=True)
    model.set_params(n_jobs=1,warm_start=False)
    atomic_binary_write(dest/'endpoint_role_model.pkl',lambda f:pickle.dump(model,f,protocol=5))
    with (dest/'endpoint_role_model.pkl').open('rb') as f:reloaded=pickle.load(f)
    assert np.array_equal(model.predict_proba(xx[:512]),reloaded.predict_proba(xx[:512]))
    write(dest/'training_receipt.json',dict(status='ENDPOINT_ROLE_COMPONENT_TRAINED',TRAIN=list(TRAIN),G=213,
        rows=total,columns=COLS,trees=64,trace=trace,threshold=.5,DEV_reads=0,STEP_reads=0,
        model_sha256=digest(dest/'endpoint_role_model.pkl'),reload_exact=True,seconds=time.perf_counter()-start,
        opposite_feature_conflicts=len(conflicts),not_final_interval_or_parent_model=True,full_contract_pass=False))
    print('ENDPOINT_TRAIN_COMPLETE',round(time.perf_counter()-start,2),'CONFLICTS',len(conflicts),flush=True)


def infer(dest,key):
    start=time.perf_counter();r=read(dest/'training_receipt.json')
    assert digest(dest/'endpoint_role_model.pkl')==r['model_sha256']
    with (dest/'endpoint_role_model.pkl').open('rb') as f:model=pickle.load(f)
    x,m,ir=inputs(key);p=np.empty(len(x),np.float32)
    for a in range(0,len(x),8192):p[a:a+8192]=model.predict_proba(x[a:a+8192])[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    folder=dest/f'scores_{key}';folder.mkdir(exist_ok=False)
    np.save(folder/'probabilities.npy',p)
    write(folder/'receipt.json',dict(status='PURE_INPUT_ENDPOINT_SCORES_FROZEN',source=key,
        probabilities_sha256=digest(folder/'probabilities.npy'),map_sha256=ir['map_sha256'],
        model_sha256=r['model_sha256'],GT_reads=0,STEP_reads=0,rows=len(p),
        positive_roles=int(np.sum(p>=.5)),seconds=time.perf_counter()-start,
        not_weld_extraction_result=True))
    x._mmap.close();print('ENDPOINT_PREDICT',key,'POSITIVE_ROLES',int(np.sum(p>=.5)),'OF',len(p),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source')
    ap.add_argument('--resume-preparation',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.stage=='train' and not args.resume_preparation:
        dest.mkdir(exist_ok=False);write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),
            TRAIN=list(TRAIN),DEV=['64'],input_run='run396',trees=64,depth=18,min_leaf=4,seed=1031,
            threshold=.5,baseline379_preserved=True,external_consultation_paused=True))
    elif args.stage=='train':
        assert not (dest/'endpoint_role_model.pkl').exists() and not (dest/'training_progress.json').exists()
        write(dest/'preparation_retry.json',dict(source_sha256=digest(__file__),fit_calls_before_retry=0,
            reason='serialize Python integer counters instead of NumPy scalars'))
    try:train(dest) if args.stage=='train' else infer(dest,args.source)
    except BaseException:write(dest/f'{args.stage}_{args.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
