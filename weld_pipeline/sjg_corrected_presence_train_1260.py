"""Correct native existence supervision, retain old and new candidate evidence.

The union is only a geometric admission pool. It is never final weld output.
Its fixed two-head rule applies to every workpiece, including development64.
"""
import argparse
import gc
import hashlib
import pickle
import shutil
import time
import traceback
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_local_presence_pilot_1234 import inputs,features
from weld_pipeline.sjg_native_whole_presence_1259 import TRAIN,paths
from weld_pipeline.sjg_whole_joint_symmetry_1231 import reverse_line_descriptors

SCHEMA='NATIVE_PRESENCE_157_FIXED_UNION_WITH384_V1'


def train(dest):
    begin=time.perf_counter();audit=read(OUT/'run419/prefit_audit.json')
    assert audit['status']=='PASS' and audit['total_conflicts']==0
    original=read(OUT/'run419/receipt.json');assert original['G']==original['covered_G']==269
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    for name in ('sjg_corrected_presence_train_1260.py','sjg_native_whole_presence_1259.py','sjg_local_presence_pilot_1234.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snapshot/name)
    write(dest/'registered_protocol.json',dict(schema=SCHEMA,TRAIN=list(TRAIN),DEV=['64'],original_G=269,
        input_run='run419',source_sha256=digest(__file__),new_trees=96,max_depth=24,min_samples_leaf=1,
        max_features=.75,seed=1031,threshold=.5,
        admission='fixed union of frozen384>=0.5 and new native-presence>=0.5 for every case',
        no_threshold_selection_on_DEV=True,best379_unchanged=True,external_consultation_paused=True,full_contract_pass=False))
    total=sum(v['known_rows'] for v in audit['bindings'])
    x=np.empty((total,157),np.float32);y=np.empty(total,np.int8);w=np.empty(total,float)
    bindings=[];offset=0
    for key in TRAIN:
        whole,local=paths(key);raw,lc,ll,case=inputs(key,whole,local)
        sr=read(OUT/f'run419/source_{key}/receipt.json');label_path=OUT/f'run419/source_{key}/presence_targets.npz'
        assert digest(label_path)==sr['labels_sha256']
        with np.load(label_path) as z:labels=z['labels']
        ix=np.flatnonzero(labels>=0);yy=labels[ix];sha=hashlib.sha256()
        for first in range(0,len(ix),4096):
            rr=ix[first:first+4096];xx=features(raw[rr],lc[rr],ll[rr])
            x[offset+first:offset+first+len(rr)]=xx;sha.update(xx.tobytes())
        bound=next(v for v in audit['bindings'] if v['source']==key)
        assert sha.hexdigest()==bound['features_sha256'] and len(ix)==bound['known_rows']
        y[offset:offset+len(ix)]=yy
        for c in (0,1):
            mask=yy==c;assert mask.any();w[offset+np.flatnonzero(mask)]=1/(len(TRAIN)*2*int(mask.sum()))
        offset+=len(ix);bindings.append(dict(source=key,labels_sha256=sr['labels_sha256'],features_sha256=sha.hexdigest()))
        raw._mmap.close();lc._mmap.close();del raw,lc,case;gc.collect();check_resources()
        print('NATIVE_PRESENCE_INPUT',key,offset,'/',total,flush=True)
    assert offset==total and int(y.sum())==sum(v['positive_rows'] for v in original['cases'])
    model=ExtraTreesClassifier(n_estimators=16,max_depth=24,min_samples_leaf=1,max_features=.75,
        bootstrap=False,n_jobs=2,random_state=1031,warm_start=True)
    trace=[]
    for trees in range(16,97,16):
        check_resources();model.set_params(n_estimators=trees);model.fit(x,y,sample_weight=w)
        trace.append(dict(trees=trees,seconds=time.perf_counter()-begin))
        write(dest/'training_progress.json',dict(trace=trace,status='TRAINING_NOT_FINAL'))
        print('NATIVE_PRESENCE TREE_BATCH',trees//16,'/6 TREES',trees,'/96 SECONDS',round(time.perf_counter()-begin,2),flush=True)
    model.set_params(n_jobs=1,warm_start=False)
    payload=dict(schema=SCHEMA,model=model,threshold=.5,prior_run='run384')
    atomic_binary_write(dest/'presence_model.pkl',lambda f:pickle.dump(payload,f,protocol=5))
    with (dest/'presence_model.pkl').open('rb') as f:loaded=pickle.load(f)
    assert np.array_equal(model.predict_proba(x[:256]),loaded['model'].predict_proba(x[:256]))
    prior=read(OUT/'run384/training_receipt.json');assert digest(OUT/'run384/whole_joint_model.pkl')==prior['model_sha256']
    write(dest/'training_receipt.json',dict(status='CORRECTED_NATIVE_PRESENCE_TRAINED',schema=SCHEMA,TRAIN=list(TRAIN),G=269,
        rows=total,positive_rows=int(y.sum()),columns=157,new_trees=96,threshold=.5,trace=trace,bindings=bindings,
        model_sha256=digest(dest/'presence_model.pkl'),prior_model_sha256=prior['model_sha256'],
        forest_state_sha256=forest_state_sha256(model),reload_exact=True,DEV_reads=0,STEP_reads=0,
        full_contract_pass=False,seconds=time.perf_counter()-begin))
    print('NATIVE_PRESENCE_TRAIN_COMPLETE',round(time.perf_counter()-begin,2),flush=True)


def infer(dest,key):
    begin=time.perf_counter();receipt=read(dest/'training_receipt.json')
    assert digest(dest/'presence_model.pkl')==receipt['model_sha256']
    assert digest(OUT/'run384/whole_joint_model.pkl')==receipt['prior_model_sha256']
    with (dest/'presence_model.pkl').open('rb') as f:new=pickle.load(f)['model']
    with (OUT/'run384/whole_joint_model.pkl').open('rb') as f:old=pickle.load(f)['models']['presence']
    whole,local=paths(key);raw,lc,ll,case=inputs(key,whole,local)
    a=np.empty(len(raw));b=np.empty(len(raw));changes=0;maxdelta=0.;line_rows=0
    for first in range(0,len(raw),4096):
        rr=raw[first:first+4096];lcpart=lc[first:first+len(rr)];lengths=ll[first:first+len(rr)]
        xx=features(rr,lcpart,lengths);pa=old.predict_proba(xx)[:,1];pb=new.predict_proba(xx)[:,1]
        a[first:first+len(rr)]=pa;b[first:first+len(rr)]=pb
        mask=rr[:,97]==1
        if mask.any():
            rev=reverse_line_descriptors(rr[mask].astype(float),lengths[mask])
            q=new.predict_proba(features(rev,lcpart[mask],lengths[mask]))[:,1]
            changes+=int(np.sum((q>=.5)!=(pb[mask]>=.5)));maxdelta=max(maxdelta,float(abs(q-pb[mask]).max()));line_rows+=int(mask.sum())
    assert changes==0 and maxdelta==0.
    scores=np.maximum(a,b);selected=np.flatnonzero(scores>=.5);original=np.flatnonzero(a>=.5)
    assert set(original.tolist())<=set(selected.tolist())
    folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False)
    # Downstream1237 consumes only selected_rows and a bound receipt. Do not
    # fabricate intervals from the old count/endpoint heads for new carriers.
    decoded=dict(parents=[],invalid_predictions=[],total_prediction_children=0,
        scope='CARRIER_ADMISSION_ONLY_NOT_WELD_OUTPUT',GT_reads=0,STEP_reads=0)
    write(folder/'decoded.json',decoded)
    np.savez(folder/'presence_scores.npz',scores=scores,selected_rows=selected,
        old_scores=a,new_scores=b,old_selected_rows=original,new_selected_rows=np.flatnonzero(b>=.5))
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_CARRIER_ADMISSION_FROZEN',schema=SCHEMA,source=key,
        decoded_sha256=digest(folder/'decoded.json'),scores_sha256=digest(folder/'presence_scores.npz'),
        model_sha256=receipt['model_sha256'],prior_model_sha256=receipt['prior_model_sha256'],
        original_admitted_rows=len(original),new_model_admitted_rows=int((b>=.5).sum()),admitted_profile_rows=len(selected),
        additional_rows=len(selected)-len(original),old_admitted_rows_all_retained=True,
        line_rows_checked=line_rows,line_reversal_presence_decision_changes=changes,line_reversal_max_probability_difference=maxdelta,
        threshold=.5,all_cases_same_fixed_union_rule=True,GT_reads=0,STEP_reads=0,
        actual_weld_extraction_NOT_EVALUATED=True,full_contract_pass=False,seconds=time.perf_counter()-begin))
    raw._mmap.close();lc._mmap.close();del case;gc.collect();reclaim_own_idle_pages();check_resources()
    print('NATIVE_PRESENCE_ADMISSION',key,'OLD',len(original),'UNION',len(selected),'ADDED',len(selected)-len(original),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source');a=ap.parse_args()
    dest=OUT/'run420'
    try:train(dest) if a.stage=='train' else infer(dest,a.source)
    except BaseException:
        if dest.exists():write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
