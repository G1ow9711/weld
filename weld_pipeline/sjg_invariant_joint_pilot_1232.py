"""Paired-representation invariant presence/count pilot on unchanged TRAIN.

The ordered category and endpoint heads stay frozen from run381. This module
does not replace the best final selector and never trains on DEV64 labels.
"""
import argparse
import gc
import hashlib
import pickle
import time
import traceback
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_whole_joint_symmetry_1231 import reverse_line_descriptors
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256

SCHEMA='PAIRED_LINE_INVARIANT_EXISTENCE_COUNT_320_V1'


def invariant_features(x,lengths):
    original=np.asarray(x,dtype=np.float64);reverse=original.copy();line=original[:,97]==1
    if line.any():reverse[line]=reverse_line_descriptors(original[line],np.asarray(lengths)[line])
    # Each coordinate describes the unordered pair of forward/reverse views.
    # Arc rows keep the old descriptor and zero differences; no arc symmetry
    # guarantee is inferred from the tested line transformation.
    return np.column_stack(((original+reverse)/2,np.abs(original-reverse)/2)).astype(np.float32)


def lengths_for(key):
    case=load_case(key);g=case['graph']
    lengths=np.repeat(g.spans[g.offsets[1:]-1,1],2)
    case['x']._mmap.close()
    return lengths,case


def train(dest):
    start=time.perf_counter();source=OUT/'run380';prior=OUT/'run381'
    assert read(prior/'carrier_reversal_raw_point_parity.json')['status']=='LINE_REVERSAL_RAW_POINT_PARITY_PASSED'
    old=read(prior/'training_receipt.json');assert digest(prior/'whole_joint_model.pkl')==old['model_sha256']
    bindings=[];total=0
    for key in TRAIN:
        folder=source/f'source_{key}';r=read(folder/'receipt.json');s=read(folder/'joint_supervision_receipt.json')
        assert digest(folder/'joint_features.npy')==r['feature_sha256']
        assert digest(folder/'joint_count_targets.npz')==s['labels_sha256']
        with np.load(folder/'joint_count_targets.npz') as z:known=int(np.sum(z['labels']>=0))
        total+=known;bindings.append(dict(source=key,rows=known,feature_sha256=r['feature_sha256'],labels_sha256=s['labels_sha256']))
    x=np.lib.format.open_memmap(dest/'train_invariant_features.npy',mode='w+',dtype=np.float32,shape=(total,320))
    y=np.empty(total,np.int32);w=np.empty(total,float);offset=0;invariance=[]
    for binding in bindings:
        key=binding['source'];folder=source/f'source_{key}';raw=np.load(folder/'joint_features.npy',mmap_mode='r')
        with np.load(folder/'joint_count_targets.npz') as z:labels=z['labels']
        indices=np.flatnonzero(labels>=0);lengths,case=lengths_for(key)
        yy=labels[indices];ww=np.zeros(len(indices));positive=yy>0
        for mask in (positive,~positive):
            if mask.any():ww[mask]=1/(len(TRAIN)*2*int(mask.sum()))
        for first in range(0,len(indices),4096):
            ix=indices[first:first+4096];x[offset+first:offset+first+len(ix)]=invariant_features(raw[ix],lengths[ix])
        y[offset:offset+len(indices)]=yy;w[offset:offset+len(indices)]=ww
        line_indices=indices[raw[indices,97]==1]
        sample=line_indices[np.linspace(0,len(line_indices)-1,min(256,len(line_indices)),dtype=int)]
        original=np.asarray(raw[sample],float);flipped=reverse_line_descriptors(original,lengths[sample])
        before=invariant_features(original,lengths[sample]);after=invariant_features(flipped,lengths[sample])
        error=float(np.abs(before-after).max(initial=0));assert np.allclose(before,after,atol=2e-6,rtol=2e-6)
        invariance.append(dict(source=key,input_rows=len(sample),maximum_feature_change=error))
        offset+=len(indices);raw._mmap.close();del case,raw;gc.collect();check_resources()
        print('INVARIANT_INPUT_READY',key,offset,'/',total,flush=True)
    assert offset==total==265796 and int((y>0).sum())==813
    x.flush();np.savez(dest/'train_labels_weights.npz',labels=y,weights=w)
    # Audit exact known count conflicts after the new invariant mapping.
    seen={};conflicts=[]
    for i in range(total):
        signature=hashlib.sha256(x[i].tobytes()).digest();previous=seen.get(signature)
        if previous is None:seen[signature]=(i,int(y[i]))
        elif previous[1]!=y[i]:conflicts.append(dict(first_row=previous[0],second_row=i,counts=[previous[1],int(y[i])]))
    del seen;gc.collect()
    write(dest/'input_control.json',dict(status='INVARIANT_INPUT_AUDITED',rows=total,positive_rows=813,bindings=bindings,
        line_reversal_checks=invariance,exact_known_count_conflicts=conflicts,DEV_reads=0,fit_calls=0,
        feature_file_sha256=digest(dest/'train_invariant_features.npy')))
    if conflicts:raise ValueError('Invariant descriptors expose contradictory complete count labels; no fit')
    with (prior/'whole_joint_model.pkl').open('rb') as stream:payload=pickle.load(stream)
    models=payload['models'];frozen={k:forest_state_sha256(models[k]) for k in ('category','endpoint')}
    settings=dict(max_depth=24,min_samples_leaf=1,max_features=.75,bootstrap=False,n_jobs=2,random_state=1031,warm_start=True)
    trace=[]
    for name,features,target,weight,trees in [('presence',x,(y>0).astype(int),w,96),('count',x[y>0],y[y>0],w[y>0],64)]:
        model=ExtraTreesClassifier(n_estimators=16,**settings)
        for count in range(16,trees+1,16):
            check_resources();model.set_params(n_estimators=count);model.fit(features,target,sample_weight=weight)
            trace.append(dict(head=name,trees=count,seconds=time.perf_counter()-start))
            write(dest/'training_progress.json',dict(status='TRAINING_NOT_FINAL',trace=trace))
            print('INVARIANT_TRAIN',name,'TREES',count,'/',trees,'SECONDS',round(time.perf_counter()-start,2),flush=True)
        model.set_params(n_jobs=1,warm_start=False);models[name]=model
    assert frozen=={k:forest_state_sha256(models[k]) for k in frozen}
    payload.update(schema=SCHEMA,models=models,invariant_presence_count=True,ordered_heads_frozen_from='run381')
    atomic_binary_write(dest/'whole_joint_model.pkl',lambda stream:pickle.dump(payload,stream,protocol=5))
    with (dest/'whole_joint_model.pkl').open('rb') as stream:reloaded=pickle.load(stream)
    for name in ('presence','count'):
        assert np.array_equal(models[name].predict_proba(x[:256]),reloaded['models'][name].predict_proba(x[:256]))
    write(dest/'training_receipt.json',dict(status='INVARIANT_PRESENCE_COUNT_TRAINED',schema=SCHEMA,TRAIN=list(TRAIN),G=213,K=payload['K'],
        new_trees=dict(presence=96,count=64),frozen_heads_sha256=frozen,model_sha256=digest(dest/'whole_joint_model.pkl'),
        same_TRAIN_rows_labels_weights_as_run381=True,threshold=.5,trace=trace,DEV_reads=0,STEP_reads=0,
        best_run379_unchanged=True,reload_exact=True,full_contract_pass=False,seconds=time.perf_counter()-start))
    x._mmap.close();print('INVARIANT_TRAIN_COMPLETE',round(time.perf_counter()-start,2),flush=True)


def infer(dest,key):
    from weld_pipeline.sjg_joint_graph_capacity_1105 import ContinuousGraphBank
    from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices
    from weld_pipeline.sjg_joint_forest_train_1109 import attributes
    start=time.perf_counter();r=read(dest/'training_receipt.json');assert digest(dest/'whole_joint_model.pkl')==r['model_sha256']
    with (dest/'whole_joint_model.pkl').open('rb') as stream:payload=pickle.load(stream)
    models=payload['models'];raw=np.load(OUT/f'run380/source_{key}/joint_features.npy',mmap_mode='r');lengths,case=lengths_for(key)
    scores=np.empty(len(raw));counts=np.empty(len(raw),np.int32);max_delta=0.;decision_changes=0;count_changes=0;line_rows=0
    for first in range(0,len(raw),4096):
        data=np.asarray(raw[first:first+4096]);ll=lengths[first:first+len(data)];features=invariant_features(data,ll)
        pp=models['presence'].predict_proba(features)[:,1];cc=models['count'].predict(features)
        scores[first:first+len(data)]=pp;counts[first:first+len(data)]=cc
        line=data[:,97]==1
        if line.any():
            reverse=reverse_line_descriptors(data[line].astype(float),ll[line]);z=invariant_features(reverse,ll[line])
            rp=models['presence'].predict_proba(z)[:,1];rc=models['count'].predict(z)
            max_delta=max(max_delta,float(np.abs(rp-pp[line]).max(initial=0)))
            decision_changes+=int(np.sum((rp>=.5)!=(pp[line]>=.5)));count_changes+=int(np.sum(rc!=cc[line]));line_rows+=int(line.sum())
    selected=np.flatnonzero(scores>=.5);rows=[];ks=[]
    for row in selected:
        for k in range(1,int(counts[row])+1):rows.append(int(row));ks.append(k)
    print('INVARIANT_PREDICT',key,'CARRIERS',len(selected),'SLOTS',len(rows),'REVERSE_DECISION_CHANGES',decision_changes,'COUNT_CHANGES',count_changes,flush=True)
    choices=[]
    for first in range(0,len(rows),4096):
        rr=rows[first:first+4096];kk=ks[first:first+4096]
        features=np.column_stack((raw[rr],np.array(kk)/payload['K'])).astype(np.float32)
        cats=models['category'].predict(features);ratios=models['endpoint'].predict(features)
        for row,cat,ratio in zip(rr,cats,ratios):
            attr=attributes(cat)
            choices.append(dict(query=len(choices),probability=float(scores[row]),child_count=1,closed=attr['closed'],
                children=[dict(profile=row//2,intervals=[0,0],fractions=ratio.tolist(),side='UNKNOWN',connection='UNKNOWN',**attr)]))
    decoded=decode_choices(choices,ContinuousGraphBank(case['graph']));folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False)
    decoded.update(runtime_model=SCHEMA,scope='RAW_FRONTEND_PROPOSAL_POOL_NOT_FINAL_SELECTED_OUTPUT',GT_reads=0,STEP_reads=0)
    write(folder/'decoded.json',decoded);np.savez(folder/'presence_scores.npz',scores=scores,selected_rows=selected)
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION',source=key,
        decoded_sha256=digest(folder/'decoded.json'),model_sha256=r['model_sha256'],admitted_profile_rows=len(selected),proposed_slots=len(rows),
        line_rows_checked=line_rows,line_reversal_max_probability_difference=max_delta,line_reversal_presence_decision_changes=decision_changes,
        line_reversal_count_changes=count_changes,GT_reads=0,STEP_reads=0,seconds=time.perf_counter()-start))
    raw._mmap.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source');ap.add_argument('--resume-prefit',action='store_true');a=ap.parse_args()
    dest=OUT/a.run
    if a.stage=='train':
        if a.resume_prefit:
            failure=read(dest/'train_all_failure.json')
            assert 'Resource guard: less than 5 GiB disk free' in failure['error']
            assert not any((dest/name).exists() for name in ('whole_joint_model.pkl','training_progress.json','training_receipt.json'))
            protocol=read(dest/'registered_protocol.json')
            assert protocol['schema']==SCHEMA and protocol['TRAIN']==list(TRAIN) and protocol['G']==213
            (dest/'train_all_failure.json').rename(dest/'prefit_failure_01.json')
            (dest/'registered_protocol.json').rename(dest/'prefit_protocol_01.json')
        else:dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(schema=SCHEMA,TRAIN=list(TRAIN),DEV=['64'],G=213,
            change='Representation of existence/count only; symmetric mean and absolute half difference, unchanged rows/weights/settings/threshold',
            ordered_category_endpoint_frozen='run381',base_best_preserved='run379',external_consultation_paused=True,source_sha256=digest(__file__)))
    try:train(dest) if a.stage=='train' else infer(dest,a.source)
    except BaseException:write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
