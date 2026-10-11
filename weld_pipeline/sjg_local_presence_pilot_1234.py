"""Controlled local-face presence pilot; count/category/endpoints frozen."""
import argparse,gc,hashlib,pickle,time,traceback
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_invariant_joint_pilot_1232 import invariant_features,lengths_for
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN
from weld_pipeline.sjg_whole_joint_symmetry_1231 import reverse_line_descriptors
from weld_pipeline.sjg_local_cross_section_1233 import SCHEMA as INPUT_SCHEMA

SCHEMA='LOCAL_SECTION_EXISTENCE_157_FROZEN_OTHER_HEADS_V1'
GEOMETRY=np.r_[np.arange(96,112),np.arange(144,160)]
INVARIANT_GEOMETRY=np.r_[GEOMETRY,GEOMETRY+160]


def inputs(key,whole_run='run380',local_run='run383'):
    folder=OUT/f'{local_run}/source_{key}';r=read(folder/'receipt.json')
    assert r['schema']==INPUT_SCHEMA and digest(folder/'local_features.npy')==r['feature_sha256']
    oldfolder=OUT/f'{whole_run}/source_{key}';old=read(oldfolder/'receipt.json')
    assert digest(oldfolder/'joint_features.npy')==old['feature_sha256']
    assert r['input_sha256']==old['input_sha256'] and r['geometry_sha256']==old['geometry_sha256']
    raw=np.load(oldfolder/'joint_features.npy',mmap_mode='r');local=np.load(folder/'local_features.npy',mmap_mode='r')
    lengths,case=lengths_for(key)
    return raw,local,lengths,case


def features(raw,local,lengths):
    return np.column_stack((local,invariant_features(raw,lengths)[:,INVARIANT_GEOMETRY])).astype(np.float32)


def train(dest):
    if (OUT/'run380/confirmed_presence_scope_conflicts.json').exists():
        raise RuntimeError('Known incomplete whole-carrier presence labels: use audited native presence supervision before refitting; frozen inference remains available')
    start=time.perf_counter();prior=OUT/'run382';pr=read(prior/'training_receipt.json')
    assert digest(prior/'whole_joint_model.pkl')==pr['model_sha256']
    x=np.empty((265796,157),np.float32);y=np.empty(265796,np.int32);w=np.empty(265796,float);offset=0;bindings=[]
    for key in TRAIN:
        raw,local,lengths,case=inputs(key)
        supervision=read(OUT/f'run380/source_{key}/joint_supervision_receipt.json')
        assert digest(OUT/f'run380/source_{key}/joint_count_targets.npz')==supervision['labels_sha256']
        with np.load(OUT/f'run380/source_{key}/joint_count_targets.npz') as z:labels=z['labels']
        ix=np.flatnonzero(labels>=0);yy=(labels[ix]>0).astype(np.int32);ww=np.zeros(len(ix))
        for mask in (yy>0,yy==0):
            if mask.any():ww[mask]=1/(len(TRAIN)*2*int(mask.sum()))
        for first in range(0,len(ix),4096):
            ii=ix[first:first+4096];x[offset+first:offset+first+len(ii)]=features(raw[ii],local[ii],lengths[ii])
        y[offset:offset+len(ix)]=yy;w[offset:offset+len(ix)]=ww;offset+=len(ix)
        bindings.append(dict(source=key,local_feature_sha256=digest(OUT/f'run383/source_{key}/local_features.npy'),
            labels_sha256=digest(OUT/f'run380/source_{key}/joint_count_targets.npz')))
        raw._mmap.close();local._mmap.close();del case,raw,local;gc.collect();check_resources()
        print('LOCAL_TRAIN_INPUT',key,offset,'/',len(x),flush=True)
    assert offset==265796 and y.sum()==813
    seen={};conflicts=[]
    for i in range(len(x)):
        sig=hashlib.sha256(x[i].tobytes()).digest();previous=seen.get(sig)
        if previous is None:seen[sig]=(i,int(y[i]))
        elif previous[1]!=y[i]:conflicts.append(dict(first=previous[0],second=i))
    del seen;gc.collect()
    write(dest/'input_control.json',dict(rows=len(x),positive_rows=int(y.sum()),columns=157,
        exact_presence_label_conflicts=conflicts,bindings=bindings,feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        fit_calls=0,DEV_reads=0,STEP_reads=0))
    if conflicts:raise ValueError('Local features expose contradictory presence labels; no fit')
    with (prior/'whole_joint_model.pkl').open('rb') as f:payload=pickle.load(f)
    models=payload['models'];frozen={k:forest_state_sha256(models[k]) for k in ('count','category','endpoint')}
    model=ExtraTreesClassifier(n_estimators=16,max_depth=24,min_samples_leaf=1,max_features=.75,
        bootstrap=False,n_jobs=2,random_state=1031,warm_start=True)
    trace=[]
    for trees in range(16,97,16):
        check_resources();model.set_params(n_estimators=trees);model.fit(x,y,sample_weight=w)
        trace.append(dict(head='presence',trees=trees,seconds=time.perf_counter()-start))
        write(dest/'training_progress.json',dict(trace=trace,status='TRAINING_NOT_FINAL'))
        print('LOCAL_TRAIN presence TREES',trees,'/96 SECONDS',round(time.perf_counter()-start,2),flush=True)
    model.set_params(n_jobs=1,warm_start=False);models['presence']=model
    assert frozen=={k:forest_state_sha256(models[k]) for k in frozen}
    payload.update(schema=SCHEMA,local_presence=True)
    atomic_binary_write(dest/'whole_joint_model.pkl',lambda f:pickle.dump(payload,f,protocol=5))
    with (dest/'whole_joint_model.pkl').open('rb') as f:reload=pickle.load(f)
    assert np.array_equal(model.predict_proba(x[:256]),reload['models']['presence'].predict_proba(x[:256]))
    write(dest/'training_receipt.json',dict(status='LOCAL_PRESENCE_TRAINED',TRAIN=list(TRAIN),G=213,K=payload['K'],
        rows=len(x),positive_rows=813,columns=157,new_trees=96,frozen_heads_sha256=frozen,
        model_sha256=digest(dest/'whole_joint_model.pkl'),threshold=.5,trace=trace,DEV_reads=0,STEP_reads=0,
        reload_exact=True,seconds=time.perf_counter()-start,full_contract_pass=False))
    print('LOCAL_TRAIN_COMPLETE',round(time.perf_counter()-start,2),flush=True)


def infer(dest,key,whole_run='run380',local_run='run383'):
    from weld_pipeline.sjg_joint_graph_capacity_1105 import ContinuousGraphBank
    from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices
    from weld_pipeline.sjg_joint_forest_train_1109 import attributes
    start=time.perf_counter();r=read(dest/'training_receipt.json');assert digest(dest/'whole_joint_model.pkl')==r['model_sha256']
    with (dest/'whole_joint_model.pkl').open('rb') as f:payload=pickle.load(f)
    models=payload['models'];raw,local,lengths,case=inputs(key,whole_run,local_run)
    scores=np.empty(len(raw));counts=np.empty(len(raw),np.int32);changes=0;maxdelta=0.;line_rows=0
    for first in range(0,len(raw),4096):
        data=raw[first:first+4096];ll=lengths[first:first+len(data)];lc=local[first:first+len(data)]
        pp=models['presence'].predict_proba(features(data,lc,ll))[:,1]
        scores[first:first+len(data)]=pp;counts[first:first+len(data)]=models['count'].predict(invariant_features(data,ll))
        line=data[:,97]==1
        if line.any():
            rev=reverse_line_descriptors(data[line].astype(float),ll[line])
            rp=models['presence'].predict_proba(features(rev,lc[line],ll[line]))[:,1]
            changes+=int(np.sum((rp>=.5)!=(pp[line]>=.5)));maxdelta=max(maxdelta,float(np.max(np.abs(rp-pp[line]))));line_rows+=int(line.sum())
    selected=np.flatnonzero(scores>=.5);choices=[];rows=[];ordinals=[]
    for row in selected:
        for k in range(1,int(counts[row])+1):rows.append(int(row));ordinals.append(k)
    for first in range(0,len(rows),4096):
        rr=rows[first:first+4096];kk=np.array(ordinals[first:first+4096])
        xx=np.column_stack((raw[rr],kk/payload['K'])).astype(np.float32)
        cats=models['category'].predict(xx);ratios=models['endpoint'].predict(xx)
        for row,cat,ratio in zip(rr,cats,ratios):
            attr=attributes(cat)
            choices.append(dict(query=len(choices),probability=float(scores[row]),child_count=1,closed=attr['closed'],
                children=[dict(profile=int(row)//2,intervals=[0,0],fractions=ratio.tolist(),side='UNKNOWN',connection='UNKNOWN',**attr)]))
    decoded=decode_choices(choices,ContinuousGraphBank(case['graph']));folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False)
    decoded.update(runtime_model=SCHEMA,scope='RAW_FRONTEND_PROPOSAL_POOL_NOT_FINAL_SELECTED_OUTPUT',GT_reads=0,STEP_reads=0)
    write(folder/'decoded.json',decoded);np.savez(folder/'presence_scores.npz',scores=scores,selected_rows=selected)
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION',source=key,
        decoded_sha256=digest(folder/'decoded.json'),model_sha256=r['model_sha256'],admitted_profile_rows=len(selected),
        proposed_slots=len(choices),line_rows_checked=line_rows,line_reversal_presence_decision_changes=changes,
        line_reversal_max_probability_difference=maxdelta,GT_reads=0,STEP_reads=0,seconds=time.perf_counter()-start))
    raw._mmap.close();local._mmap.close()
    print('LOCAL_PREDICT',key,'CARRIERS',len(selected),'SLOTS',len(choices),'REVERSE_CHANGES',changes,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source');a=ap.parse_args()
    dest=OUT/a.run
    if a.stage=='train':
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(schema=SCHEMA,TRAIN=list(TRAIN),DEV=['64'],
            change='Presence input only: local transverse support 93 + symmetric geometry/support 64. Frozen count/category/endpoints from382.',
            source_sha256=digest(__file__),threshold=.5,trees=96,best_preserved='run379',external_consultation_paused=True))
    try:train(dest) if a.stage=='train' else infer(dest,a.source)
    except BaseException:write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
