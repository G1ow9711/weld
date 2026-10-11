"""Pro377: observed frozen-policy STOP envelopes and full-data hard risk loss."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_incompatibility_head_1216 import IncompatibilityScorer,candidate_loss
from weld_pipeline.sjg_incompatibility_continuation_1219 import restore
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateRelations
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_synchronous_pool_1200 import ReadOnlyColumns
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


def risk_loss(z,labels,envelope):
    z=np.asarray(z,np.float64);labels=np.asarray(labels,np.int8);envelope=np.asarray(envelope,np.float64)
    assert z.shape==labels.shape==envelope.shape and np.isfinite(envelope).all() and np.all(envelope>=0)
    base,gradient=candidate_loss(z,labels)
    groups=[(y,np.flatnonzero(labels==y)) for y in (0,1) if np.any(labels==y)]
    risk=0.;hard={}
    for y,ids in groups:
        argument=z[ids] if y==0 else envelope[ids]-z[ids]
        q=np.logaddexp(0.,argument);local=int(np.argmax(q));index=int(ids[local]);weight=1./len(groups)
        risk+=weight*q[local];sign=1. if y==0 else -1.
        gradient[index]+=weight*sign*np.exp(-np.logaddexp(0.,-argument[local]))
        hard[str(y)]=index
    assert np.isfinite(risk) and np.isfinite(gradient).all()
    return float(base+risk),gradient,dict(base_loss=base,risk_loss=float(risk),hard_candidate_indices=hard)


def loss_control():
    z=np.array([-1.,.25,1.2,-.8,.5,-3.]);y=np.array([1,1,0,0,-1,-1]);b=np.array([.5,2.,0.,0.,8.,4.])
    value,g,detail=risk_loss(z,y,b);eps=1e-5;errors=[]
    assert detail['hard_candidate_indices']=={'0':2,'1':1}
    assert np.all(g[y==0]>0) and np.all(g[y==1]<0) and np.all(g[y<0]==0)
    for i in range(len(z)):
        a=z.copy();c=z.copy();a[i]+=eps;c[i]-=eps
        errors.append(abs((risk_loss(a,y,b)[0]-risk_loss(c,y,b)[0])/(2*eps)-g[i]))
    assert max(errors)<5e-8
    assert risk_loss(z[y>=0],y[y>=0],b[y>=0])[0]==value
    for logit,bound in ((-1.,.5),(.25,2.)):
        _,derivative,_=risk_loss(np.array([logit]),np.array([1]),np.array([bound]))
        assert derivative[0]<0
    tie_z=np.zeros(4);tie_y=np.array([0,0,1,1]);tie_b=np.zeros(4)
    tie_value,tie_gradient,tie_details=risk_loss(tie_z,tie_y,tie_b)
    base_gradient=candidate_loss(tie_z,tie_y)[1]
    assert tie_details['hard_candidate_indices']=={'0':0,'1':2}
    assert np.array_equal(tie_gradient-base_gradient,np.array([.25,0.,-.25,0.]))
    for direction in np.eye(4).tolist()+(-np.eye(4)).tolist()+[[1.,-2.,3.,-.5]]:
        d=1e-4*np.array(direction)
        assert risk_loss(tie_z+d,tie_y,tie_b)[0]>=tie_value+float(tie_gradient@d)-1e-12
    return dict(status='NEW_STOP_RISK_LOSS_CONTROL_PASSED',logit_finite_difference_max_error=max(errors),
        negative_incompatible_logit_direct_gradient=True,positive_but_below_envelope_direct_gradient=True,
        compatible_overdeduction_penalized=True,unknown_auxiliary_only_mask=True,
        exact_tie_lowest_original_candidate_index=True,tie_legal_subgradient_checked=True,optimizer_updates=0)


def compile_one(dest,key,model,mean,scale,cache):
    """No target/compatibility reads; compile using only the frozen predictor."""
    folder=dest/f'source_{key}';receipt=read(folder/'receipt.json')
    assert digest(folder/'unary.npy')==receipt['unary_sha256'] and digest(folder/'decoded.json')==receipt['decoded_sha256']
    u=(np.load(folder/'unary.npy')-mean)/scale
    parents,points=load_parents(folder);metadata=read(folder/'parent_metadata.json')
    context=DuplicateRelations(parents,metadata,model.frozen,key,cache);z,_=model.logits(u)
    envelope=np.zeros(len(parents));visits=np.zeros(len(parents),np.int64);choices=[];explicit=[];reference_control=None
    saved_path=OUT/f'run376/test_{key}/final_prediction.json';saved=read(saved_path) if key in ('21','BM79A_native_local','10') else None
    if saved is not None:assert saved['selector_sha256']==digest(OUT/'run376/selector.npz')
    reference_context=None
    try:
        for prefix in range(len(parents)+1):
            check_resources();features=context.features();base_scores,_=model.frozen.forward(u,features,context.selected)
            scores=base_scores.copy();scores[:-1]-=np.maximum(0.,z)
            available=np.ones(len(parents),bool);available[context.selected]=False
            b=base_scores[:-1]-base_scores[-1];assert np.isfinite(b).all()
            envelope[available]=np.maximum(envelope[available],b[available]);visits[available]+=1
            if key=='21':explicit.append((b.copy(),available.copy()))
            if key=='10' and prefix==1:
                old=OUT/'run376';reference=IncompatibilityScorer.load(old/'selector.npz',old/'frozen_duplicate.npz',old/'frozen_residual.npz',old/'base_selector.npz')
                reference_context=DuplicateRelations(parents,metadata,reference.frozen,key,cache)
                for index in context.selected:reference_context.add(index)
                rz,_=reference.logits(u);rs=reference.forward(u,reference_context.features(),reference_context.selected,rz)
                assert np.array_equal(z,rz) and np.array_equal(scores,rs)
                reference_control=dict(source=key,prefix_length=1,logits_and_all_candidate_STOP_scores_exact=True)
                del reference_context;reference_context=None
            chosen=model.choose_next(scores,context.selected)
            if saved is not None:
                expected=saved['selected_candidate_indices'][prefix] if prefix<len(saved['selected_candidate_indices']) else None
                assert chosen==expected
            if chosen is None:break
            assert available[chosen];choices.append(chosen);context.add(chosen)
        else:raise RuntimeError('Frozen policy failed to terminate')
        assert np.all(visits>=1) and np.isfinite(envelope).all() and np.all(envelope>=0)
        small=None
        if explicit:
            assert len(explicit)==2  # The previously sealed 21 trajectory is tiny.
            matrix=np.array([np.where(mask,b,-np.inf) for b,mask in explicit])
            direct=np.maximum(0.,matrix.max(0));assert np.array_equal(direct,envelope)
            direct_counts=np.array([mask for b,mask in explicit]).sum(0);assert np.array_equal(direct_counts,visits)
            small=dict(source=key,prefixes=len(explicit),candidates=len(envelope),explicit_available_masked_max_bit_exact=True)
        trajectory=dict(source=key,choices=choices,prefixes=len(choices)+1,initial_and_final_STOP_included=True,
            policy='Frozen run376 original argmax and STOP; no oracle correction',GT_reads=0,
            selector_sha256=digest(OUT/'run376/selector.npz'),candidate_sha256=receipt['decoded_sha256'],unary_sha256=receipt['unary_sha256'])
        path=dest/'envelopes'/f'{key}_trajectory.json';write(path,trajectory)
        npy=dest/'envelopes'/f'{key}.npy';atomic_binary_write(npy,lambda stream:np.save(stream,envelope,allow_pickle=False))
        record=dict(source=key,path=str(npy),sha256=digest(npy),trajectory_path=str(path),trajectory_sha256=digest(path),
            policy_sha256=trajectory['selector_sha256'],candidate_sha256=receipt['decoded_sha256'],unary_sha256=receipt['unary_sha256'],
            normalization_sha256=digest(OUT/'run376/selector_normalization.npz'),candidate_count=len(envelope),
            prefix_count=len(choices)+1,minimum_available_observations=int(visits.min()),GT_reads=0,
            not_a_bound_for_unobserved_states=True,saved_actions_replayed=saved is not None)
        write(dest/'envelopes'/f'{key}_binding.json',record)
        print('FROZEN376_OBSERVED_ENVELOPE',key,'CANDIDATES',len(envelope),'PREFIXES',len(choices)+1,flush=True)
        return record,small,reference_control
    finally:
        del context,reference_context,parents
        points._mmap.close()


def controls_and_envelopes(dest):
    model,adam,mean,scale,restored=restore(dest)
    assert digest(ROOT/'weld_pipeline/sjg_incompatibility_head_1216.py')==digest(OUT/'run376/source_snapshot/sjg_incompatibility_head_1216.py')
    cache=ReadOnlyColumns(dest/'envelope_relation_cache',bindings(dest,KEYS))
    reuse_columns(OUT/'run377/diagnostic_relation_cache',cache);(dest/'envelopes').mkdir()
    records=[];small=[];reference=[]
    try:
        for key in KEYS:
            record,check,ref=compile_one(dest,key,model,mean,scale,cache);records.append(record)
            if check:small.append(check)
            if ref:reference.append(ref)
        assert len(records)==7 and len(small)==len(reference)==1 and adam.step_count==112
        write(dest/'envelope_manifest.json',dict(status='SEVEN_FROZEN376_TRAIN_ENVELOPES_SEALED_BEFORE_LABEL_USE',records=records,
            GT_reads_for_policy_generation=0,model_updates=0,observed_trajectory_only=True,not_deployment_input=True,
            full_restored_checkpoint_sha256=restored['head_source_checkpoint_sha256']))
        value=loss_control();value.update(full_state_restored=True,initial_forward=reference,streaming_control=small)
        write(dest/'stop_risk_control.json',value)
        print('ENVELOPES_AND_NEW_RISK_CONTROL_PASSED',value,flush=True)
    finally:write(dest/'envelope_relation_cache_receipt.json',cache.receipt())


def load_envelopes(dest):
    manifest=read(dest/'envelope_manifest.json');assert manifest['status']=='SEVEN_FROZEN376_TRAIN_ENVELOPES_SEALED_BEFORE_LABEL_USE'
    assert [r['source'] for r in manifest['records']]==list(KEYS)
    result={}
    for r in manifest['records']:
        key=r['source'];folder=dest/f'source_{key}'
        assert digest(r['path'])==r['sha256'] and digest(r['trajectory_path'])==r['trajectory_sha256']
        assert digest(OUT/'run376/selector.npz')==r['policy_sha256']
        assert digest(folder/'decoded.json')==r['candidate_sha256'] and digest(folder/'unary.npy')==r['unary_sha256']
        assert digest(OUT/'run376/selector_normalization.npz')==r['normalization_sha256']
        b=np.load(r['path'],allow_pickle=False);assert b.shape==(r['candidate_count'],) and b.dtype==np.float64 and np.isfinite(b).all() and np.all(b>=0)
        b.setflags(write=False);result[key]=b
    return result
