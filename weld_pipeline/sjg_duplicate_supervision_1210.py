"""TRAIN-only pair matching labels, exact-input contradictions, head controls."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_pair_residual_1203 import ResidualScorer,SHAPES as RESIDUAL_SHAPES
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateScorer,DuplicateAdam,DuplicatePool,reverse_column,pack,unpack
from weld_pipeline.sjg_prefix_minibatch_1189 import restore_final


def initialize(dest):
    base,adam560,mean,scale,original=restore_final(OUT/'run367',expected_step=560)
    old=OUT/'run373';training=read(old/'selector_training.json');resume=read(old/'selector_resume_checkpoint.json')
    assert training['updates']==resume['updates']==112 and resume['next_position'] is None
    assert len(resume['trace'])==112 and [v['step'] for v in resume['trace']]==list(range(1,113))
    assert training['base_Adam_step']==resume['base_Adam_step']==560
    assert digest(old/'selector.npz')==training['selector_sha256']
    assert digest(old/'selector_resume_checkpoint.npz')==resume['checkpoint_sha256']
    assert digest(old/'selector_normalization.npz')==training['normalization_sha256']==digest(OUT/'run367/selector_normalization.npz')
    assert digest(old/'base_selector.npz')==digest(OUT/'run367/selector.npz')
    assert training['bindings']==resume['bindings']==original['bindings']
    for key,value in original['bindings'].items():
        assert value==dict(candidate_sha256=digest(dest/f'source_{key}/decoded.json'),compatibility_sha256=digest(dest/f'source_{key}/compatibility.json'))
    frozen=ResidualScorer.load(old/'selector.npz',old/'base_selector.npz')
    with np.load(old/'selector_resume_checkpoint.npz') as z:
        assert set(z.files)=={f'{prefix}_{key}' for prefix in ('parameter','first','second') for key in RESIDUAL_SHAPES}|{'updates','mean','scale'}
        assert int(z['updates'])==112 and np.array_equal(z['mean'],mean) and np.array_equal(z['scale'],scale)
        for key,shape in RESIDUAL_SHAPES.items():
            assert np.array_equal(z[f'parameter_{key}'],frozen.parameters[key])
            for prefix in ('first','second'):
                v=z[f'{prefix}_{key}'];assert v.shape==shape and np.isfinite(v).all()
                if prefix=='second':assert np.all(v>=0)
    receipt=dict(original,frozen_residual_sha256=training['selector_sha256'],
        frozen_residual_checkpoint_sha256=resume['checkpoint_sha256'],frozen_residual_Adam_step=112)
    head=DuplicateScorer(frozen,training['selector_sha256'])
    return head,DuplicateAdam(head),mean,scale,receipt


def pair_labels(edges,index,known=None):
    """Exact two-left-vertex maximum matching via its Hall condition.

    If both edge sets are nonempty, matching size is 1 iff their union has
    size 1; otherwise it is 2. Sharing one of multiple GTs is not redundant.
    """
    n=len(edges);labels=np.full(n,-1,np.int8)
    supervised=np.ones(n,bool) if known is None else np.asarray(known,bool)
    assert supervised.shape==(n,)
    if not supervised[index] or not edges[index]:return labels
    right=set(edges[index])
    for i,left in enumerate(edges):
        if i!=index and supervised[i] and left:labels[i]=int(len(set(left)|right)==1)
    return labels


class PairSupervision:
    def __init__(self,dest,keys):
        self.dest=dest;self.edges={k:read(dest/f'source_{k}/compatibility.json')['compatible_targets'] for k in keys}
        self.labels={};self.seen={};self.labelled_pairs=0;self.conflict_found=False

    def column(self,key,index,a,b):
        slot=(key,index)
        if slot in self.labels:return self.labels[slot]
        labels=pair_labels(self.edges[key],index)
        for p in np.flatnonzero(labels>=0):
            left=a[p].copy();right=b[p].copy();left[left==0]=0.;right[right==0]=0.
            aa=left.tobytes();bb=right.tobytes();code=aa+bb if aa<=bb else bb+aa
            label=int(labels[p]);witness=dict(source=key,candidate=int(p),selected=int(index),label=label)
            previous=self.seen.get(code)
            if previous is not None and previous['label']!=label:
                self.conflict_found=True
                write(self.dest/'pair_input_label_conflict.json',dict(status='EXACT_SYMMETRIC_INPUT_OPPOSITE_CONFIRMED_LABELS',
                    previous=previous,current=witness,input_forward=left.tolist(),input_reverse=right.tolist(),
                    action='STOP_HEAD_FIT_NO_MASKING_NO_ID_FEATURE',not_physical_truth=True))
                raise RuntimeError('Exact pair-input label contradiction: stop head fitting')
            if previous is None:self.seen[code]=witness
            self.labelled_pairs+=1
        self.labels[slot]=labels
        return labels

    def receipt(self):return dict(status='CONFLICT_STOPPED' if self.conflict_found else 'NO_CONTRADICTION_IN_CONSTRUCTED_SUPERVISED_COLUMNS',
        columns=len(self.labels),labelled_directional_pairs_checked=self.labelled_pairs,
        unique_symmetric_inputs=len(self.seen),unknown_pairs_masked_only_in_auxiliary=True,
        full_pair_capacity_or_generalization_not_proven=True)


def head_control():
    from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
    rng=np.random.Generator(np.random.PCG64(1210));n=8
    model=DuplicateScorer(ResidualScorer(NearestParentScorer(),'software'),'software')
    model.parameters['W2']=rng.normal(0,.03,8);model.parameters['b2'][:]=.2
    raw=rng.normal(0,.3,(n,n,16));selected=[2,0]
    edges=[[0],[0],[1],[0,1],[],[2],[0],[1,2]]
    def reverse(a):
        b=a.copy();b[:,0]=a[:,1];b[:,1]=a[:,0];return b
    available=np.ones(n+1,bool);available[selected]=False;delta=rng.normal(size=n+1)
    def make():
        pool=DuplicatePool(model,n,True)
        for index in selected:pool.add(index,raw[:,index],reverse(raw[:,index]),pair_labels(edges,index))
        return pool
    empty=DuplicatePool(model,n,True);assert np.array_equal(empty.maximum,np.zeros(n))
    p=make();loss,aux,counts=p.auxiliary(available);grad=model.backward(p,delta)
    for k in grad:grad[k]+=aux[k]
    direct_loss=[[],[]];direct_grad=[[],[]]
    for index in selected:
        a=raw[:,index];b=reverse(a);h,j=model.pair(a,b,True)
        assert np.array_equal(h,model.pair(b,a)[0])
        labels=pair_labels(edges,index)
        for i in np.flatnonzero(available[:-1]&(labels>=0)):
            y=int(labels[i]);sign=1.-2*y;v=sign*h[i]
            direct_loss[y].append(float(np.logaddexp(0.,v)))
            direct_grad[y].append(sign*np.exp(-np.logaddexp(0.,-v))*j[i])
    used=[i for i in (0,1) if direct_loss[i]]
    assert counts==[len(v) for v in direct_loss]
    assert abs(loss-np.mean([np.mean(direct_loss[i]) for i in used]))<1e-14
    assert np.max(np.abs(pack(aux)-np.mean([np.mean(direct_grad[i],axis=0) for i in used],axis=0)))<1e-14
    def objective():
        q=make();return float(-q.maximum@delta[:-1]+q.auxiliary(available)[0])
    errors={};eps=1e-5
    for key,value in model.parameters.items():
        error=0.
        for index in np.ndindex(value.shape):
            old=value[index];value[index]=old+eps;plus=objective();value[index]=old-eps;minus=objective();value[index]=old
            error=max(error,abs((plus-minus)/(2*eps)-grad[key][index]))
        errors[key]=error;assert error<5e-8,(key,error)
    tie=DuplicatePool(model,n,True);a=raw[:,0];b=reverse(a);tie.add(3,a,b);tie.add(1,a,b)
    assert np.all(tie.argmax==1)
    # Verify the exact two-vertex formula against the original oracle, including
    # multi-compatible and entirely unknown pairs, not a shared-GT shortcut.
    for s in range(n):
        labels=pair_labels(edges,s)
        for candidate in range(n):
            if candidate==s or not edges[candidate] or not edges[s]:assert labels[candidate]==-1;continue
            oracle=ParentSetOracle(edges,3);oracle.select(s);oracle.select(candidate)
            assert labels[candidate]==2-oracle.match_count
    assert pair_labels([[0],[0]],0)[1]==1
    assert pair_labels([[0],[1]],0)[1]==0
    assert pair_labels([[0,1],[0,1]],0)[1]==0
    hall=ParentSetOracle([[0],[1],[0,1]],2);hall.select(0);hall.select(1)
    assert not hall.action_masks()[1][2]
    assert pair_labels(hall.edges,0)[2]==pair_labels(hall.edges,1)[2]==0
    assert pair_labels([[0],[0]],0,known=np.array([True,False]))[1]==-1
    return dict(status='SYMMETRIC_DUPLICATE_HEAD_CONTROL_PASSED',trainable_parameters=145,
        finite_difference_max_errors=errors,incremental_auxiliary_equals_all_pair_recomputation=True,
        all_labelled_pairs_used=True,symmetric_score_exact=True,empty_set_zero=True,
        maximum_ties_lowest_candidate_index=True,two_vertex_matching_labels_verified=True,
        shared_multi_GT_not_duplicate=True,pairwise_coexistence_does_not_replace_set_oracle=True,
        unknown_pair_labels_masked_only=True,optimizer_updates=0)
