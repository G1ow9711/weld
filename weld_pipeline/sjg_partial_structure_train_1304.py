"""Same fresh-zero 80 updates as run450, adding only finite-pool set loss."""
import gc,pickle,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_unified_interval_math_1287 import loss_gradient,adam_update
from weld_pipeline.sjg_finite_structured_loss_1298 import structured_score_gradient
from weld_pipeline.sjg_partial_structure_pool_1303 import DEST,TRAIN,load_piece,failed62_scores
from weld_pipeline.sjg_partial_negative_structure_1302 import deletion_score_gradient


def train():
    begin=time.perf_counter();theta=np.zeros((4,605),np.float64)
    state=dict(step=0,m=np.zeros_like(theta),v=np.zeros_like(theta));trace=[];memories={};bindings={}
    for key in TRAIN:
        folder=DEST/f'workset_{key}';bindings[key]=digest(folder/'structure_receipt.json')
        original=read(OUT/f'run450/workset_{key}/receipt.json')
        for name,field in [('features.npy','features_sha256'),('base_margins.npy','base_sha256'),
                           ('indices_and_masks.npz','masks_sha256'),('parents.json','parents_sha256')]:assert digest(folder/name)==original[field]
    assert sum(read(DEST/f'workset_{k}/structure_receipt.json')['G'] for k in TRAIN)==269
    assert read(DEST/'observed62_partial_control.json')['status']=='PASS'
    assert read(DEST/'new_partial_controls.json')['status']=='PASS'
    for epoch in range(16):
        for piece,key in enumerate(TRAIN):
            d=load_piece(key);folder=DEST/f'workset_{key}'
            if key not in memories:memories[key]=d['memory']
            with np.load(folder/'indices_and_masks.npz') as z:a={k:z[k] for k in z.files}
            parent=read(folder/'parents.json');G=parent['G']
            fixed=np.unique(np.r_[a['positive_rows'],a['raw_bad'][:,0]])
            ids=np.r_[fixed,a['sampled'][epoch]]
            first={int(i):j for j,i in reversed(list(enumerate(ids)))}
            pp=[[[[first[int(i)],int(c)] for i,c in view] for view in p] for p in parent['parents']]
            x=np.asarray(d['x'][ids],np.float64);base=np.asarray(d['base'][ids],np.float64)
            original,gradient,meta=loss_gradient(theta,x,base,pp,a['negative'][ids],a['pairs'][ids,0],G)
            score=d['base'].astype(np.float64)+d['x']@theta.T
            structure,dscore,details=structured_score_gradient(score,d['pairs'],d['records'],memories[key])
            deletion,delete_gradient,delete_meta=deletion_score_gradient(score,d['pairs'],d['records'])
            assert delete_meta['fixed_known_wrong_carrier_denominator']==d['receipt']['known_wrong_carriers']
            gradient+=(dscore+delete_gradient).T@d['x'];theta,state,gn=adam_update(theta,gradient,state)
            row=dict(epoch=epoch+1,piece=key,update=state['step'],original_loss=original,structure_loss=structure,deletion_loss=deletion,known_wrong_carrier_denominator=delete_meta['fixed_known_wrong_carrier_denominator'],
                loss=original+structure+deletion,gradient_norm=gn,G=G,workset_rows=len(ids),pool_rows=len(d['pairs']),
                wrong_DP_carriers=sum(v['current_delta']>0 for v in details),
                eligible_carriers=len(details),maximum_structure_hinge=max((v['loss'] for v in details),default=0.),
                seconds=time.perf_counter()-begin)
            trace.append(row)
            atomic_binary_write(DEST/'latest_training_state.pkl',lambda f:pickle.dump(dict(theta=theta,adam=state,
                trace=trace,next_position=[state['step']//5,state['step']%5],workset_bindings=bindings,
                competitor_memory=memories,precompiled_sampling_rng=parent['rng_states'][epoch],
                sampling_precompiled_fixed=True),f,protocol=5))
            write(DEST/'training_progress.json',dict(completed_updates=state['step'],total_updates=80,trace=trace))
            print('EPOCH',epoch+1,'/16','PIECE',piece+1,'/5',key,'UPDATE',state['step'],'/80',
                  'ORIGINAL_LOSS',round(original,6),'STRUCTURE_LOSS',round(structure,6),
                  'DELETE_LOSS',round(deletion,6),'WRONG_POOL_SETS',row['wrong_DP_carriers'],flush=True)
            del d,a,x,base,score,dscore,delete_gradient,gradient;gc.collect();check_resources()
    atomic_binary_write(DEST/'unified_model.npz',lambda f:np.savez(f,theta=theta))
    with np.load(DEST/'unified_model.npz') as z:assert np.array_equal(z['theta'],theta)
    final=[]
    for key in TRAIN:
        d=load_piece(key);scores=d['base'].astype(np.float64)+d['x']@theta.T
        value,_,details=structured_score_gradient(scores,d['pairs'],d['records'],memories[key],False)
        delete_value,_,delete_meta=deletion_score_gradient(scores,d['pairs'],d['records'])
        failed62_scores(d,scores)
        write(DEST/f'final_pool_{key}.json',dict(source=key,structure_loss=value,deletion_loss=delete_value,deletion_details=delete_meta,carriers=details,
             finite_pool_not_full_domain=True,fit_calls=0))
        final.append(dict(source=key,structure_loss=value,deletion_loss=delete_value,wrong_DP_carriers=sum(v['current_delta']>0 for v in details)))
        del d,scores;gc.collect()
    write(DEST/'training_receipt.json',dict(status='80_FIXED_NUMPY_ADAM_UPDATES_COMPLETE',TRAIN=list(TRAIN),G=269,
        updates=80,epochs=16,learning_rate=.01,coupled_weight_decay=.0001,clip_norm=5.,final_update_only=True,
        model_sha256=digest(DEST/'unified_model.npz'),reload_exact=True,DEV_reads=0,STEP_reads=0,
        feature_dim=605,fresh_zero_theta_and_Adam=True,structure_coefficient=1.,deletion_coefficient=1.,original452_carrier_weights_frozen=True,workset_bindings=bindings,
        final_pool=final,full_contract_pass=False,seconds=time.perf_counter()-begin))


if __name__=='__main__':train()
