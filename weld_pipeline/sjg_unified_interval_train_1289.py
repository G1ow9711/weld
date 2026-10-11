"""Fixed 80 CPU NumPy Adam updates, one complete workpiece per update."""
import gc,pickle,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_unified_interval_math_1287 import controls,loss_gradient,adam_update

TRAIN=('21','BM79A_native_local','10','25','36');DEST=OUT/'run449'

def train():
    begin=time.perf_counter();write(DEST/'numerical_controls.json',controls());check_resources()
    data={};total=0
    for key in TRAIN:
        d=DEST/f'workset_{key}';r=read(d/'receipt.json');total+=r['G']
        for name,field in [('features.npy','features_sha256'),('base_margins.npy','base_sha256'),
                           ('indices_and_masks.npz','masks_sha256'),('parents.json','parents_sha256')]:assert digest(d/name)==r[field]
        assert read(d/'complete_parent_witnesses.json')['status']=='ALL_COMPLETE_PARENT_NATIVE_WITNESSES_D3_VERIFIED'
        with np.load(d/'indices_and_masks.npz') as z:arrays={k:z[k] for k in z.files}
        arrays.update(x=np.load(d/'features.npy',mmap_mode='r'),base=np.load(d/'base_margins.npy',mmap_mode='r'),
            parent=read(d/'parents.json'),receipt=r);data[key]=arrays
        if not np.isfinite(arrays['base']).all():
            raise FloatingPointError('Nonfinite frozen-base cache; repair compiler cache before any optimizer update: '+key)
    assert total==269
    theta=np.zeros((4,605),np.float64);state=dict(step=0,m=np.zeros_like(theta),v=np.zeros_like(theta));trace=[]
    for epoch in range(16):
        for piece,key in enumerate(TRAIN):
            d=data[key];parents=d['parent']['parents'];G=d['parent']['G']
            fixed=np.unique(np.r_[d['positive_rows'],d['raw_bad'][:,0]])
            ids=np.r_[fixed,d['sampled'][epoch]] # repeated uniform draws retain their mass
            first={int(i):j for j,i in reversed(list(enumerate(ids)))}
            pp=[[[[first[int(i)],int(c)] for i,c in view] for view in parent] for parent in parents]
            x=np.asarray(d['x'][ids],np.float64);base=np.asarray(d['base'][ids],np.float64)
            negative=d['negative'][ids];carriers=d['pairs'][ids,0]
            value,gradient,meta=loss_gradient(theta,x,base,pp,negative,carriers,G)
            theta,state,gn=adam_update(theta,gradient,state)
            classes=np.zeros(4,int)
            for parent,selected in zip(pp,meta['selected_whole_parent_views']):
                for i,c in parent[selected]:classes[c]+=1
            row=dict(epoch=epoch+1,piece=key,update=state['step'],loss=value,gradient_norm=gn,
                G=G,workset_rows=len(ids),original_child_mass=1.,positive_class_mass=(classes/G).tolist(),
                negative_class_mass=(negative.sum(0)/max(1,negative.sum())).tolist(),
                seconds=time.perf_counter()-begin)
            trace.append(row)
            atomic_binary_write(DEST/'latest_training_state.pkl',lambda f:pickle.dump(dict(theta=theta,adam=state,
                trace=trace,next_position=[(state['step']//5)+1,state['step']%5],
                precompiled_sampling_rng=d['parent']['rng_states'][epoch],sampling_precompiled_fixed=True),f,protocol=5))
            write(DEST/'training_progress.json',dict(completed_updates=state['step'],total_updates=80,trace=trace))
            print('EPOCH',epoch+1,'/16','PIECE',piece+1,'/5',key,'UPDATE',state['step'],'/80','LOSS',round(value,6),flush=True)
            del x,base,gradient;gc.collect();check_resources()
    assert state['step']==80 and np.isfinite(theta).all()
    atomic_binary_write(DEST/'unified_model.npz',lambda f:np.savez(f,theta=theta))
    with np.load(DEST/'unified_model.npz') as z:assert np.array_equal(z['theta'],theta)
    write(DEST/'training_receipt.json',dict(status='80_FIXED_NUMPY_ADAM_UPDATES_COMPLETE',TRAIN=list(TRAIN),G=269,
        updates=80,epochs=16,learning_rate=.01,coupled_weight_decay=.0001,clip_norm=5.,final_update_only=True,
        model_sha256=digest(DEST/'unified_model.npz'),reload_exact=True,DEV_reads=0,STEP_reads=0,
        feature_dim=605,full_contract_pass=False,seconds=time.perf_counter()-begin))

if __name__=='__main__':train()
