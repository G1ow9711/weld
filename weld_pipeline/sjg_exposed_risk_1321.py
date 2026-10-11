"""Add a fixed per-carrier sum of exposed errors to the unchanged run456 risk."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest
from weld_pipeline.sjg_owned_continuation_watchdog_1295 import atomic_json
from weld_pipeline.sjg_full_pool_soft_margin_1313 import block_loss
DEST=OUT/'run458'

def attach(p,anchor,scale):
    folder=DEST/f"mined_{p['key']}";receipt=read(folder/'receipt.json')
    for f,h in receipt['files'].items():assert digest(folder/f)==h
    with np.load(folder/'exposed_masks.npz') as z:old=z['old_negative'];new=z['new_negative']
    C=receipt['known_negative_carriers'];assert C>0 and old.shape==p['wn'].shape
    p['original_wn']=p['wn'].copy();p['exposed_mask']=old
    p['wn']=p['wn']+old*(.1/C)
    newbase=np.load(folder/'exposed_base.npy',mmap_mode='r');newx=np.load(folder/'exposed_features.npy',mmap_mode='r')
    scores=np.empty(newbase.shape)
    for a in range(0,len(scores),4096):scores[a:a+4096]=newbase[a:a+4096].astype(float)+newx[a:a+4096].astype(float)@anchor.T
    newbase._mmap.close();newx._mmap.close()
    return dict(key=p['key']+'_exposed',folder=folder,n=len(new),wp=np.zeros(new.shape),wn=new*(.1/C),
        original_wn=np.zeros(new.shape),exposed_mask=new,anchor_score=scores,
        file_pairs=[('exposed_features.npy','exposed_base.npy')]),dict(source=p['key'],Cminus=C,
        old_exposed=int(old.sum()),new_exposed=int(new.sum()),coefficient_per_exposed_error=.1/C,
        no_division_by_mined_error_count=True,original_risk_weights_preserved=True)

def controls(anchor,initial,scale,samples):
    rng=np.random.default_rng(457);x=rng.normal(size=(11,605));base=rng.normal(size=(11,4));a=base+x@anchor.T
    wp=np.zeros((11,4));wn=np.zeros((11,4));wn[0,0]=wn[1,0]=.1/7
    beta=rng.normal(size=(4,605))*.01
    loss,grad,score=block_loss(x,a,wp,wn,beta,scale)
    expected=.1/7*sum(.5*max(0,1+score[i,0])**2 for i in (0,1));assert abs(loss-expected)<1e-10
    total=0.;g=np.zeros_like(beta)
    for i in range(11):
        v,vg,_=block_loss(x[i:i+1],a[i:i+1],wp[i:i+1],wn[i:i+1],beta,scale);total+=v;g+=vg
    assert abs(total-loss)<1e-10 and abs(g-grad).max()<1e-10
    errors=[]
    for k in rng.choice(beta.size,12,replace=False):
        plus=beta.copy();minus=beta.copy();plus.flat[k]+=1e-5;minus.flat[k]-=1e-5
        v=(block_loss(x,a,wp,wn,plus,scale)[0]-block_loss(x,a,wp,wn,minus,scale)[0])/2e-5
        errors.append(abs(v-grad.flat[k]));assert errors[-1]<2e-5*max(1,abs(v))
    empty=block_loss(x,a,wp,np.zeros_like(wn),beta,scale);assert empty[0]==0 and not empty[1].any()
    # Each error is penalized separately; a negative error cannot cancel a positive one.
    independent=.5*np.maximum(0,1+np.array([-100.,2.]))**2;assert independent.sum()==4.5
    beta0=(initial-anchor)*scale;maximum=0.
    for sx,sa,sp,sn,sbase in samples:
        actual=sa+(sx/scale)@beta0.T;direct=sbase+sx@initial.T
        maximum=max(maximum,float(abs(actual-direct).max(initial=0.)))
    assert maximum<1e-9
    result=dict(status='PASS',dense_stream_loss_error=abs(total-loss),dense_stream_gradient_error=float(abs(g-grad).max()),
        finite_difference_max_error=max(errors),empty_and_unknown_mask_gradient_zero=True,
        separate_error_hinges_do_not_cancel=True,initial456_export_max_error=maximum,
        actual_native_BM_mask_control='mined_BM79A_native_local/BM_previous_raw_coverage_control.json')
    atomic_json(DEST/'controls.json',result)
