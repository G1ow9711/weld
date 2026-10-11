"""Query the original frozen model's actual score blocks, including its clamps.

No duplicate formula for span/occupancy is maintained here. Positive native
double-precision intervals can collapse in float32; the original forward owns
that numerical convention and must also own these cached training margins.
"""
import numpy as np

def query_encoded(model,h,q,p,length,closed,ii,jj,chunk=64):
    import torch
    ii=np.asarray(ii,np.int64);jj=np.asarray(jj,np.int64);n=len(q)
    assert ii.shape==jj.shape and np.all(ii>=0) and np.all(ii<jj) and np.all(jj<n)
    result=np.empty((len(ii),len(model.classes)-1),np.float32)
    with torch.no_grad():
        for block in np.unique(ii//chunk):
            lo=int(block)*chunk;hi=min(lo+chunk,n-1);pick=np.flatnonzero(ii//chunk==block)
            full=model.scores(h[:n],h[n:],q,p,length,closed,lo,hi)
            a=torch.as_tensor(ii[pick]-lo,device=q.device);b=torch.as_tensor(jj[pick],device=q.device)
            original=full[a,b];result[pick]=(original[:,1:]-original[:,:1]).cpu().numpy()
    if not np.isfinite(result).all():raise FloatingPointError('Original frozen score query produced a nonfinite margin')
    return result

def margins(rt,r,ii,jj):
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import tensors,CHUNK
    x,c,q,p=tensors(rt['s'],r,rt['payload']['norm'],rt['device'])
    with torch.no_grad():h=rt['model'].encode(x,c)
    return query_encoded(rt['model'],h,q,p,r['curve_length_mm'],r['is_closed'],ii,jj,CHUNK)

def controls():
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import PairModel
    torch.set_num_threads(1);torch.manual_seed(1292);model=PairModel([0,1,2,3,4]).eval()
    native=np.array([0.,500.,500.+1e-8,500.+1e-7,501.],np.float64)
    q=torch.tensor(native,dtype=torch.float32);p=q*.5;n=len(q)
    assert np.all(np.diff(native)>0) and q[1]==q[2]==q[3]
    x=torch.randn(2*n,180);context=torch.randn(160);ii,jj=np.triu_indices(n,1)
    with torch.no_grad():
        h=model.encode(x,context);full=model.scores(h[:n],h[n:],q,p,float(native[-1]),False,0,n-1)
        reference=(full[ii,jj,1:]-full[ii,jj,:1]).numpy()
        ti=torch.tensor(ii,dtype=torch.long);tj=torch.tensor(jj,dtype=torch.long)
        broken=(p[tj]-p[ti])/(q[tj]-q[ti])
    assert not torch.isfinite(broken).all() and np.isfinite(reference).all()
    maximum=0.
    for chunk in (1,2,64):
        actual=query_encoded(model,h,q,p,float(native[-1]),False,ii,jj,chunk)
        maximum=max(maximum,float(abs(actual-reference).max()));assert np.allclose(actual,reference,rtol=0,atol=1e-5)
    return dict(status='PASS',positive_native_intervals_retained=True,float32_collapsed_span_fixture=True,
        reproduces_old_missing_clamp_nonfinite=True,original_forward_clamp_used_without_formula_duplication=True,
        original_full_vs_query_max_abs=maximum,all_queried_margins_finite=True,fit_calls=0,
        original_frozen_model_or_contract_gates_changed=False)

if __name__=='__main__':
    from weld_pipeline.sjg_local_runtime_904 import OUT,write
    result=controls();write(OUT/'run449/score_query_repair_controls.json',result);print(result,flush=True)
