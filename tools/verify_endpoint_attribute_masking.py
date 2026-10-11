"""Real compiled labels: unknown NaNs and optional V fields cannot affect loss."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import read,write
from weld_pipeline.server_endpoint_attribute_supervision import masked_endpoint_loss
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1/endpoint_attribute_supervision_v1';torch.set_num_threads(2);parts=[]
for r in read(BASE/'summary.json')['cases']:
 with np.load(BASE/r['key']/'labels.npz') as z:parts.append({k:z[k] for k in z.files})
data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};state=torch.tensor(data['state'],dtype=torch.long);mm=torch.tensor(data['exact_mm']);sm=torch.tensor(data['state_mask']);nm=torch.tensor(data['numeric_mask']);torch.manual_seed(1031);logits=torch.randn(*state.shape,4,requires_grad=True);values=torch.randn(*state.shape,dtype=torch.float64,requires_grad=True);a,b,counts=masked_endpoint_loss(logits,values,state,mm,sm,nm);(a+b).backward();assert torch.all(logits.grad[~sm]==0) and torch.all(values.grad[~nm]==0)
changed_logits=logits.detach().clone();changed_values=values.detach().clone();changed_logits[~sm]=float('nan');changed_values[~nm]=float('nan');c,d,_=masked_endpoint_loss(changed_logits,changed_values,state,mm,sm,nm);assert torch.equal(a,c) and torch.equal(b,d)
# All-unlabeled batch must have finite zero losses, not NaN * zero.
l=torch.full((2,8,4),float('nan'),requires_grad=True);v=torch.full((2,8),float('nan'),requires_grad=True);unknown=torch.full((2,8),-1,dtype=torch.long);missing=torch.full((2,8),float('nan'));mask=torch.zeros((2,8),dtype=torch.bool);x,y,_=masked_endpoint_loss(l,v,unknown,missing,mask,mask);assert x==0 and y==0;(x+y).backward();assert torch.all(l.grad==0) and torch.all(v.grad==0)
write(BASE/'loss_masking_controls.json',dict(status='PASS',children=len(state),**counts,unknown_and_nonrequired_gradients_zero=True,unknown_prediction_nan_does_not_change_loss=True,all_unknown_batch_zero_loss_and_gradient=True,real_label_states_used=True,model_fit_calls=0));print(counts,flush=True)
